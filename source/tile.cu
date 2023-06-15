#include "tile.h"

#ifndef WINGDIAPI
#define WINGDIAPI
#endif
#ifndef APIENTRY
#define APIENTRY
#endif
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

constexpr unsigned int MAX_TILE_MODIFY_DISTANCE_X = 2;
constexpr unsigned int MAX_TILE_MODIFY_DISTANCE_Y = 1;
constexpr unsigned int MAX_OFFSET_X = MAX_TILE_MODIFY_DISTANCE_X * 2 + 1;
constexpr unsigned int MAX_OFFSET_Y = MAX_TILE_MODIFY_DISTANCE_Y * 2 + 1;
constexpr int BLOCK_SIZE = 16;
const dim3 THREADS{ BLOCK_SIZE, BLOCK_SIZE };
const dim3 BASE_GRID{ (GRID_WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE, (GRID_HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE };
const dim3 UPDATE_GRID{ (BASE_GRID.x + MAX_OFFSET_X - 1) / MAX_OFFSET_X, (BASE_GRID.y + MAX_OFFSET_Y - 1) / MAX_OFFSET_Y };

struct XORRand
{
public:
	__host__ __device__ uint32_t operator()()
	{
		state ^= state << 13;
		state ^= state >> 17;
		state ^= state << 5;

		return state;
	}

	__host__ __device__ void SetState(uint32_t state)
	{
		this->state = state;
	}

private:
	uint32_t state;
};

class TileGrid
{
public:
	__host__ __device__ Tile* Tile(unsigned int x, unsigned int y)
	{
		if ((x >= GRID_WIDTH) | (y >= GRID_HEIGHT))
		{
			return nullptr;
		}
		else
		{
			return &tiles[x + y * GRID_WIDTH];
		}
	}

	__host__ __device__ XORRand& Random(unsigned int x, unsigned int y)
	{
		return rands[x + y * GRID_WIDTH];
	}

	__host__ __device__ unsigned int GetTick()
	{
		return tick;
	}

	__host__ void SetTick(unsigned int t)
	{
		tick = t;
	}

private:
	::Tile tiles[GRID_WIDTH * GRID_HEIGHT];
	XORRand rands[GRID_WIDTH * GRID_HEIGHT];
	unsigned int tick;
};

__device__ void SwapTiles(Tile* __restrict__ a, Tile* __restrict__ b)
{
	Tile copy = *a;
	*a = *b;
	*b = copy;
}

//__device__ void Move(TileGrid& grid, unsigned int x, unsigned int y, int dx, int dy)
//{
//	__builtin_assume(x < GRID_WIDTH);
//	__builtin_assume(y < GRID_HEIGHT);
//	__builtin_assume((dx < 0 ? -dx : dx) <= MAX_TILE_MODIFY_DISTANCE_X);
//	__builtin_assume((dy < 0 ? -dy : dy) <= MAX_TILE_MODIFY_DISTANCE_Y);
//}

typedef void (*TileUpdate)(TileGrid&, unsigned int, unsigned int);
__device__ void AirUpdate(TileGrid& grid, unsigned int x, unsigned int y) { }
__device__ void SandUpdate(TileGrid& grid, unsigned int x, unsigned int y)
{
	Tile* me = grid.Tile(x, y);

	{
		Tile* below = grid.Tile(x, y - 1);

		if (below && below->Density() < me->Density())
		{
			SwapTiles(me, below);

			return;
		}
	}

	if (grid.Random(x, y)() % 2 == 0)
	{
		Tile* downLeft = grid.Tile(x - 1, y - 1);

		if (downLeft && downLeft->Density() < me->Density())
		{
			SwapTiles(me, downLeft);

			return;
		}
	}
	else
	{
		Tile* downRight = grid.Tile(x + 1, y - 1);

		if (downRight && downRight->Density() < me->Density())
		{
			SwapTiles(me, downRight);

			return;
		}
	}
}

__device__ void BasicLiquidUpdate(TileGrid& grid, unsigned int x, unsigned int y)
{
	Tile* me = grid.Tile(x, y);
	Tile* below = grid.Tile(x, y - 1);

	if (below && below->Density() < me->Density())
	{
		SwapTiles(me, below);
	}
	else
	{
		unsigned int r = grid.Random(x, y)();
		Tile* moveTo;
		switch (r % 7)
		{
		case 0:
		case 1:
			moveTo = (grid.Tile(x - 1, y) && grid.Tile(x - 1, y)->Density() < me->Density()) ? grid.Tile(x - 2, y) : nullptr;
			break;

		case 2:
			moveTo = grid.Tile(x - 1, y);
			break;

		case 3:
			moveTo = grid.Tile(x + 1, y);
			break;

		case 4:
		case 5:
			moveTo = (grid.Tile(x + 1, y) && grid.Tile(x + 1, y)->Density() < me->Density()) ? grid.Tile(x + 2, y) : nullptr;
			break;

		default:
			moveTo = nullptr;
			break;
		}

		if (moveTo && moveTo->Density() < me->Density())
		{
			SwapTiles(me, moveTo);
		}
	}
}

__device__ void WaterUpdate(TileGrid& grid, unsigned int x, unsigned int y)
{
	if (grid.Tile(x, y)->data.temperature > 100.0f)
	{
		*grid.Tile(x, y) = Tile(Tile::Air);
	}
	else
	{
		BasicLiquidUpdate(grid, x, y);
	}
}

__constant__ TileUpdate tileUpdateFns[Tile::__TypeCount] =
{
	AirUpdate,
	SandUpdate,
	WaterUpdate,
	BasicLiquidUpdate
};

static TileGrid* grid = nullptr;
static cudaGraphicsResource_t resource;
static cudaArray_t array;
static cudaSurfaceObject_t surface;

__global__ void _Update(TileGrid* grid, unsigned int ofx, unsigned int ofy)
{
	const unsigned int x = (blockDim.x * blockIdx.x + threadIdx.x) * MAX_OFFSET_X + ofx;
	const unsigned int y = (blockDim.y * blockIdx.y + threadIdx.y) * MAX_OFFSET_Y + ofy;

	Tile* t = grid->Tile(x, y);
	if (t && t->lastUpdated <= grid->GetTick())
	{
		t->lastUpdated = grid->GetTick() + 1;
		tileUpdateFns[t->type](*grid, x, y);
	}
}

__global__ void _UpdateTemp(TileGrid* grid)
{
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

	float newTemperature;
	if (x < GRID_WIDTH && y < GRID_HEIGHT)
	{
		Tile me = *grid->Tile(x, y);
		newTemperature = me.data.temperature;

		for (int dx = -1; dx <= 1; dx++)
		{
			for (int dy = -1; dy <= 1; dy++)
			{
				const Tile* t = grid->Tile(x + dx, y + dy);
				if (t)
				{
					const float diff = t->Energy() - me.Energy();
					const float transferAmount = diff * fminf(t->ThermalDiffusity(), me.ThermalDiffusity());

					newTemperature += transferAmount;
				}
			}
		}
	}

	__syncthreads();

	if (x < GRID_WIDTH && y < GRID_HEIGHT)
	{
		grid->Tile(x, y)->data.temperature = newTemperature;
	}
}

__global__ void _Render(TileGrid* grid, cudaSurfaceObject_t surface)
{
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < GRID_WIDTH && y < GRID_HEIGHT)
	{
		surf2Dwrite(grid->Tile(x, y)->Color(), surface, x * sizeof(uint32_t), y);
	}
}

__global__ void _Setup(TileGrid* grid)
{
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < GRID_WIDTH && y < GRID_HEIGHT)
	{
		*grid->Tile(x, y) = Tile(Tile::Air);
		grid->Tile(x, y)->lastUpdated = grid->GetTick();
		grid->Random(x, y).SetState(x + y * GRID_WIDTH);
	}
}

void SetupGrid(unsigned int textureID)
{
	if (!grid)
	{
		cudaMallocManaged((void**)&grid, sizeof(TileGrid));
		cudaGraphicsGLRegisterImage(&resource, textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
		cudaGraphicsMapResources(1, &resource);
		cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0);
		cudaGraphicsUnmapResources(1, &resource);

		cudaResourceDesc desc;
		desc.resType = cudaResourceTypeArray;
		desc.res.array.array = array;

		cudaCreateSurfaceObject(&surface, &desc);

		grid->SetTick(0);

		_Setup<<<BASE_GRID, THREADS>>>(grid);
		cudaDeviceSynchronize();
	}
}

void DestroyGrid()
{
	cudaFree(grid);
	cudaDestroySurfaceObject(surface);
	grid = nullptr;
}

void SetTile(unsigned int x, unsigned int y, Tile t)
{
	*grid->Tile(x, y) = t;
}

Tile GetTile(unsigned int x, unsigned int y)
{
	return *grid->Tile(x, y);
}

void Update()
{
	_UpdateTemp<<<BASE_GRID, THREADS>>>(grid);
	cudaDeviceSynchronize();

	for (unsigned int ofx = 0; ofx < MAX_OFFSET_X; ofx++)
	{
		for (unsigned int ofy = 0; ofy < MAX_OFFSET_Y; ofy++)
		{
			_Update<<<UPDATE_GRID, THREADS>>>(grid, ofx, ofy);
			cudaDeviceSynchronize();
		}
	}

	grid->SetTick(grid->GetTick() + 1);
}

void Render()
{
	_Render<<<BASE_GRID, THREADS>>>(grid, surface);
	cudaDeviceSynchronize();
}