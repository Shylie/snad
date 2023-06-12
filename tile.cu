#include "tile.h"

#define WINGDIAPI
#define APIENTRY
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <device_atomic_functions.h>

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
		if (x >= GRID_WIDTH || y >= GRID_HEIGHT)
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

	unsigned int tick;

private:
	::Tile tiles[GRID_WIDTH * GRID_HEIGHT];
	XORRand rands[GRID_WIDTH * GRID_HEIGHT];
};

__device__ void SwapTiles(Tile* a, Tile* b)
{
	Tile copy = *a;
	*a = *b;
	*b = copy;
}

typedef void (*TileUpdate)(TileGrid&, unsigned int, unsigned int);
__device__ void AirUpdate(TileGrid& grid, unsigned int x, unsigned int y) { }
__device__ void SandUpdate(TileGrid& grid, unsigned int x, unsigned int y)
{
	Tile* me = grid.Tile(x, y);

	{
		Tile* below = grid.Tile(x, y - 1);

		if (below && below->State() < me->State() && below->Density() < me->Density())
		{
			SwapTiles(me, below);

			below->lastUpdated = grid.tick + 1;

			return;
		}
	}

	{
		Tile* downLeft = grid.Tile(x - 1, y - 1);

		if (downLeft && downLeft->State() < me->State() && downLeft->Density() < me->Density())
		{
			SwapTiles(me, downLeft);

			downLeft->lastUpdated = grid.tick + 1;

			return;
		}
	}

	{
		Tile* downRight = grid.Tile(x + 1, y - 1);

		if (downRight && downRight->State() < me->State() && downRight->Density() < me->Density())
		{
			SwapTiles(me, downRight);

			downRight->lastUpdated = grid.tick + 1;

			return;
		}
	}
}

__device__ void BasicLiquidUpdate(TileGrid& grid, unsigned int x, unsigned int y)
{
	Tile* me = grid.Tile(x, y);
	Tile* below = grid.Tile(x, y - 1);

	if (below && below->State() <= me->State() && below->Density() < me->Density())
	{
		SwapTiles(me, below);

		below->lastUpdated = grid.tick + 1;
	}
	else
	{
		unsigned int r = grid.Random(x, y)();
		Tile* moveTo = nullptr;
		switch (r % 7)
		{
		case 0:
		case 1:
			moveTo = (grid.Tile(x - 1, y) && grid.Tile(x - 1, y)->State() <= me->State()) ? grid.Tile(x - 2, y) : nullptr;
			break;

		case 2:
			moveTo = grid.Tile(x - 1, y);
			break;

		case 3:
			moveTo = grid.Tile(x + 1, y);
			break;

		case 4:
		case 5:
			moveTo = (grid.Tile(x + 1, y) && grid.Tile(x + 1, y)->State() <= me->State()) ? grid.Tile(x + 2, y) : nullptr;
			break;
		}

		if (moveTo && moveTo->State() <= me->State())
		{
			SwapTiles(me, moveTo);
			moveTo->lastUpdated = grid.tick + 1;
		}
	}
}

__constant__ TileUpdate tileUpdateFns[Tile::TCount] =
{
	AirUpdate,
	SandUpdate,
	BasicLiquidUpdate,
	BasicLiquidUpdate
};

static TileGrid* grid = nullptr;
static cudaGraphicsResource_t resource;
static cudaArray_t array;
static cudaSurfaceObject_t surface;

__global__ void _Update(TileGrid* grid, unsigned int ofx, unsigned int ofy)
{
	unsigned int x = (blockDim.x * blockIdx.x + threadIdx.x) * MAX_OFFSET_X + ofx;
	unsigned int y = (blockDim.y * blockIdx.y + threadIdx.y) * MAX_OFFSET_Y + ofy;

	if (x < GRID_WIDTH && y < GRID_HEIGHT)
	{
		Tile t = *grid->Tile(x, y);
		if (t.lastUpdated < grid->tick)
		{
			tileUpdateFns[t.type](*grid, x, y);
		}
	}
}

__global__ void _Render(TileGrid* grid, cudaSurfaceObject_t surface)
{
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < GRID_WIDTH && y < GRID_HEIGHT)
	{
		surf2Dwrite(grid->Tile(x, y)->Color(), surface, x * sizeof(uint32_t), y);
	}
}

__global__ void _Setup(TileGrid* grid)
{
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < GRID_WIDTH && y < GRID_HEIGHT)
	{
		*grid->Tile(x, y) = { Tile::TAir };
		grid->Tile(x, y)->lastUpdated = grid->tick;
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

		grid->tick = 0;

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
	for (unsigned int ofx = 0; ofx < MAX_OFFSET_X; ofx++)
	{
		for (unsigned int ofy = 0; ofy < MAX_OFFSET_Y; ofy++)
		{
			_Update<<<UPDATE_GRID, THREADS>>>(grid, ofx, ofy);
			cudaDeviceSynchronize();
		}
	}
	grid->tick++;
}

void Render()
{
	_Render<<<BASE_GRID, THREADS>>>(grid, surface);
	cudaDeviceSynchronize();
}