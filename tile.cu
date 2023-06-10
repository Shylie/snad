#include "tile.h"

#define WINGDIAPI
#define APIENTRY
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

constexpr unsigned int MAX_TILE_MODIFY_DISTANCE = 1;
constexpr unsigned int MAX_OFFSET = 2 * MAX_TILE_MODIFY_DISTANCE + 1;
constexpr int BLOCK_SIZE = 16;
const dim3 THREADS{ BLOCK_SIZE, BLOCK_SIZE };
const dim3 BASE_GRID{ (GRID_WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE, (GRID_HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE };
const dim3 UPDATE_GRID{ (BASE_GRID.x + MAX_OFFSET - 1) / MAX_OFFSET, (BASE_GRID.y + MAX_OFFSET - 1) / MAX_OFFSET };

class TileGrid
{
public:
	__host__ __device__ Tile* operator()(unsigned int x, unsigned int y)
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

	unsigned int tick;

private:
	Tile tiles[GRID_WIDTH * GRID_HEIGHT];
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
	Tile* me = grid(x, y);

	{
		Tile* below = grid(x, y - 1);

		if (below && below->type == Tile::Air)
		{
			SwapTiles(me, below);

			below->lastUpdated = grid.tick + 1;

			return;
		}
	}

	{
		Tile* downLeft = grid(x - 1, y - 1);

		if (downLeft && downLeft->type == Tile::Air)
		{
			SwapTiles(me, downLeft);

			downLeft->lastUpdated = grid.tick + 1;

			return;
		}
	}

	{
		Tile* downRight = grid(x + 1, y - 1);

		if (downRight && downRight->type == Tile::Air)
		{
			SwapTiles(me, downRight);

			downRight->lastUpdated = grid.tick + 1;

			return;
		}
	}
}

__device__ void WaterUpdate(TileGrid& grid, unsigned int x, unsigned int y)
{
	Tile* me = grid(x, y);
	Tile* below = grid(x, y - 1);

	if (below && below->type == Tile::Air)
	{
		SwapTiles(me, below);

		below->lastUpdated = grid.tick + 1;
	}
	else
	{
		if ((grid.tick + x) % 2 == 0)
		{
			Tile* left = grid(x - 1, y);

			if (left && left->type == Tile::Air)
			{
				SwapTiles(me, left);

				left->lastUpdated = grid.tick + 1;
			}
		}
		else if ((grid.tick + x) % 2 == 1)
		{
			Tile* right = grid(x + 1, y);

			if (right && right->type == Tile::Air)
			{
				SwapTiles(me, right);

				right->lastUpdated = grid.tick + 1;
			}
		}
	}
}

__constant__ TileUpdate tileUpdateFns[Tile::TypeCount] =
{
	AirUpdate,
	SandUpdate,
	WaterUpdate
};

static TileGrid* grid = nullptr;
static cudaGraphicsResource_t resource;
static cudaArray_t array;
static cudaSurfaceObject_t surface;

__global__ void _Update(TileGrid* grid, unsigned int ofx, unsigned int ofy)
{
	unsigned int x = (blockDim.x * blockIdx.x + threadIdx.x) * MAX_OFFSET + ofx;
	unsigned int y = (blockDim.y * blockIdx.y + threadIdx.y) * MAX_OFFSET + ofy;

	if (x < GRID_WIDTH && y < GRID_HEIGHT)
	{
		Tile t = *(*grid)(x, y);
		if (t.lastUpdated < grid->tick)
		{
			tileUpdateFns[(*grid)(x, y)->type](*grid, x, y);
		}
	}
}

__global__ void _Render(TileGrid* grid, cudaSurfaceObject_t surface)
{
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < GRID_WIDTH && y < GRID_HEIGHT)
	{
		surf2Dwrite((*grid)(x, y)->Color(), surface, x * sizeof(uint32_t), y);
	}
}

__global__ void _Setup(TileGrid* grid)
{
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < GRID_WIDTH && y < GRID_HEIGHT)
	{
		(*grid)(x, y)->lastUpdated = grid->tick;
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
	*(*grid)(x, y) = t;
}

Tile GetTile(unsigned int x, unsigned int y)
{
	return *(*grid)(x, y);
}

void Update()
{
	for (unsigned int ofx = 0; ofx < MAX_OFFSET; ofx++)
	{
		for (unsigned int ofy = 0; ofy < MAX_OFFSET; ofy++)
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