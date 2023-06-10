#pragma once

#include <cstdint>
#include <type_traits>

constexpr int GRID_WIDTH = 128;
constexpr int GRID_HEIGHT = 128;
constexpr int TILE_SIZE = 10;
constexpr int SCREEN_WIDTH = GRID_WIDTH * TILE_SIZE;
constexpr int SCREEN_HEIGHT = GRID_HEIGHT * TILE_SIZE;

#ifdef __CUDA_ARCH__
#define LOC_HD __host__ __device__
#else
#define LOC_HD
#endif

struct Tile
{
	enum Type : unsigned int
	{
		Air,
		Sand,
		Water,
		TypeCount
	} type;

	union Data
	{
		struct Air
		{
			LOC_HD uint32_t Color() { return 0xFFFFFFFF; }
		} air;
		struct Sand
		{
			LOC_HD uint32_t Color() { return 0xFF80B2C2; }
		} sand;
		struct Water
		{
			LOC_HD uint32_t Color() { return 0xFFDA8923; }
		} water;
	} data;
	
	unsigned int lastUpdated;

	LOC_HD uint32_t Color()
	{
		switch (type)
		{
		case Air:
			return data.air.Color();

		case Sand:
			return data.sand.Color();

		case Water:
			return data.water.Color();

		default:
			return 0xFF000000;
		}
	}
};

void SetupGrid(unsigned int textureID);
void DestroyGrid();
void Update();
void Render();
Tile GetTile(unsigned int x, unsigned int y);
void SetTile(unsigned int x, unsigned int y, Tile t);