#pragma once

#include <cstdint>

constexpr int GRID_WIDTH = 960;
constexpr int GRID_HEIGHT = 960;
constexpr int TILE_SIZE = 1;
constexpr int SCREEN_WIDTH = GRID_WIDTH * TILE_SIZE;
constexpr int SCREEN_HEIGHT = GRID_HEIGHT * TILE_SIZE;

#ifdef __CUDA_ARCH__
#define LOC_HD __host__ __device__
#else
#define LOC_HD
#endif

struct Tile
{
	enum Type: unsigned char
	{
		TAir,
		TSand,
		TWater,
		TLava,
		TCount
	} type;

	enum State: unsigned char
	{
		SGas,
		SLiquid,
		SSolid
	};

	union Data
	{
		struct TAir
		{
			LOC_HD uint32_t Color() const { return 0xFFFFFFFF; }
			LOC_HD State State() const { return SGas; }
			LOC_HD float Density() const { return 0.5f; }
		} air;
		struct TSand
		{
			LOC_HD uint32_t Color() const { return 0xFF80B2C2; }
			LOC_HD State State() const { return SSolid; }
			LOC_HD float Density() const { return 10.0f; }
		} sand;
		struct TWater
		{
			LOC_HD uint32_t Color() const { return 0xFFDA8923; }
			LOC_HD State State() const { return SLiquid; }
			LOC_HD float Density() const { return 1.0f; }
		} water;
		struct TLava
		{
			LOC_HD uint32_t Color() const { return 0xFF1535FF; }
			LOC_HD State State() const { return SLiquid; }
			LOC_HD float Density() const { return 2.0f; }
		} lava;
	} data;
	
	unsigned int lastUpdated;

	LOC_HD uint32_t Color() const
	{
		switch (type)
		{
		case TAir:
			return data.air.Color();

		case TSand:
			return data.sand.Color();

		case TWater:
			return data.water.Color();

		case TLava:
			return data.lava.Color();

		default:
			return 0xFF000000;
		}
	}

	LOC_HD State State() const
	{
		switch (type)
		{
		case TAir:
			return data.air.State();

		case TSand:
			return data.sand.State();

		case TWater:
			return data.water.State();

		case TLava:
			return data.lava.State();

		default:
			return SGas;
		}
	}

	LOC_HD float Density() const
	{
		switch (type)
		{
		case TAir:
			return data.air.Density();

		case TSand:
			return data.sand.Density();

		case TWater:
			return data.water.Density();

		case TLava:
			return data.lava.Density();

		default:
			return 0.0f;
		}
	}
};

void SetupGrid(unsigned int textureID);
void DestroyGrid();
void Update();
void Render();
Tile GetTile(unsigned int x, unsigned int y);
void SetTile(unsigned int x, unsigned int y, Tile t);