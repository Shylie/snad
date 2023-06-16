#ifndef TILE_H
#define TILE_H

#include <cstdint>
#include <cmath>
#include <cstdio>

constexpr int GRID_WIDTH = 256;
constexpr int GRID_HEIGHT = 256;
constexpr int TILE_SIZE = 4;
constexpr int SCREEN_WIDTH = GRID_WIDTH * TILE_SIZE;
constexpr int SCREEN_HEIGHT = GRID_HEIGHT * TILE_SIZE;

#ifdef __CUDA_ARCH__
#define LOC_HD __host__ __device__
#else
#define LOC_HD
#endif

LOC_HD inline float Clamp(float x, float min, float max)
{
	const float t = x < min ? min : x;
	return t > max ? max : x;
}

template <typename T>
LOC_HD inline float Lerp(T a, T b, float t01)
{
	return (1 - t01) * a + t01 * b;
}

LOC_HD inline uint32_t Mix(uint32_t ca, uint32_t cb, float t01)
{
	const unsigned char r = Lerp(ca & 0xFF, cb & 0xFF, t01);
	const unsigned char g = Lerp((ca >> 8) & 0xFF, (cb >> 8) & 0xFF, t01);
	const unsigned char b = Lerp((ca >> 16) & 0xFF, (cb >> 16) & 0xFF, t01);

	return 0xFF000000 | (b << 16) | (g << 8) | r;
}

struct Tile
{
	enum Type : unsigned char
	{
		Air,
		Sand,
		Water,
		Lava,
		Stone,
		__TypeCount
	} type;

	struct Data
	{
		union
		{
			struct Air
			{
				LOC_HD uint32_t Color(const Data&) const { return 0xFFFFFFFF; }
				LOC_HD float Density(const Data&) const { return 5.0f; }
				LOC_HD float SpecificHeat(const Data&) const { return 1000.0f; }
				LOC_HD float ThermalConductivity(const Data&) const { return 0.104f; }

				LOC_HD static float DefaultTemperature() { return 25.0f; }
			} air;
			struct Sand
			{
				LOC_HD uint32_t Color(const Data&) const { return 0xFF80B2C2; }
				LOC_HD float Density(const Data&) const { return 50.0f; }
				LOC_HD float SpecificHeat(const Data&) const { return 1000.0f; }
				LOC_HD float ThermalConductivity(const Data&) const { return 1.0f; }

				LOC_HD static float DefaultTemperature() { return 25.0f; }
			} sand;
			struct Water
			{
				LOC_HD uint32_t Color(const Data&) const { return 0xFFDA8923; }
				LOC_HD float Density(const Data&) const { return 25.0f; }
				LOC_HD float SpecificHeat(const Data&) const { return 4184.0f; }
				LOC_HD float ThermalConductivity(const Data&) const { return 2.4356f; }

				LOC_HD static float DefaultTemperature() { return 25.0f; }
			} water;
			struct Lava
			{
				LOC_HD uint32_t Color(const Data& d) const
				{
					constexpr float TEMP_MIN = 700.0f;
					constexpr float TEMP_MAX = 1200.0f;

					const float t = Clamp(Clamp(d.temperature - TEMP_MIN, 0, TEMP_MAX - TEMP_MIN) / (TEMP_MAX - TEMP_MIN), 0, 1);

					return Mix(0x20A0, 0x37FF, t);
				}
				LOC_HD float Density(const Data&) const { return 40.0f; }
				LOC_HD float SpecificHeat(const Data&) const { return 1100.0f; }
				LOC_HD float ThermalConductivity(const Data&) const { return 5.2f; }

				LOC_HD static float DefaultTemperature() { return 1200.0f; }
			} lava;
			struct Stone
			{
				LOC_HD uint32_t Color(const Data& d) const
				{
					constexpr float TEMP_MIN = 0.0f;
					constexpr float TEMP_MAX = 700.0f;

					const float t = Clamp(Clamp(d.temperature - TEMP_MIN, 0, TEMP_MAX - TEMP_MIN) / (TEMP_MAX - TEMP_MIN), 0, 1);

					return Mix(0x404040, 0x20A0, t);
				}
				LOC_HD float Density(const Data&) const { return 65.0f; }
				LOC_HD float SpecificHeat(const Data&) const { return 1100.0f; }
				LOC_HD float ThermalConductivity(const Data&) const { return 5.2f; }

				LOC_HD static float DefaultTemperature() { return 25.0f; }
			} stone;
		};

		struct
		{
			float temperature;
		};
	} data;

	unsigned int lastUpdated;

	LOC_HD Tile() { }

	LOC_HD Tile(Tile::Type type) :
		type(type),
		data(MakeData(type)),
		lastUpdated(0)
	{ }

	LOC_HD Tile& SetTemperature(float temperature)
	{
		data.temperature = temperature;

		return *this;
	}

	LOC_HD uint32_t Color() const
	{
		switch (type)
		{
		case Air:
			return data.air.Color(data);

		case Sand:
			return data.sand.Color(data);

		case Water:
			return data.water.Color(data);

		case Lava:
			return data.lava.Color(data);

		case Stone:
			return data.stone.Color(data);

		default:
#ifdef __CUDA_ARCH__
			__builtin_unreachable();
#else
			return 0xFF000000;
#endif
		}
	}

	LOC_HD float Density() const
	{
		switch (type)
		{
		case Air:
			return data.air.Density(data);

		case Sand:
			return data.sand.Density(data);

		case Water:
			return data.water.Density(data);

		case Lava:
			return data.lava.Density(data);

		case Stone:
			return data.stone.Density(data);

		default:
#ifdef __CUDA_ARCH__
			__builtin_unreachable();
#else
			return 0;
#endif
		}
	}

	LOC_HD float SpecificHeat() const
	{
		switch (type)
		{
		case Air:
			return data.air.SpecificHeat(data);

		case Sand:
			return data.sand.SpecificHeat(data);

		case Water:
			return data.water.SpecificHeat(data);

		case Lava:
			return data.lava.SpecificHeat(data);

		case Stone:
			return data.stone.SpecificHeat(data);

		default:
#ifdef __CUDA_ARCH__
			__builtin_unreachable();
#else
			return 0;
#endif
		}
	}

	LOC_HD float ThermalConductivity() const
	{
		switch (type)
		{
		case Air:
			return data.air.ThermalConductivity(data);

		case Sand:
			return data.sand.ThermalConductivity(data);

		case Water:
			return data.water.ThermalConductivity(data);

		case Lava:
			return data.lava.ThermalConductivity(data);

		case Stone:
			return data.stone.ThermalConductivity(data);

		default:
#ifdef __CUDA_ARCH__
			__builtin_unreachable();
#else
			return 0;
#endif
		}
	}

private:
	LOC_HD static Data MakeData(Type type)
	{
		Data d{ };

		switch (type)
		{
		case Air:
			d.air = { };
			d.temperature = decltype(d.air)::DefaultTemperature();
			break;

		case Sand:
			d.sand = { };
			d.temperature = decltype(d.sand)::DefaultTemperature();
			break;

		case Water:
			d.water = { };
			d.temperature = decltype(d.water)::DefaultTemperature();
			break;

		case Lava:
			d.lava = { };
			d.temperature = decltype(d.lava)::DefaultTemperature();
			break;
		}

		return d;
	}
};

void SetupGrid(unsigned int textureID);
void DestroyGrid();
void Update();
void Render();
Tile GetTile(unsigned int x, unsigned int y);
void SetTile(unsigned int x, unsigned int y, Tile t);

#endif//TILE_H