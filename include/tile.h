#ifndef TILE_H
#define TILE_H

#include <cstdint>
#include <cmath>

constexpr int GRID_WIDTH = 640;
constexpr int GRID_HEIGHT = 360;
constexpr int TILE_SIZE = 3;
constexpr int SCREEN_WIDTH = GRID_WIDTH * TILE_SIZE;
constexpr int SCREEN_HEIGHT = GRID_HEIGHT * TILE_SIZE;

#ifdef __CUDA_ARCH__
#define LOC_HD __host__ __device__
#else
#define LOC_HD
#endif

template <typename T>
LOC_HD float Lerp(T a, T b, float t01)
{
	return (1 - t01) * a + t01 * b;
}

inline LOC_HD uint32_t Mix(uint32_t ca, uint32_t cb, float t01)
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
				LOC_HD float SpecificHeat(const Data&) const { return 1.0f; }
				LOC_HD float ThermalConductivity(const Data&) const { return 0.052f; }

				LOC_HD static float DefaultTemperature() { return 25.0f; }
			} air;
			struct Sand
			{
				LOC_HD uint32_t Color(const Data&) const { return 0xFF80B2C2; }
				LOC_HD float Density(const Data&) const { return 50.0f; }
				LOC_HD float SpecificHeat(const Data&) const { return 1.0f; }
				LOC_HD float ThermalConductivity(const Data&) const { return 0.5f; }

				LOC_HD static float DefaultTemperature() { return 25.0f; }
			} sand;
			struct Water
			{
				LOC_HD uint32_t Color(const Data&) const { return 0xFFDA8923; }
				LOC_HD float Density(const Data&) const { return 25.0f; }
				LOC_HD float SpecificHeat(const Data&) const { return 1.0f; }
				LOC_HD float ThermalConductivity(const Data&) const { return 1.2178f; }

				LOC_HD static float DefaultTemperature() { return 25.0f; }
			} water;
			struct Lava
			{
				LOC_HD uint32_t Color(const Data& d) const
				{
					constexpr float TEMP_MIN = 0.0f;
					constexpr float TEMP_MAX = 1000.0f;

					const float t = d.temperature < TEMP_MIN ? TEMP_MIN : d.temperature;
					const float t2 = t > TEMP_MAX ? TEMP_MAX : d.temperature;
					const float t01 = t2 / (TEMP_MAX - TEMP_MIN);

					return Mix(0x40AFFF, 0x0000FF, t01);
				}
				LOC_HD float Density(const Data&) const { return 40.0f; }
				LOC_HD float SpecificHeat(const Data&) const { return 1.0f; }
				LOC_HD float ThermalConductivity(const Data&) const { return 2.6f; }

				LOC_HD static float DefaultTemperature() { return 1000.0f; }
			} lava;
		};

		struct
		{
			float temperature;
		};
	} data;

	unsigned int lastUpdated;

	LOC_HD Tile(Tile::Type type) :
		type(type),
		data(MakeData(type))
	{ }

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

		default:
#ifdef __CUDA_ARCH__
			__builtin_unreachable();
#else
			return 0;
#endif
		}
	}

	LOC_HD float Energy() const
	{
		return data.temperature * SpecificHeat();
	}

	LOC_HD float ThermalDiffusity() const
	{
		return ThermalConductivity() / (SpecificHeat() * Density());
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