#include <raylib.h>

#include "tile.h"

int main(int argc, char** argv)
{
	InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "snad");
	//SetTargetFPS(60);

	Texture texture = LoadTextureFromImage(Image{ nullptr, GRID_WIDTH, GRID_HEIGHT, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 });

	SetupGrid(texture.id);

	float szf = 0;

	unsigned int selected = Tile::Type::TSand;
	uint32_t selectedColor = Tile{ static_cast<Tile::Type>(selected), { } }.Color();

	while (!WindowShouldClose())
	{
		for (int i = 0; i < 5; i++)
		{
			Update();
		}

		szf += GetMouseWheelMove();
		if (szf < 0) { szf = 0; }
		const int sz = static_cast<int>(szf);

		unsigned int x = GetMouseX() / TILE_SIZE;
		unsigned int y = GRID_HEIGHT - GetMouseY() / TILE_SIZE - 1;

		if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) || IsMouseButtonDown(MOUSE_BUTTON_RIGHT))
		{

			if (IsMouseButtonDown(MOUSE_BUTTON_LEFT))
			{
				for (int dx = -sz; dx <= sz; dx++)
				{
					for (int dy = -sz; dy <= sz; dy++)
					{
						if (x + dx < GRID_WIDTH && y + dy < GRID_HEIGHT)
						{
							SetTile(x + dx, y + dy, { static_cast<Tile::Type>(selected), { } });
						}
					}
				}
			}
			else if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT))
			{
				for (int dx = -sz; dx <= sz; dx++)
				{
					for (int dy = -sz; dy <= sz; dy++)
					{
						if (x + dx < GRID_WIDTH && y + dy < GRID_HEIGHT)
						{
							SetTile(x + dx, y + dy, { Tile::TAir, { } });
						}
					}
				}
			}
		}

		if (IsKeyPressed(KEY_COMMA) && selected - 1 > Tile::Type::TAir)
		{
			selected -= 1;
			selectedColor = Tile{ static_cast<Tile::Type>(selected), { } }.Color();
		}
		if (IsKeyPressed(KEY_PERIOD) && selected + 1 < Tile::Type::TCount)
		{
			selected += 1;
			selectedColor = Tile{ static_cast<Tile::Type>(selected), { } }.Color();
		}

		Render();

		BeginDrawing();
		DrawTexturePro(texture, { 0, 0, GRID_WIDTH, -GRID_HEIGHT }, { 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT }, { 0, 0 }, 0, WHITE);
		DrawRectangleLines(TILE_SIZE * (x - sz), TILE_SIZE * (GRID_HEIGHT - y - sz - 1), TILE_SIZE * (2 * sz + 1), TILE_SIZE * (2 * sz + 1), Color{ 0, 0, 0, 127 });
		DrawRectangle(20, 20, 20, 20, Fade(*reinterpret_cast<Color*>(&selectedColor), 0.5f));
		DrawRectangleLines(20, 20, 20, 20, Color{ 0, 0, 0, 127 });
		DrawFPS(20, 60);
		EndDrawing();
	}

	DestroyGrid();

	UnloadTexture(texture);

	CloseWindow();

	return 0;
}