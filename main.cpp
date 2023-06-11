#include <raylib.h>

#include "tile.h"

int main(int argc, char** argv)
{
	InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "snad");
	SetTargetFPS(60);

	Texture texture = LoadTextureFromImage(Image{ nullptr, GRID_WIDTH, GRID_HEIGHT, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 });

	SetupGrid(texture.id);

	unsigned int selected = Tile::Type::TSand;
	uint32_t selectedColor = Tile{ static_cast<Tile::Type>(selected), { } }.Color();

	while (!WindowShouldClose())
	{
		Update();

		if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) || IsMouseButtonDown(MOUSE_BUTTON_RIGHT))
		{
			unsigned int x = GetMouseX() / TILE_SIZE;
			unsigned int y = GRID_HEIGHT - GetMouseY() / TILE_SIZE - 1;

			if (IsMouseButtonDown(MOUSE_BUTTON_LEFT))
			{
				if (x < GRID_WIDTH && y < GRID_HEIGHT)
				{
					SetTile(x, y, { static_cast<Tile::Type>(selected), { } });
				}
			}
			else if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT))
			{
				if (x < GRID_WIDTH && y < GRID_HEIGHT)
				{
					SetTile(x, y, { Tile::TAir, { } });
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
		DrawRectangle(20, 20, 20, 20, *reinterpret_cast<Color*>(&selectedColor));
		DrawRectangleLines(20, 20, 20, 20, Color{ 0, 0, 0, 255 });
		DrawFPS(20, 60);
		EndDrawing();
	}

	DestroyGrid();

	UnloadTexture(texture);

	CloseWindow();

	return 0;
}