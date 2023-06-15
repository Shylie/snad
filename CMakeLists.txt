cmake_minimum_required(VERSION 3.26)

project(snad LANGUAGES CUDA CXX)

include(FetchContent)
FetchContent_Declare(
	raylib
	GIT_REPOSITORY https://github.com/raysan5/raylib.git
	GIT_TAG 4.5.0
)

FetchContent_MakeAvailable(
	raylib
)

add_executable(snad
	include/tile.h
	source/tile.cu
	source/main.cpp
)

target_link_libraries(snad
	raylib
)

target_include_directories(snad
	PRIVATE
		include
		source
)

if (WIN32)
	set(SUBSYSTEM_LINKER_OPTIONS "/SUBSYSTEM:WINDOWS;/ENTRY:mainCRTStartup")
else()
	set(SUBSYSTEM_LINKER_OPTIONS "-mwindows")
endif()

target_link_options(snad
	PRIVATE ${SUBSYSTEM_LINKER_OPTIONS}
)

set_property(
	TARGET
		snad
	PROPERTY
		CUDA_ARCHITECTURES native
)

target_compile_options(snad PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:
		$<$<CONFIG:Debug>:
			-G
		>
		$<$<CONFIG:RelWithDebInfo>:
			-G
		>
		$<$<CONFIG:MinSizeRel>:
			-lineinfo
		>
		$<$<CONFIG:Release>:
			-lineinfo
		>
	>
)