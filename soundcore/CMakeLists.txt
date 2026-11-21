# The shared native sound engine: one implementation for the Raspberry Pi player, Open Blue
# (through JNI) and offline rendering.
cmake_minimum_required(VERSION 3.16)
project(soundcore CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(soundcore STATIC
  src/dct.cpp src/voices.cpp src/flat6.cpp src/hypercar.cpp src/jet.cpp src/warp.cpp
  src/effects.cpp src/engine.cpp)
target_include_directories(soundcore PUBLIC src)
target_compile_options(soundcore PRIVATE -Wall -Wextra)

# offline: render a drive to WAV and report levels and speed
add_executable(sc_render tools/render.cpp)
target_link_libraries(sc_render soundcore)

enable_testing()
add_executable(sc_tests tests/test_core.cpp)
target_link_libraries(sc_tests soundcore)
add_test(NAME soundcore COMMAND sc_tests ${CMAKE_SOURCE_DIR}/../../open-blue-android/app/src/main/assets/effects)

# the Pi player (ALSA); only where ALSA is available
find_package(ALSA QUIET)
if(ALSA_FOUND)
  add_executable(sc_player tools/player.cpp)
  target_link_libraries(sc_player soundcore ALSA::ALSA pthread)
endif()
