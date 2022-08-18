#Copyright 2011, Bas Fagginger Auer.
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.
cmake_minimum_required(VERSION 2.6)
project(MATCH LANGUAGES CUDA CXX)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${MATCH_SOURCE_DIR})

find_package(CUDA REQUIRED)
message(STATUS "CUDA_INCLUDE_DIR: ${CUDA_INCLUDE_DIR}")
message(STATUS "CUDA_INCLUDE_DIRS	: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CUDA_LIBRARY: ${CUDA_LIBRARY}")
message(STATUS "CUDA_LIBRARY: ${CUDA_LIBRARIES}")

find_package(TBB REQUIRED)
message(STATUS "TBB_INCLUDE_DIR: ${TBB_INCLUDE_DIR}")
message(STATUS "TBB_INCLUDE_DIRS: ${TBB_INCLUDE_DIRS}")
message(STATUS "TBB_ROOT_DIR: ${TBB_ROOT_DIR}")
message(STATUS "TBB::tbb: ${TBB::tbb}")
message(STATUS "TBB_LIBRARIES: ${TBB_LIBRARIES}")

message(STATUS "TBB_LIBRARY: ${TBB_LIBRARY}")

find_package(Curses REQUIRED)
message(STATUS "CURSES_INCLUDE_DIR: ${CURSES_INCLUDE_DIR}")
message(STATUS "CURSES_LIBRARIES: ${CURSES_LIBRARIES}")
message(STATUS "CURSES_LIBRARY: ${CURSES_LIBRARY}")
add_definitions(-w)
include_directories(${CURSES_INCLUDE_DIR})

#set(CMAKE_CXX_FLAGS "-O2 -g -Wall -Wextra -Wshadow -Werror -ansi")
set(CMAKE_CXX_FLAGS "-O3 -Wall -ffast-math -DNDEBUG")
#set(CUDA_NVCC_FLAGS -g -G)
set(CUDA_NVCC_FLAGS -Xptxas -dlcm=cg)

add_subdirectory(${MATCH_SOURCE_DIR}/src/)
add_subdirectory(${MATCH_SOURCE_DIR}/bin/)
add_subdirectory(${MATCH_SOURCE_DIR}/DotWriter/lib/)
#enable_testing()
#include(gtest/GoogleTest.cmake)
#add_subdirectory(gtest)

