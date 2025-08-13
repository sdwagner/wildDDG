cmake_minimum_required(VERSION 3.16.3)
project(wildDDG)

# project version
set(CMAKE_PROJECT_VERSION 1.0.0)
set(CMAKE_PROJECT_VERSION_MAJOR 1)
set(CMAKE_PROJECT_VERSION_MINOR 0)
set(CMAKE_PROJECT_VERSION_PATCH 0)

# set output paths
set(PROJECT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# For homebrew OpenMP on macOS
set(CMAKE_POLICY_DEFAULT_CMP0074 NEW)


# set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# PMP options
set(PMP_BUILD_APPS     OFF CACHE BOOL "")
set(PMP_BUILD_EXAMPLES OFF CACHE BOOL "")
set(PMP_BUILD_TESTS    OFF CACHE BOOL "")
set(PMP_BUILD_DOCS     OFF CACHE BOOL "")
set(PMP_INSTALL        OFF CACHE BOOL "")
add_definitions(-DPMP_SCALAR_TYPE_64)
add_definitions(-DINCLUDE_TEMPLATES)

# Geogram Options
set(GEOGRAM_WITH_GRAPHICS OFF CACHE BOOL "")

# Geometry-Central Options
set(SUITESPARSE OFF CACHE BOOL "")

# CinoLib Options
set(cinolib_DIR ${PROJECT_SOURCE_DIR}/include/cinolib)
find_package(cinolib REQUIRED)

set(LIBIGL_PREDICATES ON CACHE BOOL "")

# include CGAL
find_package(CGAL REQUIRED)
find_package(MPFR REQUIRED)
find_package(GMP REQUIRED)


# add include directories
include_directories(${PROJECT_SOURCE_DIR}/include/spectra/include)
include_directories(${CGAL_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include/libigl/include)
include_directories(${PROJECT_SOURCE_DIR}/include/vcglib)
include_directories(${PROJECT_SOURCE_DIR}/include/geogram/src/lib)
include_directories(${PROJECT_SOURCE_DIR}/include/cinolib/include)


if(WIN32)
  if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_MATH_DEFINES -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_MATH_DEFINES -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS -fopenmp")
  endif()
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

# make a release build by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-template-arg-list-after-template-kw")


if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb")
endif()

# which directories to process
add_subdirectory(include/pmp-library)
add_subdirectory(include/geometry-central)
add_subdirectory(include/OpenMesh)
add_subdirectory(include/libigl)
add_subdirectory(include/vcglib)
add_subdirectory(include/geogram)
add_subdirectory(src/)
