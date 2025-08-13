#pragma once

#include "Grid.h"
#include <pmp/surface_mesh.h>

using namespace pmp;

//=============================================================================

void marching_cubes(const Grid& _grid, SurfaceMesh& _mesh, Scalar _isoval=0, bool polygon_mesh = false);


//=============================================================================
