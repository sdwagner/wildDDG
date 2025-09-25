// =============================================================================
//  Copyright (c) 2025 Sven D. Wagner, Mario Botsch.
//  Distributed under MIT license, see file LICENSE for details.
// =============================================================================

#pragma once
//=============================================================================

#include <pmp/surface_mesh.h>

#include "enums.h"
using namespace pmp;

//=============================================================================

bool parameterize_boundary(SurfaceMesh& mesh);

void parameterize_direct(SurfaceMesh &mesh, LaplaceConfig config);

//=============================================================================