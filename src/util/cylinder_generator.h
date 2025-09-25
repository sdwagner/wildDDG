// =============================================================================
//  Copyright (c) 2025 Sven D. Wagner, Mario Botsch.
//  Distributed under MIT license, see file LICENSE for details.
// =============================================================================

#pragma once
#include "pmp/mat_vec.h"
#include "marching_cubes/Grid.h"
#include "marching_cubes/MarchingCubes.h"
#include "pmp/surface_mesh.h"


inline double gen_cylinder_sdf(dvec3 p, const double r, const double h)
{
    using namespace pmp;
    dvec2 d = dvec2(norm(dvec2(p[0], p[2])),abs(p[1])) - dvec2(r,h);
    return fmin(fmax(d[0],d[1]),0.0) + norm(max(d,dvec2(0.0, 0.0)));
}

inline void gen_cylinder_grid(const double r, const double h, Grid& grid)
{
    for (int i = 0; i < grid.x_resolution(); i++)
    {
        for (int j = 0; j < grid.y_resolution(); j++)
        {
            for (int k = 0; k < grid.z_resolution(); k++)
            {
                const dvec3 p = grid.point(i, j, k);
                grid(i, j, k) = gen_cylinder_sdf(p, r, h);
            }
        }
    }
}

inline void gen_cylinder_mesh(const double r, const double h, const int resolution, SurfaceMesh& mesh, bool polygon_mesh)
{

    Grid grid(dvec3(-2*r, -2*h, -2*r),
              dvec3(4*r, 0, 0),
              dvec3(0, 4*h, 0),
              dvec3(0, 0, 4*r),
              resolution, resolution, resolution);
    gen_cylinder_grid(r, h, grid);
    marching_cubes(grid, mesh, 0, polygon_mesh);
}