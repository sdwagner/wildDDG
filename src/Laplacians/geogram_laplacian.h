// =============================================================================
//  Copyright (c) 2025 Sven D. Wagner, Mario Botsch.
//  Distributed under MIT license, see file LICENSE for details.
// =============================================================================


#pragma once

#include "geogram/mesh/mesh.h"
#include "geogram/mesh/mesh_geometry.h"
#include <Eigen/Sparse>

// See include/geogram/src/lib/geogram/mesh/mesh_manifold_harmonics.cpp for original definition and license
inline double P1_FEM_coefficient(
    const GEO::Mesh& M, GEO::index_t f, GEO::index_t v1, GEO::index_t v2
)
{
    using namespace GEO;
    index_t v3 = NO_VERTEX;
    for (index_t lv = 0; lv < M.facets.nb_vertices(f); ++lv)
    {
        index_t v = M.facets.vertex(f, lv);
        if (v != v1 && v != v2)
        {
            v3 = v;
            break;
        }
    }
    geo_assert(v3 != NO_VERTEX);
    const vec3& p1 = M.vertices.point(v1);
    const vec3& p2 = M.vertices.point(v2);
    const vec3& p3 = M.vertices.point(v3);
    vec3 V1 = p1 - p3;
    vec3 V2 = p2 - p3;
    return 1.0 / ::tan(Geom::angle(V1, V2));
}


inline void geogram_laplacian(GEO::Mesh& M, Eigen::SparseMatrix<double>& L)
{
    std::vector<Eigen::Triplet<double>> triplets;
    using namespace GEO;

    // Sum of row coefficient associated with each vertex
    vector<double> v_row_sum(M.vertices.nb(), 0.0);

    for (index_t f : M.facets)
    {
        index_t fnv = M.facets.nb_vertices(f);
        for (index_t lv = 0; lv < fnv; ++lv)
        {
            index_t v1 = M.facets.vertex(f, lv);
            index_t v2 = M.facets.vertex(f, (lv + 1) % fnv);
            double w = 0.5 * P1_FEM_coefficient(M, f, v1, v2);
            triplets.emplace_back(v1, v2, w);
            triplets.emplace_back(v2, v1, w);
            v_row_sum[v1] += w;
            v_row_sum[v2] += w;
        }
    }
    for (index_t v : M.vertices)
    {
        // Diagonal term is minus row sum
        // plus small number to make M non-singular
        triplets.emplace_back(v, v, -v_row_sum[v] + 1e-6); // Why exactly?
    }
    L.setFromTriplets(triplets.begin(), triplets.end());
}
