// =============================================================================
//  Copyright (c) 2025 Sven D. Wagner, Mario Botsch.
//  Distributed under MIT license, see file LICENSE for details.
// =============================================================================

#include "parameterization.h"

#include "enums.h"
#include "../Laplacians/optimized_laplace.h"
#include "pmp/algorithms/differential_geometry.h"
#include "pmp/algorithms/laplace.h"

//=============================================================================

bool parameterize_boundary(SurfaceMesh &mesh)
{
    // get properties
    auto points = mesh.vertex_property<Point>("v:point");
    auto tex = mesh.vertex_property<TexCoord>("v:tex");

    // Initialize all texture coordinates to the origin.
    for (auto v : mesh.vertices())
        tex[v] = TexCoord(0, 0);

    // find a boundary vertex
    SurfaceMesh::VertexIterator vit, vend = mesh.vertices_end();
    for (vit = mesh.vertices_begin(); vit != vend; ++vit)
        if (mesh.is_boundary(*vit))
            break;

    // fail if no boundary vertex could be found
    if (vit == vend)
    {
        std::cerr << "Mesh has no boundary." << std::endl;
        return false;
    }

    // walk along boundary halfedges, collect boundary loop
    std::vector<Vertex> loop;
    Vertex v = *vit;
    Halfedge h = mesh.halfedge(v);
    do
    {
        loop.push_back(mesh.to_vertex(h));
        h = mesh.next_halfedge(h);
    } while (h != mesh.halfedge(v));

    // compute length of boundary loop
    const unsigned int n = loop.size();
    double length(0);
    for (int i = 0; i < n; ++i)
    {
        length += distance(points[loop[i]], points[loop[(i + 1) % n]]);
    }

    // map length intervalls to unit circle intervals
    double l(0);
    for (int i = 0; i < n; ++i)
    {
        double angle = -2.0 * M_PI * l / length;
        tex[loop[i]] = TexCoord(cos(angle), sin(angle));
        l += distance(points[loop[i]], points[loop[(i + 1) % n]]);
    }

    // map from unit circle [-1,1]^2 to [0,1]^2
    for (auto v : mesh.vertices())
    {
        tex[v] *= 0.5;
        tex[v] += TexCoord(0.5, 0.5);
    }

    return true;
}

//-----------------------------------------------------------------------------

void parameterize_direct(SurfaceMesh &mesh, LaplaceConfig config)
{
    const int n = mesh.n_vertices();

    // we assume that boundary constraints are precomputed!
    auto tex = mesh.vertex_property<TexCoord>("v:tex");

    // build matrix
    SparseMatrix S;
    tri_laplace_matrix(mesh, S, config);

    // build right-hand side
    DenseMatrix B(n, 2);
    for (auto v : mesh.vertices())
    {
        B.row(v.idx()) = mesh.is_boundary(v) ? static_cast<Eigen::Vector2d>(tex[v]) : Eigen::Vector2d(0, 0);
    }

    // selector matrices
    auto is_constrained = [&](int v){return mesh.is_boundary(pmp::Vertex(v));};
    Eigen::MatrixXd tex_mat = cholesky_solve(S, DenseMatrix::Zero(n, 2), is_constrained, B);
    for (int i = 0; i < n; ++i)
        tex[pmp::Vertex(i)] = TexCoord(tex_mat(i, 0), tex_mat(i, 1));
}
//=============================================================================