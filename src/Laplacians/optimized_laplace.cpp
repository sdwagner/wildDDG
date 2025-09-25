// =============================================================================
//  Copyright (c) 2025 Sven D. Wagner, Mario Botsch.
//  Distributed under MIT license, see file LICENSE for details.
// =============================================================================

#include "optimized_laplace.h"
#include <pmp/algorithms/laplace.h>

#include "../util/mesh_converter.h"

int static_counter = 0;
int dynamic_counter = 0;
int both_counter = 0;


double double_triarea(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1,
                      const Eigen::Vector3d& p2, LaplaceConfig config, AreaComputation area_computation)
{
    double area = 0;

    switch (area_computation)
    {
    case CrossProduct:
        {
            area = (p1 - p0).cross(p2 - p0).norm();
            break;
        }
    case Heron_Sorted:
        {
            std::array<double, 3> l{};
            // Kahan's Heron's formula from "Miscalculating Area and Angles of a Needle-like Triangle"
            // https://www.cs.unc.edu/~snoeyink/c/c205/Triangle.pdf §2
            // Should work better for needles
            l[0] = (p1 - p2).norm();
            l[1] = (p0 - p2).norm();
            l[2] = (p0 - p1).norm();
            std::ranges::sort(l, std::ranges::greater());

            //Parentheses are important here!
            const double arg = (l[0] + (l[1] + l[2])) * (l[2] - (l[0] - l[1])) *
                               (l[2] + (l[0] - l[1])) * (l[0] + (l[1] - l[2]));
            area = 0.5 * sqrt(arg);
            break;
        }
    case Max_Area:
        {
            area = (p1 - p0).cross(p2 - p0).norm();
            area = fmax(area, (p0 - p2).cross(p1 - p2).norm());
            area = fmax(area, (p0 - p1).cross(p2 - p1).norm());
            break;
        }
    }

    //TFEM implementations
    if (config.tfem_method == TFEMDynamic)
    {
        std::array<double, 3> l{};
        l[0] = (p1 - p2).squaredNorm();
        l[1] = (p0 - p2).squaredNorm();
        l[2] = (p0 - p1).squaredNorm();
        double h = pow((sqrt(l[0]) + sqrt(l[1]) + sqrt(l[2]))/3.0, 2);
        const double C = config.tfem_dynamic_constant;
        area = fmax(area, C*h);
    }
    else if (config.tfem_method == TFEMDynamicFailsafe)
    {
        std::array<double, 3> l{};
        l[0] = (p1 - p2).squaredNorm();
        l[1] = (p0 - p2).squaredNorm();
        l[2] = (p0 - p1).squaredNorm();
        double h = pow((sqrt(l[0]) + sqrt(l[1]) + sqrt(l[2]))/3.0, 2);
        h = fmax(1e-20, h);
        const double C = config.tfem_dynamic_constant;
        area = fmax(area, C*h);
    }
    else if (config.tfem_method == TFEMStatic)
    {
        area = fmax(config.tfem_static_constant, area);
        if (area == config.tfem_static_constant)
        {
            static_counter++;
        }
    }
    return area;
}

void tri_mass_matrix(const pmp::SurfaceMesh& mesh, pmp::DiagonalMatrix& M, LaplaceConfig config)
{
    const int nv = mesh.n_vertices();
    std::vector<pmp::Vertex> vertices; // polygon vertices
    pmp::DenseMatrix polygon;          // positions of polygon vertices
    pmp::DiagonalMatrix Mpoly;         // local mass matrix

    M.setZero(nv);

    for (pmp::Face f : mesh.faces())
    {
        // collect polygon vertices
        vertices.clear();
        for (pmp::Vertex v : mesh.vertices(f))
        {
            vertices.push_back(v);
        }
        assert(vertices.size() == 3);

        // collect their positions
        auto p0 = static_cast<Eigen::Vector3d>(mesh.position(vertices[0]));
        auto p1 = static_cast<Eigen::Vector3d>(mesh.position(vertices[1]));
        auto p2 = static_cast<Eigen::Vector3d>(mesh.position(vertices[2]));
        if (config.lib != OptimizedLaplacian_Cross_Dot_Mass)
            config.tfem_method = TFEMNormal;
        const double area = double_triarea(p0, p1, p2, config, area_computation);

        M.diagonal()[vertices[0].idx()] += area / 6.0;
        M.diagonal()[vertices[1].idx()] +=  area / 6.0;
        M.diagonal()[vertices[2].idx()] +=  area / 6.0;

    }
}

void triangle_gradient_matrix(const Eigen::Vector3d& p0,
                                     const Eigen::Vector3d& p1,
                                     const Eigen::Vector3d& p2, pmp::DenseMatrix& G, LaplaceConfig config, bool verbose)
{
    G.resize(3, 3);
    Eigen::Vector3d n = (p1 - p0).cross(p2 - p0);
    n.normalize();
    n /= double_triarea(p0, p1, p2, config, area_computation);
    G.col(0) = n.cross(p2 - p1);
    G.col(1) = n.cross(p0 - p2);
    G.col(2) = n.cross(p1 - p0);
}

void triangle_laplace_matrix(const Eigen::Vector3d& p0,
                                    const Eigen::Vector3d& p1,
                                    const Eigen::Vector3d& p2, pmp::DenseMatrix& Ltri, LaplaceConfig config)
{
    Ltri.resize(3, 3);
    Ltri.setZero();
    std::array<double, 3> cot{};
    double area = double_triarea(p0, p1, p2, config, area_computation);


    if (!config.clamp_zero_area || area > 0)
    {
        if (dot_sqlength)
        {
            const double inv_area = 0.5 / area;
            cot[0] = inv_area * (p1-p0).dot(p2-p0);
            cot[1] = inv_area * (p0-p1).dot(p2-p1);
            cot[2] = inv_area * (p0-p2).dot(p1-p2);
        }
        else
        {
            std::array<double, 3> l2{};
            // squared edge lengths
            l2[0] = (p1 - p2).squaredNorm();
            l2[1] = (p0 - p2).squaredNorm();
            l2[2] = (p0 - p1).squaredNorm();
            const double inv_area = 0.25 / area;
            cot[0] = inv_area * (l2[1] + l2[2] - l2[0]);
            cot[1] = inv_area * (l2[2] + l2[0] - l2[1]);
            cot[2] = inv_area * (l2[0] + l2[1] - l2[2]);
        }
        if (config.clamp_negative)
        {
            cot[0] = fmax(cot[0], 0.0);
            cot[1] = fmax(cot[1], 0.0);
            cot[2] = fmax(cot[2], 0.0);
        }
        Ltri(0, 0) = cot[1] + cot[2];
        Ltri(1, 1) = cot[0] + cot[2];
        Ltri(2, 2) = cot[0] + cot[1];
        Ltri(1, 0) = Ltri(0, 1) = -cot[2];
        Ltri(2, 0) = Ltri(0, 2) = -cot[1];
        Ltri(2, 1) = Ltri(1, 2) = -cot[0];
    }
}

void divmass_matrix(const pmp::SurfaceMesh& mesh, pmp::DiagonalMatrix& M, LaplaceConfig config)
{
    // how many virtual triangles will we have after refinement?
    unsigned int nt = mesh.n_faces();

    // initialize global matrix
    M.resize(3 * nt);
    auto& diag = M.diagonal();

    std::vector<pmp::Vertex> vertices; // polygon vertices
    pmp::DenseMatrix polygon;          // positions of polygon vertices

    unsigned int idx = 0;

    for (pmp::Face f : mesh.faces())
    {
        // collect polygon vertices
        vertices.clear();
        for (pmp::Vertex v : mesh.vertices(f))
        {
            vertices.push_back(v);
        }
        const int n = vertices.size();

        // collect their positions
        polygon.resize(n, 3);
        for (int i = 0; i < n; ++i)
        {
            polygon.row(i) = (Eigen::Vector3d)mesh.position(vertices[i]);
        }

        const double area = 0.5 *
            double_triarea(polygon.row(0), polygon.row(1), polygon.row(2), config, area_computation);
        diag[idx++] = area;
        diag[idx++] = area;
        diag[idx++] = area;
    }

    assert(idx == 3 * nt);
}
void tri_gradient_matrix(const pmp::SurfaceMesh& mesh, pmp::SparseMatrix& G, LaplaceConfig config)
{
    const int nv = mesh.n_vertices();
    const int nt = mesh.n_faces();


    std::vector<pmp::Vertex> vertices; // polygon vertices
    pmp::DenseMatrix Gpoly;            // local gradient matrix

    std::vector<pmp::Triplet> triplets;
    triplets.reserve(9 * nt);

    unsigned int n_rows = 0;

    for (pmp::Face f : mesh.faces())
    {
        // collect polygon vertices
        vertices.clear();
        for (pmp::Vertex v : mesh.vertices(f))
        {
            vertices.push_back(v);
        }
        assert(vertices.size() == 3);

        // collect their positions
        auto p0 = (Eigen::Vector3d)mesh.position(vertices[0]);
        auto p1 = (Eigen::Vector3d)mesh.position(vertices[1]);
        auto p2 = (Eigen::Vector3d)mesh.position(vertices[2]);

        // setup local element matrix
        triangle_gradient_matrix(p0, p1, p2, Gpoly, config);

        // assemble to global matrix
        for (int j = 0; j < Gpoly.cols(); ++j)
        {
            for (int i = 0; i < Gpoly.rows(); ++i)
            {
                triplets.emplace_back(n_rows + i, vertices[j].idx(),
                                      Gpoly(i, j));
            }
        }

        n_rows += Gpoly.rows();
    }
    assert(n_rows == 3 * nt);

    // build sparse matrix from triplets
    G.resize(n_rows, nv);
    G.setFromTriplets(triplets.begin(), triplets.end());

}

void tri_divergence_matrix(const pmp::SurfaceMesh& mesh, pmp::SparseMatrix& D, LaplaceConfig config)
{
    pmp::SparseMatrix G;
    tri_gradient_matrix(mesh, G, config);
    pmp::DiagonalMatrix M;
    divmass_matrix(mesh, M, config);
    D = -G.transpose() * M;
}

void get_divergence_and_gradient_matrix(const pmp::SurfaceMesh& mesh, pmp::SparseMatrix& D, pmp::SparseMatrix& G, LaplaceConfig config)
{
    tri_gradient_matrix(mesh, G, config);
    pmp::DiagonalMatrix M;
    divmass_matrix(mesh, M,config);
    D = -G.transpose() * M;
}

void tri_laplace_matrix(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, pmp::SparseMatrix& L, LaplaceConfig config)
{
    const int nv = V.rows();
    const int nf = F.rows();
    pmp::DenseMatrix Lpoly;            // local laplace matrix

    std::vector<pmp::Triplet> triplets;
    triplets.reserve(9 * nf); // estimate for triangle meshes

    static_counter = 0;
    dynamic_counter = 0;
    both_counter = 0;
    for (auto f : F.rowwise())
    {
        // collect their positions
        Eigen::Vector3d p0 = V.row(f[0]);
        Eigen::Vector3d p1 = V.row(f[1]);
        Eigen::Vector3d p2 = V.row(f[2]);

        // setup local laplace matrix
        triangle_laplace_matrix(p0, p1, p2, Lpoly, config);

        // assemble to global laplace matrix
        for (int j = 0; j < 3; ++j)
        {
            for (int k = 0; k < 3; ++k)
            {
                triplets.emplace_back(f[k], f[j],
                                      -Lpoly(k, j));
            }
        }
    }

    // build sparse matrix from triplets
    L.resize(nv, nv);
    L.setFromTriplets(triplets.begin(), triplets.end());

    // clamp negative off-diagonal entries to zero
    if (config.clamp_negative)
    {
        for (unsigned int k = 0; k < L.outerSize(); k++)
        {
            double diag_offset(0.0);

            for (pmp::SparseMatrix::InnerIterator iter(L, k); iter; ++iter)
            {
                if (iter.row() != iter.col() && iter.value() < 0.0)
                {
                    diag_offset += -iter.value();
                    iter.valueRef() = 0.0;
                }
            }
            for (pmp::SparseMatrix::InnerIterator iter(L, k); iter; ++iter)
            {
                if (iter.row() == iter.col() && iter.value() < 0.0)
                    iter.valueRef() -= diag_offset;
            }
        }
    }
}
void tri_laplace_matrix(const pmp::SurfaceMesh& mesh, pmp::SparseMatrix& L, LaplaceConfig config)
{
    const int nv = mesh.n_vertices();
    const int nf = mesh.n_faces();
    pmp::DenseMatrix Lpoly;            // local laplace matrix

    std::vector<pmp::Triplet> triplets;
    triplets.reserve(9 * nf); // estimate for triangle meshes

    for (auto f : mesh.faces())
    {
        // collect their positions
        auto it = mesh.vertices(f);
        const pmp::Vertex verts[3] = {*it++, *it++, *it++};
        Eigen::Vector3d p0 = mesh.position(verts[0]);
        Eigen::Vector3d p1 = mesh.position(verts[1]);
        Eigen::Vector3d p2 = mesh.position(verts[2]);

        // setup local laplace matrix
        triangle_laplace_matrix(p0, p1, p2, Lpoly, config);

        // assemble to global laplace matrix
        for (int j = 0; j < 3; ++j)
        {
            for (int k = 0; k < 3; ++k)
            {
                triplets.emplace_back(verts[k].idx(), verts[j].idx(),
                                      -Lpoly(k, j));
            }
        }
    }

    // build sparse matrix from triplets
    L.resize(nv, nv);
    L.setFromTriplets(triplets.begin(), triplets.end());

    // clamp negative off-diagonal entries to zero
    if (config.clamp_negative)
    {
        for (unsigned int k = 0; k < L.outerSize(); k++)
        {
            double diag_offset(0.0);

            for (pmp::SparseMatrix::InnerIterator iter(L, k); iter; ++iter)
            {
                if (iter.row() != iter.col() && iter.value() < 0.0)
                {
                    diag_offset += -iter.value();
                    iter.valueRef() = 0.0;
                }
            }
            for (pmp::SparseMatrix::InnerIterator iter(L, k); iter; ++iter)
            {
                if (iter.row() == iter.col() && iter.value() < 0.0)
                    iter.valueRef() -= diag_offset;
            }
        }
    }
}