// =============================================================================
//  Copyright (c) 2025 Sven D. Wagner, Mario Botsch.
//  Distributed under MIT license, see file LICENSE for details.
// =============================================================================

#pragma once

#include "enums.h"
#include "pmp/algorithms/differential_geometry.h"
#include "pmp/algorithms/numerics.h"

inline std::pair<double, double> implicit_smoothing_error(const Eigen::SparseMatrix<double>& S, const Eigen::DiagonalMatrix<double, Eigen::Dynamic>& M,
                                      const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Solve_Function function)
{
    try
    {
        int n = S.rows();
        pmp::SurfaceMesh mesh;
        to_pmp_mesh(V, F, mesh);
        auto is_boundary = [&mesh](int i){return mesh.is_boundary(pmp::Vertex(i));};
        Eigen::MatrixXd X = pmp::cholesky_solve(Eigen::SparseMatrix<double>(M) - 0.1 * S, M * V, is_boundary, V);
        Eigen::VectorXd dists = X.rowwise().norm();
        double avg_dist = dists.mean();
        double error = sqrt((dists - avg_dist * Eigen::VectorXd::Ones(n)).squaredNorm() / n);
        std::cout << "Smoothing RMSE error: " << error << std::endl;
        return {error, NAN};
    }
    catch (const std::exception& e)
    {
        std::cout << "Smoothing RMSE error: " << NAN << std::endl;
        return {NAN, NAN};
    }

}

inline void implicit_smoothing(const Eigen::SparseMatrix<double>& S, const Eigen::DiagonalMatrix<double, Eigen::Dynamic>& M, Eigen::MatrixXd& V, const Eigen::MatrixXi& F, double dt, bool rescale)
{
    pmp::Point center_before;
    pmp::Scalar area_before(0);
    pmp::SurfaceMesh mesh;
    to_pmp_mesh(V, F, mesh);
    auto is_boundary = [&mesh](int i){return mesh.is_boundary(pmp::Vertex(i));};

    if (rescale)
    {
        center_before = centroid(mesh);
        area_before = pmp::surface_area(mesh);
    }
    V = pmp::cholesky_solve(Eigen::SparseMatrix<double>(M) - dt * S, M * V, is_boundary, V);
    if (rescale)
    {
        to_pmp_mesh(V, F, mesh);
        // restore original center
        const pmp::Point center_after = centroid(mesh);
        for (int i = 0; i < V.rows(); i++)
            V.row(i) = Eigen::Vector3d(V.row(i)) - Eigen::Vector3d(center_after);
        // restore original surface area
        const double area_after = surface_area(mesh);
        const double scale = sqrt(area_before / area_after);
        V = scale * V;

        for (int i = 0; i < V.rows(); i++)
            V.row(i) = Eigen::Vector3d(V.row(i)) + Eigen::Vector3d(center_before);
    }

}

inline void explicit_smoothing(const Eigen::SparseMatrix<double>& S, const Eigen::DiagonalMatrix<double, Eigen::Dynamic>& M, Eigen::MatrixXd& V, double dt)
{
    try
    {
        int n = S.rows();
        V = pmp::cholesky_solve(Eigen::SparseMatrix<double>(M), (Eigen::SparseMatrix<double>(M) + dt * S) * V);
        Eigen::VectorXd dists = V.rowwise().norm();
        double avg_dist = dists.mean();
        double error = sqrt((dists - avg_dist * Eigen::VectorXd::Ones(n)).squaredNorm() / n);
        std::cout << "Smoothing RMSE error: " << error << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cout << "Smoothing RMSE error: " << NAN << std::endl;
    }

}