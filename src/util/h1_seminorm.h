// =============================================================================
//  Copyright (c) 2025 Sven D. Wagner, Mario Botsch.
//  Distributed under MIT license, see file LICENSE for details.
// =============================================================================

#pragma once

#include "enums.h"
#include "../Laplacians/optimized_laplace.h"
#include "mesh_converter.h"

inline Eigen::Vector3d evaluate_grad(const Eigen::Vector3d& p, const Solve_Function function)
{
    Eigen::Vector3d grad(0.0, 0.0, 0.0);
    switch (function)
    {
    case Franke2d:
    case BiHarm_Franke:
        {
            const double x = p(0);
            const double y = p(1);
            grad[0] = -20.25 * x * exp(-1.0 / 4.0 * pow(9 * x - 7, 2) - 9.0 / 4.0 * pow(3 * y - 1, 2)) +
                32.4 * x * exp(-pow(9 * x - 4, 2) - pow(9 * y - 7, 2)) - 30.375 * x *
                exp(-1.0 / 4.0 * pow(9 * x - 2, 2) - 1.0 / 4.0 * pow(9 * y - 2, 2)) - 2.4795918367346936 * x *
                exp(-9.0 / 10.0 * y - 1.0 / 49.0 * pow(9 * x + 1, 2) - 1.0 / 10.0) + 15.75 *
                exp(-1.0 / 4.0 * pow(9 * x - 7, 2) - 9.0 / 4.0 * pow(3 * y - 1, 2)) - 14.4 *
                exp(-pow(9 * x - 4, 2) - pow(9 * y - 7, 2)) + 6.75 *
                exp(-1.0 / 4.0 * pow(9 * x - 2, 2) - 1.0 / 4.0 * pow(9 * y - 2, 2)) - 0.27551020408163263 * exp(
                    -9.0 / 10.0 * y - 1.0 / 49.0 * pow(9 * x + 1, 2) - 1.0 / 10.0);
            grad[1] = -0.5 * ((81.0 / 2.0) * y - 27.0 / 2.0) *
                exp(-1.0 / 4.0 * pow(9 * x - 7, 2) - 1.0 / 4.0 * pow(9 * y - 3, 2)) - 0.75 * ((81.0 / 2.0) * y - 9) *
                exp(-1.0 / 4.0 * pow(9 * x - 2, 2) - 1.0 / 4.0 * pow(9 * y - 2, 2)) + 0.2 * (162 * y -
                    126) * exp(-pow(9 * x - 4, 2) - pow(9 * y - 7, 2)) - 0.675 * exp(
                    -9.0 / 10.0 * y - 1.0 / 49.0 * pow(9 * x + 1, 2) - 1.0 / 10.0);
            grad[2] = 0.0;
            break;
        }
    default:
        break;
    }
    return grad;
}

// Compare FEM gradient of solution to analytical gradient evaluated at face center
inline double h1_seminorm(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const LaplaceConfig& config,
                          const Solve_Function function, const Eigen::VectorXd& solution)
{
    pmp::SurfaceMesh mesh;
    to_pmp_mesh(V, F, mesh);
    Eigen::SparseMatrix<double> D, G;
    get_divergence_and_gradient_matrix(mesh, D, G, config);
    Eigen::VectorXd grad = G * solution;
    const Eigen::MatrixXd grad_mat = grad.reshaped(3, F.rows()).transpose();
    Eigen::MatrixXd grad_gt_mat = Eigen::MatrixXd::Zero(F.rows(), 3);
    for (int i = 0; i < F.rows(); ++i)
    {
        Eigen::Vector3d p = V.row(F(i, 0));
        p += V.row(F(i, 1));
        p += V.row(F(i, 2));
        p /= 3.0;
        grad_gt_mat.row(i) = evaluate_grad(p, function);
    }
    return sqrt((grad_gt_mat - grad_mat).rowwise().squaredNorm().mean());
}
