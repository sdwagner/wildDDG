#pragma once

#include "geodesics_solver.h"
#include "bipoisson_solver.h"
#include "poisson_solver.h"
#include "../Laplacians/construct_laplace.h"
#include "condition_number.h"
#include "smoothing_solver.h"

inline std::pair<double, double> solve_system_lib(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const LaplaceConfig config,
                               const Solve_Function function,
                               const int l, const int m)
{
    pmp::SparseMatrix L;
    pmp::DiagonalMatrix M;
    setup_laplacian(L, M, V, F, config);
    pmp::SurfaceMesh mesh;
    to_pmp_mesh(V, F, mesh);
    switch (type_map[function])
    {
    case BiHarm_Poisson:
        {
            // Constrain two rows of vertices
            auto is_constrained = [&mesh](int index)
            {
                const auto v = pmp::Vertex(index);
                if (mesh.is_boundary(v))
                    return true;
                return std::ranges::any_of(mesh.vertices(v), [&mesh](const pmp::Vertex vp) -> bool { return mesh.is_boundary(vp); });
            };
            return solve_bipoisson_system(L, M, V, F, is_constrained, function, config);
        }
    case Spherical_Poisson:
        {
            return solve_SH_poisson_system(L, M, V, l, m);
        }
    case Poisson:
        {
            auto is_constrained = [&mesh](int index)
            {
                return mesh.is_boundary(pmp::Vertex(index));
            };
            return solve_poisson_system(L, M, V, F, is_constrained, function, config);
        }
    case Condition_Number:
        {
            auto is_constrained = [&mesh](int index)
            {
                return mesh.is_boundary(pmp::Vertex(index));
            };
            return {condition_number(L, Eigen::SparseMatrix<double>(M), is_constrained, function), NAN};
        }
    case Smoothing:
        {
            return implicit_smoothing_error(L, M, V, F, function);
        }
    case Geodesics:
        {
            return compute_geodesics(config, V, F, function);
        }
    case MeanCurvature:
        {
            int nv = V.rows();
            double rms = 0.0;
            Eigen::VectorXd H = 0.5 * (M.inverse() * L * V).rowwise().norm();
            //     compute mean curvature
            if (function == MeanCurvaturePlane)
            {
                for (unsigned int i = 0; i < nv; i++)
                {
                    if (!mesh.is_boundary(pmp::Vertex(i)))
                        rms += H(i);
                }
            }
            else if (function == MeanCurvatureSphere)
            {
                for (unsigned int i = 0; i < nv; i++)
                {
                    if (!mesh.is_boundary(pmp::Vertex(i)))
                        rms += abs(H(i)-1);
                }
            }
            rms /= (double) nv;
            std::cout << "MeanCurvature: " << rms << std::endl;
            return {rms, NAN};
        }
    }
    return {NAN, NAN};
}
inline void smooth_lib(Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const LaplaceConfig config, bool implicit, double dt, bool rescale)
{
    pmp::SparseMatrix L;
    pmp::DiagonalMatrix M;
    setup_laplacian(L, M, V, F, config);
    if (implicit)
        implicit_smoothing(L, M, V, F, dt, rescale);
    else
        explicit_smoothing(L, M, V, dt);
}