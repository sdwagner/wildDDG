#pragma once

#include "pmp/algorithms/numerics.h"
#include "poisson_solver.h"
#include "enums.h"


inline double bilaplace_franke_function(double x, double y)
{
    return ((0.013657366490187605 * pow(9 * x + 1, 4) - 1.8436761893428761 * pow(9 * x + 1, 2) + 21.068638931695126)
            * exp((1.0 / 4.0) * pow(9 * x - 7, 2) + pow(9 * x - 4, 2) + (1.0 / 4.0) * pow(9 * x - 2, 2) + (9.0 / 4.0) *
                pow(3 * y - 1, 2) + pow(9 * y - 7, 2) + (1.0 / 4.0) * pow(9 * y - 2, 2))
            + (205.03125 * pow(9 * x - 7, 4) + 3690.5625 * pow(9 * x - 7, 2) * pow(3 * y - 1, 2)
                - 3280.5 * pow(9 * x - 7, 2) + 16607.53125 * pow(3 * y - 1, 4) - 29524.5 * pow(3 * y - 1, 2) + 6561.0)
            * exp((9.0 / 10.0) * y + pow(9 * x - 4, 2) + (1.0 / 4.0) * pow(9 * x - 2, 2) + (1.0 / 49.0) *
                pow(9 * x + 1, 2) + pow(9 * y - 7, 2)
                + (1.0 / 4.0) * pow(9 * y - 2, 2) + 1.0 / 10.0) + (-20995.2 * pow(9 * x - 4, 4) - 41990.4 *
                pow(9 * x - 4, 2) * pow(9 * y - 7, 2)
                + 83980.8 * pow(9 * x - 4, 2) - 20995.2 * pow(9 * y - 7, 4) + 83980.8 * pow(9 * y - 7, 2) - 41990.4) *
            exp((9.0 / 10.0) * y
                + (1.0 / 4.0) * pow(9 * x - 7, 2) + (1.0 / 4.0) * pow(9 * x - 2, 2) + (1.0 / 49.0) * pow(9 * x + 1, 2) +
                (9.0 / 4.0) * pow(3 * y - 1, 2) + (1.0 / 4.0) * pow(9 * y - 2, 2) + 1.0 / 10.0)
            + (307.546875 * pow(9 * x - 2, 4) + 615.09375 * pow(9 * x - 2, 2) * pow(9 * y - 2, 2) - 4920.75 *
                pow(9 * x - 2, 2) + 307.546875 * pow(9 * y - 2, 4) - 4920.75 * pow(9 * y - 2, 2) + 9841.5)
            * exp((9.0 / 10.0) * y + (1.0 / 4.0) * pow(9 * x - 7, 2) + pow(9 * x - 4, 2) + (1.0 / 49.0) *
                pow(9 * x + 1, 2) + (9.0 / 4.0) * pow(3 * y - 1, 2) + pow(9 * y - 7, 2) + 1.0 / 10.0))
        * exp(-9.0 / 10.0 * y - 1.0 / 4.0 * pow(9 * x - 7, 2) - pow(9 * x - 4, 2) - 1.0 / 4.0 * pow(9 * x - 2, 2) - 1.0
            / 49.0 * pow(9 * x + 1, 2) - 9.0 / 4.0 * pow(3 * y - 1, 2) - pow(9 * y - 7, 2)
            - 1.0 / 4.0 * pow(9 * y - 2, 2) - 1.0 / 10.0);
}


inline double bilaplacian(const Eigen::Vector3d& p, Solve_Function function)
{
    switch (function)
    {
    case BiHarm_Franke:
        {
            return bilaplace_franke_function(p.x(), p.y());
        }
    default:
        {
            std::cerr << "Bilaplacian function not implemented" << std::endl;
            return 0.0;
        }
    }
}

inline double bilaplace_function(const Eigen::Vector3d& p, Solve_Function function)
{
    switch (function)
    {
    case BiHarm_Franke:
        {
            return franke_function(p.x(), p.y());
        }
    default:
        {
            std::cerr << "Bilaplacian function not implemented" << std::endl;
            return 0.0;
        }
    }
}

inline std::pair<double, double> solve_bipoisson_system(const Eigen::SparseMatrix<double>& S,
                                      const Eigen::DiagonalMatrix<double, Eigen::Dynamic>& M,
                                      const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                                      const std::function<bool(unsigned int)>& is_constrained,
                                      const Solve_Function function, const LaplaceConfig& config)
{
    const int nv = static_cast<int>(S.cols());
    Eigen::VectorXd b(nv), analytic_solution(nv);

    for (int i = 0; i < nv; i++)
    {
        b(i) = bilaplacian(V.row(i), function);
        analytic_solution(i) = bilaplace_function(V.row(i), function);
    }
    b = M * b;

    try
    {
        const Eigen::VectorXd x = pmp::cholesky_solve(S * M.inverse() * S, b, is_constrained, analytic_solution);
        const double error = sqrt((x - analytic_solution).squaredNorm() / static_cast<double>(nv));
        std::cout << "RMSE error inner vertices: " << error << std::endl;
        double grad_error = h1_seminorm(V, F, config, function, x);
        std::cout << "RMSE h1 error: " << grad_error << std::endl;
        return {error, grad_error};
    }
    catch (const std::exception& e)
    {
        std::cout << "RMSE error inner vertices: " << NAN << std::endl;
        std::cout << "RMSE h1 error: " << NAN << std::endl;
        return {NAN, NAN};
    }
}
