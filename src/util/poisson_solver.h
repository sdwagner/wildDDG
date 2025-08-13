//=============================================================================
// Copyright 2023 Astrid Bunge, Mario Botsch.
// Distributed under MIT license, see file LICENSE for details.
//=============================================================================
#pragma once
#include "pmp/algorithms/numerics.h"
#include "enums.h"
#include "h1_seminorm.h"


//-----------------------------------------------------------------------------


inline double franke_function(double x, double y)
{
    double cx2 = (9. * x - 2.) * (9. * x - 2.);
    double cy2 = (9. * y - 2.) * (9. * y - 2.);

    double cx1 = (9. * x + 1.) * (9. * x + 1.);
    double cx7 = (9. * x - 7.) * (9. * x - 7.);

    double cy3 = (9. * y - 3.) * (9. * y - 3.);
    double cx4 = (9. * x - 4.) * (9. * x - 4.);

    double cy7 = (9. * y - 7.) * (9. * y - 7.);

    return (3. / 4.) * exp(-(1. / 4.) * cx2 - (1. / 4.) * cy2) +
        (3. / 4.) * exp(-(1. / 49.) * cx1 - (9. / 10.) * y - 1. / 10.) +
        (1. / 2.) * exp(-(1. / 4.) * cx7 - (1. / 4.) * cy3) -
        (1. / 5.) * exp(-cx4 - cy7);
}

//-----------------------------------------------------------------------------

inline double laplace_franke_function(double x, double y)
{
    double mathematica =
        64.8 * exp(-pow(-4. + 9. * x, 2.0) - pow(-7. + 9. * y, 2.0)) -
        40.5 * exp(0.25 * (-pow(-7. + 9. * x, 2) - pow(-3. + 9. * y, 2))) -
        60.75 * exp(0.25 * (-pow(-2. + 9. * x, 2) - pow(-2. + 9. * y, 2))) -
        1.8720918367346937 * exp(-0.02040816326530612 * pow(1. + 9. * x, 2) -
            0.1 * (1. + 9. * y)) +
        10.125 * exp(0.25 * (-pow(-7. + 9. * x, 2) - pow(-3. + 9. * y, 2))) *
        pow(-7. + 9. * x, 2) -
        64.8 * exp(-pow(-4. + 9. * x, 2) - pow(-7. + 9. * y, 2)) *
        pow(-4. + 9. * x, 2) +
        15.1875 * exp(0.25 * (-pow(-2. + 9. * x, 2) - pow(-2. + 9. * y, 2))) *
        pow(-2. + 9. * x, 2) +
        0.1012078300708038 *
        exp(-0.02040816326530612 * pow(1. + 9. * x, 2) -
            0.1 * (1. + 9. * y)) *
        pow(1. + 9. * x, 2) -
        64.8 * exp(-pow(-4. + 9. * x, 2) - pow(-7. + 9. * y, 2)) *
        pow(-7. + 9. * y, 2) +
        10.125 * exp(0.25 * (-pow(-7. + 9. * x, 2) - pow(-3. + 9. * y, 2))) *
        pow(-3. + 9. * y, 2) +
        15.1875 * exp(0.25 * (-pow(-2. + 9. * x, 2) - pow(-2. + 9. * y, 2))) *
        pow(-2. + 9. * y, 2);
    return mathematica;
}

inline double poisson_function(const Eigen::Vector3d& p, Solve_Function function)
{
    switch (function)
    {
    case Franke2d:
        return franke_function(p[0], p[1]);
    case XpY:
        return p[0] + p[1];
    case X2pY2:
        return (p[0] + 1) * (p[0] + 1) + (p[1] + 1) * (p[1] + 1);
    case Runge:
        return 1.0 / (1 + p[0] * p[0] + p[1] * p[1]);
    default:
        std::cerr << "Function not implemented" << std::endl;
        return -50.0;
    }
}

//-----------------------------------------------------------------------------

inline double laplace_of_poisson_function(const Eigen::Vector3d& p, Solve_Function function)
{
    switch (function)
    {
    case Franke2d:
        return laplace_franke_function(p[0], p[1]);
    case XpY:
        return 0;
    case X2pY2:
        return 4;
    case Runge:
        return 4.0 * (-1 + p[0] * p[0] + p[1] * p[1]) / pow(1.0 + p[0] * p[0] + p[1] * p[1], 3.0);
    default:
        std::cerr << "Function not implemented" << std::endl;
        return -50.0;
    }
}


inline std::pair<double, double> solve_poisson_system(const Eigen::SparseMatrix<double>& S, const Eigen::DiagonalMatrix<double, Eigen::Dynamic>& M,
                                   const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                                   const std::function<bool(unsigned int)>& is_constrained, const Solve_Function function, const LaplaceConfig& config)
{
    const int nv = static_cast<int>(S.cols());
    Eigen::VectorXd b(nv), analytic_solution(nv);

    for (int i = 0; i < nv; i++)
    {
        b(i) = laplace_of_poisson_function(V.row(i), function);
        analytic_solution(i) = poisson_function(V.row(i), function);
    }
    b = M * b;

    try
    {
        const Eigen::VectorXd x = pmp::cholesky_solve(S, b, is_constrained, analytic_solution);
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


inline double factorial(int n)
{
    if (n == 0)
        return 1.0;
    return (double)n * factorial(n - 1);
}

//----------------------------------------------------------------------------

inline double scale(int l, int m)
{
    double temp = ((2.0 * (double)l + 1.0) * factorial(l - m)) /
                  (4.0 * M_PI * factorial(l + m));
    return sqrt(temp);
}

//----------------------------------------------------------------------------

inline double legendre_Polynomial(int l, int m, double x)
{
    // evaluate an Associated Legendre Polynomial P(l,m,x) at x
    double pmm = 1.0;
    if (m > 0)
    {
        double somx2 = sqrt((1.0 - x) * (1.0 + x));
        double fact = 1.0;
        for (int i = 1; i <= m; i++)
        {
            pmm *= (-fact) * somx2;
            fact += 2.0;
        }
    }
    if (l == m)
        return pmm;
    double pmmp1 = x * (2.0 * (double)m + 1.0) * pmm;
    if (l == m + 1)
        return pmmp1;
    double pll = 0.0;
    for (int ll = m + 2; ll <= l; ++ll)
    {
        pll = ((2.0 * (double)ll - 1.0) * x * pmmp1 -
               ((double)ll + (double)m - 1.0) * pmm) /
              ((double)ll - (double)m);
        pmm = pmmp1;
        pmmp1 = pll;
    }
    return pll;
}



inline double sphericalHarmonic(pmp::Point p, int l, int m)
{
    // l is the band, range [0..n]
    // m in the range [-l..l]
    // transform cartesian to spherical coordinates, assuming r = 1

    double phi = atan2(p[0], p[2]) + M_PI;
    double cos_theta = p[1] / norm(p);
    const double sqrt2 = sqrt(2.0);
    if (m == 0)
        return scale(l, 0) * legendre_Polynomial(l, m, cos_theta);
    if (m > 0)
        return sqrt2 * scale(l, m) * cos((double)m * phi) *
               legendre_Polynomial(l, m, cos_theta);
    return sqrt2 * scale(l, -m) * sin(-(double)m * phi) *
               legendre_Polynomial(l, -m, cos_theta);
}

inline std::pair<double, double> solve_SH_poisson_system(const Eigen::SparseMatrix<double>& S, const Eigen::DiagonalMatrix<double, Eigen::Dynamic>& M,
                                      const Eigen::MatrixXd& V, const int l, const int m)
{
    const int nv = static_cast<int>(S.cols());
    Eigen::VectorXd analytic_solution(nv);

    for (int i = 0; i < nv; i++)
    {
        analytic_solution(i) = sphericalHarmonic(V.row(i), l, m);
    }

    try
    {
        Eigen::MatrixXd X = pmp::cholesky_solve(Eigen::SparseMatrix<double>(M), S * analytic_solution);
        double eval = -l * (l + 1);
        double error = (analytic_solution - 1.0 / eval * X).transpose() * M *
            (analytic_solution - 1.0 / eval * X);
        error = sqrt(error / double(nv));
        std::cout << "error spherical harmonics "
            << "Y_" << l << "^" << m << ": " << error << std::endl;
        return {error, NAN};
    }
    catch (const std::exception& e)
    {
        std::cout << "error spherical harmonics "
            << "Y_" << l << "^" << m << ": " << NAN << std::endl;
        return {NAN, NAN};
    }
}
