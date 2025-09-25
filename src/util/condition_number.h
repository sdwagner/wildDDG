// =============================================================================
//  Copyright (c) 2025 Sven D. Wagner, Mario Botsch.
//  Distributed under MIT license, see file LICENSE for details.
// =============================================================================

//=============================================================================
// Copyright 2023 Astrid Bunge, Mario Botsch.
// Distributed under MIT license, see file LICENSE for details.
//=============================================================================

#pragma once

#include "enums.h"
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/MatOp/SparseCholesky.h>
#include <Spectra/Util/GEigsMode.h>
#include <Spectra/Util/SelectionRule.h>
#include <Spectra/Util/CompInfo.h>
#include <Spectra/SymGEigsShiftSolver.h>
#include <Spectra/SymGEigsSolver.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/SymEigsSolver.h>
#include <iostream>

inline double condition_number(const Eigen::SparseMatrix<double>& S,
                               const Eigen::SparseMatrix<double>& M,
                               const std::function<bool(unsigned int)>& is_constrained,
                               const Solve_Function function)
{
    int num_vertices = S.rows();
    //slice matrices so that only rows and cols for inner vertices remain
    std::vector<int> innerVertIdxs;
    for (int i = 0; i < num_vertices; ++i)
    {
        if (!is_constrained(i))
        {
            innerVertIdxs.push_back(i);
        }
    }
    int nInnerVertIdxs = innerVertIdxs.size();

    Eigen::SparseMatrix<double> S_in_in(nInnerVertIdxs, nInnerVertIdxs);
    Eigen::SparseMatrix<double> M_in_in(nInnerVertIdxs, nInnerVertIdxs);
    if (nInnerVertIdxs == num_vertices)
    {
        S_in_in = S;
        M_in_in = M;
    }
    else
    {
        Eigen::SparseMatrix<double> S_columns(S.rows(), nInnerVertIdxs);
        Eigen::SparseMatrix<double> M_columns(M.rows(), nInnerVertIdxs);
        Eigen::SparseMatrix<double, Eigen::RowMajor> S_rows(nInnerVertIdxs,
                                                            nInnerVertIdxs);
        Eigen::SparseMatrix<double, Eigen::RowMajor> M_rows(nInnerVertIdxs,
                                                            nInnerVertIdxs);

        // process rows and columns separately for linear runtime
        for (int i = 0; i < nInnerVertIdxs; i++)
        {
            S_columns.col(i) = S.col(innerVertIdxs[i]);
            M_columns.col(i) = M.col(innerVertIdxs[i]);
        }
        for (int i = 0; i < nInnerVertIdxs; i++)
        {
            S_rows.row(i) = S_columns.row(innerVertIdxs[i]);
            M_rows.row(i) = M_columns.row(innerVertIdxs[i]);
        }
        S_in_in = S_rows;
        M_in_in = M_rows;
    }

    int numEigValues = 3;
    int convergenceSpeed = std::min(40 * numEigValues, (int)S_in_in.rows());

    Eigen::VectorXd eigValsMax;
    Eigen::VectorXd eigValsMin;
    try
    {
        if (function == ConditioningGeneralized)
        {
            // Construct generalized eigen solver object, requesting the largest generalized eigenvalue
            Spectra::SparseSymMatProd<double> sOpMax(-S_in_in);
            Spectra::SparseCholesky<double> sBOpMax(M_in_in);
            Spectra::SymGEigsSolver<Spectra::SparseSymMatProd<double>,
                                    Spectra::SparseCholesky<double>,
                                    Spectra::GEigsMode::Cholesky>
            eigSolverMax(sOpMax, sBOpMax, 1, convergenceSpeed);
            eigSolverMax.init();
            eigSolverMax.compute(Spectra::SortRule::LargestAlge);
            eigValsMax = eigSolverMax.eigenvalues();

            if (eigSolverMax.info() != Spectra::CompInfo::Successful)
            {
                std::cout << "Condition Number: " << NAN << std::endl;
                return NAN;
            }

            // Construct generalized eigen solver object, seeking three generalized
            // eigenvalues that are closest to zero. This is equivalent to specifying
            // a shift sigma = 0.0 combined with the SortRule::LargestMagn selection rule
            Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse> sOpMin(
                -S_in_in, M_in_in);
            Spectra::SparseSymMatProd<double> sBOpMin(M_in_in);
            Spectra::SymGEigsShiftSolver<
                    Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>,
                    Spectra::SparseSymMatProd<double>, Spectra::GEigsMode::ShiftInvert>
                eigSolverMin(sOpMin, sBOpMin, numEigValues, convergenceSpeed, -0.1);
            eigSolverMin.init();
            eigSolverMin.compute(Spectra::SortRule::LargestMagn);
            eigValsMin = eigSolverMin.eigenvalues();

            if (eigSolverMin.info() != Spectra::CompInfo::Successful)
            {
                std::cout << "Condition Number: " << NAN << std::endl;
                return NAN;
            }
        }
        else
        {
            // Max Eigenvalue solver
            Spectra::SparseSymMatProd<double> sOpMax(-S_in_in);
            Spectra::SymEigsSolver eigSolverMax(sOpMax, 1, convergenceSpeed);
            eigSolverMax.init();
            eigSolverMax.compute(Spectra::SortRule::LargestAlge);
            eigValsMax = eigSolverMax.eigenvalues();

            if (eigSolverMax.info() != Spectra::CompInfo::Successful)
            {
                std::cout << "Condition Number: " << NAN << std::endl;
                return NAN;
            }

            // Min Eigenvalue solver
            Spectra::SparseSymShiftSolve<double> sOpMin(-S_in_in);
            Spectra::SymEigsShiftSolver eigSolverMin(sOpMin, numEigValues, convergenceSpeed, -0.1);
            eigSolverMin.init();
            eigSolverMin.compute(Spectra::SortRule::LargestMagn);
            eigValsMin = eigSolverMin.eigenvalues();

            if (eigSolverMin.info() != Spectra::CompInfo::Successful)
            {
                std::cout << "Condition Number: " << NAN << std::endl;
                return NAN;
            }
        }

        //std::cout << "Min eigenvalues: " << eigValsMin << std::endl;
        //std::cout << "Max eigenvalue: " << eigValsMax << std::endl;

        Eigen::Vector3d values;

        values(0) = eigValsMax.coeff(0);
        values(1) =
            eigValsMin.coeff(numEigValues - 1 - (num_vertices == innerVertIdxs.size()));
        values(2) = values(0) / values(1);
        values(2) = (values(2) < 0.0) ? NAN : values(2);
        std::cout << "Condition Number: " << values(2) << std::endl;

        return values(2);
    }
    catch (std::exception& e)
    {
        std::cout << "Condition Number: " << NAN << std::endl;
        return NAN;
    }
}
