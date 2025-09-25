// =============================================================================
//  Copyright (c) 2025 Sven D. Wagner, Mario Botsch.
//  Distributed under MIT license, see file LICENSE for details.
// =============================================================================

#pragma once


#include "../util/mesh_converter.h"
#include "../util/enums.h"
#include "optimized_laplace.h"

inline double MOLLIFICATION_FACTOR = 1e-6;
inline double elapsed = 0;

void setup_laplacian(Eigen::SparseMatrix<double>& L, Eigen::DiagonalMatrix<double, Eigen::Dynamic>& M,
                            const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, LaplaceConfig config);
