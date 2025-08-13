//=============================================================================
// Copyright 2023 Astrid Bunge, Mario Botsch.
// Distributed under MIT license, see file LICENSE for details.
//=============================================================================
#pragma once
#include <Eigen/Sparse>
#include <pmp/surface_mesh.h>
#include "../util/enums.h"
#include "pmp/algorithms/numerics.h"

void poly_laplace_matrix(const pmp::SurfaceMesh& mesh, Eigen::SparseMatrix<double>& L, LaplaceConfig config);
void poly_mass_matrix(const pmp::SurfaceMesh& mesh, Eigen::DiagonalMatrix<double, Eigen::Dynamic>& M, LaplaceConfig config);
void poly_divergence_and_gradient_matrix(const pmp::SurfaceMesh& mesh, Eigen::SparseMatrix<double>& D, Eigen::SparseMatrix<double>& G, LaplaceConfig config);
void compute_virtual_vertex(const pmp::DenseMatrix& poly, Eigen::VectorXd& weights);