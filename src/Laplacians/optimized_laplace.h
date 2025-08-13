#pragma once
#include <pmp/surface_mesh.h>
#include <pmp/algorithms/numerics.h>
#include "../util/enums.h"

inline AreaComputation area_computation = CrossProduct;
inline bool dot_sqlength = false;

double double_triarea(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1,
                      const Eigen::Vector3d& p2, LaplaceConfig config, AreaComputation area_computation);
void triangle_gradient_matrix(const Eigen::Vector3d& p0,
                                     const Eigen::Vector3d& p1,
                                     const Eigen::Vector3d& p2, pmp::DenseMatrix& G, LaplaceConfig config, bool verbose = false);
void triangle_laplace_matrix(const Eigen::Vector3d& p0,
                                    const Eigen::Vector3d& p1,
                                    const Eigen::Vector3d& p2, pmp::DenseMatrix& Ltri, LaplaceConfig config);
void divmass_matrix(const pmp::SurfaceMesh& mesh, pmp::DiagonalMatrix& M, LaplaceConfig config);
void tri_gradient_matrix(const pmp::SurfaceMesh& mesh, pmp::SparseMatrix& G, LaplaceConfig config);
void tri_divergence_matrix(const pmp::SurfaceMesh& mesh, pmp::SparseMatrix& D, LaplaceConfig config);
void tri_laplace_matrix(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, pmp::SparseMatrix& L, LaplaceConfig config);
void tri_laplace_matrix(const pmp::SurfaceMesh& mesh, pmp::SparseMatrix& L, LaplaceConfig config);
void tri_mass_matrix(const pmp::SurfaceMesh& mesh, pmp::DiagonalMatrix& M, LaplaceConfig config);
void get_divergence_and_gradient_matrix(const pmp::SurfaceMesh& mesh, pmp::SparseMatrix& D, pmp::SparseMatrix& G, LaplaceConfig config);