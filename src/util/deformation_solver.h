#pragma once

#include <Eigen/Core>
#include "../Laplacians/optimized_laplace.h"


void solve_deformation(pmp::SurfaceMesh& mesh, const std::function<bool(int)>& is_fixed, const std::function<bool(int)>& is_handle, const Eigen::Matrix4d& transform, LaplaceConfig config);
void tri_gradient_matrix_rotated(const pmp::SurfaceMesh& mesh, Eigen::MatrixXd& G, LaplaceConfig config);
Eigen::Vector3d affine_transform(const Eigen::Matrix4d& m, const Eigen::Vector3d& v);