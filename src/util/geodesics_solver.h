#pragma once

#include "enums.h"

std::pair<double, double> compute_geodesics(const LaplaceConfig& config, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Solve_Function function);
