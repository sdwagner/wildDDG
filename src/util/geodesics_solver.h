// =============================================================================
//  Copyright (c) 2025 Sven D. Wagner, Mario Botsch.
//  Distributed under MIT license, see file LICENSE for details.
// =============================================================================

#pragma once

#include "enums.h"

std::pair<double, double> compute_geodesics(const LaplaceConfig& config, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Solve_Function function);
