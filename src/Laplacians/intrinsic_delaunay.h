// =============================================================================
//  Copyright (c) 2025 Sven D. Wagner, Mario Botsch.
//  Distributed under MIT license, see file LICENSE for details.
// =============================================================================

#pragma once

#include "geometrycentral/surface/vertex_position_geometry.h"

enum class FlipType { Euclidean = 0, Hyperbolic };
size_t flipToDelaunay(geometrycentral::surface::SurfaceMesh& mesh, geometrycentral::surface::EdgeData<double>& edgeLengths, FlipType flipType = FlipType::Euclidean,
                      double delaunayEPS = 1e-6);
std::tuple<Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>>
idt_laplacian(const geometrycentral::surface::SurfaceMesh& mesh,
                                geometrycentral::surface::EmbeddedGeometryInterface& geom,
                                double relativeMollificationFactor, bool buildDelaunay = true);
