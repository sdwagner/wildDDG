// =============================================================================
//  Copyright (c) 2025 Sven D. Wagner, Mario Botsch.
//  Distributed under MIT license, see file LICENSE for details.
// =============================================================================

//=============================================================================
// Copyright 2023 Astrid Bunge, Mario Botsch.
// Distributed under MIT license, see file LICENSE for details.
//=============================================================================
#pragma once
//=============================================================================

#include <pmp/surface_mesh.h>
#include <Eigen/Sparse>

//=============================================================================

using namespace pmp;

//=============================================================================

class Curvature
{
public:
    Curvature(SurfaceMesh& mesh, bool compare)
        : mesh_(mesh), compare_to_sphere(compare) {}

    //! Visualizes the mean curvature of our mesh.
    void visualize_curvature(const Eigen::SparseMatrix<double> &S, const Eigen::DiagonalMatrix<double, Eigen::Dynamic> &M);


private:
    SurfaceMesh& mesh_;
    bool compare_to_sphere;

    //! convert curvature values ("v:curv") to 1D texture coordinates
    void curvature_to_texture_coordinates() const;
};

//=============================================================================
