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

#include "enums.h"
//=============================================================================


enum DiffusionStep
{
    MeanEdge = 0,
    MaxEdge = 1,
    MaxDiagonal = 2
};

class GeodesicsInHeat
{
public:
    GeodesicsInHeat(pmp::SurfaceMesh& mesh,
                    bool geodist, bool euklid,
                    DiffusionStep diffusion = MeanEdge)
      : mesh_(mesh),
      geodist_sphere_(geodist),
      geodist_cube_(euklid),
      diffusionStep_(diffusion)  {};
    ~GeodesicsInHeat() = default;

    double getDistance(int vertex, Eigen::VectorXd& dist,
                       Eigen::VectorXd& orthodist, bool verbose = true);

    void distance_to_texture_coordinates() const;

    void compute_geodesics(LaplaceConfig config, bool lumped = true);

private:
    pmp::SurfaceMesh& mesh_;

    Eigen::MatrixXd pos;

    Eigen::SparseMatrix<double> divOperator, gradOperator;

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> cholL, cholA;

    static double edgeLength(const pmp::SurfaceMesh& mesh, pmp::Edge e);
    static double averageEdgeLength(const pmp::SurfaceMesh& mesh);
    static double maxEdgeLength(const pmp::SurfaceMesh& mesh);
    static double maxDiagonalLength(const pmp::SurfaceMesh& mesh);

    double great_circle_distance(pmp::Vertex v, pmp::Vertex vv, double r = 1.0);
    double haversine_distance(pmp::Vertex v, pmp::Vertex vv, double r = 1.0);
    double vincenty_distance(pmp::Vertex v, pmp::Vertex vv, double r = 1.0);

    bool geodist_sphere_, geodist_cube_;

    DiffusionStep diffusionStep_;
};
