// =============================================================================
//  Copyright (c) 2025 Sven D. Wagner, Mario Botsch.
//  Distributed under MIT license, see file LICENSE for details.
// =============================================================================

//=============================================================================
// Copyright 2023 Astrid Bunge, Mario Botsch.
// Distributed under MIT license, see file LICENSE for details.
//=============================================================================

#include <pmp/surface_mesh.h>
#include "GeodesicsInHeat.h"
#include <Eigen/Sparse>
#include <iostream>
#include <pmp/algorithms/normals.h>
#include <cfloat>

#include "mesh_converter.h"
#include "../Laplacians/bunge_poly_laplace.h"
#include "../Laplacians/optimized_laplace.h"

//=============================================================================


//-----------------------------------------------------------------------------

double GeodesicsInHeat::edgeLength(const pmp::SurfaceMesh& mesh, pmp::Edge e)
{
    auto v0 = mesh.vertex(e, 0);
    auto v1 = mesh.vertex(e, 1);
    return pmp::distance(mesh.position(v0), mesh.position(v1));
}

//-----------------------------------------------------------------------------

double GeodesicsInHeat::averageEdgeLength(const pmp::SurfaceMesh& mesh)
{
    double avgLen = 0.;

    for (auto e : mesh.edges())
    {
        avgLen += edgeLength(mesh, e);
    }

    return avgLen / (double)mesh.n_edges();
}
//-----------------------------------------------------------------------------

double GeodesicsInHeat::maxEdgeLength(const pmp::SurfaceMesh& mesh)
{
    double maxLen = 0.;
    double currLen;
    for (auto e : mesh.edges())
    {
        currLen = edgeLength(mesh, e);
        if (currLen > maxLen)
        {
            maxLen = currLen;
        }
    }

    return maxLen;
}
//-----------------------------------------------------------------------------

double GeodesicsInHeat::maxDiagonalLength(const pmp::SurfaceMesh& mesh)
{
    double maxDiag = 0.;
    double currLen;
    for (auto f : mesh.faces())
    {
        for (auto v : mesh.vertices(f))
        {
            for (auto vv : mesh.vertices(f))
            {
                currLen = distance(mesh.position(v), mesh.position(vv));
                if (currLen > maxDiag)
                {
                    maxDiag = currLen;
                }
            }
        }
    }
    return maxDiag;
}
//-----------------------------------------------------------------------------

void GeodesicsInHeat::compute_geodesics(LaplaceConfig config, bool lumped)
{
    pos.resize((int)mesh_.n_vertices(), 3);

    for (int i = 0; i < (int)mesh_.n_vertices(); ++i)
        for (int j = 0; j < 3; ++j)
            pos(i, j) = mesh_.positions()[i][j];
    Eigen::SparseMatrix<double> S;
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> M;

    int max_valence = 3;
    for (auto f: mesh_.faces())
    {
        max_valence = fmax(max_valence, mesh_.valence(f));
    }
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    pmp_to_arrays(V, F, mesh_);
    if (max_valence == 3)
    {
        get_divergence_and_gradient_matrix(mesh_, divOperator, gradOperator, config);
        tri_laplace_matrix(mesh_, S, config);
        tri_mass_matrix(mesh_, M, config);
    }
    else
    {
        poly_divergence_and_gradient_matrix(mesh_, divOperator, gradOperator, config);
        poly_laplace_matrix(mesh_, S, config);
        poly_mass_matrix(mesh_, M, config);
    }

    double h;
    if (diffusionStep_ == MeanEdge)
    {
        h = pow(averageEdgeLength(mesh_), 2);
        std::cout << "Mean edge length diffusion \n";
    }
    else if (diffusionStep_ == MaxEdge)
    {
        h = pow(maxEdgeLength(mesh_), 2);
        std::cout << "Max edge length diffusion \n";
    }
    else
    {
        h = pow(maxDiagonalLength(mesh_), 2);
        std::cout << "Max diagonal length diffusion \n";
    }

    Eigen::SparseMatrix<double> A = pmp::SparseMatrix(M) - h * S;

    cholL.analyzePattern(S);
    cholL.factorize(S);

    cholA.analyzePattern(A);
    cholA.factorize(A);
}

//-----------------------------------------------------------------------------

double GeodesicsInHeat::getDistance(const int vertex, Eigen::VectorXd& dist,
                                    Eigen::VectorXd& orthodist, bool verbose)
{
    // diffuse heat
    const int N = (int)mesh_.n_vertices();

    auto distances = mesh_.add_vertex_property<pmp::Scalar>("v:dist");

    Eigen::SparseVector<double> b(N);
    b.coeffRef(vertex) = 1.;

    // compute gradients
    Eigen::VectorXd heat = cholA.solve(b);
    Eigen::VectorXd grad = gradOperator * heat;

    for (int i = 0; i < grad.rows(); i += 3)
    {
        grad.block(i, 0, 3, 1).normalize();
    }
    dist = cholL.solve(divOperator * (-grad));
    dist = dist.array() - dist.minCoeff();
    orthodist.resize(dist.size());

    int k = 0;
    pmp::Vertex v0 = pmp::Vertex(vertex);
    double rms = 0.0;
    double radius = norm(mesh_.position(v0));
    for (auto v : mesh_.vertices())
    {
        distances[v] = dist[k];

        if (geodist_sphere_)
        {
            orthodist(k) = great_circle_distance(v0, v, radius);
            rms += (dist(k) - orthodist(k)) * (dist(k) - orthodist(k));
        }

        if (geodist_cube_)
        {
            orthodist(k) = norm(mesh_.position(v0) - mesh_.position(v));
            rms += (dist(k) - orthodist(k)) * (dist(k) - orthodist(k));
        }

        k++;
    }
    std::string meshtype;
    if (geodist_sphere_)
    {
        rms /= (double)mesh_.n_vertices();
        rms = sqrt(rms);
        rms /= radius;
        meshtype = " sphere ";
    }
    else if (geodist_cube_)
    {
        rms /= (double)mesh_.n_vertices();
        rms = sqrt(rms);
        meshtype = " plane ";
    }
    if ((geodist_cube_ || geodist_sphere_) && verbose)
    {

        std::cout << "Distance deviation" + meshtype + ": " << rms << std::endl;
    }
    distance_to_texture_coordinates();
    mesh_.remove_vertex_property<pmp::Scalar>(distances);
    return rms;
}

//-----------------------------------------------------------------------------

void GeodesicsInHeat::distance_to_texture_coordinates() const
{
    //     remove per-halfedge texture coordinates
    auto htex = mesh_.get_halfedge_property<pmp::TexCoord>("h:tex");
    if (htex)
        mesh_.remove_halfedge_property(htex);

    auto distances = mesh_.get_vertex_property<pmp::Scalar>("v:dist");
    assert(distances);

    // find maximum distance
    pmp::Scalar maxdist(0);
    for (auto v : mesh_.vertices())
    {
        if (distances[v] <= FLT_MAX)
        {
            maxdist = std::max(maxdist, distances[v]);
        }
    }

    auto tex = mesh_.vertex_property<pmp::TexCoord>("v:tex");
    for (auto v : mesh_.vertices())
    {
        if (distances[v] <= FLT_MAX)
        {
            tex[v] = pmp::TexCoord(distances[v] / maxdist, 0.0);
        }
        else
        {
            tex[v] = pmp::TexCoord(1.0, 0.0);
        }
    }
}

//-----------------------------------------------------------------------------

double GeodesicsInHeat::great_circle_distance(pmp::Vertex v, pmp::Vertex vv, double r)
{
    double dis;
    if (v == vv)
    {
        return 0.0;
    }
    pmp::Normal n = pmp::vertex_normal(mesh_, v);
    pmp::Normal nn = pmp::vertex_normal(mesh_, vv);
    double delta_sigma = acos(dot(n, nn));
    if (std::isnan(delta_sigma))
    {
        dis = haversine_distance(v, vv, r);
        if (std::isnan(delta_sigma))
        {
            dis = vincenty_distance(v, vv, r);
        }
        return dis;
    }

    return r * delta_sigma;
}

//-----------------------------------------------------------------------------

double GeodesicsInHeat::haversine_distance(pmp::Vertex v, pmp::Vertex vv, double r)
{
    pmp::Point p = mesh_.position(v);
    pmp::Point pp = mesh_.position(vv);

    double lamda1 = atan2(p[1], p[0]) + std::numbers::pi;
    double phi1 = std::numbers::pi / 2.0 - acos(p[2] / r);

    double lamda2 = atan2(pp[1], pp[0]) + std::numbers::pi;
    double phi2 = std::numbers::pi / 2.0 - acos(pp[2] / r);

    double d_lamda = fabs(lamda1 - lamda2);
    double d_phi = fabs(phi1 - phi2);

    double a = pow(sin(d_phi / 2), 2) +
               cos(phi1) * cos(phi2) * pow(sin(d_lamda / 2), 2);

    double d_sigma = 2 * asin(sqrt(a));

    return r * d_sigma;
}

//-----------------------------------------------------------------------------

double GeodesicsInHeat::vincenty_distance(pmp::Vertex v, pmp::Vertex vv, double r)
{
    //  special case of the Vincenty formula for an ellipsoid with equal major and minor axes
    pmp::Point p = mesh_.position(v);
    pmp::Point pp = mesh_.position(vv);

    double lamda1 = atan2(p[1], p[0]) + std::numbers::pi;
    double phi1 = std::numbers::pi / 2.0 - acos(p[2] / r);

    double lamda2 = atan2(pp[1], pp[0]) + std::numbers::pi;
    double phi2 = std::numbers::pi / 2.0 - acos(pp[2] / r);

    double d_lamda = fabs(lamda1 - lamda2);

    // Numerator
    double a = pow(cos(phi2) * sin(d_lamda), 2);

    double b = cos(phi1) * sin(phi2);
    double c = sin(phi1) * cos(phi2) * cos(d_lamda);
    double d = pow(b - c, 2);

    double e = sqrt(a + d);

    // Denominator
    double f = sin(phi1) * sin(phi2);
    double g = cos(phi1) * cos(phi2) * cos(d_lamda);

    double h = f + g;

    double d_sigma = atan2(e, h);

    return r * d_sigma;
}

//=============================================================================
