// =============================================================================
//  Copyright (c) 2025 Sven D. Wagner, Mario Botsch.
//  Distributed under MIT license, see file LICENSE for details.
// =============================================================================

#include "geometrycentral/surface/heat_method_distance.h"
#include "GeodesicsInHeat.h"
#include "geodesics_solver.h"

#include "mesh_converter.h"


void distance_to_texture_coordinates(pmp::SurfaceMesh& mesh_)
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


std::pair<double, double> compute_geodesics(const LaplaceConfig& config, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Solve_Function function)
{
    try
    {
        pmp::SurfaceMesh mesh;
        to_pmp_mesh(V, F, mesh);
        if (config.lib == Intrinsic_Delaunay_Mollification)
        {
            double error = 0;
            std::unique_ptr<geometrycentral::surface::SurfaceMesh> gc_mesh;
            std::unique_ptr<geometrycentral::surface::VertexPositionGeometry> gc_geometry;
            to_gc_mesh(V, F, gc_mesh, gc_geometry);
            auto solver = geometrycentral::surface::HeatMethodDistanceSolver(*gc_geometry, 1, true);
            auto v0 = geometrycentral::surface::Vertex(gc_mesh.get(), 0);
            auto vertex_data = solver.computeDistance(v0);
            for (auto v : gc_mesh->vertices())
            {
                if (function == Planar_Geodesics)
                {
                    const double dist = norm(gc_geometry->vertexPositions[v] - gc_geometry->vertexPositions[v0]);
                    error += (vertex_data[v] - dist) * (vertex_data[v] - dist);
                }
            }
            std::cout << "Geodesics RMSE error: " << error << std::endl;
            return {error, NAN};

        }
        GeodesicsInHeat heat(mesh, function == Spherical_Geodesics, function == Planar_Geodesics, MaxDiagonal);
        Eigen::VectorXd dist, geodist;
        heat.compute_geodesics(config);
        double error = heat.getDistance(0, dist, geodist);
        return {error, NAN};
    }
    catch (const std::exception& e)
    {
        std::cout << "Geodesics RMSE error: " << NAN << std::endl;
        return {NAN, NAN};
    }

}
