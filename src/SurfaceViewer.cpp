#include "SurfaceViewer.h"
#include <imgui.h>

#include "geometrycentral/surface/exact_geodesics.h"
#include "geometrycentral/surface/heat_method_distance.h"
#include "util/bipoisson_solver.h"
#include "util/grid_generator.h"
#include "util/mesh_converter.h"
#include "Laplacians/construct_laplace.h"
#include "pmp/bounding_box.h"
#include "pmp/stop_watch.h"
#include "pmp/algorithms/laplace.h"
#include "pmp/algorithms/subdivision.h"
#include "pmp/algorithms/utilities.h"
#include "pmp/io/io.h"
#include "util/cylinder_generator.h"
#include "util/system_solver.h"
#include "util/hole_filling.h"
#include "util/Curvature.h"
#include "util/deformation_solver.h"
#include "./Laplacians/bunge_poly_laplace.h"

int main()
{
    SurfaceViewer window("Polygon Modeling", 800, 600);
    return window.run();
}


void SurfaceViewer::process_imgui()
{
    MeshViewer::process_imgui();

    ImGui::Spacing();
    ImGui::Spacing();

    if (ImGui::CollapsingHeader("Grid", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::SliderInt("Grid Size (Can only be odd)", &grid_size, 7, 201);
        ImGui::SliderFloat("Monomial Degree", &mono_degree, 1, 51);
        ImGui::SliderInt("log_10(Edge Length Epsilon)", &exp_eps, -30, 0);
        ImGui::Checkbox("Large Angle or Small Edge", &angle_edge);
        ImGui::Checkbox("Spherify?", &sphere_);
        ImGui::Checkbox("Cylindrify?", &cylinder_);
        float eps = pow(10, exp_eps);
        grid_size = grid_size % 2 == 0 ? grid_size +1 : grid_size;
        if(ImGui::Button("Generate Plane"))
        {
            generate_tri_grid(V, F, grid_size, eps, angle_edge);
            to_pmp_mesh(V, F, mesh_);
            mesh_.garbage_collection();
            pmp::BoundingBox bb = bounds(mesh_);
            set_scene((pmp::vec3)bb.center(), 0.5 * bb.size());
            update_mesh();
        }
        if (ImGui::Button("Generate Plane Sigmoid"))
        {
            generate_tri_grid_inconsistent(V, F, grid_size, mono_degree, angle_edge, sphere_, cylinder_, false);
            to_pmp_mesh(V, F, mesh_);
            update_mesh();
            mesh_.garbage_collection();
            pmp::BoundingBox bb = bounds(mesh_);
            set_scene((pmp::vec3)bb.center(), 0.5 * bb.size());
            update_mesh();

        }
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::SliderInt("Vertices x60", &delaunay_factor_, 1, 1000);
        if (ImGui::Button("Generate Delaunay Plane"))
        {
            generate_unstructured_grid(V, F, delaunay_factor_);
            to_pmp_mesh(V, F, mesh_);
            update_mesh();
            mesh_.garbage_collection();
            pmp::BoundingBox bb = bounds(mesh_);
            set_scene((pmp::vec3)bb.center(), 0.5 * bb.size());
            update_mesh();

        }
        if (ImGui::Button("Destroy Mesh"))
        {
            destroy_mesh(mesh_, eps, 0.1, 0.5);
            update_mesh();
            mesh_.garbage_collection();
            pmp::BoundingBox bb = bounds(mesh_);
            set_scene((pmp::vec3)bb.center(), 0.5 * bb.size());
            update_mesh();

        }
        ImGui::Spacing();
        ImGui::Spacing();
        if (ImGui::Button("Generate Quad Plane"))
        {
            generate_quad_grid(V, F, grid_size, eps);
            to_pmp_mesh(V, F, mesh_);
            update_mesh();
            mesh_.garbage_collection();
            pmp::BoundingBox bb = bounds(mesh_);
            set_scene((pmp::vec3)bb.center(), 0.5 * bb.size());
            update_mesh();

        }
        if (ImGui::Button("Generate Quad Plane Sigmoid"))
        {
            generate_tri_quad_grid_inconsistent(V, F, grid_size, mono_degree, sphere_);
            to_pmp_mesh(V, F, mesh_);
            update_mesh();
            mesh_.garbage_collection();
            pmp::BoundingBox bb = bounds(mesh_);
            set_scene((pmp::vec3)bb.center(), 0.5 * bb.size());
            update_mesh();

        }
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::SliderInt("Marching Cubes Grid Size", &marching_cubes_resolution_, 5, 301);
        ImGui::Checkbox("Polygonal Marching Cubes", &polygon_marching_cubes_);
        if (ImGui::Button("Generate Cylinder"))
        {
            mesh_.clear();
            gen_cylinder_mesh(1, 5, marching_cubes_resolution_, mesh_, polygon_marching_cubes_);
            pmp_to_arrays(V, F, mesh_);
            update_mesh();
            mesh_.garbage_collection();
            pmp::BoundingBox bb = bounds(mesh_);
            set_scene((pmp::vec3)bb.center(), 0.5 * bb.size());
            update_mesh();

        }
    }

    ImGui::Spacing();
    ImGui::RadioButton("PMP", &laplace, PMP);
    ImGui::RadioButton("Open Mesh", &laplace, Open_Mesh);
    ImGui::RadioButton("IGL", &laplace, IGL);
    ImGui::RadioButton("Geometry Central", &laplace, Geometry_Central);
    ImGui::RadioButton("CGAL", &laplace, Cgal);
    ImGui::RadioButton("VCGLib", &laplace, VCGLib);
    ImGui::RadioButton("Geogram", &laplace, Geogram);
    ImGui::RadioButton("CinoLib", &laplace, CinoLib);
    ImGui::RadioButton("Intrinsic Mollification", &laplace, Intrinsic_Mollification);
    ImGui::RadioButton("Intrinsic Delaunay", &laplace, Intrinsic_Delaunay);
    ImGui::RadioButton("Intrinsic Delaunay Mollification", &laplace, Intrinsic_Delaunay_Mollification);
    ImGui::RadioButton("Optimized Laplacian", &laplace, OptimizedLaplacian_Cross_Dot);
    ImGui::RadioButton("Optimized Laplacian + Mass Matrix Tempering", &laplace, OptimizedLaplacian_Cross_Dot_Mass);
    ImGui::RadioButton("Bunge Polygon Laplacian", &laplace, BungeLaplace_Cross_Dot);

    if (laplace == OptimizedLaplacian_Cross_Dot || laplace == OptimizedLaplacian_Cross_Dot_Mass || laplace == BungeLaplace_Cross_Dot)
    {
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::RadioButton("No TFEM", &tfem_method_, TFEMNormal);
        ImGui::RadioButton("Static TFEM", &tfem_method_, TFEMStatic);
        ImGui::RadioButton("Dynamic TFEM", &tfem_method_, TFEMDynamic);
        if (laplace == OptimizedLaplacian_Cross_Dot_Mass)
            ImGui::RadioButton("D-TFEM", &tfem_method_, TFEMDynamicFailsafe);
        else
            ImGui::RadioButton("Dynamic TFEM + Failsafe", &tfem_method_, TFEMDynamicFailsafe);
    }
    ImGui::Spacing();
    ImGui::Spacing();

    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Text("Poisson Planar");
    ImGui::RadioButton("Franke", &function, Franke2d);
    ImGui::RadioButton("X + Y", &function, XpY);
    ImGui::RadioButton("X^2 + Y^2", &function, X2pY2);
    ImGui::RadioButton("Runge", &function, Runge);
    ImGui::Text("Poisson Spherical");
    ImGui::RadioButton("Spherical Harmonics", &function, Spherical_Harmonics);
    ImGui::Text("Biharmonic Poisson Planar");
    ImGui::RadioButton("Franke Biharmonic", &function, BiHarm_Franke);
    ImGui::Text("Mean Curvature Planar");
    ImGui::RadioButton("Mean Curvature Plane", &function, MeanCurvaturePlane);
    ImGui::RadioButton("Mean Curvature Sphere", &function, MeanCurvatureSphere);

    if (ImGui::Button("Solve System"))
    {
        pmp_to_arrays(V, F, mesh_);
        double stat_const = 1e-2*pow(mean_edge_length(mesh_), 2);
        if (tfem_method_ == TFEMStatic)
            stat_const = pow(mean_edge_length(mesh_), 3);
        solve_system_lib(V, F, {static_cast<Lib>(laplace), static_cast<TFEMMethod>(tfem_method_), stat_const, sqrt(3)/4*0.1}, static_cast<Solve_Function>(function), 4, 2);
    }
    if (ImGui::CollapsingHeader("Geodesics in Heat",
                                ImGuiTreeNodeFlags_DefaultOpen))
    {
        static int geodesics = 2;
        ImGui::RadioButton("Compare distances to arc lengths", &geodesics, 0);
        ImGui::RadioButton("Compare to euclidean distances", &geodesics, 1);
        ImGui::RadioButton("No comparison", &geodesics, 2);

        if (geodesics == 0)
        {
            compare_sphere = true;
            compare_cube = false;
        }
        else if (geodesics == 1)
        {
            compare_sphere = false;
            compare_cube = true;
        }
        else if (geodesics == 2)
        {
            compare_sphere = false;
            compare_cube = false;
        }

        ImGui::Spacing();
        ImGui::Spacing();


        static int ts = 0;
        ImGui::Text("Choose your diffusion time step ");

        ImGui::RadioButton("Mean edge length", &ts, 0);
        ImGui::RadioButton("Max edge length", &ts, 1);
        ImGui::RadioButton("Max diagonal length", &ts, 2);
        time_step_ = DiffusionStep(ts);

        if (ImGui::Button("Compute Exact Distances from vertex 0"))
        {

                pmp_to_arrays(V, F, mesh_);
                std::unique_ptr<geometrycentral::surface::SurfaceMesh> gc_mesh;
                std::unique_ptr<geometrycentral::surface::VertexPositionGeometry> gc_geometry;
                to_gc_mesh(V, F, gc_mesh, gc_geometry);
                auto v0 = geometrycentral::surface::Vertex(gc_mesh.get(), 0);
                geometrycentral::surface::VertexData<double> distances = exactGeodesicDistance(*gc_mesh, *gc_geometry, v0);
                double maxdist(0);
                for (auto v : gc_mesh->vertices())
                {
                    if (distances[v] <= FLT_MAX)
                    {
                        maxdist = std::max(maxdist, distances[v]);
                    }
                }
                double error = 0;
                for (auto v : gc_mesh->vertices())
                {
                    if (compare_cube)
                    {
                        const double dist = norm(gc_geometry->vertexPositions[v] - gc_geometry->vertexPositions[v0]);
                        error += (distances[v] - dist) * (distances[v] - dist);
                    }
                }
                error /= (double)gc_mesh->nVertices();
                error = sqrt(error);
                std::cout << "Geodesics RMSE error: " << error << std::endl;

                auto tex = mesh_.vertex_property<TexCoord>("v:tex");
                for (auto v : gc_mesh->vertices())
                {
                    if (distances[v] <= FLT_MAX)
                    {
                        tex[Vertex(v.getIndex())] = TexCoord(distances[v] / maxdist, 0.0);
                    }
                    else
                    {
                        tex[Vertex(v.getIndex())] = TexCoord(1.0, 0.0);
                    }
                }
            update_mesh();
            renderer_.use_checkerboard_texture();
            set_draw_mode("Texture");
        }

        if (ImGui::Button("Compute Geodesic Distances from vertex 0"))
        {

            if (laplace == Intrinsic_Delaunay_Mollification)
            {
                pmp_to_arrays(V, F, mesh_);
                std::unique_ptr<geometrycentral::surface::SurfaceMesh> gc_mesh;
                std::unique_ptr<geometrycentral::surface::VertexPositionGeometry> gc_geometry;
                to_gc_mesh(V, F, gc_mesh, gc_geometry);
                pmp::StopWatch stop_watch;
                auto solver = geometrycentral::surface::HeatMethodDistanceSolver(*gc_geometry, 1, true);
                stop_watch.start();
                auto v0 = geometrycentral::surface::Vertex(gc_mesh.get(), 0);
                auto distances = solver.computeDistance(v0);
                double maxdist(0);
                for (auto v : gc_mesh->vertices())
                {
                    if (distances[v] <= FLT_MAX)
                    {
                        maxdist = std::max(maxdist, distances[v]);
                    }
                }
                double error = 0;
                for (auto v : gc_mesh->vertices())
                {
                    if (compare_cube)
                    {
                        const double dist = norm(gc_geometry->vertexPositions[v] - gc_geometry->vertexPositions[v0]);
                        error += (distances[v] - dist) * (distances[v] - dist);
                    }
                }
                error /= (double)gc_mesh->nVertices();
                error = sqrt(error);
                std::cout << "Geodesics RMSE error: " << error << std::endl;

                auto tex = mesh_.vertex_property<TexCoord>("v:tex");
                for (auto v : gc_mesh->vertices())
                {
                    if (distances[v] <= FLT_MAX)
                    {
                        tex[Vertex(v.getIndex())] = TexCoord(distances[v] / maxdist, 0.0);
                    }
                    else
                    {
                        tex[Vertex(v.getIndex())] = TexCoord(1.0, 0.0);
                    }
                }
                stop_watch.stop();
                std::cout << stop_watch.elapsed() << std::endl;

            }
            else
            {
                double stat_const = 1e-8;
                if (tfem_method_ == TFEMStatic)
                    stat_const = pow(mean_edge_length(mesh_), 3);

                pmp::StopWatch stop_watch;
                GeodesicsInHeat heat(mesh_, compare_sphere, compare_cube, time_step_);
                Eigen::VectorXd dist, geodist;
                heat.compute_geodesics({OptimizedLaplacian_Cross_Dot_Mass, static_cast<TFEMMethod>(tfem_method_), stat_const, 1e-2});
                stop_watch.start();
                heat.getDistance(0, dist, geodist);
                stop_watch.stop();
                std::cout << stop_watch.elapsed() << std::endl;
            }
            update_mesh();
            renderer_.use_checkerboard_texture();
            set_draw_mode("Texture");
        }
    }
    ImGui::Spacing();
    ImGui::SliderInt("log_10(Time Step)", &exp_dt, -8, 0);
    float dt = pow(10, exp_dt);

    if (ImGui::Button("Implicit Smoothing"))
    {
        pmp_to_arrays(V, F, mesh_);
        smooth_lib(V, F, {static_cast<Lib>(laplace), static_cast<TFEMMethod>(tfem_method_)}, true, dt, true);
        to_pmp_mesh(V, F, mesh_);
        update_mesh();
    }
    if (ImGui::Button("Explicit Smoothing"))
    {
        pmp_to_arrays(V, F, mesh_);
        smooth_lib(V, F, {static_cast<Lib>(laplace), static_cast<TFEMMethod>(tfem_method_)}, false, dt, false);
        to_pmp_mesh(V, F, mesh_);
        update_mesh();
    }

    if (ImGui::Button("Close smallest hole"))
    {
        // find smallest hole
        Halfedge hmin;
        unsigned int lmin(mesh_.n_halfedges());
        for (auto h : mesh_.halfedges())
        {
            if (mesh_.is_boundary(h))
            {
                Scalar l(0);
                Halfedge hh = h;
                do
                {
                    ++l;
                    if (!mesh_.is_manifold(mesh_.to_vertex(hh)))
                    {
                        l += lmin + 42; // make sure this hole is not chosen
                        break;
                    }
                    hh = mesh_.next_halfedge(hh);
                } while (hh != h);

                if (l < lmin)
                {
                    lmin = l;
                    hmin = h;
                }
            }
        }

        // close smallest hole
        if (hmin.is_valid())
        {
            try
            {
                fill_hole(mesh_, hmin, {static_cast<Lib>(laplace), static_cast<TFEMMethod>(tfem_method_)});
            }
            catch (const InvalidInputException& e)
            {
                std::cerr << e.what() << std::endl;
                return;
            }
            update_mesh();
        }
        else
        {
            std::cerr << "No manifold boundary loop found\n";
        }
    }
    if (ImGui::Button("Make Bigger"))
    {
        for (auto v : mesh_.vertices())
            mesh_.position(v) *= 10;
        pmp::BoundingBox bb = bounds(mesh_);
        set_scene((pmp::vec3)bb.center(), 0.5 * bb.size());
        update_mesh();
    }
    if (ImGui::Button("Make Smaller"))
    {
        for (auto v : mesh_.vertices())
            mesh_.position(v) *= 0.1;
        pmp::BoundingBox bb = bounds(mesh_);
        set_scene((pmp::vec3)bb.center(), 0.5 * bb.size());
        update_mesh();
    }
    if (ImGui::Button("Dualize"))
    {
        dualize(mesh_);
        mesh_.garbage_collection();
        pmp_to_arrays(V, F, mesh_);
        update_mesh();
    }

    // curvature visualization
    if (ImGui::CollapsingHeader("Curvature", ImGuiTreeNodeFlags_DefaultOpen))
    {
        static bool curvature_sphere_ = false;
        ImGui::Checkbox("Compare to unit sphere curvatures",
                        &curvature_sphere_);

        if (ImGui::Button("Mean Curvature"))
        {
            Curvature analyzer(mesh_, curvature_sphere_);

            pmp::SparseMatrix L;
            pmp::DiagonalMatrix M;
            setup_laplacian(L, M, V, F, {static_cast<Lib>(laplace), static_cast<TFEMMethod>(tfem_method_)});
            analyzer.visualize_curvature(L, M);
            renderer_.use_cold_warm_texture();
            update_mesh();
            set_draw_mode("Texture");
        }
    }
    if (ImGui::Button("Triangulate naively"))
    {
        pmp_to_arrays(V, F, mesh_);
        std::vector<std::array<int, 3>> triangles;
        for (int i = 0; i < F.rows(); i++)
        {
            for (int j = 2; j < F.cols(); j++)
            {
                if (F(i,j) != -1)
                {
                    std::array arr{ F(i, 0), F(i,j-1), F(i,j) };
                    triangles.emplace_back(arr);
                }
                else
                    break;
            }
        }
        F.resize(triangles.size(), 3);
        for (int i = 0; i < F.rows(); i++)
        {
            F(i, 0) = triangles.at(i)[0];
            F(i, 1) = triangles.at(i)[1];
            F(i, 2) = triangles.at(i)[2];
        }
        to_pmp_mesh(V, F, mesh_);
        update_mesh();
    }
    if (ImGui::Button("Subdivide"))
    {
        pmp::linear_subdivision(mesh_);
        update_mesh();
        pmp_to_arrays(V, F, mesh_);
    }

    ImGui::Spacing();
    ImGui::Spacing();

    ImGui::SliderFloat("Deformation Angle", &deformation_angle_, 0, 2 * M_PI);

    if (ImGui::Button("Deform"))
    {
        pmp::BoundingBox bb = bounds(mesh_);
        auto is_fixed = [&](int i){return abs(mesh_.position(pmp::Vertex(i))[1] - bb.min()[1]) < 0.5;};
        auto is_handle = [&](int i){return abs(mesh_.position(pmp::Vertex(i))[1] - bb.max()[1]) < 0.5;};
        Eigen::Matrix4d transform, scaling;
        transform << cos(deformation_angle_), -sin(deformation_angle_), 0, 0,
                     sin(deformation_angle_), cos(deformation_angle_), 0, 0,
                     0, 0, 1, 0,
                     0, 0, 0, 1;
        scaling << 1, 0, 0, 0,
                   0, 1, 0, 0,
                   0, 0, 1, 0,
                   0, 0, 0, 1;
        try
        {
            solve_deformation(mesh_, is_fixed, is_handle, transform*scaling, {(Lib)laplace, (TFEMMethod)tfem_method_});
        }
        catch (const std::exception& e)
        {
            std::cerr << e.what() << std::endl;
        }
        update_mesh();
    }

    if (ImGui::Button("Insert virtual vertex"))
    {
        for (auto f : mesh_.faces())
        {
            Eigen::MatrixXd poly = Eigen::MatrixXd::Zero(mesh_.valence(f), 3);
            int i = 0;
            for (auto v : mesh_.vertices(f))
            {
                poly.row(i) = (Eigen::Vector3d)mesh_.position(v);
                i++;
            }
            Eigen::VectorXd weights;
            compute_virtual_vertex(poly, weights);
            Eigen::Vector3d pos = weights.transpose() * poly;
            mesh_.split(f, pos);
        }
        update_mesh();
    }
}

void SurfaceViewer::keyboard(int key, int scancode, int action, int mods)
{
    if (action != GLFW_PRESS && action != GLFW_REPEAT)
        return;

    switch (key)
    {
    case GLFW_KEY_W: // write mesh
        {
            IOFlags flags;
            auto v_tex = mesh_.get_vertex_property<pmp::TexCoord>("v:tex");
            auto h_tex = mesh_.get_halfedge_property<pmp::TexCoord>("h:tex");
            if (v_tex && !h_tex)
            {
                h_tex = mesh_.add_halfedge_property<TexCoord>("h:tex");
                for (const auto h : mesh_.halfedges())
                {
                    h_tex[h] = v_tex[mesh_.to_vertex(h)];
                }
            }
            flags.use_halfedge_texcoords = true;
            flags.use_vertex_texcoords = true;
            pmp::write(mesh_, "output.obj", flags);
            break;
        }
    default:
        {
            MeshViewer::keyboard(key, scancode, action, mods);
            break;
        }
    }
}