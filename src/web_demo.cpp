// =============================================================================
//  Copyright (c) 2025 Sven D. Wagner, Mario Botsch.
//  Distributed under MIT license, see file LICENSE for details.
// =============================================================================

#include <imgui.h>

#include "pmp/bounding_box.h"
#include "pmp/algorithms/hole_filling.h"
#include "pmp/algorithms/utilities.h"
#include "pmp/visualization/mesh_viewer.h"
#include "util/cylinder_generator.h"
#include "util/deformation_solver.h"
#include "util/GeodesicsInHeat.h"
#include "util/grid_generator.h"
#include "util/hole_filling.h"
#include "util/parameterization.h"

enum Mode
{
    None = 0,
    Holefilling = 1,
    Parametrization = 2,
    Parametrization2 = 3,
    Deformation = 4,
    GeodesicsHeat = 5,
    Polygon = 6
};

class SurfaceViewer : public pmp::MeshViewer{

public: // public methods
    SurfaceViewer(const char* title, int width, int height, Mode mode)
        : MeshViewer(title, width, height), mode_(mode)
    {
        set_draw_mode("Hidden Line");
        if (mode == Deformation || mode == GeodesicsHeat)
            crease_angle_ = 0;
        renderer_.set_crease_angle(crease_angle_);

        if (mode_ == Parametrization || mode_ == GeodesicsHeat)
            set_draw_mode("Smooth Shading");

        if (mode_ == Holefilling)
            MeshViewer::load_mesh("holefilling.obj");
        if (mode_ == Parametrization)
            MeshViewer::load_mesh("mario.obj");

        if (mode == Polygon)
        {
            laplace = BungeLaplace_Cross_Dot;
            tfem_method_ = TFEMNormal;
            generate_tri_quad_grid_inconsistent(V, F, grid_size, mono_degree, sphere_);
            to_pmp_mesh(V, F, mesh_);
            mesh_.garbage_collection();
            pmp::BoundingBox bb = bounds(mesh_);
            set_scene((pmp::vec3)bb.center(), 0.5 * bb.size());
            MeshViewer::update_mesh();
        }

        if (mode == Deformation)
        {
            generate_tri_grid_inconsistent(V, F, grid_size, mono_degree, angle_edge, sphere_, cylinder_, false);
            to_pmp_mesh(V, F, mesh_);
            mesh_.garbage_collection();
            pmp::BoundingBox bb = bounds(mesh_);
            set_scene((pmp::vec3)bb.center(), 0.55 * bb.size());
            MeshViewer::update_mesh();

        }

        if (mode == Parametrization2)
        {
            generate_tri_grid_inconsistent(V, F, grid_size, mono_degree, angle_edge, sphere_, false, false);
            to_pmp_mesh(V, F, mesh_);
            mesh_.garbage_collection();
            pmp::BoundingBox bb = bounds(mesh_);
            set_scene((pmp::vec3)bb.center(), 0.55 * bb.size());
            MeshViewer::update_mesh();

        }

        if (mode_ == GeodesicsHeat)
        {
            MeshViewer::load_mesh("voromesh_bob_poly.obj");
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
            MeshViewer::update_mesh();
        }

        if (mode_ == Parametrization || mode_ == Parametrization2)
        {
            clamp_zero_area_ = false;
            parameterize_boundary(mesh_);
            MeshViewer::update_mesh();
            renderer_.use_checkerboard_texture();
        }
    }

protected:
    void process_imgui() override;
    void keyboard(int key, int scancode, int action, int mods) override;
    void draw(const std::string& drawMode) override;

    int grid_size = 81;
    int exp_eps = -1;
    bool angle_edge = false;
    bool sphere_ = false;
    float mono_degree = 11.331f;
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    int laplace = OptimizedLaplacian_Cross_Dot;
    int function = 0;
    bool compare_sphere = false, compare_cube = false;
    DiffusionStep time_step_ = MaxDiagonal;
    int tfem_method_ = TFEMNormal;
    int marching_cubes_resolution_ = 101;
    int delaunay_factor_ = 10;
    int exp_dt = -3;
    bool polygon_marching_cubes_ = false;
    bool clamp_zero_area_ = true;
    float deformation_angle_ = 0.0f;
    bool cylinder_ = true;
    Mode mode_ = None;
    pmp::SurfaceMesh undeformed_mesh_;
};

void SurfaceViewer::process_imgui()
{
    if (ImGui::CollapsingHeader("Mesh Info", ImGuiTreeNodeFlags_CollapsingHeader))
    {
        // mesh statistics
        ImGui::BulletText("%d vertices", (int)mesh_.n_vertices());
        ImGui::BulletText("%d edges", (int)mesh_.n_edges());
        ImGui::BulletText("%d faces", (int)mesh_.n_faces());

        // draw mode
        ImGui::PushItemWidth(130 * imgui_scaling());
        const char* current_item = draw_mode_names_[draw_mode_].c_str();
        if (ImGui::BeginCombo("Draw Mode", current_item))
        {
            for (unsigned int i = 0; i < n_draw_modes_; ++i)
            {
                const char* item = draw_mode_names_[i].c_str();
                bool is_selected = (current_item == item);
                if (ImGui::Selectable(item, is_selected))
                    draw_mode_ = i;
                if (is_selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        ImGui::PopItemWidth();

        if (draw_mode_names_[draw_mode_] == "Points")
        {
            // point size
            ImGui::PushItemWidth(120 * imgui_scaling());
            ImGui::SliderInt("Point Size", &point_size_, 1, 20);
            ImGui::PopItemWidth();
            if (point_size_ != renderer_.point_size())
            {
                renderer_.set_point_size(point_size_);
            }
        }
        else
        {
            // crease angle
            ImGui::PushItemWidth(120);
            ImGui::SliderFloat("Crease Angle", &crease_angle_, 0.0f, 180.0f,
                               "%.0f");
            ImGui::PopItemWidth();
            if (crease_angle_ != renderer_.crease_angle())
            {
                renderer_.set_crease_angle(crease_angle_);
            }
        }
    }

    ImGui::Spacing();
    ImGui::Spacing();

    if (ImGui::CollapsingHeader("Grid", ImGuiTreeNodeFlags_CollapsingHeader))
    {
        ImGui::Text("Grid Size (Can only be odd)");
        ImGui::SetNextItemWidth(200.0f);
        ImGui::SliderInt("##Grid Size (Can only be odd)", &grid_size, 7, 201);
        ImGui::Text("Monomial Degree");
        ImGui::SetNextItemWidth(200.0f);
        ImGui::SliderFloat("##Monomial Degree", &mono_degree, 1, 51);
        ImGui::Text("log_10(Edge Length Epsilon)");
        ImGui::SetNextItemWidth(200.0f);
        ImGui::SliderInt("##log_10(Edge Length Epsilon)", &exp_eps, -30, 0);
        ImGui::Checkbox("Large Angle or Small Edge", &angle_edge);
        ImGui::Checkbox("Spherify?", &sphere_);
        ImGui::Checkbox("Cylindrify?", &cylinder_);
        float eps = pow(10, exp_eps);
        grid_size = grid_size % 2 == 0 ? grid_size +1 : grid_size;
        if(ImGui::Button("Generate Tri Plane Isolated"))
        {
            generate_tri_grid(V, F, grid_size, eps, angle_edge);
            to_pmp_mesh(V, F, mesh_);
            mesh_.garbage_collection();
            pmp::BoundingBox bb = bounds(mesh_);
            set_scene((pmp::vec3)bb.center(), 0.5 * bb.size());
            update_mesh();
        }
        if (ImGui::Button("Generate Tri Plane Banded"))
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
        if (ImGui::Button("Generate Quad Plane Isolated"))
        {
            generate_quad_grid(V, F, grid_size, eps);
            to_pmp_mesh(V, F, mesh_);
            update_mesh();
            mesh_.garbage_collection();
            pmp::BoundingBox bb = bounds(mesh_);
            set_scene((pmp::vec3)bb.center(), 0.5 * bb.size());
            update_mesh();

        }
        if (ImGui::Button("Generate Quad Plane Banded"))
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
        ImGui::Text("Marching Cubes Grid Size");
        ImGui::SetNextItemWidth(200.0f);
        ImGui::SliderInt("##Marching Cubes Grid Size", &marching_cubes_resolution_, 5, 301);
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
    ImGui::Spacing();
    int laplace_prev = laplace;
    int tfem_method_prev = tfem_method_;
    if (ImGui::CollapsingHeader("Laplacians", (mode_ == Polygon || mode_ == Deformation || mode_ == Parametrization || mode_ == Parametrization2 || mode_ == GeodesicsHeat) ? ImGuiTreeNodeFlags_DefaultOpen : ImGuiTreeNodeFlags_CollapsingHeader))
    {
        ImGui::Spacing();
        if (ImGui::RadioButton("FEM/Cotangent Laplacian", &laplace, OptimizedLaplacian_Cross_Dot))
        {
            tfem_method_ = TFEMNormal;
        }
        if (ImGui::RadioButton("TFEM", &laplace, TFEM_Cross_Dot))
        {
            tfem_method_ = TFEMStatic;
        }
        if (ImGui::RadioButton("D-TFEM", &laplace, OptimizedLaplacian_Cross_Dot_Mass))
        {
            tfem_method_ = TFEMDynamicFailsafe;
        }
        ImGui::Spacing();
        ImGui::Checkbox("Stability Trick", &clamp_zero_area_);
    }
    ImGui::Spacing();
    ImGui::Spacing();

    if (ImGui::CollapsingHeader("Geodesics in Heat",
                                (mode_ == Polygon || mode_ == GeodesicsHeat) ? ImGuiTreeNodeFlags_DefaultOpen : ImGuiTreeNodeFlags_CollapsingHeader))
    {

        if (ImGui::Button("Compute Distances"))
        {
            GeodesicsInHeat heat(mesh_, compare_sphere, compare_cube, time_step_);
            Eigen::VectorXd dist, geodist;
            LaplaceConfig config = {static_cast<Lib>(laplace), static_cast<TFEMMethod>(tfem_method_), pow(mean_edge_length(mesh_), 3)};
            config.clamp_zero_area = clamp_zero_area_;
            heat.compute_geodesics(config);
            heat.getDistance(0, dist, geodist);
            update_mesh();
            renderer_.use_checkerboard_texture();
            set_draw_mode("Texture");
        }
    }
    ImGui::Spacing();
    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Deformation",
                                    (mode_ == Deformation) ? ImGuiTreeNodeFlags_DefaultOpen : ImGuiTreeNodeFlags_CollapsingHeader))
    {
        if (ImGui::Button("Reset Undeformed"))
        {
            undeformed_mesh_ = mesh_;
        }

        ImGui::Text("Deformation Angle");
        ImGui::SetNextItemWidth(200.0f);
        int def_degree = (int)(deformation_angle_ * 180.0f / M_PI);
        ImGui::SliderInt("##Deformation Angle", &def_degree, 0, 360, "%d°");
        float deformation_angle = def_degree * M_PI / 180.0f;


        if (deformation_angle_ != deformation_angle || laplace_prev != laplace || tfem_method_prev != tfem_method_)
        {
            deformation_angle_ = deformation_angle;
            if (undeformed_mesh_.n_vertices() != 0)
            {
                mesh_ = undeformed_mesh_;
            }
            else
            {
                undeformed_mesh_ = mesh_;
            }
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
                LaplaceConfig config = {static_cast<Lib>(laplace), static_cast<TFEMMethod>(tfem_method_), pow(mean_edge_length(mesh_), 3)};
                config.clamp_zero_area = clamp_zero_area_;
                solve_deformation(mesh_, is_fixed, is_handle, transform*scaling, config);
            }
            catch (const std::exception& e)
            {
                std::cerr << e.what() << std::endl;
            }
            update_mesh();
        }

        ImGui::Spacing();
        ImGui::Spacing();
    }
    ImGui::Spacing();
    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Holefilling",
                                    (mode_ == Holefilling) ? ImGuiTreeNodeFlags_DefaultOpen : ImGuiTreeNodeFlags_CollapsingHeader))
    {
        if (ImGui::Button("Close smallest hole - Regular"))
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
                    pmp::fill_hole(mesh_, hmin);
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

        if (ImGui::Button("Close smallest hole - Robust"))
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
                    LaplaceConfig config = {static_cast<Lib>(laplace), static_cast<TFEMMethod>(tfem_method_), pow(mean_edge_length(mesh_), 3)};
                    config.clamp_zero_area = clamp_zero_area_;
                    fill_hole(mesh_, hmin, config);
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
    }

    ImGui::Spacing();
    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Parametrization",
                                    (mode_ == Parametrization || mode_ == Parametrization2) ? ImGuiTreeNodeFlags_DefaultOpen : ImGuiTreeNodeFlags_CollapsingHeader))
    {
        if (ImGui::Button("Compute Harmonic Parametrization"))
        {
            if (parameterize_boundary(mesh_))
            {
                LaplaceConfig config = {static_cast<Lib>(laplace), static_cast<TFEMMethod>(tfem_method_), pow(mean_edge_length(mesh_), 3)};
                config.clamp_zero_area = clamp_zero_area_;
                parameterize_direct(mesh_, config);

            }
            update_mesh();
            renderer_.use_checkerboard_texture();
            set_draw_mode("Texture");
        }
    }
}

void SurfaceViewer::draw(const std::string& drawMode)
{
    MeshViewer::draw(drawMode);



    /*
    // draw the UV coordinates of the mesh in the upper right corner
    if (mode_ == Parametrization || mode_ == Parametrization2)
    {
        glClear(GL_DEPTH_BUFFER_BIT);

        // setup viewport
        GLint size = std::min(width(), height()) / 3;
        glViewport(width() - size - 1, height() - size - 1, size, size);

        // setup matrices
        mat4 P = ortho_matrix(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
        mat4 M = mat4::identity();

        // draw mesh once more
        renderer_.draw(P, M, "Texture Layout");

        // restore viewport
        glViewport(0, 0, width(), height());
    }
    */
}

void SurfaceViewer::keyboard(int key, int scancode, int action, int mods)
{
    if (action != GLFW_PRESS && action != GLFW_REPEAT)
        return;

    switch (key)
    {
    case GLFW_KEY_BACKSPACE: // reload model
        {
            if (mode_ == Holefilling)
                MeshViewer::load_mesh("holefilling.obj");
            if (mode_ == Parametrization)
                MeshViewer::load_mesh("mario.obj");


            if (mode_ == Parametrization2 || mode_ == GeodesicsHeat)
            {
                generate_tri_grid_inconsistent(V, F, grid_size, mono_degree, angle_edge, sphere_, false, false);
                to_pmp_mesh(V, F, mesh_);
                mesh_.garbage_collection();
                pmp::BoundingBox bb = bounds(mesh_);
                set_scene((pmp::vec3)bb.center(), 0.55 * bb.size());
                MeshViewer::update_mesh();

            }

            if (mode_ == Deformation)
            {
                generate_tri_grid_inconsistent(V, F, grid_size, mono_degree, angle_edge, sphere_, cylinder_, false);
                to_pmp_mesh(V, F, mesh_);
                mesh_.garbage_collection();
                pmp::BoundingBox bb = bounds(mesh_);
                set_scene((pmp::vec3)bb.center(), 0.55 * bb.size());
                MeshViewer::update_mesh();
                deformation_angle_ = 0.0f;

            }
            break;
        }

    default:
        {
            TrackballViewer::keyboard(key, scancode, action, mods);
            break;
        }
    }
}

int main(int argc, char** argv)
{
    Mode mode = None;
    std::cout << argv[1] << std::endl;
    if (argc == 2)
    {
        std::string s = argv[1];
        if (s == "holefilling")
            mode = Holefilling;
        else if (s == "deformation")
            mode = Deformation;
        else if (s == "parametrization")
            mode = Parametrization;
        else if (s == "parametrization2")
            mode = Parametrization2;
        else if (s == "geodesics")
            mode = GeodesicsHeat;
        else if (s == "polygon")
            mode = Polygon;
        else
            mode = None;
    }
    SurfaceViewer window("Polygon Modeling", 800, 600, mode);

    return window.run();
}