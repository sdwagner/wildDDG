#pragma once

#include <pmp/visualization/mesh_viewer.h>
#include "util/GeodesicsInHeat.h"

class SurfaceViewer : public pmp::MeshViewer{

public: // public methods
    SurfaceViewer(const char* title, int width, int height)
        : MeshViewer(title, width, height)
    {
        set_draw_mode("Hidden Line");
    }

protected:
    void process_imgui() override;
    void keyboard(int key, int scancode, int action, int mods) override;

    int grid_size = 9;
    int exp_eps = -1;
    bool angle_edge = false;
    bool sphere_ = false;
    float mono_degree = 3;
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    int laplace = 0;
    int function = 0;
    bool compare_sphere, compare_cube;
    DiffusionStep time_step_;
    int tfem_method_ = TFEMNormal;
    int marching_cubes_resolution_ = 101;
    int delaunay_factor_ = 10;
    int exp_dt = -3;
    bool polygon_marching_cubes_ = false;
    float deformation_angle_ = M_PI_2;
    bool cylinder_ = false;
};
