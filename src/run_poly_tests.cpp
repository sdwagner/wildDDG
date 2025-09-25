// =============================================================================
//  Copyright (c) 2025 Sven D. Wagner, Mario Botsch.
//  Distributed under MIT license, see file LICENSE for details.
// =============================================================================

#include "util/grid_generator.h"
#include "util/mesh_converter.h"
#include "util/system_solver.h"
#include "Laplacians/construct_laplace.h"


struct Run
{
    std::string name;
    bool band = true;
    bool sphere = false;
    bool degree_dims = true;
    Solve_Function function = Franke2d;
    double min_degree = 1;
    double max_degree = 51;
    double inc_degree = 0.5;
    double std_degree = 9;
    double min_el = 1e-30;
    double max_el = 1;
    double inc_el = 0.2;
    double std_el = 1e-10;
    int min_dims = 9;
    int max_dims = 241;
    int inc_dims = 4;
    int std_dims = 51;
    int sh_l = 4;
    int sh_m = 2;
};

void eval(const Run& run, int dims, double deg, double el, std::ofstream& error_file, std::vector<Lib> libs,
          std::vector<Lib> tfem_libs)
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    if (run.band)
        generate_tri_quad_grid_inconsistent(V, F, dims, deg, run.sphere);
    else
        generate_quad_grid(V, F, dims, el);
    pmp::SurfaceMesh mesh;
    to_pmp_mesh(V, F, mesh);

    for (const auto lib : libs)
    {
        if (std::ranges::find(tfem_libs, lib) != tfem_libs.end())
        {
            std::cout << lib_map[lib] << ", TFEM " << tfem_method_map[TFEMNormal] << ": ";
            error_file << solve_system_lib(V, F, {lib, TFEMNormal}, run.function, run.sh_l, run.sh_m).first << ",";
            std::cout << lib_map[lib] << ", TFEM " << tfem_method_map[TFEMStatic] << ": ";
            error_file << solve_system_lib(V, F, {lib, TFEMStatic}, run.function, run.sh_l, run.sh_m).first << ",";
            std::cout << lib_map[lib] << ", TFEM " << tfem_method_map[TFEMDynamic] << ": ";
            error_file << solve_system_lib(V, F, {lib, TFEMDynamic}, run.function, run.sh_l, run.sh_m).first << ",";
            std::cout << lib_map[lib] << ", TFEM " << tfem_method_map[TFEMDynamicFailsafe] << ": ";
            error_file << solve_system_lib(V, F, {lib, TFEMDynamicFailsafe}, run.function, run.sh_l, run.sh_m).first << ",";
        }
        else
        {
            std::cout << lib_map[lib] << ": ";
            error_file << solve_system_lib(V, F, {lib, TFEMNormal}, run.function, run.sh_l, run.sh_m).first << ",";
        }
    }
    error_file << dims << "," << deg << "," << el << std::endl;
}


void do_run(const Run& run)
{
    std::vector libs = {
        BungeLaplace_Cross_Dot,
        BungeLaplace_Heron_l2Sq, BungeLaplace_Max_Dot
    };
    if (type_map[run.function] == Geodesics)
        libs = {BungeLaplace_Cross_Dot, BungeLaplace_Max_Dot};
    const std::vector tfem_libs = {
        BungeLaplace_Cross_Dot, BungeLaplace_Max_Dot
    };


    const std::string path = "../out/grid_poly_tests/";
    std::filesystem::create_directories(path);
    std::ofstream error_file(path + run.name);

    int dims = run.std_dims;
    double deg = run.std_degree;
    double el = run.std_el;


    for (const auto lib : libs)
    {
        if (std::ranges::find(tfem_libs, lib) != tfem_libs.end())
        {
            error_file << lib_map[lib] << " (TFEM " << tfem_method_map[TFEMNormal] << "),";
            error_file << lib_map[lib] << " (TFEM " << tfem_method_map[TFEMStatic] << "),";
            error_file << lib_map[lib] << " (TFEM " << tfem_method_map[TFEMDynamic] << "),";
            error_file << lib_map[lib] << " (TFEM " << tfem_method_map[TFEMDynamicFailsafe] << "),";
        }
        else
        {
            error_file << lib_map[lib] << ",";
        }
    }
    error_file << "Grid Resolution" << "," << "Monomial Degree" << "," << "Edge Length" << std::endl;


    if (run.degree_dims)
    {
        if (run.band)
        {
            for (deg = run.min_degree; deg <= run.max_degree; deg += run.inc_degree)
            {
                eval(run, dims, deg, el, error_file, libs, tfem_libs);
            }
        }
        else
        {
            for (el = run.max_el; el >= run.min_el; el *= run.inc_el)
            {
                eval(run, dims, deg, el, error_file, libs, tfem_libs);
            }
        }
    }
    else
    {
        for (dims = run.min_dims; dims <= run.max_dims; dims += run.inc_dims)
        {
            eval(run, dims, deg, el, error_file, libs, tfem_libs);
        }
    }
}

int main()
{
    MOLLIFICATION_FACTOR = 1e-6;
    Run run;

    run = Run();
    run.sphere = false;
    run.band = false;
    run.function = Franke2d;
    run.name = "laplace_comparison_el_needle_plane_single.csv";
    run.degree_dims = true;
    do_run(run);

    run = Run();
    run.sphere = false;
    run.band = false;
    run.function = Franke2d;
    run.name = "laplace_comparison_dims_needle_plane_single.csv";
    run.degree_dims = false;
    do_run(run);


    run = Run();
    run.sphere = false;
    run.function = Franke2d;
    run.name = "laplace_comparison_degree_needle_plane.csv";
    run.degree_dims = true;
    do_run(run);

    run = Run();
    run.sphere = false;
    run.function = Franke2d;
    run.name = "laplace_comparison_dims_needle_plane.csv";
    run.degree_dims = false;
    do_run(run);


    run = Run();
    run.sphere = true;
    run.function = Spherical_Harmonics;
    run.name = "laplace_comparison_degree_needle_sphere.csv";
    run.degree_dims = true;
    do_run(run);

    run = Run();
    run.sphere = true;
    run.function = Spherical_Harmonics;
    run.name = "laplace_comparison_dims_needle_sphere.csv";
    run.degree_dims = false;
    do_run(run);


    run = Run();
    run.sphere = false;
    run.function = BiHarm_Franke;
    run.name = "bilaplace_comparison_degree_needle_plane.csv";
    run.degree_dims = true;
    do_run(run);

    run = Run();
    run.sphere = false;
    run.function = BiHarm_Franke;
    run.name = "bilaplace_comparison_dims_needle_plane.csv";
    run.degree_dims = false;
    do_run(run);


    // Conditioning

    run = Run();
    run.sphere = false;
    run.function = ConditioningStiffness;
    run.name = "conditioning_comparison_degree_needle_plane.csv";
    run.degree_dims = true;
    do_run(run);


    run = Run();
    run.sphere = false;
    run.function = ConditioningStiffness;
    run.max_dims = 150;
    run.name = "conditioning_comparison_dims_needle_plane.csv";
    run.degree_dims = false;
    do_run(run);


    run = Run();
    run.sphere = true;
    run.function = ConditioningStiffness;
    run.name = "conditioning_comparison_degree_needle_sphere.csv";
    run.degree_dims = true;
    do_run(run);

    run = Run();
    run.sphere = true;
    run.function = ConditioningStiffness;
    run.max_dims = 150;
    run.name = "conditioning_comparison_dims_needle_sphere.csv";
    run.degree_dims = false;
    do_run(run);

    // Smoothing


    run = Run();
    run.sphere = true;
    run.function = Spherical_Smoothing;
    run.name = "smoothing_comparison_degree_needle_sphere.csv";
    run.degree_dims = true;
    do_run(run);

    run = Run();
    run.sphere = true;
    run.function = Spherical_Smoothing;
    run.name = "smoothing_comparison_dims_needle_sphere.csv";
    run.degree_dims = false;
    do_run(run);

    run = Run();
    run.sphere = false;
    run.function = Planar_Geodesics;
    run.name = "geodesics_comparison_degree_needle_plane.csv";
    run.degree_dims = true;
    do_run(run);

    run = Run();
    run.sphere = false;
    run.function = Planar_Geodesics;
    run.name = "geodesics_comparison_dims_needle_plane.csv";
    run.degree_dims = false;
    do_run(run);

    run = Run();
    run.sphere = true;
    run.function = Spherical_Geodesics;
    run.name = "geodesics_comparison_degree_needle_sphere.csv";
    run.degree_dims = true;
    do_run(run);

    run = Run();
    run.sphere = true;
    run.function = Spherical_Geodesics;
    run.name = "geodesics_comparison_dims_needle_sphere.csv";
    run.degree_dims = false;
    do_run(run);
}
