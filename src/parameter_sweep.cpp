#include <fstream>
#include "util/enums.h"
#include <Eigen/Dense>

#include "pmp/algorithms/utilities.h"
#include "util/grid_generator.h"
#include "util/system_solver.h"

struct Run
{
    std::string name;
    bool caps = true;
    bool band = true;
    bool sphere = false;
    bool degree_dims = true;
    bool dynamic = false;
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
    double min_dynamic_const = 1e-10;
    double inc_dynamic_const = 0.1;
    double max_dynamic_const = 10;
    double std_dynamic_const = 1e-2;
    double min_static_const = 1e-20;
    double inc_static_const = 0.1;
    double max_static_const = 1e-3;
    double std_static_const = 1e-8;
    int sh_l = 4;
    int sh_m = 2;
};

void eval(const Run& run, int dims, double deg, double el, std::ofstream& error_file, LaplaceConfig config)
{

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    if (run.band)
        generate_tri_grid_inconsistent(V, F, dims, deg, run.caps, run.sphere, false);
    else
        generate_tri_grid(V, F, dims, el, run.caps);
    pmp::SurfaceMesh mesh;
    to_pmp_mesh(V, F, mesh);

    std::cout << lib_map[config.lib] << ", TFEM " << tfem_method_map[config.tfem_method] << ": ";
    error_file << solve_system_lib(V, F, config, run.function, run.sh_l, run.sh_m).first << ",";
    error_file << dims << "," << deg << "," << el << "," << ((config.tfem_method == TFEMStatic) ? config.tfem_static_constant : config.tfem_dynamic_constant) << std::endl;
}


void do_run(const Run& run)
{
    const std::string path = "../out/tri_parameter_sweep/";
    std::filesystem::create_directories(path);
    std::ofstream error_file(path+run.name);

    int dims = run.std_dims;
    double deg = run.std_degree;
    double el = run.std_el;
    double dynamic_const = run.std_dynamic_const;
    double static_const = run.std_static_const;

    if (run.degree_dims)
    {
        if (run.band)
        {
            for (deg = run.min_degree; deg <= run.max_degree; deg += run.inc_degree)
            {
                if (run.dynamic)
                    for (dynamic_const = run.max_dynamic_const; dynamic_const >= run.min_dynamic_const; dynamic_const *= run.inc_dynamic_const)
                        eval(run, dims, deg, el, error_file, {OptimizedLaplacian_Cross_Dot, TFEMDynamicFailsafe, static_const, dynamic_const});
                else
                    for (static_const = run.max_static_const; static_const >= run.min_static_const; static_const *= run.inc_static_const)
                        eval(run, dims, deg, el, error_file, {OptimizedLaplacian_Cross_Dot, TFEMStatic, static_const, dynamic_const});
            }
        }
        else
        {
            for (el = run.max_el; el >= run.min_el; el *= run.inc_el)
            {
                if (run.dynamic)
                    for (dynamic_const = run.max_dynamic_const; dynamic_const >= run.min_dynamic_const; dynamic_const *= run.inc_dynamic_const)
                        eval(run, dims, deg, el, error_file, {OptimizedLaplacian_Cross_Dot, TFEMDynamicFailsafe, static_const, dynamic_const});
                else
                    for (static_const = run.max_static_const; static_const >= run.min_static_const; static_const *= run.inc_static_const)
                        eval(run, dims, deg, el, error_file, {OptimizedLaplacian_Cross_Dot, TFEMStatic, static_const, dynamic_const});
            }
        }
    }
    else
    {
        for (dims = run.min_dims; dims <= run.max_dims; dims += run.inc_dims)
        {
            if (run.dynamic)
                for (dynamic_const = run.max_dynamic_const; dynamic_const >= run.min_dynamic_const; dynamic_const *= run.inc_dynamic_const)
                    eval(run, dims, deg, el, error_file, {OptimizedLaplacian_Cross_Dot, TFEMDynamicFailsafe, static_const, dynamic_const});
            else
                for (static_const = run.max_static_const; static_const >= run.min_static_const; static_const *= run.inc_static_const)
                    eval(run, dims, deg, el, error_file, {OptimizedLaplacian_Cross_Dot, TFEMStatic, static_const, dynamic_const});
        }
    }
}

int main()
{

    Run run;

    run = Run();
    run.caps = true;
    run.sphere = false;
    run.band = false;
    run.function = Franke2d;
    run.name = "laplace_comparison_el_caps_plane_single_stat.csv";
    run.degree_dims = true;
    run.dynamic = false;
    do_run(run);

    run = Run();
    run.caps = false;
    run.sphere = false;
    run.band = false;
    run.function = Franke2d;
    run.name = "laplace_comparison_el_needle_plane_single_stat.csv";
    run.degree_dims = true;
    run.dynamic = false;
    do_run(run);

    run = Run();
    run.caps = true;
    run.sphere = false;
    run.band = false;
    run.function = Franke2d;
    run.name = "laplace_comparison_dims_caps_plane_single_stat.csv";
    run.degree_dims = false;
    run.dynamic = false;
    do_run(run);

    run = Run();
    run.caps = false;
    run.sphere = false;
    run.band = false;
    run.function = Franke2d;
    run.name = "laplace_comparison_dims_needle_plane_single_stat.csv";
    run.degree_dims = false;
    run.dynamic = false;
    do_run(run);

    run = Run();
    run.caps = true;
    run.sphere = false;
    run.function = Franke2d;
    run.name = "laplace_comparison_degree_caps_plane_stat.csv";
    run.degree_dims = true;
    run.dynamic = false;
    do_run(run);

    run = Run();
    run.caps = false;
    run.sphere = false;
    run.function = Franke2d;
    run.name = "laplace_comparison_degree_needle_plane_stat.csv";
    run.degree_dims = true;
    run.dynamic = false;
    do_run(run);

    run = Run();
    run.caps = true;
    run.sphere = false;
    run.function = Franke2d;
    run.name = "laplace_comparison_dims_caps_plane_stat.csv";
    run.degree_dims = false;
    run.dynamic = false;
    do_run(run);

    run = Run();
    run.caps = false;
    run.sphere = false;
    run.function = Franke2d;
    run.name = "laplace_comparison_dims_needle_plane_stat.csv";
    run.degree_dims = false;
    run.dynamic = false;
    do_run(run);

    run = Run();
    run.caps = true;
    run.sphere = false;
    run.band = false;
    run.function = Franke2d;
    run.name = "laplace_comparison_el_caps_plane_single_dyn.csv";
    run.degree_dims = true;
    run.dynamic = true;
    do_run(run);

    run = Run();
    run.caps = false;
    run.sphere = false;
    run.band = false;
    run.function = Franke2d;
    run.name = "laplace_comparison_el_needle_plane_single_dyn.csv";
    run.degree_dims = true;
    run.dynamic = true;
    do_run(run);

    run = Run();
    run.caps = true;
    run.sphere = false;
    run.band = false;
    run.function = Franke2d;
    run.name = "laplace_comparison_dims_caps_plane_single_dyn.csv";
    run.degree_dims = false;
    run.dynamic = true;
    do_run(run);

    run = Run();
    run.caps = false;
    run.sphere = false;
    run.band = false;
    run.function = Franke2d;
    run.name = "laplace_comparison_dims_needle_plane_single_dyn.csv";
    run.degree_dims = false;
    run.dynamic = true;
    do_run(run);

    run = Run();
    run.caps = true;
    run.sphere = false;
    run.function = Franke2d;
    run.name = "laplace_comparison_degree_caps_plane_dyn.csv";
    run.degree_dims = true;
    run.dynamic = true;
    do_run(run);

    run = Run();
    run.caps = false;
    run.sphere = false;
    run.function = Franke2d;
    run.name = "laplace_comparison_degree_needle_plane_dyn.csv";
    run.degree_dims = true;
    run.dynamic = true;
    do_run(run);

    run = Run();
    run.caps = true;
    run.sphere = false;
    run.function = Franke2d;
    run.name = "laplace_comparison_dims_caps_plane_dyn.csv";
    run.degree_dims = false;
    run.dynamic = true;
    do_run(run);

    run = Run();
    run.caps = false;
    run.sphere = false;
    run.function = Franke2d;
    run.name = "laplace_comparison_dims_needle_plane_dyn.csv";
    run.degree_dims = false;
    run.dynamic = true;
    do_run(run);

}