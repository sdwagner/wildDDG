// =============================================================================
//  Copyright (c) 2025 Sven D. Wagner, Mario Botsch.
//  Distributed under MIT license, see file LICENSE for details.
// =============================================================================

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include "pmp/surface_mesh.h"
#include "pmp/io/io.h"

#include "eigenlib/Eigen/src/Core/Matrix.h"
#include "Laplacians/construct_laplace.h"
#include "pmp/algorithms/numerics.h"
#include "util/enums.h"
#include "util/system_solver.h"

int main()
{
    std::vector libs = {
        PMP, Open_Mesh, IGL, Geometry_Central, Cgal, VCGLib, Geogram, CinoLib, Intrinsic_Mollification,
        Intrinsic_Delaunay_Mollification, OptimizedLaplacian_Cross_Dot, OptimizedLaplacian_Cross_l2Sq, OptimizedLaplacian_Heron_Dot,
        OptimizedLaplacian_Heron_l2Sq, OptimizedLaplacian_Max_Dot, OptimizedLaplacian_Max_l2Sq,
        OptimizedLaplacian_Cross_Dot_Mass
    };
    const std::vector tfem_libs = {
        OptimizedLaplacian_Cross_Dot, OptimizedLaplacian_Max_Dot, OptimizedLaplacian_Cross_Dot_Mass
    };


    const std::string path = "../out/thingi10k_tests/";
    std::filesystem::create_directories(path);
    std::ofstream error_file(path + "summary.csv");


    for (const auto lib : libs)
    {
        if (std::ranges::find(tfem_libs, lib) != tfem_libs.end())
        {
            error_file << lib_map[lib] << " (TFEM " << tfem_method_map[TFEMNormal] << "),";
            error_file << lib_map[lib] << " (TFEM " << tfem_method_map[TFEMNormal] << ") Time,";
            error_file << lib_map[lib] << " (TFEM " << tfem_method_map[TFEMStatic] << "),";
            error_file << lib_map[lib] << " (TFEM " << tfem_method_map[TFEMStatic] << ") Time,";
            error_file << lib_map[lib] << " (TFEM " << tfem_method_map[TFEMDynamic] << "),";
            error_file << lib_map[lib] << " (TFEM " << tfem_method_map[TFEMDynamic] << ") Time,";
            error_file << lib_map[lib] << " (TFEM " << tfem_method_map[TFEMDynamicFailsafe] << "),";
            error_file << lib_map[lib] << " (TFEM " << tfem_method_map[TFEMDynamicFailsafe] << ") Time,";
        }
        else
        {
            error_file << lib_map[lib] << ",";
            error_file << lib_map[lib] << " Time,";
        }
    }
    error_file << std::endl;
    for (int i = 0; i <= 6123; i++)
    {
        std::cout << i << std::endl;
        std::string obj_path = "../thingi10k/" + std::to_string(i) + ".obj";
        if (std::filesystem::exists(obj_path))
        {
            try
            {
                pmp::SurfaceMesh mesh;
                pmp::read(mesh, obj_path);
                Eigen::MatrixXd V;
                Eigen::MatrixXi F;
                pmp::mesh_to_matrices(mesh, V, F);

                double error;
                for (const auto lib : libs)
                {
                    Eigen::SparseMatrix<double> L;
                    Eigen::DiagonalMatrix<double, Eigen::Dynamic> M;
                    if (std::ranges::find(tfem_libs, lib) != tfem_libs.end())
                    {
                        setup_laplacian(L, M, V, F, {lib, TFEMNormal});
                        error_file << (isfinite(L.sum()) ? 1 : 0) << "," << elapsed << ",";
                        setup_laplacian(L, M, V, F, {lib, TFEMStatic});
                        error_file << (isfinite(L.sum()) ? 1 : 0) << "," << elapsed << ",";
                        setup_laplacian(L, M, V, F, {lib, TFEMDynamic});
                        error_file << (isfinite(L.sum()) ? 1 : 0) << "," << elapsed << ",";
                        setup_laplacian(L, M, V, F, {lib, TFEMDynamicFailsafe});
                        error_file << (isfinite(L.sum()) ? 1 : 0) << "," << elapsed << ",";
                    }
                    else
                    {
                        setup_laplacian(L, M, V, F, {lib, TFEMNormal});
                        error_file << (isfinite(L.sum()) ? 1 : 0) << "," << elapsed << ",";
                    }
                }
            }
            catch (const std::exception& e)
            {
                std::cout << path << std::endl;
            }
            error_file << std::endl;
        }
    }
    error_file.close();
    return 0;
}