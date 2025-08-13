#include "igl/intrinsic_delaunay_cotmatrix.h"
#include "construct_laplace.h"
#include "bunge_poly_laplace.h"
#include "intrinsic_delaunay.h"
#include "cgal_laplacian.h"
#include "openmesh_laplacian.h"
#include "vcglib_laplacian.h"
#include "geogram_laplacian.h"
#include "igl/cotmatrix.h"
#include "pmp/algorithms/laplace.h"
#include "geogram/mesh/mesh.h"
#include "cinolib/laplacian.h"
#include "pmp/stop_watch.h"

void setup_laplacian(Eigen::SparseMatrix<double>& L, Eigen::DiagonalMatrix<double, Eigen::Dynamic>& M,
                     const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, LaplaceConfig config)
{
    L.resize(V.rows(), V.rows());
    L.setZero();
    M.resize(V.rows());
    M.setZero();
    pmp::SparseMatrix M_S;
    pmp::StopWatch stop_watch;
    pmp::SurfaceMesh pmp_mesh;
    to_pmp_mesh(V, F, pmp_mesh);
    area_computation = CrossProduct;
    stop_watch.start();
    tri_mass_matrix(pmp_mesh, M, {OptimizedLaplacian_Cross_Dot, TFEMNormal});
    stop_watch.stop();
    switch (config.lib)
    {
    case PMP:
        {
            // PMP Laplace and Mixed Cells Mass Matrix
            pmp::LAPLACE_VERSION = TFEMNormal;
            stop_watch.resume();
            pmp::laplace_matrix(pmp_mesh, L);
            stop_watch.stop();
            break;
        }
    case Open_Mesh:
        {
            // OpenMesh, only Laplace Weights in Smoother (no mass matrix)
            DefaultTriMesh open_mesh;
            to_open_mesh(V, F, open_mesh);
            stop_watch.resume();
            openmesh_laplacian<DefaultTriMesh>(open_mesh, L);
            stop_watch.stop();
            break;
        }

    case Cgal:
        {
            // CGAL, only Laplace Matrix in Example (no mass matrix)
            CGALMesh cgal_mesh;
            to_cgal_mesh(V, F, cgal_mesh);
            stop_watch.resume();
            cgal_laplacian(cgal_mesh, L);
            stop_watch.stop();
            break;
        }

    case Geometry_Central:
        {
            // Geometry-Central Laplace and Barycentric Cells Mass Matrix
            std::unique_ptr<geometrycentral::surface::SurfaceMesh> gc_mesh;
            std::unique_ptr<geometrycentral::surface::VertexPositionGeometry> gc_geometry;
            to_gc_mesh(V, F, gc_mesh, gc_geometry);
            //gc_geometry->requireVertexLumpedMassMatrix();
            stop_watch.resume();
            gc_geometry->requireCotanLaplacian();
            L = -gc_geometry->cotanLaplacian;
            stop_watch.stop();
            break;
        }

    case IGL:
        {
            // LIBIGL Laplace and Mixed Cells Mass Matrix
            stop_watch.resume();
            igl::cotmatrix(V, F, L);
            stop_watch.stop();
            break;
        }
    case Intrinsic_Delaunay:
        {
            Eigen::MatrixXi F_int;
            Eigen::MatrixXd l_int;
            stop_watch.start();
            igl::intrinsic_delaunay_cotmatrix(V, F, L, l_int, F_int);
            massmatrix_intrinsic(l_int, F_int, igl::MASSMATRIX_TYPE_BARYCENTRIC, M_S);
            M = M_S.diagonal().asDiagonal();
            stop_watch.stop();
            break;
        }
    case Intrinsic_Delaunay_Mollification:
        {
            // Geometry-Central Laplace and Barycentric Cells Mass Matrix
            std::unique_ptr<geometrycentral::surface::SurfaceMesh> gc_mesh;
            std::unique_ptr<geometrycentral::surface::VertexPositionGeometry> gc_geometry;
            to_gc_mesh(V, F, gc_mesh, gc_geometry);
            stop_watch.start();
            auto [Lp, Mp] = idt_laplacian(*gc_mesh, *gc_geometry, MOLLIFICATION_FACTOR);
            L = -Lp;
            M = Mp.diagonal().asDiagonal();
            stop_watch.stop();
            break;
        }
    case Intrinsic_Mollification:
        {
            // Geometry-Central Laplace and Barycentric Cells Mass Matrix
            std::unique_ptr<geometrycentral::surface::SurfaceMesh> gc_mesh;
            std::unique_ptr<geometrycentral::surface::VertexPositionGeometry> gc_geometry;
            to_gc_mesh(V, F, gc_mesh, gc_geometry);
            stop_watch.start();
            auto [Lp, Mp] = idt_laplacian(*gc_mesh, *gc_geometry, MOLLIFICATION_FACTOR, false);
            L = -Lp;
            M = Mp.diagonal().asDiagonal();
            stop_watch.stop();
            break;
        }
    case OptimizedLaplacian_Cross_Dot:
        {
            area_computation = CrossProduct;
            dot_sqlength = true;
            stop_watch.resume();
            tri_laplace_matrix(V, F, L, config);
            stop_watch.stop();
            break;
        }
    case OptimizedLaplacian_Cross_Dot_Mass:
        {
            area_computation = CrossProduct;
            dot_sqlength = true;
            stop_watch.start();
            tri_mass_matrix(pmp_mesh, M, config);
            tri_laplace_matrix(V, F, L, config);
            stop_watch.stop();
            break;
        }
    case OptimizedLaplacian_Cross_l2Sq:
        {
            area_computation = CrossProduct;
            dot_sqlength = false;
            stop_watch.resume();
            tri_laplace_matrix(V, F, L, config);
            stop_watch.stop();
            break;
        }
    case OptimizedLaplacian_Heron_Dot:
        {
            area_computation = Heron_Sorted;
            dot_sqlength = true;
            stop_watch.resume();
            tri_laplace_matrix(V, F, L, config);
            stop_watch.stop();
            break;
        }
    case OptimizedLaplacian_Heron_l2Sq:
        {
            area_computation = Heron_Sorted;
            dot_sqlength = false;
            stop_watch.resume();
            tri_laplace_matrix(V, F, L, config);
            stop_watch.stop();
            break;
        }
    case OptimizedLaplacian_Max_Dot:
        {
            area_computation = Max_Area;
            dot_sqlength = true;
            stop_watch.resume();
            tri_laplace_matrix(V, F, L, config);
            stop_watch.stop();
            break;
        }
    case OptimizedLaplacian_Max_l2Sq:
        {
            area_computation = Max_Area;
            dot_sqlength = false;
            stop_watch.resume();
            tri_laplace_matrix(V, F, L, config);
            stop_watch.stop();
            break;
        }
    case VCGLib:
        {
            VCGMesh vcglib_mesh;
            to_vcglib_mesh(V, F, vcglib_mesh);
            stop_watch.resume();
            vcglib_laplacian(vcglib_mesh, L);
            stop_watch.stop();
            break;
        }
    case Geogram:
        {
            GEO::Mesh geo_mesh;
            to_geogram_mesh(V, F, geo_mesh);
            stop_watch.resume();
            geogram_laplacian(geo_mesh, L);
            stop_watch.stop();
            break;
        }
    case CinoLib:
        {
            cinolib::Trimesh cinolib_mesh;
            to_cinolib_mesh(V, F, cinolib_mesh);
            stop_watch.resume();
            L = laplacian(cinolib_mesh, cinolib::COTANGENT);
            stop_watch.stop();
            break;
        }
    case BungeLaplace_Cross_Dot:
        {
            area_computation = CrossProduct;
            dot_sqlength = true;
            stop_watch.start();
            poly_laplace_matrix(pmp_mesh, L, config);
            poly_mass_matrix(pmp_mesh, M, config);
            stop_watch.stop();
            break;
        }
    case BungeLaplace_Cross_l2Sq:
        {
            area_computation = CrossProduct;
            dot_sqlength = false;
            stop_watch.start();
            poly_laplace_matrix(pmp_mesh, L, config);
            poly_mass_matrix(pmp_mesh, M, config);
            stop_watch.stop();
            break;
        }
    case BungeLaplace_Heron_Dot:
        {
            area_computation = Heron_Sorted;
            dot_sqlength = true;
            stop_watch.start();
            poly_laplace_matrix(pmp_mesh, L, config);
            poly_mass_matrix(pmp_mesh, M, config);
            stop_watch.stop();
            break;
        }
    case BungeLaplace_Heron_l2Sq:
        {
            area_computation = Heron_Sorted;
            dot_sqlength = false;
            stop_watch.start();
            poly_laplace_matrix(pmp_mesh, L, config);
            poly_mass_matrix(pmp_mesh, M, config);
            stop_watch.stop();
            break;
        }
    case BungeLaplace_Max_Dot:
        {
            area_computation = Max_Area;
            dot_sqlength = true;
            stop_watch.start();
            poly_laplace_matrix(pmp_mesh, L, config);
            poly_mass_matrix(pmp_mesh, M, config);
            stop_watch.stop();
            break;
        }
    case BungeLaplace_Max_l2Sq:
        {
            area_computation = Max_Area;
            dot_sqlength = false;
            stop_watch.start();
            poly_laplace_matrix(pmp_mesh, L, config);
            poly_mass_matrix(pmp_mesh, M, config);
            stop_watch.stop();
            break;
        }
    }
    elapsed = stop_watch.elapsed();
    std::cout << lib_map[config.lib] << ": " << stop_watch.elapsed() << " ms" << std::endl;
}
