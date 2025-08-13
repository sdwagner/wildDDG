#pragma once
//
// Created by Sven Wagner on 26.11.24.
//
#include <CGAL/Surface_mesh/Surface_mesh.h>
#include <CGAL/Weights/discrete_harmonic_weights.h>
#include <CGAL/Eigen_matrix.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>

using Kernel  = CGAL::Exact_predicates_exact_constructions_kernel;
using FT      = Kernel::FT;
using Point_3 = Kernel::Point_3;

using CGALMesh = CGAL::Surface_mesh<Point_3>;
using VD   = boost::graph_traits<CGALMesh>::vertex_descriptor;
using HD   = boost::graph_traits<CGALMesh>::halfedge_descriptor;
using CGALVIndex = CGALMesh::Vertex_index;

template<typename PointMap>
FT get_w_ij(const CGALMesh& mesh, HD he, PointMap pmap);

void cgal_laplacian(const CGALMesh& mesh, Eigen::SparseMatrix<double>& L);
