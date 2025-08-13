#pragma once
#include "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"
#include <OpenMesh/Core/Geometry/EigenVectorT.hh>
#include <Eigen/Sparse>


struct EigenTraits : OpenMesh::DefaultTraits {
  using Point = Eigen::Vector3d;
  using Normal = Eigen::Vector3d;

  using TexCoord2D = Eigen::Vector2d;
};


typedef OpenMesh::TriMesh_ArrayKernelT<EigenTraits>  DefaultTriMesh;

template <class Mesh>
void openmesh_laplacian(Mesh& mesh_, Eigen::SparseMatrix<double>& L)
{
  typename Mesh::VertexIter        v_it, v_end(mesh_.vertices_end());
  typename Mesh::EdgeIter          e_it, e_end(mesh_.edges_end());
  typename Mesh::HalfedgeHandle    heh0, heh1, heh2;
  typename Mesh::VertexHandle      v0, v1;
  typename Mesh::Normal            d0, d1;
  typename Mesh::Scalar            weight, lb(-1.0), ub(1.0);

  L.resize(mesh_.n_vertices(), mesh_.n_vertices());


  std::vector<Eigen::Triplet<double>> triplets;
  for (e_it=mesh_.edges_begin(); e_it!=e_end; ++e_it)
  {
    const typename Mesh::Point       *p0, *p1, *p2;

    weight = 0.0;

    heh0   = mesh_.halfedge_handle(*e_it, 0);
    v0     = mesh_.to_vertex_handle(heh0);
    int i0 = v0.idx();
    p0     = &mesh_.point(v0);

    heh1   = mesh_.halfedge_handle(*e_it, 1);
    v1     = mesh_.to_vertex_handle(heh1);
    p1     = &mesh_.point(v1);
    int i1 = v1.idx();

    if (!mesh_.is_boundary(heh0))
    {
      heh2   = mesh_.next_halfedge_handle(heh0);
      p2     = &mesh_.point(mesh_.to_vertex_handle(heh2));
      d0     = (*p0 - *p2); normalize(d0);
      d1     = (*p1 - *p2); normalize(d1);
      weight += 1.0 / tan(acos(std::max(lb, std::min(ub, dot(d0,d1) ))));
    }

    if (!mesh_.is_boundary(heh1))
    {
      heh2   = mesh_.next_halfedge_handle(heh1);
      p2     = &mesh_.point(mesh_.to_vertex_handle(heh2));
      d0     = (*p0 - *p2); normalize(d0);
      d1     = (*p1 - *p2); normalize(d1);
      weight += 1.0 / tan(acos(std::max(lb, std::min(ub, dot(d0,d1) ))));
    }
    weight /= 2.0;

    triplets.emplace_back(i0, i1, weight);
    triplets.emplace_back(i1, i0, weight);
    triplets.emplace_back(i0, i0, -weight);
    triplets.emplace_back(i1, i1, -weight);

  }
  L.setFromTriplets(triplets.begin(), triplets.end());


}