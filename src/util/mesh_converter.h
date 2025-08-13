#pragma once

#include <Eigen/Core>
#include <pmp/surface_mesh.h>
#include <geometrycentral/surface/surface_mesh.h>
#include <geometrycentral/surface/surface_mesh_factories.h>
#include "../Laplacians/openmesh_laplacian.h"
#include "../Laplacians/cgal_laplacian.h"
#include "../Laplacians/vcglib_laplacian.h"
#include "geogram/mesh/mesh.h"
#include <cinolib/meshes/trimesh.h>

#include "pmp/algorithms/differential_geometry.h"

inline void to_pmp_mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, pmp::SurfaceMesh& mesh)
{
    mesh.clear();
    mesh.garbage_collection();
    int n_vertices = V.rows();
    int n_faces = F.rows();
    for (int i = 0; i < n_vertices; i++)
    {
        mesh.add_vertex(V.row(i));
    }
    for (int i = 0; i < n_faces; i++)
    {
        std::vector<pmp::Vertex> indices;
        for (int j = 0; j < F.cols(); j++)
        {
            if (F(i,j) != -1)
                indices.emplace_back(F(i,j));
        }
        mesh.add_face(indices);
    }
}

inline void pmp_to_arrays(Eigen::MatrixXd& V, Eigen::MatrixXi& F, const pmp::SurfaceMesh& mesh)
{
    V.resize(mesh.n_vertices(), 3);
    int max_valence = 3;
    for (auto f: mesh.faces())
    {
        max_valence = fmax(max_valence, mesh.valence(f));
    }
    F.resize(mesh.n_faces(), max_valence);
    int i = 0;
    for (auto v : mesh.vertices())
    {
        V.row(i) = Eigen::Vector3d(mesh.position(v));
        i++;
    }

    i = 0;
    for (auto f : mesh.faces())
    {
        Eigen::VectorXi indices = -1 * Eigen::VectorXi::Ones(max_valence);
        int j = 0;
        for (auto v : mesh.vertices(f))
        {
            indices[j] = v.idx();
            j++;
        }
        F.row(i) = indices;
        i++;
    }
}

inline void to_gc_mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, std::unique_ptr<geometrycentral::surface::SurfaceMesh>& mesh, std::unique_ptr<geometrycentral::surface::VertexPositionGeometry>& geometry)
{
    std::tie(mesh, geometry) = geometrycentral::surface::makeSurfaceMeshAndGeometry(V, F);
}

inline void to_open_mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, DefaultTriMesh& mesh)
{
    mesh.clear();
    mesh.garbage_collection();
    int n_vertices = V.rows();
    int n_faces = F.rows();
    for (int i = 0; i < n_vertices; i++)
    {
        mesh.add_vertex(V.row(i));
    }
    for (int i = 0; i < n_faces; i++)
    {
        const auto v1 = OpenMesh::VertexHandle(F.row(i)[0]);
        const auto v2 = OpenMesh::VertexHandle(F.row(i)[1]);
        const auto v3 = OpenMesh::VertexHandle(F.row(i)[2]);
        mesh.add_face(v1, v2, v3);
    }
}

inline void to_cgal_mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, CGALMesh& mesh)
{
    mesh.clear();
    int n_vertices = V.rows();
    int n_faces = F.rows();
    for (int i = 0; i < n_vertices; i++)
    {
        mesh.add_vertex(Point_3(V.row(i)[0], V.row(i)[1],V.row(i)[2]));
    }
    for (int i = 0; i < n_faces; i++)
    {
        const auto v1 = CGALVIndex(F.row(i)[0]);
        const auto v2 = CGALVIndex(F.row(i)[1]);
        const auto v3 = CGALVIndex(F.row(i)[2]);
        mesh.add_face(v1, v2, v3);
    }
}

inline void to_vcglib_mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, VCGMesh& mesh)
{
    mesh.Clear();

    auto v_iter = vcg::tri::Allocator<VCGMesh>::AddVertices(mesh, V.rows());
    for (int i = 0; i < V.rows(); i++, ++v_iter)
    {
        Eigen::Vector3d pt = V.row(i);
        v_iter->P() = vcg::Point3(pt[0], pt[1], pt[2]);
    }
    for (int i = 0; i < F.rows(); i++)
    {
        Eigen::Vector3i indices = F.row(i);
        vcg::tri::Allocator<VCGMesh>::AddFace(mesh, &mesh.vert[indices[0]], &mesh.vert[indices[1]], &mesh.vert[indices[2]]);
    }
}

inline void to_geogram_mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, GEO::Mesh& mesh)
{
    mesh.clear();
    GEO::vector<double> vert_list;
    for (int i = 0; i < V.rows(); i++)
    {
        vert_list.push_back(V.row(i)[0]);
        vert_list.push_back(V.row(i)[1]);
        vert_list.push_back(V.row(i)[2]);
    }
    mesh.vertices.create_vertices(V.rows());
    mesh.vertices.assign_points(vert_list, 3, false);
    for (int i = 0; i < F.rows(); i++)
    {
        Eigen::Vector3i indices = F.row(i);
        mesh.facets.create_triangle(indices[0], indices[1], indices[2]);
    }
}

inline void to_cinolib_mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, cinolib::Trimesh<>& mesh)
{

    std::vector<cinolib::vec3d> verts;
    std::vector<std::vector<uint>> tris;
    for (int i = 0; i < V.rows(); i++)
    {
        verts.emplace_back(V.row(i)[0], V.row(i)[1], V.row(i)[2]);
    }

    for (int i = 0; i < F.rows(); i++)
    {
        std::vector tri = {static_cast<unsigned>(F.row(i)[0]), static_cast<unsigned>(F.row(i)[1]), static_cast<unsigned>(F.row(i)[2])};
        tris.push_back(tri);
    }
    mesh.clear();
    mesh.init(verts, tris);
}

inline void dualize(pmp::SurfaceMesh& mesh)
{
    // the new dualized mesh
    pmp::SurfaceMesh tmp;

    // remember new vertices per face
    auto fvertex = mesh.add_face_property<pmp::Vertex>("f:vertex");

    // add centroid for each face
    for (auto f : mesh.faces())
        fvertex[f] = tmp.add_vertex(centroid(mesh, f));

    // add new face for each vertex
    for (auto v : mesh.vertices())
    {
        std::vector<pmp::Vertex> vertices;
        for (auto f : mesh.faces(v))
            vertices.push_back(fvertex[f]);

        tmp.add_face(vertices);
    }

    // swap old and new meshes, don't copy properties
    mesh.assign(tmp);
}