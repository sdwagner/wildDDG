// =============================================================================
//  Copyright (c) 2025 Sven D. Wagner, Mario Botsch.
//  Distributed under MIT license, see file LICENSE for details.
// =============================================================================

#pragma once

#include <Eigen/Core>
#include <map>

#include "mesh_converter.h"

#ifndef __EMSCRIPTEN__
#include "igl/predicates/delaunay_triangulation.h"
#endif


inline void collapse(Eigen::MatrixXd& V, Eigen::MatrixXi& F, int from_idx, int to_idx)
{
    // Simple halfedge collapse using PMP (much overhead)
    pmp::SurfaceMesh mesh;
    to_pmp_mesh(V, F, mesh);
    pmp::Halfedge h = mesh.find_halfedge(pmp::Vertex(from_idx), pmp::Vertex(to_idx));
    mesh.collapse(h);
    mesh.garbage_collection();
    V.resize(mesh.n_vertices(), 3);
    F.resize(mesh.n_faces(), 3);
    for (int i = 0; i < mesh.n_vertices(); i++)
    {
        V.row(i) = Eigen::Vector3d(mesh.position(pmp::Vertex(i)));
    }
    for (int i = 0; i < mesh.n_faces(); i++)
    {
        auto f = mesh.vertices(pmp::Face(i)).begin();
        pmp::Vertex v1 = *f++;
        pmp::Vertex v2 = *f++;
        pmp::Vertex v3 = *f;
        F.row(i) = Eigen::Vector3i(v1.idx(), v2.idx(), v3.idx());
    }
}

inline void generate_tri_grid(Eigen::MatrixXd& V, Eigen::MatrixXi& F, int dims, float edge_length, bool caps)
{
    V.resize(dims*dims, 3);
    F.resize(2*(dims-1)*(dims-1), 3);
    double dx = 2.0 / (dims-1);

    int face_index = 0;
    for (int i = 0; i < dims; i++)
    {
        for (int j = 0; j < dims; j++)
        {
            V.row(dims*i+j) = Eigen::Vector3d(dx*i-1, dx*j-1, 0);
            if (j != 0 && i != 0)
            {
                if (i <= dims / 2 ^ j <= dims / 2)
                {
                    F.row(face_index) = Eigen::Vector3i(j+i*dims, j+(i-1)*dims, j+(i-1)*dims-1);
                    face_index++;
                    F.row(face_index) = Eigen::Vector3i(j+i*dims, j+(i-1)*dims-1, j+i*dims-1);
                    face_index++;
                }
                else
                {
                    F.row(face_index) = Eigen::Vector3i(j+(i-1)*dims, j+i*dims-1, j+i*dims);
                    face_index++;
                    F.row(face_index) = Eigen::Vector3i(j+(i-1)*dims, j+(i-1)*dims-1, j+i*dims-1);
                    face_index++;
                }
            }
        }
    }
    int v1 = dims*dims/2;
    int v2 = dims*dims/2 +1;

    Eigen::Vector3d vec = V.row(v2) - V.row(v1);
    Eigen::Vector3d p1 = V.row(v1);
    V.row(v2) = p1 + edge_length * vec;

    if (caps)
    {
        collapse(V, F, v1, v1+dims);
    }
}

inline double mapping(const double x, const double deg)
{
    return (x < 0 ? -1.0 : 1.0) * pow(abs(x), deg);
}

inline Eigen::Vector3d map_to_sphere(const Eigen::Vector3d& p)
{
    double angle_1 = 2 * M_PI * (0.5*p[0] + 0.5);
    double angle_2 = acos(p[1]);

    return {sin(angle_2)*sin(angle_1), -cos(angle_2), -sin(angle_2)*cos(angle_1)};

}

inline Eigen::Vector3d map_to_cylinder(const Eigen::Vector3d& p)
{
    double angle_1 = 2 * M_PI * (0.5*p[0] + 0.5);

    return {sin(angle_1), 5*p[1], cos(angle_1)};

}

inline void generate_tri_grid_inconsistent(Eigen::MatrixXd& V, Eigen::MatrixXi& F, int dims, double deg, bool caps, bool sphere, bool cylinder, bool compress_both = false)
{
    V.resize(dims*dims, 3);
    V.setZero();
    F.resize(2*(dims-1)*(dims-1), 3);
    double dx = 2.0 / (dims-1);

    if (sphere)
    {
        V.row(0) = map_to_sphere(Eigen::Vector3d(-1, -1, 0));
        V.row(dims) = map_to_sphere(Eigen::Vector3d(1, 1, 0));
    }
    else if (cylinder)
    {
        V.row(0) = (Eigen::Vector3d(0, 5*mapping(dx-1, deg), 0));
        V.row(dims) = (Eigen::Vector3d(0, 5*mapping(dx*(dims-2)-1, deg), 0));
    }

    int face_index = 0;
    for (int i = 0; i < dims; i++)
    {
        for (int j = 0; j < dims; j++)
        {

            if (caps)
            {
                if ((sphere || cylinder) && (j == 0 || j == dims -1))
                {
                    continue;
                }
                int ind = i;
                if ((sphere || cylinder) && i == dims - 1)
                {
                    ind = 0;
                }
                else
                {
                    Eigen::Vector3d p;
                    if (compress_both)
                        p = Eigen::Vector3d(mapping(dx*i-1, deg), mapping(dx*j-1, deg), 0);
                    else
                        p = Eigen::Vector3d(dx*i-1, mapping(dx*j-1, deg), 0);
                    if (sphere)
                    {
                        p = map_to_sphere(p);
                    }
                    else if (cylinder)
                    {
                        p = map_to_cylinder(p);
                    }
                    V.row(dims*i+j) = p;
                }
                if (j == 0 || i == 0)
                    continue;
                if ((sphere || cylinder) && j == 1 && (i % 2 == 0))
                {
                    F.row(face_index) = Eigen::Vector3i(0, j+ind*dims, j+(i-2)*dims);
                    face_index++;
                    continue;
                }
                if ((sphere || cylinder) && j == 1)
                    continue;
                if ((sphere || cylinder) && j == dims-2 && (i % 2 == 0))
                {
                    F.row(face_index) = Eigen::Vector3i(dims, j+(i-2)*dims, j+ind*dims);
                    face_index++;
                }
                if (j % 2 == 0)
                {
                    if (i % 2 == 0)
                    {
                         F.row(face_index) = Eigen::Vector3i((j+(i-1)*dims), (j+(i-2)*dims-1), (j+ind*dims-1));
                        face_index++;
                    }
                    else if (i == 1)
                    {
                         F.row(face_index) = Eigen::Vector3i((j+ind*dims), (j+(i-1)*dims), (j+(i-1)*dims-1));
                        face_index++;
                    }
                    else
                    {
                         F.row(face_index) = Eigen::Vector3i((j+ind*dims), (j+(i-2)*dims), (j+(i-1)*dims-1));
                        face_index++;
                    }
                    if (i == dims-1)
                    {
                         F.row(face_index) = Eigen::Vector3i((j+ind*dims), (j+(i-1)*dims), (j+ind*dims-1));
                        face_index++;
                    }

                }
                else
                {
                    if (i % 2 == 0)
                    {
                         F.row(face_index) = Eigen::Vector3i((j+ind*dims), (j+(i-2)*dims), (j+(i-1)*dims-1));
                        face_index++;
                    }
                    else if (i == 1)
                    {
                         F.row(face_index) = Eigen::Vector3i((j+(i-1)*dims), (j+(i-1)*dims-1), (j+ind*dims-1));
                        face_index++;
                    }
                    else
                    {
                         F.row(face_index) = Eigen::Vector3i((j+(i-1)*dims), (j+(i-2)*dims-1), (j+ind*dims-1));
                        face_index++;
                    }
                    if (i == dims-1)
                    {
                         F.row(face_index) = Eigen::Vector3i((j+ind*dims), (j+(i-1)*dims-1), (j+ind*dims-1));
                        face_index++;
                    }

                }
            }
            else
            {
                if ((sphere || cylinder) && (j == 0 || j == dims -1))
                {
                    continue;
                }
                int ind = i;
                if ((sphere || cylinder) && i == dims - 1)
                {
                    ind = 0;
                }
                else
                {
                    Eigen::Vector3d p;
                    if (compress_both)
                        p = Eigen::Vector3d(mapping(dx*i-1, deg), mapping(dx*j-1, deg), 0);
                    else
                        p = Eigen::Vector3d(dx*i-1, mapping(dx*j-1, deg), 0);
                    if (sphere)
                    {
                        p = map_to_sphere(p);
                    }
                    else if (cylinder)
                    {
                        p = map_to_cylinder(p);
                    }
                    V.row(dims*i+j) = p;
                }
                if (j == 0 || i == 0)
                    continue;
                if ((sphere || cylinder) && j == 1)
                {
                    F.row(face_index) = Eigen::Vector3i(0, j+ind*dims, j+(i-1)*dims);
                    face_index++;
                    continue;
                }
                if ((sphere || cylinder) && j == dims-2)
                {
                    F.row(face_index) = Eigen::Vector3i(dims, j+(i-1)*dims, j+ind*dims);
                    face_index++;
                }
                if (i <= dims / 2)
                {
                    F.row(face_index) = Eigen::Vector3i(j+ind*dims, j+(i-1)*dims, j+(i-1)*dims-1);
                    face_index++;
                    F.row(face_index) = Eigen::Vector3i(j+ind*dims, j+(i-1)*dims-1, j+ind*dims-1);
                    face_index++;
                }
                else
                {
                    F.row(face_index) = Eigen::Vector3i(j+(i-1)*dims, j+ind*dims-1, j+ind*dims);
                    face_index++;
                    F.row(face_index) = Eigen::Vector3i(j+(i-1)*dims, j+(i-1)*dims-1, j+ind*dims-1);
                    face_index++;
                }
            }
        }
    }


    if (caps){
        Eigen::MatrixXi F_temp = F.topLeftCorner(face_index, 3);
        int n = 0;
        for (int i = 0; i < dims; i++)
        {
            for (int j = 0; j < dims; j++)
            {
                if ((j % 2 == 0 || i % 2 == 0) && (j % 2 == 1 || i % 2 == 1 || i == 0 || i == dims-1))
                {
                    if ((sphere || cylinder))
                    {
                        if (Eigen::Vector3d(V.row(j+i*dims)).norm() > 1e-8)
                            n++;
                    }
                    else
                    {
                        n++;
                    }
                }

            }
        }

        Eigen::MatrixXd V_temp(n, 3);
        std::map<int, int> idx_map;
        int index = 0;
        for (int i = 0; i < dims; i++)
        {
            for (int j = 0; j < dims; j++)
            {
                if ((j % 2 == 0 || i % 2 == 0) && (j % 2 == 1 || i % 2 == 1 || i == 0 || i == dims-1))
                {
                    if ((sphere || cylinder))
                    {
                        if (Eigen::Vector3d(V.row(j+i*dims)).norm() > 1e-8)
                        {
                            idx_map.insert(std::pair(j+i*dims, index));
                            V_temp.row(index) = V.row(j+i*dims);
                            index++;
                        }
                    }
                    else
                    {
                        idx_map.insert(std::pair(j+i*dims, index));
                        V_temp.row(index) = V.row(j+i*dims);
                        index++;

                    }
                }
            }
        }
        V = V_temp;

        for (int i = 0; i < F_temp.rows(); i++)
        {
            for (auto j = 0; j < F_temp.cols(); j++)
            {
                F_temp(i,j) = idx_map[F_temp(i,j)];
            }
        }
        F = F_temp;
    }
    else if ((sphere || cylinder))
    {

        Eigen::MatrixXi F_temp = F.topLeftCorner(face_index, 3);
        int n = 0;
        for (int i = 0; i < dims; i++)
        {
            for (int j = 0; j < dims; j++)
            {
                if (Eigen::Vector3d(V.row(j+i*dims)).norm() > 1e-8)
                    n++;

            }
        }

        Eigen::MatrixXd V_temp(n, 3);
        std::map<int, int> idx_map;
        int index = 0;
        for (int i = 0; i < dims; i++)
        {
            for (int j = 0; j < dims; j++)
            {
                if (Eigen::Vector3d(V.row(j+i*dims)).norm() > 1e-8)
                {
                    idx_map.insert(std::pair(j+i*dims, index));
                    V_temp.row(index) = V.row(j+i*dims);
                    index++;
                }
            }
        }
        V = V_temp;

        for (int i = 0; i < F_temp.rows(); i++)
        {
            for (auto j = 0; j < F_temp.cols(); j++)
            {
                F_temp(i,j) = idx_map[F_temp(i,j)];
            }
        }
        F = F_temp;
    }
}

inline void generate_quad_grid(Eigen::MatrixXd& V, Eigen::MatrixXi& F, int dims, float edge_length)
{
    V.resize(dims*dims, 3);
    F.resize((dims-1)*(dims-1), 4);
    double dx = 2.0 / (dims-1);

    int face_index = 0;
    for (int i = 0; i < dims; i++)
    {
        for (int j = 0; j < dims; j++)
        {
            V.row(dims*i+j) = Eigen::Vector3d(dx*i-1, dx*j-1, 0);
            if (j != 0 && i != 0)
            {
                F.row(face_index) = Eigen::Vector4i(j+i*dims, j+(i-1)*dims, j+(i-1)*dims-1, j+i*dims-1);
                face_index++;
            }
        }
    }
    int v1 = dims*dims/2;
    int v2 = dims*dims/2 +1;

    Eigen::Vector3d vec = V.row(v2) - V.row(v1);
    Eigen::Vector3d p1 = V.row(v1);
    V.row(v2) = p1 + edge_length * vec;
}

inline void generate_tri_quad_grid_inconsistent(Eigen::MatrixXd& V, Eigen::MatrixXi& F, int dims, double deg, bool sphere)
{
    V.resize(dims*dims, 3);
    V.setZero();
    F.resize((dims-1)*(dims-1), 4);
    double dx = 2.0 / (dims-1);

    if (sphere)
    {
        V.row(0) = map_to_sphere(Eigen::Vector3d(-1, -1, 0));
        V.row(dims) = map_to_sphere(Eigen::Vector3d(1, 1, 0));
    }

    int face_index = 0;
    for (int i = 0; i < dims; i++)
    {
        for (int j = 0; j < dims; j++)
        {

            if (sphere && (j == 0 || j == dims -1))
            {
                continue;
            }
            int ind = i;
            if (sphere && i == dims - 1)
            {
                ind = 0;
            }
            else
            {
                Eigen::Vector3d p = Eigen::Vector3d(dx*i-1, mapping(dx*j-1, deg), 0);
                if (sphere)
                {
                    p = map_to_sphere(p);
                }
                V.row(dims*i+j) = p;
            }
            if (j == 0 || i == 0)
                continue;
            if (sphere && j == 1)
            {
                F.row(face_index) = Eigen::Vector4i(0, j+ind*dims, j+(i-1)*dims, -1);
                face_index++;
                continue;
            }
            if (sphere && j == dims-2)
            {
                F.row(face_index) = Eigen::Vector4i(dims, j+(i-1)*dims, j+ind*dims, -1);
                face_index++;
            }
            F.row(face_index) = Eigen::Vector4i(j+ind*dims, j+(i-1)*dims, j+(i-1)*dims-1, j+ind*dims-1);
            face_index++;
        }
    }

    if (sphere)
    {

        Eigen::MatrixXi F_temp = F.topLeftCorner(face_index, 4);
        int n = 0;
        for (int i = 0; i < dims; i++)
        {
            for (int j = 0; j < dims; j++)
            {
                if (Eigen::Vector3d(V.row(j+i*dims)).norm() > 1e-8)
                    n++;

            }
        }

        Eigen::MatrixXd V_temp(n, 3);
        std::map<int, int> idx_map;
        idx_map.insert(std::pair(-1, -1));
        int index = 0;
        for (int i = 0; i < dims; i++)
        {
            for (int j = 0; j < dims; j++)
            {
                if (Eigen::Vector3d(V.row(j+i*dims)).norm() > 1e-8)
                {
                    idx_map.insert(std::pair(j+i*dims, index));
                    V_temp.row(index) = V.row(j+i*dims);
                    index++;
                }
            }
        }
        V = V_temp;

        for (int i = 0; i < F_temp.rows(); i++)
        {
            for (auto j = 0; j < F_temp.cols(); j++)
            {
                F_temp(i,j) = idx_map[F_temp(i,j)];
            }
        }
        F = F_temp;
    }
}

inline void generate_random_array(int seed, int n, int m, Eigen::MatrixXd& rand_mat)
{
    srand(seed);
    rand_mat.resize(n, m);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            rand_mat(i,j) = 2 * static_cast<double>(rand()) / RAND_MAX - 1;
        }
    }
}
#ifndef __EMSCRIPTEN__
inline void generate_unstructured_grid(Eigen::MatrixXd& V, Eigen::MatrixXi& F, const int factor, const int seed=0)
{

    const int inner_points = 52*factor;
    const int boundary_points = 2*factor;
    Eigen::MatrixXd points;
    generate_random_array(seed, inner_points,2, points);
    Eigen::MatrixXd points_corner(4,2);
    points_corner << -1,-1,
                     -1,1,
                     1,-1,
                     1,1;

    std::random_device rd{};
    std::mt19937 gen{rd()};

    // Values near the mean are the most likely. Standard deviation
    // affects the dispersion of generated values from the mean.
        int irr_points = 111 * 0;
    auto distr = [&gen, irr_points](double mean)
    {
        std::normal_distribution d_x{0.0, 0.03};
        std::normal_distribution d_y{mean, 0.03};
        auto random_double_x = [&d_x, &gen]{ return d_x(gen); };
        auto random_double_y = [&d_y, &gen]{ return d_y(gen); };
        Eigen::MatrixXd high_dense = Eigen::MatrixXd::Zero(irr_points,2);
        for (int i = 0; i < irr_points; i++)
        {
            double x = std::clamp(random_double_x(), -0.99, 0.99);
            double y = std::clamp(random_double_y(), -0.99, 0.99)-mean;
            double norm = 1;
            high_dense(i,0) = std::clamp(x * norm, -0.99, 0.99);
            high_dense(i,1) = std::clamp(y * norm+mean, -0.99, 0.99);
        }
        return high_dense;
    };

    Eigen::MatrixXd high_dense = distr(0.0);


    Eigen::MatrixXd points_bxmin, points_bxmax, points_bymin, points_bymax;
    generate_random_array(seed+1, boundary_points,2, points_bxmin);
    points_bxmin.col(1) = -Eigen::VectorXd::Ones(boundary_points);
    generate_random_array(seed+2, boundary_points,2, points_bxmax);
    points_bxmax.col(1) = Eigen::VectorXd::Ones(boundary_points);
    generate_random_array(seed+3, boundary_points,2, points_bymin);
    points_bymin.col(0) = -Eigen::VectorXd::Ones(boundary_points);
    generate_random_array(seed+4, boundary_points,2, points_bymax);
    points_bymax.col(0) = Eigen::VectorXd::Ones(boundary_points);
    Eigen::MatrixXd con_points(inner_points+4*boundary_points+4+irr_points,2);
    con_points << points_corner,
                  points,
                  points_bxmin,
                  points_bxmax,
                  points_bymin,
                  points_bymax,
                  high_dense;
    V.resize(inner_points+4*boundary_points+4+irr_points, 3);
    V.col(0) = con_points.col(0);
    V.col(1) = con_points.col(1);
    V.col(2) = Eigen::VectorXd::Zero(inner_points+4*boundary_points+4+irr_points);
    igl::predicates::delaunay_triangulation(con_points, F);
}
#endif

inline bool is_collapse_legal(const pmp::SurfaceMesh& mesh, const pmp::Vertex v, const pmp::Point& pos_after)
{
    const pmp::Point p0 = mesh.position(v);

    for (const auto h : mesh.halfedges(v))
    {
        const pmp::Halfedge n_h = mesh.next_halfedge(h);
        pmp::Point p2 = mesh.position(mesh.from_vertex(n_h));
        pmp::Point p3 = mesh.position(mesh.to_vertex(n_h));

        pmp::Point before_normal = normalize(cross(p2-p0, p2-p3));
        pmp::Point after_normal = normalize(cross(p2-pos_after, p2-p3));
        if(dot(before_normal, after_normal) <= 0.5)
        {
            return false;
        }
    }

    return true;
}

inline void destroy_mesh(pmp::SurfaceMesh& mesh, const double el, const double degen_prob, const double needle_prob, const int seed = 0)
{
    srand(seed);
    for (const pmp::Halfedge h : mesh.halfedges())
    {
        if (mesh.is_boundary(mesh.from_vertex(h)))
            continue;
        if (static_cast<double>(rand()) / RAND_MAX < degen_prob)
        {
            const pmp::Vertex v1 = mesh.from_vertex(h);
            const pmp::Vertex v2 = mesh.to_vertex(h);

            pmp::Point p1 = mesh.position(v1);
            pmp::Point p2 = mesh.position(v2);

            if (static_cast<double>(rand()) / RAND_MAX < needle_prob)
            {

                pmp::Point e = p2-p1;

                pmp::Point pos_after = mesh.position(v1) + normalize(e) * (norm(e) - el);

                if (is_collapse_legal(mesh, v1, pos_after))
                    mesh.position(v1) = pos_after;

            }
            else
            {
                const pmp::Halfedge hn = mesh.next_halfedge(h);
                const pmp::Vertex v3 = mesh.to_vertex(hn);
                pmp::Point p3 = mesh.position(v3);

                pmp::Point e1 = p1-p2;
                pmp::Point e2 = p3-p2;
                pmp::Point e3 = dot(e1, e2) * e2 - e1;

                pmp::Point pos_after = mesh.position(v1) + normalize(e3) * (norm(e3) - el);

                if (is_collapse_legal(mesh, v1, pos_after))
                    mesh.position(v1) = pos_after;

            }
        }
    }
}