#include "deformation_solver.h"
#include "pmp/algorithms/laplace.h"

void solve_deformation(pmp::SurfaceMesh& mesh, const std::function<bool(int)>& is_fixed, const std::function<bool(int)>& is_handle, const Eigen::Matrix4d& transform, LaplaceConfig config)
{
    Eigen::SparseMatrix<double> L, D;
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> M;
    tri_divergence_matrix(mesh, D, config);
    tri_laplace_matrix(mesh, L, config);
    tri_mass_matrix(mesh, M, config);

    auto svd = transform.block(0,0,3,3).jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> Sigma = svd.singularValues().asDiagonal();
    Eigen::Matrix3d R = U * V.transpose();
    std::cout << "Rotation:\n" << R << std::endl;
    Eigen::Matrix3d S = V * Sigma * V.transpose();
    std::cout << "Shearing/Scaling:\n" << S << std::endl;


    Eigen::VectorXd s = 0.5 * Eigen::VectorXd::Ones(mesh.n_vertices());
    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(mesh.n_vertices(), 3);
    for (int i = 0; i < mesh.n_vertices(); ++i)
    {
        if (is_fixed(i))
        {
            s(i) = 0;
            X.row(i) =  (Eigen::Vector3d)mesh.position(pmp::Vertex(i));

        }
        else if (is_handle(i))
        {
            s(i) = 1;
            X.row(i) = affine_transform(transform, mesh.position(pmp::Vertex(i)));
        }
    }
    std::function is_constrained = [&](unsigned int i) { return is_fixed(i) || is_handle(i); };
    s = pmp::cholesky_solve(L, Eigen::VectorXd::Zero(mesh.n_vertices()), is_constrained, s);
    Eigen::Matrix3d rot = R.block(0,0,3,3);
    auto q = Eigen::Quaternion<double>(rot);
    Eigen::Quaternion<double> ident;
    ident.setIdentity();

    auto face_transform = mesh.add_face_property<Eigen::Matrix3d>("f:transform");
    for (pmp::Face f : mesh.faces())
    {

        double s_i = 0;
        for (auto v : mesh.vertices(f))
            s_i += s(v.idx());
        s_i /= 3;
        Eigen::Quaternion<double> r_i = q.slerp(1-s_i, ident);
        Eigen::Matrix3d T_i = r_i.toRotationMatrix() * (s_i * S + (1.0-s_i) * Eigen::Matrix3d::Identity());
        face_transform[f] = T_i.transpose();

    }
    Eigen::MatrixXd g_x;
    tri_gradient_matrix_rotated(mesh, g_x, config);

    Eigen::MatrixXd sol = pmp::cholesky_solve(L, D * g_x, is_constrained, X);
    pmp::matrix_to_coordinates(sol, mesh);

    mesh.remove_face_property(face_transform);

}

void tri_gradient_matrix_rotated(const pmp::SurfaceMesh& mesh, Eigen::MatrixXd& G, LaplaceConfig config)
{
    G.resize(3*mesh.n_faces(), 3);
    const int nt = mesh.n_faces();


    std::vector<pmp::Vertex> vertices; // polygon vertices
    pmp::DenseMatrix Gpoly;            // local gradient matrix

    std::vector<pmp::Triplet> triplets;
    triplets.reserve(9 * nt);

    unsigned int n_rows = 0;
    auto face_transform = mesh.get_face_property<Eigen::Matrix3d>("f:transform");

    for (pmp::Face f : mesh.faces())
    {
        // collect polygon vertices
        vertices.clear();
        for (pmp::Vertex v : mesh.vertices(f))
        {
            vertices.push_back(v);
        }
        assert(vertices.size() == 3);

        // collect their positions
        Eigen::Matrix3d p = Eigen::Matrix3d::Zero();
        p.row(0) = (Eigen::Vector3d)mesh.position(vertices[0]);
        p.row(1) = (Eigen::Vector3d)mesh.position(vertices[1]);
        p.row(2) = (Eigen::Vector3d)mesh.position(vertices[2]);

        // setup local element matrix
        triangle_gradient_matrix(p.row(0), p.row(1), p.row(2), Gpoly, config);
        Gpoly = Gpoly * p * face_transform[f];
        G.middleRows(n_rows, 3) = Gpoly;


        n_rows += Gpoly.rows();
    }
    assert(n_rows == 3 * nt);

}

Eigen::Vector3d affine_transform(const Eigen::Matrix4d& m, const Eigen::Vector3d& v)
{

    const double x = m(0, 0) * v[0] + m(0, 1) * v[1] + m(0, 2) * v[2] + m(0, 3);
    const double y = m(1, 0) * v[0] + m(1, 1) * v[1] + m(1, 2) * v[2] + m(1, 3);
    const double z = m(2, 0) * v[0] + m(2, 1) * v[1] + m(2, 2) * v[2] + m(2, 3);
    return {x, y, z};
}