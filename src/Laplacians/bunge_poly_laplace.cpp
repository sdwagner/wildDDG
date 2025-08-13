//=============================================================================
// Copyright 2023 Astrid Bunge, Mario Botsch.
// Distributed under MIT license, see file LICENSE for details.
//=============================================================================
#include "bunge_poly_laplace.h"
#include "optimized_laplace.h"

typedef Eigen::MatrixXd DenseMatrix;
typedef Eigen::SparseMatrix<double> SparseMatrix;

void compute_virtual_vertex(const DenseMatrix& poly, Eigen::VectorXd& weights)
{
    const int n = poly.rows();

    // setup array of positions and edges
    std::vector<Eigen::Vector3d> x(n), d(n);
    for (int i = 0; i < n; ++i)
        x[i] = poly.row(i);
    for (int i = 0; i < n; ++i)
        d[i] = x[(i + 1) % n] - x[i];

    // setup matrix A and rhs b
    // see Equation (38) of "Polygon Laplacian made simple", Eurographics 2020
    DenseMatrix A(n + 1, n);
    Eigen::VectorXd b(n + 1);
    for (int j = 0; j < n; ++j)
    {
        for (int i = j; i < n; ++i)
        {
            double Aij(0.0), bi(0.0);
            for (int k = 0; k < n; ++k)
            {
                Aij += (x[j].cross(d[k])).dot(x[i].cross(d[k]));
                bi += (x[i].cross(d[k])).dot(x[k].cross(d[k]));
            }
            A(i, j) = A(j, i) = Aij;
            b(i) = bi;
        }
    }
    for (int j = 0; j < n; ++j)
    {
        A(n, j) = 1.0;
    }
    b(n) = 1.0;

    weights = A.completeOrthogonalDecomposition().solve(b).topRows(n);
}


void polygon_laplace_matrix(const DenseMatrix& polygon, DenseMatrix& Lpoly, LaplaceConfig config)
{
    const int n = (int)polygon.rows();
    Lpoly = DenseMatrix::Zero(n, n);

    // shortcut for triangles
    if (n == 3)
    {
        triangle_laplace_matrix(polygon.row(0), polygon.row(1), polygon.row(2),
                                Lpoly, config);
        return;
    }

    // compute position of virtual vertex
    Eigen::VectorXd vweights;
    compute_virtual_vertex(polygon, vweights);
    //find_trace_minimizer_weights(polygon, vweights, false);
    Eigen::Vector3d vvertex = polygon.transpose() * vweights;

    // laplace matrix of refined triangle fan
    DenseMatrix Lfan = DenseMatrix::Zero(n + 1, n + 1);
    DenseMatrix Ltri(3, 3);
    for (int i = 0; i < n; ++i)
    {
        const int j = (i + 1) % n;

        // build laplace matrix of one triangle
        triangle_laplace_matrix(polygon.row(i), polygon.row(j), vvertex, Ltri, config);

        // assemble to laplace matrix for refined triangle fan
        Lfan(i, i) += Ltri(0, 0);
        Lfan(i, j) += Ltri(0, 1);
        Lfan(i, n) += Ltri(0, 2);
        Lfan(j, i) += Ltri(1, 0);
        Lfan(j, j) += Ltri(1, 1);
        Lfan(j, n) += Ltri(1, 2);
        Lfan(n, i) += Ltri(2, 0);
        Lfan(n, j) += Ltri(2, 1);
        Lfan(n, n) += Ltri(2, 2);
    }

    // build prolongation matrix
    DenseMatrix P(n + 1, n);
    P.setIdentity();
    P.row(n) = vweights;

    // build polygon laplace matrix by sandwiching
    Lpoly = P.transpose() * Lfan * P;
}

void poly_laplace_matrix(const pmp::SurfaceMesh& mesh, SparseMatrix& L, LaplaceConfig config)
{
    const int nv = mesh.n_vertices();
    const int nf = mesh.n_faces();
    std::vector<pmp::Vertex> vertices; // polygon vertices
    DenseMatrix polygon;          // positions of polygon vertices
    DenseMatrix Lpoly;            // local laplace matrix

    std::vector<pmp::Triplet> triplets;
    triplets.reserve(9 * nf); // estimate for triangle meshes

    for (pmp::Face f : mesh.faces())
    {
        // collect polygon vertices
        vertices.clear();
        for (pmp::Vertex v : mesh.vertices(f))
        {
            vertices.push_back(v);
        }
        const int n = vertices.size();

        // collect their positions
        polygon.resize(n, 3);
        for (int i = 0; i < n; ++i)
        {
            polygon.row(i) = (Eigen::Vector3d)mesh.position(vertices[i]);
        }

        // setup local laplace matrix
        polygon_laplace_matrix(polygon, Lpoly, config);

        // assemble to global laplace matrix
        for (int j = 0; j < n; ++j)
        {
            for (int k = 0; k < n; ++k)
            {
                triplets.emplace_back(vertices[k].idx(), vertices[j].idx(),
                                      -Lpoly(k, j));
            }
        }
    }

    // build sparse matrix from triplets
    L.resize(nv, nv);
    L.setFromTriplets(triplets.begin(), triplets.end());

    // clamp negative off-diagonal entries to zero
    if (config.clamp_negative)
    {
        for (unsigned int k = 0; k < L.outerSize(); k++)
        {
            double diag_offset(0.0);

            for (SparseMatrix::InnerIterator iter(L, k); iter; ++iter)
            {
                if (iter.row() != iter.col() && iter.value() < 0.0)
                {
                    diag_offset += -iter.value();
                    iter.valueRef() = 0.0;
                }
            }
            for (SparseMatrix::InnerIterator iter(L, k); iter; ++iter)
            {
                if (iter.row() == iter.col() && iter.value() < 0.0)
                    iter.valueRef() -= diag_offset;
            }
        }
    }
}

void polygon_mass_matrix(const DenseMatrix& polygon, pmp::DiagonalMatrix& Mpoly, LaplaceConfig config)
{
    const int n = (int)polygon.rows();

    // shortcut for triangles
    if (n == 3)
    {
        const double area = double_triarea(polygon.row(0), polygon.row(1), polygon.row(2), config, area_computation);
        Eigen::Vector3d m_diag = area / 6.0 * Eigen::Vector3d::Ones();
        Mpoly = m_diag.asDiagonal();
        return;
    }

    // compute position of virtual vertex
    Eigen::VectorXd vweights;
    compute_virtual_vertex(polygon, vweights);
    Eigen::Vector3d vvertex = polygon.transpose() * vweights;

    // laplace matrix of refined triangle fan
    DenseMatrix Mfan = DenseMatrix::Zero(n + 1, n + 1);
    pmp::DiagonalMatrix Mtri;
    for (int i = 0; i < n; ++i)
    {
        const int j = (i + 1) % n;

        const double area = double_triarea(polygon.row(i), polygon.row(j), vvertex, config, area_computation);
        Eigen::Vector3d m_diag = area / 6.0 * Eigen::Vector3d::Ones();
        Mtri = m_diag.asDiagonal();

        // assemble to laplace matrix for refined triangle fan
        // (we are dealing with diagonal matrices)
        Mfan.diagonal()[i] += Mtri.diagonal()[0];
        Mfan.diagonal()[j] += Mtri.diagonal()[1];
        Mfan.diagonal()[n] += Mtri.diagonal()[2];
    }

    // build prolongation matrix
    DenseMatrix P(n + 1, n);
    P.setIdentity();
    P.row(n) = vweights;

    // build polygon laplace matrix by sandwiching
    DenseMatrix PMP = P.transpose() * Mfan * P;
    Mpoly = PMP.rowwise().sum().asDiagonal();
}

void poly_mass_matrix(const pmp::SurfaceMesh& mesh, pmp::DiagonalMatrix& M, LaplaceConfig config)
{
    const int nv = mesh.n_vertices();
    std::vector<pmp::Vertex> vertices; // polygon vertices
    DenseMatrix polygon;          // positions of polygon vertices
    pmp::DiagonalMatrix Mpoly;         // local mass matrix

    M.setZero(nv);

    for (pmp::Face f : mesh.faces())
    {
        // collect polygon vertices
        vertices.clear();
        for (pmp::Vertex v : mesh.vertices(f))
        {
            vertices.push_back(v);
        }
        const int n = vertices.size();

        // collect their positions
        polygon.resize(n, 3);
        for (int i = 0; i < n; ++i)
        {
            polygon.row(i) = (Eigen::Vector3d)mesh.position(vertices[i]);
        }

        // setup local mass matrix
        polygon_mass_matrix(polygon, Mpoly, config);

        // assemble to global mass matrix
        for (int k = 0; k < n; ++k)
        {
            M.diagonal()[vertices[k].idx()] += Mpoly.diagonal()[k];
        }
    }
}


void polygon_gradient_matrix(const DenseMatrix& polygon, DenseMatrix& Gpoly, LaplaceConfig config)
{
    const int n = (int)polygon.rows();

    // compute position of virtual vertex
    Eigen::VectorXd vweights;
    compute_virtual_vertex(polygon, vweights);
    Eigen::Vector3d vvertex = polygon.transpose() * vweights;

    DenseMatrix Gfan = DenseMatrix::Zero(3 * n, n + 1);
    DenseMatrix Gtri(3, 3);
    int row = 0;
    for (int i = 0; i < n; ++i)
    {
        const int j = (i + 1) % n;

        // build laplace matrix of one triangle
        triangle_gradient_matrix(polygon.row(i), polygon.row(j), vvertex, Gtri, config);

        // assemble to matrix for triangle fan
        for (int k = 0; k < 3; ++k)
        {
            Gfan(row + k, i) += Gtri(k, 0);
            Gfan(row + k, j) += Gtri(k, 1);
            Gfan(row + k, n) += Gtri(k, 2);
        }

        row += 3;
    }

    // build prolongation matrix
    DenseMatrix P(n + 1, n);
    P.setIdentity();
    P.row(n) = vweights;

    // build polygon gradient matrix by sandwiching (from left only)
    Gpoly = Gfan * P;
}

void poly_divmass_matrix(const pmp::SurfaceMesh& mesh, pmp::DiagonalMatrix& M, LaplaceConfig config)
{
    // how many virtual triangles will we have after refinement?
    unsigned int nt = 0;
    for (auto f : mesh.faces())
    {
        nt += mesh.valence(f);
    }

    // initialize global matrix
    M.resize(3 * nt);
    auto& diag = M.diagonal();

    std::vector<pmp::Vertex> vertices; // polygon vertices
    DenseMatrix polygon;          // positions of polygon vertices

    unsigned int idx = 0;

    for (pmp::Face f : mesh.faces())
    {
        // collect polygon vertices
        vertices.clear();
        for (pmp::Vertex v : mesh.vertices(f))
        {
            vertices.push_back(v);
        }
        const int n = vertices.size();

        // collect their positions
        polygon.resize(n, 3);
        for (int i = 0; i < n; ++i)
        {
            polygon.row(i) = (Eigen::Vector3d)mesh.position(vertices[i]);
        }

        // compute position of virtual vertex
        Eigen::VectorXd vweights;
        compute_virtual_vertex(polygon, vweights);
        Eigen::Vector3d vvertex = polygon.transpose() * vweights;

        for (int i = 0; i < n; ++i)
        {
            const double area = 0.5 *
                double_triarea(polygon.row(i), polygon.row((i + 1) % n), vvertex, config, CrossProduct);

            diag[idx++] = area;
            diag[idx++] = area;
            diag[idx++] = area;
        }
    }

    assert(idx == 3 * nt);
}

void poly_gradient_matrix(const pmp::SurfaceMesh& mesh, SparseMatrix& G, LaplaceConfig config)
{
    const int nv = mesh.n_vertices();

    // how many virtual triangles will we have after refinement?
    // triangles are not refined, other polygons are.
    unsigned int nt = 0;
    for (auto f : mesh.faces())
    {
        nt += mesh.valence(f);
    }

    std::vector<pmp::Vertex> vertices; // polygon vertices
    DenseMatrix polygon;          // positions of polygon vertices
    DenseMatrix Gpoly;            // local gradient matrix

    std::vector<pmp::Triplet> triplets;
    triplets.reserve(9 * nt);

    unsigned int n_rows = 0;

    for (pmp::Face f : mesh.faces())
    {
        // collect polygon vertices
        vertices.clear();
        for (pmp::Vertex v : mesh.vertices(f))
        {
            vertices.push_back(v);
        }
        const int n = vertices.size();

        // collect their positions
        polygon.resize(n, 3);
        for (int i = 0; i < n; ++i)
        {
            polygon.row(i) = (Eigen::Vector3d)mesh.position(vertices[i]);
        }

        // setup local element matrix
        polygon_gradient_matrix(polygon, Gpoly, config);

        // assemble to global matrix
        for (int j = 0; j < Gpoly.cols(); ++j)
        {
            for (int i = 0; i < Gpoly.rows(); ++i)
            {
                triplets.emplace_back(n_rows + i, vertices[j].idx(),
                                      Gpoly(i, j));
            }
        }

        n_rows += Gpoly.rows();
    }
    assert(n_rows == 3 * nt);

    // build sparse matrix from triplets
    G.resize(n_rows, nv);
    G.setFromTriplets(triplets.begin(), triplets.end());
}

void poly_divergence_matrix(const pmp::SurfaceMesh& mesh, SparseMatrix& D, LaplaceConfig config)
{
    SparseMatrix G;
    poly_gradient_matrix(mesh, G, config);
    pmp::DiagonalMatrix M;
    poly_divmass_matrix(mesh, M, config);
    D = -G.transpose() * M;
}

void poly_divergence_and_gradient_matrix(const pmp::SurfaceMesh& mesh, pmp::SparseMatrix& D, pmp::SparseMatrix& G, LaplaceConfig config)
{
    poly_gradient_matrix(mesh, G, config);
    pmp::DiagonalMatrix M;
    poly_divmass_matrix(mesh, M, config);
    D = -G.transpose() * M;
}