#pragma once
#include "vcg/complex/algorithms/mesh_to_matrix.h"

class VCGVertex; class VCGEdge; class VCGFace;
struct VCGUsedTypes : public vcg::UsedTypes<vcg::Use<VCGVertex>::AsVertexType,
                                           vcg::Use<VCGEdge>::AsEdgeType,
                                           vcg::Use<VCGFace>::AsFaceType>{};

class VCGVertex  : public vcg::Vertex<VCGUsedTypes, vcg::vertex::Coord3d, vcg::vertex::Normal3d, vcg::vertex::BitFlags> {};
class VCGFace    : public vcg::Face<VCGUsedTypes, vcg::face::FFAdj, vcg::face::VertexRef, vcg::face::BitFlags> {};
class VCGEdge    : public vcg::Edge<VCGUsedTypes> {};

class VCGMesh    : public vcg::tri::TriMesh<std::vector<VCGVertex>, std::vector<VCGFace>, std::vector<VCGEdge>> {};

inline void vcglib_laplacian(VCGMesh& mesh, Eigen::SparseMatrix<double>& L)
{
    std::vector<std::pair<int, int>> indices;
    std::vector<double> entries;
    std::vector<Eigen::Triplet<double>> triplets;
    vcg::tri::MeshToMatrix<VCGMesh>::GetLaplacianMatrix(mesh, indices, entries, true, 1, false);

    L.resize(mesh.VN(), mesh.VN());

    for (int i=0; i<indices.size(); i++)
    {
        triplets.emplace_back(indices[i].first, indices[i].second, entries[i]);
    }
    L.setFromTriplets(triplets.begin(), triplets.end());
    L = -L;
}