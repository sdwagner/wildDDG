#include "cgal_laplacian.h"
#include <CGAL/Weights.h>

template<typename PointMap>
FT get_w_ij(const CGALMesh& mesh, const HD he, const PointMap pmap) {

  const VD v0 = target(he, mesh);
  const VD v1 = source(he, mesh);

  const auto& q  = get(pmap, v0); // query
  const auto& p1 = get(pmap, v1); // neighbor j

  if (is_border_edge(he, mesh)) {
    const HD he_cw = opposite(next(he, mesh), mesh);
    VD v2 = source(he_cw, mesh);

    if (is_border_edge(he_cw, mesh)) {
      const HD he_ccw = prev(opposite(he, mesh), mesh);
      v2 = source(he_ccw, mesh);

      const auto& p2 = get(pmap, v2); // neighbor jp
      return CGAL::Weights::cotangent(p1, p2, q);
    } else {
      const auto& p0 = get(pmap, v2); // neighbor jm
      return CGAL::Weights::cotangent(q, p0, p1);
    }
  }

  const HD he_cw = opposite(next(he, mesh), mesh);
  const VD v2 = source(he_cw, mesh);
  const HD he_ccw = prev(opposite(he, mesh), mesh);
  const VD v3 = source(he_ccw, mesh);

  const auto& p0 = get(pmap, v2); // neighbor jm
  const auto& p2 = get(pmap, v3); // neighbor jp
  return CGAL::Weights::cotangent_weight(p0, p1, p2, q) / 2.0;
}

void cgal_laplacian(const CGALMesh& mesh, Eigen::SparseMatrix<double>& L) {

  const auto pmap = get(CGAL::vertex_point, mesh); // vertex to point map
  const auto imap = get(CGAL::vertex_index, mesh); // vertex to index map

  std::vector<Eigen::Triplet<double>> L_tuple;
  // Fill the matrix.
  for (const HD he : halfedges(mesh)) {
    const VD vi = source(he, mesh);
    const VD vj = target(he, mesh);

    const std::size_t i = get(imap, vi);
    const std::size_t j = get(imap, vj);
    if (i > j) { continue; }

    const double w_ij = to_double(get_w_ij(mesh, he, pmap) / 2.0);
    L_tuple.emplace_back(i, j,  w_ij);
    L_tuple.emplace_back(j, i,  w_ij);
    L_tuple.emplace_back(i, i, -w_ij);
    L_tuple.emplace_back(j, j, -w_ij);
  }
  L.setFromTriplets(L_tuple.begin(), L_tuple.end());
}
