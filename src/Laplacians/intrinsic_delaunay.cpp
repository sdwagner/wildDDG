// =============================================================================
//  Copyright (c) 2025 Sven D. Wagner, Mario Botsch.
//  Distributed under MIT license, see file LICENSE for details.
// =============================================================================

#include "intrinsic_delaunay.h"

#include <deque>
#include <geometrycentral/utilities/elementary_geometry.ipp>

#include "geometrycentral/surface/intrinsic_mollification.h"
#include "geometrycentral/surface/edge_length_geometry.h"

// See include/geometry-central/src/surface/intrinsic_triangulation.cpp for original definition and license
size_t flipToDelaunay(geometrycentral::surface::SurfaceMesh& mesh, geometrycentral::surface::EdgeData<double>& edgeLengths, FlipType flipType, double delaunayEPS) {

  using namespace geometrycentral::surface;
  using namespace geometrycentral;
  // TODO all of these helpers are duplicated from signpost_intrinsic_triangulation

  auto flippedEdgeLen = [&](Halfedge iHe) {
    // Gather index values
    Halfedge iHeA0 = iHe;
    Halfedge iHeA1 = iHeA0.next();
    Halfedge iHeA2 = iHeA1.next();
    Halfedge iHeB0 = iHe.twin();
    Halfedge iHeB1 = iHeB0.next();
    Halfedge iHeB2 = iHeB1.next();

    // Handle non-oriented edges
    if (iHeA0.orientation() == iHeB0.orientation()) {
      std::swap(iHeB1, iHeB2);
    }

    // Gather length values
    double l01 = edgeLengths[iHeA1.edge()];
    double l12 = edgeLengths[iHeA2.edge()];
    double l23 = edgeLengths[iHeB1.edge()];
    double l30 = edgeLengths[iHeB2.edge()];
    double l02 = edgeLengths[iHeA0.edge()];

    switch (flipType) {
    case FlipType::Hyperbolic: {
      double newLen = (l01 * l23 + l12 * l30) / l02;
      return newLen;
      break;
    }
    case FlipType::Euclidean: {
      Vector2 p3{0., 0.};
      Vector2 p0{l30, 0.};
      Vector2 p2 = layoutTriangleVertex(p3, p0, l02, l23); // involves more arithmetic than strictly necessary
      Vector2 p1 = layoutTriangleVertex(p2, p0, l01, l12);
      double newLen = (p1 - p3).norm();
      return newLen;
      break;
    }
    }
    return -1.; // unreachable
  };

  auto area = [&](Face f) {
    Halfedge he = f.halfedge();
    double a = edgeLengths[he.edge()];
    he = he.next();
    double b = edgeLengths[he.edge()];
    he = he.next();
    double c = edgeLengths[he.edge()];
    return triangleArea(a, b, c);
  };

  auto halfedgeCotanWeight = [&](Halfedge heI) {
    if (heI.isInterior()) {
      Halfedge he = heI;
      double l_ij = edgeLengths[he.edge()];
      he = he.next();
      double l_jk = edgeLengths[he.edge()];
      he = he.next();
      double l_ki = edgeLengths[he.edge()];
      he = he.next();
      GC_SAFETY_ASSERT(he == heI, "faces must be triangular");
      double areaV = area(he.face());
      double cotValue = (-l_ij * l_ij + l_jk * l_jk + l_ki * l_ki) / (4. * areaV);
      return cotValue / 2;
    } else {
      return 0.;
    }
  };

  auto edgeCotanWeight = [&](Edge e) {
    return halfedgeCotanWeight(e.halfedge()) + halfedgeCotanWeight(e.halfedge().twin());
  };


  auto shouldFlipEdge = [&](Edge e) {
    if (e.isBoundary()) return false;

    switch (flipType) {
    case FlipType::Hyperbolic: {

      double score = 0.;
      for (Halfedge he : {e.halfedge(), e.halfedge().twin()}) {

        double lA = edgeLengths[he.edge()];
        double lB = edgeLengths[he.next().edge()];
        double lC = edgeLengths[he.next().next().edge()];

        score -= lA / (lB * lC);
        score += lB / (lC * lA);
        score += lC / (lA * lB);
      }

      return score < -delaunayEPS;
      break;
    }
    case FlipType::Euclidean: {
      double cWeight = edgeCotanWeight(e);
      return (cWeight < -delaunayEPS);
      break;
    }
    }
    return false; // unreachable
  };

  auto flipEdgeIfNotDelaunay = [&](Edge e) {
    // Can't flip
    //if (e.isBoundary()) return false;
    //if (e.isBoundary()) throw std::runtime_error("boundary");
    if (!e.isManifold()) throw std::runtime_error("nonmanifold");

    // Don't want to flip
    if (!shouldFlipEdge(e)) return false;

    // Get geometric data
    Halfedge he = e.halfedge();
    double newLength = flippedEdgeLen(he);

    // If we're going to create a non-finite edge length, abort the flip
    // (only happens if you're in a bad numerical place)
    if (!std::isfinite(newLength)) {
      return false;
    }

    // Combinatorial flip
    bool flipped = mesh.flip(e, false);

    // Should always be possible, something unusual is going on if we end up here
    if (!flipped) {
      return false;
    }

    // Assign the new edge lengths
    edgeLengths[e] = newLength;

    return true;
  };

  std::deque<Edge> edgesToCheck;
  EdgeData<char> inQueue(mesh, true);
  for (Edge e : mesh.edges()) {
    edgesToCheck.push_back(e);
  }

  size_t nFlips = 0;
  while (!edgesToCheck.empty()) {

    // Get the top element from the queue of possibily non-Delaunay edges
    Edge e = edgesToCheck.front();
    edgesToCheck.pop_front();
    inQueue[e] = false;

    bool wasFlipped = flipEdgeIfNotDelaunay(e);

    if (!wasFlipped) continue;

    // Handle the aftermath of a flip
    nFlips++;

    // Add neighbors to queue, as they may need flipping now
    Halfedge he = e.halfedge();
    Halfedge heN = he.next();
    Halfedge heT = he.twin();
    Halfedge heTN = heT.next();
    std::vector<Edge> neighEdges = {heN.edge(), heN.next().edge(), heTN.edge(), heTN.next().edge()};
    for (Edge nE : neighEdges) {
      if (!inQueue[nE]) {
        edgesToCheck.push_back(nE);
        inQueue[nE] = true;
      }
    }
  }

  // std::cout << "nFlips = " << nFlips << std::endl;
  return nFlips;
}


std::tuple<Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>>
idt_laplacian(const geometrycentral::surface::SurfaceMesh& mesh,
                                geometrycentral::surface::EmbeddedGeometryInterface& geom,
                                double relativeMollificationFactor, bool buildDelaunay)
{
    // Create a copy of the mesh / geometry to operate on
    std::unique_ptr<geometrycentral::surface::SurfaceMesh> idt_mesh = mesh.copyToSurfaceMesh();
    geom.requireVertexPositions();
    geometrycentral::surface::VertexPositionGeometry idt_geom(*idt_mesh,
                                                                geom.vertexPositions.reinterpretTo(*idt_mesh));
    idt_geom.requireEdgeLengths();
    geometrycentral::surface::EdgeData<double> tuftedEdgeLengths = idt_geom.edgeLengths;

    // Mollify, if requested
    if (relativeMollificationFactor > 0)
    {
        mollifyIntrinsic(*idt_mesh, tuftedEdgeLengths, relativeMollificationFactor);
    }

    if (buildDelaunay)
    {
      // Flip to delaunay
      flipToDelaunay(*idt_mesh, tuftedEdgeLengths);
    }

    // Build the matrices
    geometrycentral::surface::EdgeLengthGeometry tuftedIntrinsicGeom(*idt_mesh, tuftedEdgeLengths);
    tuftedIntrinsicGeom.requireCotanLaplacian();
    tuftedIntrinsicGeom.requireVertexLumpedMassMatrix();

    return std::make_tuple(tuftedIntrinsicGeom.cotanLaplacian, tuftedIntrinsicGeom.vertexLumpedMassMatrix);
}
