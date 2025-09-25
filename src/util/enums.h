// =============================================================================
//  Copyright (c) 2025 Sven D. Wagner, Mario Botsch.
//  Distributed under MIT license, see file LICENSE for details.
// =============================================================================

#pragma once

#include <map>

enum Solve_Function
{
    Franke2d = 0,
    XpY = 1,
    X2pY2 = 2,
    Runge = 3,
    Spherical_Harmonics = 4,
    BiHarm_Franke = 5,
    ConditioningStiffness = 6,
    ConditioningGeneralized = 7,
    Spherical_Smoothing = 8,
    Planar_Geodesics = 9,
    Spherical_Geodesics = 10,
    MeanCurvaturePlane = 11,
    MeanCurvatureSphere = 12,
};

enum Test_Type
{
    Poisson = 0,
    Spherical_Poisson = 1,
    BiHarm_Poisson = 2,
    Condition_Number = 3,
    Smoothing = 4,
    Geodesics = 5,
    MeanCurvature = 6,
};

inline std::map<Solve_Function, Test_Type> type_map = {
    {Franke2d, Poisson}, {XpY, Poisson}, {X2pY2, Poisson},
    {Runge, Poisson}, {Spherical_Harmonics, Spherical_Poisson}, {BiHarm_Franke, BiHarm_Poisson},
    {ConditioningStiffness, Condition_Number}, {ConditioningGeneralized, Condition_Number},
    {Spherical_Smoothing, Smoothing}, {Spherical_Geodesics, Geodesics}, {Planar_Geodesics, Geodesics},
    {MeanCurvaturePlane, MeanCurvature}, {MeanCurvatureSphere, MeanCurvature}
};

enum Lib
{
    PMP = 0,
    Open_Mesh = 1,
    IGL = 2,
    Geometry_Central = 3,
    Cgal = 4,
    VCGLib = 5,
    Geogram = 6,
    CinoLib = 7,
    Intrinsic_Mollification = 8,
    Intrinsic_Delaunay = 9,
    Intrinsic_Delaunay_Mollification = 10,
    OptimizedLaplacian_Cross_Dot = 11,
    OptimizedLaplacian_Cross_l2Sq = 12,
    OptimizedLaplacian_Heron_Dot = 13,
    OptimizedLaplacian_Heron_l2Sq = 14,
    OptimizedLaplacian_Max_Dot = 15,
    OptimizedLaplacian_Max_l2Sq = 16,
    OptimizedLaplacian_Cross_Dot_Mass = 17,
    BungeLaplace_Cross_Dot = 18,
    BungeLaplace_Cross_l2Sq = 19,
    BungeLaplace_Heron_Dot = 20,
    BungeLaplace_Heron_l2Sq = 21,
    BungeLaplace_Max_Dot = 22,
    BungeLaplace_Max_l2Sq = 23,
    TFEM_Cross_Dot = 24,
};

enum AreaComputation
{
    CrossProduct = 0,
    Heron_Sorted = 1,
    Max_Area = 2,
};

enum TFEMMethod
{
    TFEMNormal = 0,
    TFEMStatic = 1,
    TFEMDynamic = 2,
    TFEMDynamicFailsafe = 3,
};

inline std::map<TFEMMethod, std::string> tfem_method_map = {
    {TFEMNormal, "Normal"},
    {TFEMStatic, "Static"},
    {TFEMDynamic, "Dynamic"},
    {TFEMDynamicFailsafe, "Dynamic-Failsafe"}
};

inline std::map<Lib, std::string> lib_map = {
    {PMP, "PMP"},
    {Open_Mesh, "OpenMesh"},
    {IGL, "IGL"},
    {VCGLib, "VCGLib"},
    {Geogram, "Geogram"},
    {CinoLib, "CinoLib"},
    {Geometry_Central, "Geometry Central"},
    {Cgal, "CGAL"},
    {Intrinsic_Mollification, "Intrinsic Mollification"},
    {Intrinsic_Delaunay, "Intrinsic Delaunay"},
    {OptimizedLaplacian_Cross_Dot, "Optimized Laplace Cross Dot"},
    {OptimizedLaplacian_Cross_Dot_Mass, "Optimized Laplace Cross Dot Mass"},
    {OptimizedLaplacian_Cross_l2Sq, "Optimized Laplace Cross l2Sq"},
    {OptimizedLaplacian_Heron_Dot, "Optimized Laplace Heron Dot"},
    {OptimizedLaplacian_Heron_l2Sq, "Optimized Laplace Heron l2Sq"},
    {OptimizedLaplacian_Max_Dot, "Optimized Laplace Max Dot"},
    {OptimizedLaplacian_Max_l2Sq, "Optimized Laplace Max l2Sq"},
    {BungeLaplace_Cross_Dot, "Bunge Laplace Cross Dot"},
    {BungeLaplace_Cross_l2Sq, "Bunge Laplace Cross Dot"},
    {BungeLaplace_Cross_l2Sq, "Bunge Laplace Cross l2Sq"},
    {BungeLaplace_Heron_Dot, "Bunge Laplace Heron Dot"},
    {BungeLaplace_Heron_l2Sq, "Bunge Laplace Heron l2Sq"},
    {BungeLaplace_Max_Dot, "Bunge Laplace Max Dot"},
    {BungeLaplace_Max_l2Sq, "Bunge Laplace Max l2Sq"},
    {Intrinsic_Delaunay_Mollification, "Intrinsic Delaunay Mollification"}
};

struct LaplaceConfig
{
    Lib lib = OptimizedLaplacian_Cross_Dot;
    TFEMMethod tfem_method = TFEMNormal;
    double tfem_static_constant = 1e-8;
    double tfem_dynamic_constant = 1e-3;
    bool clamp_negative = false;
    bool clamp_zero_area = true;
};