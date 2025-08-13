#pragma once

#include <pmp/mat_vec.h>
#include <vector>

using namespace pmp;

//=============================================================================

/** Simple class for a 3D regular grid of float values */
class Grid
{
public:

    /** construct grid with origin and three axes of bounding box, as well as
        with the grid resolution in x, y, z direction */
    Grid(const dvec3&  _origin = dvec3(0,0,0),
         const dvec3&  _x_axis = dvec3(1,0,0),
         const dvec3&  _y_axis = dvec3(0,1,0),
         const dvec3&  _z_axis = dvec3(0,0,1),
         unsigned int  _x_res = 10,
         unsigned int  _y_res = 10,
         unsigned int  _z_res = 10);


    /// return grid's origin
    const dvec3& origin() const { return origin_; }
    /// return grid's x-axis
    const dvec3& x_axis() const { return x_axis_; }
    /// return grid's y-axis
    const dvec3& y_axis() const { return y_axis_; }
    /// return grid's z-axis
    const dvec3& z_axis() const { return z_axis_; }

    /// return grid's x-resolution
    unsigned int x_resolution() const { return x_res_; }
    /// return grid's y-resolution
    unsigned int y_resolution() const { return y_res_; }
    /// return grid's z-resolution
    unsigned int z_resolution() const { return z_res_; }


    /// return position of grid point at index (x,y,z)
    dvec3 point(unsigned int x, unsigned int y, unsigned int z) const {
        return origin_ + dx_*x + dy_*y + dz_*z;
    }
    /// return position of grid point at index xyz
    dvec3 point(const ivec3& xyz) const {
        return origin_ + dx_*xyz[0] + dy_*xyz[1] + dz_*xyz[2];
    }


    /// return reference to scalar value at grid position/index (x,y,z)
    double& operator()(unsigned int x, unsigned int y, unsigned int z) {
        return values_[z + y*z_res_ + x*z_res_*y_res_];
    }
    /// return scalar value at grid position/index (x,y,z)
    double operator()(unsigned int x, unsigned int y, unsigned int z) const {
        return values_[z + y*z_res_ + x*z_res_*y_res_];
    }
    /// return reference to scalar value at grid position/index xyz
    double& operator()(const ivec3& xyz) {
        return values_[xyz[2] + xyz[1]*z_res_ + xyz[0]*z_res_*y_res_];
    }
    /// return scalar value at grid position/index xyz
    double operator()(const ivec3& xyz) const {
        return values_[xyz[2] + xyz[1]*z_res_ + xyz[0]*z_res_*y_res_];
    }


private:

    dvec3               origin_, x_axis_, y_axis_, z_axis_, dx_, dy_, dz_;
    unsigned int        x_res_, y_res_, z_res_;
    std::vector<double>  values_;
};

//=============================================================================
