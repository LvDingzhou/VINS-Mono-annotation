#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "../utility/utility.h"

class PoseLocalParameterization : public ceres::LocalParameterization
{
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    virtual int GlobalSize() const { return 7; };//4个旋转+3个平移
    virtual int LocalSize() const { return 6; };//实际的自由度
};
