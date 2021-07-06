#include "tools.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth)
{
   VectorXd rmse(4);
   rmse << 0, 0, 0, 0;

   // the estimation vector size should not be zero
   if (estimations.size() == 0)
   {
      std::cout << "Estimation vector has size zero" << std::endl;
      return rmse;
   }
   // the estimation vector size should equal ground truth vector size
   if (estimations.size() != ground_truth.size())
   {
      std::cout << "Estimation vector has different size than the ground truth vector" << std::endl;
      return rmse;
   }

   VectorXd tmp(4);
   // accumulate squared residuals
   for (int i = 0; i < estimations.size(); ++i)
   {
      tmp = estimations[i] - ground_truth[i];
      tmp = tmp.array() * tmp.array();
      rmse = rmse + tmp;
   }
   // calculate the mean
   rmse = rmse / estimations.size();
   // calculate the squared root
   rmse = rmse.array().sqrt();
   // return the result
   return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state)
{
   MatrixXd Hj(3, 4);
   // recover state parameters
   float px = x_state(0);
   float py = x_state(1);
   float vx = x_state(2);
   float vy = x_state(3);

   float px2_py2 = px * px + py * py;
   float px2_py2_sqrt = sqrt(px2_py2);

   // check division by zero
   if (fabs(px2_py2) < 0.0001)
   {
      std::cout << "Error calculating the Jacobian" << std::endl;
      return Hj;
   }

   // compute the Jacobian matrix
   Hj << px / px2_py2_sqrt, py / px2_py2_sqrt, 0, 0,
       -py / px2_py2, px / px2_py2, 0, 0,
       py * (vx * py - vy * px) / pow(px2_py2, 1.5), px * (vy * px - vx * py) / pow(px2_py2, 1.5), px / px2_py2_sqrt, py / px2_py2_sqrt;
   return Hj;
}
