#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF()
{
    is_initialized_ = false;

    previous_timestamp_ = 0;

    /* initializing matrices */
    // Laser
    R_laser_ = MatrixXd(2, 2);
    //measurement covariance matrix - laser
    R_laser_ << 0.0225, 0,
        0, 0.0225;
    H_laser_ = MatrixXd(2, 4);
    H_laser_ << 1, 0, 0, 0, 0, 1, 0, 0;

    // Radar
    R_radar_ = MatrixXd(3, 3);
    //measurement covariance matrix - radar
    R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;
    Hj_ = MatrixXd(3, 4);

    // state covariance initial value
    float pos_uncertainty = 10;
    float vel_uncertainty = 1000;
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << pos_uncertainty, 0, 0, 0, 0, pos_uncertainty, 0, 0, 0, 0, vel_uncertainty, 0, 0, 0, 0, vel_uncertainty;

    // state transition matrix
    ekf_.F_ = MatrixXd(4, 4);
    // has to be updated with new time step values
    ekf_.F_ << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1;
}

FusionEKF::~FusionEKF() {}

VectorXd FusionEKF::radar_fn(const VectorXd &x_state)
{
    VectorXd h(3, 1);
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);
    h << sqrt(px * px + py * py), atan2(py, px), (px * vx + py * vy) / sqrt(px * px + py * py);
    return h;
}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack)
{
    // Initialization
    if (!is_initialized_)
    {
        // Initialize the state ekf_.x_ with the first measurement.
        ekf_.x_ = VectorXd(4);
        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR)
        {
            float rho = measurement_pack.raw_measurements_(0);
            float phi = measurement_pack.raw_measurements_(1);
            float rho_dot = measurement_pack.raw_measurements_(2);

            float px = rho * cos(phi);
            float py = rho * sin(phi);
            float vx = rho_dot * cos(phi);
            float vy = rho_dot * sin(phi);
            ekf_.x_ << px, py, vx, vy;
        }
        else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER)
        {
            float px = measurement_pack.raw_measurements_(0);
            float py = measurement_pack.raw_measurements_(1);
            float vx = 0;
            float vy = 0;
            ekf_.x_ << px, py, vx, vy;
        }

        // done initializing, no need to predict or update
        previous_timestamp_ = measurement_pack.timestamp_;
        is_initialized_ = true;
        return;
    }

    /* Prediction */
    // compute the time elapsed between the current and previous measurements, dt - expressed in seconds
    float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
    previous_timestamp_ = measurement_pack.timestamp_;

    // Update the state transition matrix F according to the new elapsed time.
    // Modify the F matrix so that the time is integrated
    ekf_.F_(0, 2) = dt;
    ekf_.F_(1, 3) = dt;

    // Update the process noise covariance matrix.
    // Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
    float dt_2 = dt * dt;
    float dt_3 = dt_2 * dt;
    float dt_4 = dt_3 * dt;
    float noise_ax = 9;
    float noise_ay = 9;
    ekf_.Q_ = MatrixXd(4, 4);
    ekf_.Q_ << dt_4 / 4 * noise_ax, 0, dt_3 / 2 * noise_ax, 0,
        0, dt_4 / 4 * noise_ay, 0, dt_3 / 2 * noise_ay,
        dt_3 / 2 * noise_ax, 0, dt_2 * noise_ax, 0,
        0, dt_3 / 2 * noise_ay, 0, dt_2 * noise_ay;

    ekf_.Predict();

    /* Update */
    // Update the state and covariance matrices.
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR)
    {
        ekf_.H_ = tools.CalculateJacobian(ekf_.x_);

        ekf_.R_ = R_radar_;

        VectorXd z_pred = radar_fn(ekf_.x_);

        VectorXd y = measurement_pack.raw_measurements_ - z_pred; // passed as z to UpdateEKF

        // normalize y to (-pi, pi)
        float pi = 3.141592;
        while (y(1) < pi)
        {
            y(1) += 2 * pi;
            cout << "1";
        }
        while (y(1) > pi)
        {
            y(1) -= 2 * pi;
            cout << "1";
        }

        ekf_.UpdateEKF(y);
    }
    else
    {
        ekf_.H_ = H_laser_;
        ekf_.R_ = R_laser_;
        ekf_.Update(measurement_pack.raw_measurements_);
    }

    // print the output
    cout << "x_ = " << ekf_.x_ << endl;
    cout << "P_ = " << ekf_.P_ << endl;
}
