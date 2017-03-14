#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);

  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
        0, 1, 0, 0;

  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  //create a 4D state vector, we don't know yet the values of the x state
	ekf_.x_ = VectorXd(4);

	//state covariance matrix P
	ekf_.P_ = MatrixXd(4, 4);
	ekf_.P_ << 1, 0, 0, 0,
			  0, 1, 0, 0,
			  0, 0, 1000, 0,
			  0, 0, 0, 1000;

	//the initial transition matrix F_
	ekf_.F_ = MatrixXd(4, 4);
	ekf_.F_ << 1, 0, 1, 0,
			  0, 1, 0, 1,
			  0, 0, 1, 0,
			  0, 0, 0, 1;

  noise_ax_ = 9;
  noise_ay_ = 9;

  //set the process covariance matrix Q
  ekf_.Q_ = MatrixXd(4, 4);


}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_ ) {
    //set the state with the initial location and zero velocity

    // first measurement
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float p = measurement_pack.raw_measurements_[0];
      float theta = measurement_pack.raw_measurements_[1];
      float dp = measurement_pack.raw_measurements_[2] ;

      float x = cos(theta) * p;
      float y = sin(theta) * p;
      float vx = cos(theta) * dp;
      float vy = sin(theta) * dp;

      ekf_.x_ << x,y, vx,vy;

    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }

    previous_timestamp_ = measurement_pack.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

   //compute the time elapsed between the current and previous measurements
 	float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds

  if(dt > 60){
    cout << "dt too large:" << dt <<" resetting filter" << endl;
    is_initialized_ = false;
    return;
  }

  previous_timestamp_ = measurement_pack.timestamp_;

 	float dt_2 = dt * dt;
 	float dt_3 = dt_2 * dt;
 	float dt_4 = dt_3 * dt;

 	//Modify the F matrix so that the time is integrated
 	ekf_.F_(0, 2) = dt;
 	ekf_.F_(1, 3) = dt;

  ekf_.Q_ <<  dt_4/4*noise_ax_, 0, dt_3/2*noise_ax_, 0,
         0, dt_4/4*noise_ay_, 0, dt_3/2*noise_ay_,
         dt_3/2*noise_ax_, 0, dt_2*noise_ax_, 0,
         0, dt_3/2*noise_ay_, 0, dt_2*noise_ay_;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    try{
      ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
      ekf_.R_ = R_radar_;

      /* Not implemented here since the data is gaussian

      // Residual checks on the solution can be used to validate inputs against
      // outliers. If the Residual(r) value below is large the model and the measurements
      // are not in sync. Resetting the filter should correct it
      MatrixXd rms_delta = measurement_pack.raw_measurements_ - ekf_.H_*ekf_.x_;
      float r = (rms_delta.transpose()*rms_delta)(0);
      if(r >= 3){ // parameter should be based on noise parameters
        // reset filter
        is_initialized_ = false;
        cout << "Resetting filter:" << endl;
      }*/

      ekf_.UpdateEKF(measurement_pack.raw_measurements_);
    }catch(const std::invalid_argument& e){
      cout << "Invalid Jacobian, skipping update" << endl;
    }

  } else {
    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    /* Not implemented here since the data is gaussian

    // Residual checks on the solution can be used to validate inputs against
    // outliers. If the Residual(r) value below is large the model and the measurements
    // are not in sync. Resetting the filter should correct it
    MatrixXd rms_delta = measurement_pack.raw_measurements_ - ekf_.H_*ekf_.x_;
    float r = (rms_delta.transpose()*rms_delta)(0);
    if(r >= 3){ // parameter should be based on noise parameters
      // reset filter
      is_initialized_ = false;
      cout << "Resetting filter:" << endl;
    }*/
    ekf_.Update(measurement_pack.raw_measurements_);
  }


  // print the output
  //cout << "x_ = " << ekf_.x_ << endl;
  //cout << "P_ = " << ekf_.P_ << endl;

}
