#pragma once

#include <vector>
#include <cstdio>
#include <fstream>
#include <eigen3/Eigen/Dense> //must before opencv eigen.hpp
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "opencv2/core/utility.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/flann.hpp>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <cv_bridge/cv_bridge.h>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "CameraPoseVisualization.h"
#include "ceres/ceres.h"
#include "line.h"
#include "afm/lines2d.h"

#define WINDOW_SIZE 10
using namespace Eigen;
using namespace std;
using namespace camodocal;
extern std::string TRACK_RESULT_PATH;  //save result


struct PointMatch {
    PointMatch(Vector3d p, Vector3d q, Vector3d n):
    p(p), q(q), n(n) {}
    Vector3d p;
    Vector3d q;
    Vector3d n;  
};

class estimator
{
public:
	estimator(){};
	void setParameters(const string &calib_file, vector<Vector6d> &_lines3d);
	void processImage(double _time_stamp, Vector3d &_vio_T, Matrix3d &_vio_R, cv::Mat &_image, vector<line2d> &_lines2d);
	void loadExtrinsictf(Vector3d &_w2gb_T, Matrix3d &_w2gb_R);
	vector<line2d> undistortedPoints(vector<line2d> &_lines2d);
	void showUndistortion(const string &name);

	vector<line3d> updatemaplines_3d(Vector3d &_vio_T, Matrix3d &_vio_R);

	void jointoptimization();
	void slideWindow();
	void fuse_pose();
	void savelines_2d3d(const bool &save);
	void savematches(const vector<pairsmatch> &matches, int &frame, Matrix3d &delta_R_i, Vector3d &delta_t_i, const bool &optimized);

	enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };
	SolverFlag solver_flag;
	double time_stamp[WINDOW_SIZE+1]; 
	int frame_count, index;

	Eigen::Vector3d vio_T[WINDOW_SIZE+1]; 
	Eigen::Matrix3d vio_R[WINDOW_SIZE+1]; 
	Eigen::Vector3d T_w[WINDOW_SIZE+1];
	Eigen::Matrix3d R_w[WINDOW_SIZE+1];
	Eigen::Vector3d delta_T[WINDOW_SIZE+1];		
	Eigen::Matrix3d delta_R[WINDOW_SIZE+1];
	Eigen::Vector3d w2gb_T;
	Eigen::Matrix3d w2gb_R;
	Eigen::Vector3d b2c_T;
	Eigen::Matrix3d b2c_R;
	Eigen::Matrix3d K;
	cv::Mat image[WINDOW_SIZE+1];
	cv::Mat cv_CMatrix, new_Matrix, cv_dist;
	camodocal::CameraPtr m_camera;

	vector<line3d> lines3d[WINDOW_SIZE+1];
	vector<line2d> lines2d[WINDOW_SIZE+1]; 
	vector<line2d> undist_lines2d[WINDOW_SIZE+1];
	vector<pairsmatch> matches2d3d[WINDOW_SIZE+1];
	vector<Vector6d> lines3d_map;

	int iterations;
	int per_inliers;
	double lamda; 
	double threshold;
	bool save;
};

struct RegistrationError
{
	RegistrationError(Vector3d param2d, Vector3d ptstart, Vector3d ptend, Matrix3d K, Matrix3d b2c_R, Vector3d b2c_T, Matrix3d delta_R, Vector3d delta_T)
		: param2d(param2d), ptstart(ptstart), ptend(ptend), K(K), b2c_R(b2c_R), b2c_T(b2c_T), delta_R(delta_R), delta_T(delta_T) {}
	template <typename T>
	bool operator()(const T *const rotation,
					const T *const translation,
					T *residuals) const
	{
		Matrix<T, 3, 3> R_w = Quaternion<T>(rotation[0], rotation[1], rotation[2], rotation[3]).normalized().toRotationMatrix();
		Matrix<T, 3, 1> T_w = Matrix<T, 3, 1>(translation[0], translation[1], translation[2]);
		Matrix<T, 3, 3> R_W_i = delta_R.cast<T>() * R_w;
		Matrix<T, 3, 1> T_w_i = delta_R.cast<T>() * T_w + delta_T.cast<T>();
		Matrix<T, 3, 3> R = b2c_R.transpose().cast<T>() * R_W_i.transpose();
		Matrix<T, 3, 1> t = -R * T_w_i - (b2c_R.transpose() * b2c_T).cast<T>(); //can be optimized
		// Matrix<T, 3, 1> ptstart_tf3d=;
		// Matrix<T, 3, 1> ptend_tf3d=;
		// T z_dist=(ptstart_tf3d[2]+ptend_tf3d[2])/2.0;
		const Matrix<T, 3, 1> ptstart_tf = K.cast<T>() * (R * ptstart.cast<T>() + t);
		const Matrix<T, 3, 1> ptend_tf = K.cast<T>() * (R * ptend.cast<T>() + t);
		const T dist_st = (T(param2d.x()) * ptstart_tf.x() / ptstart_tf.z() + T(param2d.y()) * ptstart_tf.y() / ptstart_tf.z() + T(param2d.z()));
		const T dist_ed = (T(param2d.x()) * ptend_tf.x() / ptend_tf.z() + T(param2d.y()) * ptend_tf.y() / ptend_tf.z() + T(param2d.z()));
		residuals[0] = dist_st;
		residuals[1] = dist_ed;

		return true;
	}

	static ceres::CostFunction *Create(const Vector3d param2d,
									   const Vector3d ptstart,
									   const Vector3d ptend,
									   const Matrix3d K,
									   const Matrix3d b2c_R,
									   const Vector3d b2c_T,
									   const Matrix3d delta_R,
									   const Vector3d delta_T)
	{
		return (new ceres::AutoDiffCostFunction<RegistrationError, 2, 4, 3>(
			new RegistrationError(param2d, ptstart, ptend, K, b2c_R, b2c_T, delta_R, delta_T)));
	}
	Vector3d param2d;
	Vector3d ptstart;
	Vector3d ptend;
	//line3d l3d;
	Eigen::Matrix3d K;
	Matrix3d b2c_R;
	Vector3d b2c_T;
	Matrix3d delta_R;
	Vector3d delta_T;
};
