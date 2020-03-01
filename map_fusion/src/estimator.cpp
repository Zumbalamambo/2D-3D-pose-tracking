#include "estimator.h"
#include <omp.h>
#include <stdio.h>
std::string TRACK_RESULT_PATH;

void estimator::setParameters(const string &calib_file, vector<Vector6d> &_lines3d)
{
	frame_count=-1;
	index=0;
	solver_flag=INITIAL;
	lines3d_map= _lines3d;
	cv::FileStorage fsSettings(calib_file, cv::FileStorage::READ);
	cv::FileNode ns = fsSettings["projection_parameters"];
    double m_fx = static_cast<double>(ns["fx"]);
    double m_fy = static_cast<double>(ns["fy"]);
	double m_cx = static_cast<double>(ns["cx"]);
	double m_cy = static_cast<double>(ns["cy"]);
	cv_CMatrix = (cv::Mat_<double>(3, 3) << m_fx, 0, m_cx, 0, m_fy, m_cy, 0, 0, 1);
	

	ns = fsSettings["distortion_parameters"];
	double m_k1 = static_cast<double>(ns["k1"]);
	double m_k2 = static_cast<double>(ns["k2"]);
	double m_p1 = static_cast<double>(ns["p1"]);
	double m_p2 = static_cast<double>(ns["p2"]);
	cv_dist = (cv::Mat_<double>(1, 4) << m_k1, m_k2, m_p1, m_p2);

	int width=static_cast<int>(fsSettings["width"]);
	int height=static_cast<int>(fsSettings["height"]);
	new_Matrix=getOptimalNewCameraMatrix(cv_CMatrix, cv_dist, cv::Size(width, height), 0, cv::Size(width, height));
	
	cv::cv2eigen(cv_CMatrix, K);
	//camera frame to body frame
    cv::Mat cv_b2c_R, cv_b2c_T;
    fsSettings["extrinsicRotation"] >> cv_b2c_R;
    fsSettings["extrinsicTranslation"] >> cv_b2c_T;
    cv::cv2eigen(cv_b2c_R, b2c_R);
    cv::cv2eigen(cv_b2c_T, b2c_T);

	m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);

	//optimization parameters
	iterations= static_cast<int>(fsSettings["iterations"]);
	per_inliers= static_cast<int>(fsSettings["per_inliers"]);
	threshold= static_cast<double>(fsSettings["threshold"]);
	lamda= static_cast<double>(fsSettings["lamda"]);
	save=static_cast<int>(fsSettings["savefile"]);

	ROS_INFO("Finishing setting params for sliding window...");
}
// load transformation of 3D map to global body, body to camera tranform.
void estimator::loadExtrinsictf(Vector3d &_w2gb_T, Matrix3d &_w2gb_R)
{
	w2gb_T = _w2gb_T;
	w2gb_R = _w2gb_R; 
	ROS_INFO("Finishing load extrinsic...");
}

// create estimator online
void estimator::processImage(double _time_stamp, Vector3d &_vio_T, Matrix3d &_vio_R, cv::Mat &_image, vector<line2d> &_lines2d)
{
	if (frame_count < WINDOW_SIZE)
		frame_count++;
	time_stamp[frame_count] = _time_stamp;
	vio_T[frame_count]=_vio_T;
	vio_R[frame_count] = _vio_R;
	image[frame_count]= _image.clone();
	lines2d[frame_count]= _lines2d;
	undist_lines2d[frame_count]=undistortedPoints(_lines2d);
	if (frame_count > 0)
		{  //P(n-1)=delta_R*P(n)+delta_T
			delta_R[frame_count - 1] = vio_R[frame_count-1] * vio_R[frame_count].transpose();
			delta_T[frame_count - 1] = vio_T[frame_count-1] - delta_R[frame_count - 1] * vio_T[frame_count];
		}
	delta_R[frame_count]<<1,0,0,
						  0,1,0,
						  0,0,1;
	delta_T[frame_count]<<0.0,0.0,0.0;

	if (solver_flag == INITIAL)
	{
		//find local 3d lines
		T_w[frame_count] = _vio_T;
		R_w[frame_count] = _vio_R;
		lines3d[frame_count] = updatemaplines_3d(_vio_T, _vio_R);
		Matrix3d tempRot = b2c_R.transpose() * R_w[frame_count].transpose();
		Vector3d tempTrans = -tempRot * T_w[frame_count] - b2c_R.transpose() * b2c_T;
		matches2d3d[frame_count] = updatecorrespondence(lines3d[frame_count], undist_lines2d[frame_count], K, tempRot, tempTrans, lamda, threshold);
		fuse_pose();
		if (frame_count == WINDOW_SIZE-1 || WINDOW_SIZE==0)
		{
			solver_flag = NON_LINEAR; 
		}
	} 
	else
	{
		//predict current frame
		R_w[frame_count]=delta_R[frame_count - 1].transpose()*R_w[frame_count - 1];
        T_w[frame_count]=delta_R[frame_count - 1].transpose()*(T_w[frame_count-1]-delta_T[frame_count - 1]);
		lines3d[frame_count] = updatemaplines_3d(T_w[frame_count], R_w[frame_count]);
		Matrix3d tempRot = b2c_R.transpose() * R_w[frame_count].transpose();
		Vector3d tempTrans = -tempRot * T_w[frame_count] - b2c_R.transpose() * b2c_T;
		matches2d3d[frame_count] = updatecorrespondence(lines3d[frame_count], undist_lines2d[frame_count], K, tempRot, tempTrans, lamda, threshold);
		//solve optimization
		jointoptimization();
		//slideWindow
		slideWindow();
	}
	index++;

	if (index < 2)
	{
		ROS_INFO("Starting time: %f", _time_stamp);
		Matrix3d tempRot = b2c_R.transpose() * R_w[frame_count].transpose();
		Vector3d tempTrans = -tempRot * T_w[frame_count] - b2c_R.transpose() * b2c_T;
		vector<pairsmatch> match1;
		match1 = updatecorrespondence(lines3d[frame_count], undist_lines2d[frame_count], K, tempRot, tempTrans, lamda, threshold);
		savelines_2d3d(true);
		savematches(match1,index,delta_R[frame_count], delta_T[frame_count], false);
	}
}

vector<line3d> estimator::updatemaplines_3d(Vector3d &_vio_T, Matrix3d &_vio_R)
{
	vector<line3d> tmp_lines3d;
	//update transformations: 3d map points to camera frame
	Eigen::Matrix3d R = b2c_R.transpose() * _vio_R.transpose() * w2gb_R;
	Eigen::Vector3d T = b2c_R.transpose() * (_vio_R.transpose() * (w2gb_T - _vio_T) - b2c_T);

	for (size_t i = 0; i < lines3d_map.size(); i++)
	{
		bool start_flag = false, end_flag = false;
		Eigen::Vector3d start_pt = Eigen::Vector3d(lines3d_map[i][0], lines3d_map[i][1], lines3d_map[i][2]);
		Eigen::Vector3d end_pt = Eigen::Vector3d(lines3d_map[i][3], lines3d_map[i][4], lines3d_map[i][5]);

		Eigen::Vector3d tf_start_pt = R * start_pt + T;
		Eigen::Vector3d tf_end_pt = R * end_pt + T;

		if (tf_start_pt[2] > 0)
		{
			double xx=K(0,0)*tf_start_pt[0]/tf_start_pt[2]+K(0,2);
			double yy=K(1,0)*tf_start_pt[1]/tf_start_pt[2]+K(1,2);
			if (xx*(xx+1-image[frame_count].cols) <= 0 && yy*(yy+1-image[frame_count].rows) <= 0)
					start_flag = true;
		}

		if (tf_end_pt[2] > 0)
		{
			double xx=K(0,0)*tf_end_pt[0]/tf_end_pt[2]+K(0,2);
			double yy=K(1,0)*tf_end_pt[1]/tf_end_pt[2]+K(1,2);
			if (xx*(xx+1-image[frame_count].cols) <= 0 && yy*(yy+1-image[frame_count].rows) <= 0)
					end_flag = true;
		}

		if (start_flag && end_flag) // both end points are in FOV
		{
			//map to vio frame, point cloud
			Vector3d pt1 = w2gb_R * start_pt + w2gb_T;
			Vector3d pt2 = w2gb_R * end_pt + w2gb_T;
			line3d l3d(pt1, pt2);
			tmp_lines3d.push_back(l3d);
		}
		else if (start_flag) //only start point is in FOV
		{
			Eigen::Vector3d dirvec = tf_end_pt - tf_start_pt;
			double t = 0.1;
			bool inFOV=true;
			while (inFOV)
			{
				Eigen::Vector3d temp_tf_end_pt = tf_start_pt + t * dirvec;
				if (temp_tf_end_pt[2] > 0)
				{
					double xx = K(0, 0) * temp_tf_end_pt[0] / temp_tf_end_pt[2] + K(0, 2);
					double yy = K(1, 0) * temp_tf_end_pt[1] / temp_tf_end_pt[2] + K(1, 2);
					if (xx * (xx + 1 - image[frame_count].cols) <= 0 && yy * (yy + 1 - image[frame_count].rows) <= 0)
						t += 0.1;
					else
						inFOV=false;
				}
				else
					inFOV=false;	
			}

			end_pt=start_pt+(t-0.1)*(end_pt-start_pt);
			Vector3d pt1 = w2gb_R * start_pt + w2gb_T;
			Vector3d pt2 = w2gb_R * end_pt + w2gb_T;
			line3d l3d(pt1, pt2);
			tmp_lines3d.push_back(l3d);
		}
		else if (end_flag) // only end point is in FOV
		{
			Eigen::Vector3d dirvec = tf_start_pt- tf_end_pt;
			double t = 0.1;
			bool inFOV = true;
			while (inFOV)
			{
				Eigen::Vector3d temp_tf_start_pt = tf_end_pt + t * dirvec;
				if (temp_tf_start_pt[2] > 0)
				{
					double xx = K(0, 0) * temp_tf_start_pt[0] / temp_tf_start_pt[2] + K(0, 2);
					double yy = K(1, 0) * temp_tf_start_pt[1] / temp_tf_start_pt[2] + K(1, 2);
					if (xx * (xx + 1 - image[frame_count].cols) <= 0 && yy * (yy + 1 - image[frame_count].rows) <= 0)
						t += 0.1;
					else
						inFOV = false;
				}
				else
					inFOV = false;
			}

			start_pt = end_pt + (t - 0.1)*(start_pt - end_pt);
			Vector3d pt1 = w2gb_R * start_pt + w2gb_T;
			Vector3d pt2 = w2gb_R * end_pt + w2gb_T;
			line3d l3d(pt1, pt2);
			tmp_lines3d.push_back(l3d);
		}
		//for the cases of two end points are not in FOV, but a part of line visible, not considered
	}

	return tmp_lines3d;
}

void estimator::showUndistortion(const string &name)
{
	int ROW = image[frame_count].rows;
	int COL = image[frame_count].cols;
	int FOCAL_LENGTH = 460;
	cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
	vector<Eigen::Vector2d> distortedp, undistortedp;
	for (int i = 0; i < COL; i++)
		for (int j = 0; j < ROW; j++)
		{
			Eigen::Vector2d a(i, j);
			Eigen::Vector3d b;
			m_camera->liftProjective(a, b);
			distortedp.push_back(a);
			undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
			//printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
		}
	for (int i = 0; i < int(undistortedp.size()); i++)
	{ 
		cv::Mat pp(3, 1, CV_32FC1);
		pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
		pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
		pp.at<float>(2, 0) = 1.0;
		//cout << trackerData[0].K << endl;
		//printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
		//printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
		if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
		{
			undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = image[frame_count].at<uchar>(distortedp[i].y(), distortedp[i].x());
		}
		else
		{
			//ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
		}
	}
	cv::imwrite(name, undistortedImg);
	cv::waitKey(2);
}

vector<line2d> estimator::undistortedPoints(vector<line2d> &_lines2d)
{
	vector<line2d> _undist_lines2d;
	for (unsigned int i = 0; i < _lines2d.size(); i++)
	{
		line2d kl = _lines2d[i];
		Eigen::Vector3d b;
		m_camera->liftProjective(kl.ptstart, b);
		Eigen::Vector3d pt_start = K * b;

		Eigen::Vector3d d;
		m_camera->liftProjective(kl.ptend, d);
		Eigen::Vector3d pt_end = K * d;
		Vector4d l2d(pt_start.x() / pt_start.z(), pt_start.y() / pt_start.z(), pt_end.x() / pt_end.z(), pt_end.y() / pt_end.z());
		line2d undist_line2d(l2d);
		_undist_lines2d.push_back(undist_line2d);
	}
	return _undist_lines2d;
}

// using extracted 2D and 3D line features to get the correspondence and refine the camera pose
void estimator::jointoptimization()
{

	Matrix3d delta_R_n[frame_count+1];
	Vector3d delta_T_n[frame_count+1];
	delta_R_n[frame_count]=delta_R[frame_count];
	delta_T_n[frame_count]=delta_T[frame_count];
	
	savelines_2d3d(save);
	double theta=lamda;
	double reject_threshod=threshold;
	for (int iter = 0; iter < iterations; iter++)
	{
		for (int nframe = frame_count - 1; nframe >= 0; nframe--)
		{
			// ROS_INFO("obtian %d", nframe);
			delta_R_n[nframe] = delta_R[nframe] * delta_R_n[nframe + 1];
			delta_T_n[nframe] = delta_R[nframe] * delta_T_n[nframe + 1] + delta_T[nframe];
		}
		// obtain 2d-3d correspondences
		int Num_matches = 0;
		for (int nframe = frame_count; nframe >= 0; nframe--)
		{
			// Matrix3d R_w_n = delta_R_n[nframe] * R_w[frame_count];
			// Vector3d T_w_n = delta_R_n[nframe] * T_w[frame_count] + delta_T_n[nframe];
			// Matrix3d tempRot = b2c_R.transpose() * R_w_n.transpose();
			// Vector3d tempTrans = -tempRot * T_w_n - b2c_R.transpose() * b2c_T;
			// matches2d3d[nframe] = updatecorrespondence(lines3d[nframe], undist_lines2d[nframe], K, tempRot, tempTrans, theta, reject_threshod);
			Num_matches += matches2d3d[nframe].size();
		}
		if (save && iter == 0)
		{
			savematches(matches2d3d[frame_count], frame_count, delta_R_n[frame_count], delta_T_n[frame_count], false);
		}
		if (Num_matches < frame_count * per_inliers) //current frame feature is not stable, skip, use the vio pose
		{
			ROS_WARN("feature matching is not enough");
			break;
		}
		//ceres optimization using 2d-3d correspondences
		ceres::Problem problem;
		Eigen::Quaterniond updateQuat(R_w[frame_count]);
		std::vector<double> ceres_rotation = std::vector<double>({updateQuat.w(), updateQuat.x(), updateQuat.y(), updateQuat.z()});
		std::vector<double> ceres_translation = std::vector<double>({T_w[frame_count].x(), T_w[frame_count].y(), T_w[frame_count].z()});
		ceres::LocalParameterization *quaternion_parameterization =
			new ceres::QuaternionParameterization;
		ceres::LossFunction *loss_func(new ceres::HuberLoss(3.0));

		for (int nframe = frame_count; nframe >=0; nframe--)
			for (unsigned int i = 0; i < matches2d3d[nframe].size(); i++)
			{
				Vector3d param2d(matches2d3d[nframe][i].line2dt.A / matches2d3d[nframe][i].line2dt.A2B2,
								 matches2d3d[nframe][i].line2dt.B / matches2d3d[nframe][i].line2dt.A2B2, matches2d3d[nframe][i].line2dt.C / matches2d3d[nframe][i].line2dt.A2B2);
				ceres::CostFunction *cost_function =
					RegistrationError::Create(param2d, matches2d3d[nframe][i].line3dt.ptstart, matches2d3d[nframe][i].line3dt.ptend, K,
											  b2c_R, b2c_T, delta_R_n[nframe], delta_T_n[nframe]);

				problem.AddResidualBlock(
					cost_function,
					loss_func,
					&(ceres_rotation[0]), 
					&(ceres_translation[0]));
				problem.SetParameterization(&(ceres_rotation[0]), quaternion_parameterization);
			}

		ceres::Solver::Options options;
		options.linear_solver_type = ceres::SPARSE_SCHUR;
		options.minimizer_progress_to_stdout = false; 
		options.max_num_iterations = 30;
		options.num_threads = 12;
		// options.logging_type = SILENT;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		// std::cout << summary.BriefReport() << "\n";
		Eigen::Quaterniond q(ceres_rotation[0], ceres_rotation[1], ceres_rotation[2], ceres_rotation[3]);
		Eigen::Vector3d t(ceres_translation[0], ceres_translation[1], ceres_translation[2]);
		R_w[frame_count] = q.normalized().toRotationMatrix();
		T_w[frame_count] = t;

		if (save && iter == iterations - 1)
		{
			savematches(matches2d3d[frame_count], frame_count, delta_R_n[frame_count], delta_T_n[frame_count], true);
		}
		//more restrict threshold for inlier correspondences
		theta=0.9*theta;
		reject_threshod=0.9*reject_threshod;
		Matrix3d tempRot = b2c_R.transpose() * R_w[frame_count].transpose();
		Vector3d tempTrans = -tempRot * T_w[frame_count] - b2c_R.transpose() * b2c_T;
		matches2d3d[frame_count] = updatecorrespondence(lines3d[frame_count], undist_lines2d[frame_count], K, tempRot, tempTrans, theta, reject_threshod);

	}
	fuse_pose();
}

void estimator::fuse_pose()
{
	Eigen::Matrix3d deR = R_w[frame_count - 1] * R_w[frame_count].transpose();
	Eigen::Vector3d deT = T_w[frame_count - 1]- deR*T_w[frame_count];
	
	if ((deT-delta_T[frame_count - 1]).norm()>0.8)
	{
		ROS_WARN("correspondence error...");
		R_w[frame_count]=delta_R[frame_count - 1].transpose()*R_w[frame_count - 1];
        T_w[frame_count]=delta_R[frame_count - 1].transpose()*(T_w[frame_count-1]-delta_T[frame_count - 1]);
	}
	// write result to file
	ofstream foutC(TRACK_RESULT_PATH, ios::app);
	foutC.setf(ios::fixed, ios::floatfield);
	foutC.precision(0);
	foutC << time_stamp[frame_count] * 1e9 << ",";
	foutC.precision(5);
	Eigen::Quaterniond tmp_Q(R_w[frame_count]);
	foutC << T_w[frame_count].x() << ","
		  << T_w[frame_count].y() << ","
		  << T_w[frame_count].z() << ","
		  << tmp_Q.w() << ","
		  << tmp_Q.x() << ","
		  << tmp_Q.y() << ","
		  << tmp_Q.z() << "," << endl;
	foutC.close();
}

void estimator::slideWindow()
{
	// cout<<"frame_count="<<frame_count<<endl;
	for (size_t i = 0; i < WINDOW_SIZE; i++)
	{
		time_stamp[i] = time_stamp[i+1];
		vio_T[i] = vio_T[i+1];
		vio_R[i] = vio_R[i+1];
		delta_R[i]= delta_R[i+1];
		delta_T[i]= delta_T[i+1];
		T_w[i] = T_w[i+1];
		R_w[i] =R_w[i+1];
		//image[i] = image[i+1];
		//lines2d[i] = lines2d[i+1];
		undist_lines2d[i] = undist_lines2d[i+1];
		lines3d[i] = lines3d[i+1];
		matches2d3d[i]=matches2d3d[i+1];
	}
}

void estimator::savematches(const vector<pairsmatch> &matches, int &frame,
							 Matrix3d &delta_R_i, Vector3d &delta_t_i, const bool &optimized)
{
	ROS_INFO("Number of matches: %d", int(matches.size()));
	char filename2d[100];
	if (optimized)
		sprintf(filename2d, "optmlines2d_%d_%d.txt", index, frame);
	else
	{
		sprintf(filename2d, "bfmlines2d_%d_%d.txt", index, frame);
	}
	ofstream out2d(filename2d);
	char filename3d[100];
	if (optimized)
		sprintf(filename3d, "optmlines3d_%d_%d.txt", index, frame);
	else
	{
		sprintf(filename3d, "bfmlines3d_%d_%d.txt", index, frame);
	}
	ofstream out3d(filename3d);

	Matrix3d R_w_n = delta_R_i * R_w[frame_count];
	Vector3d T_w_n = delta_R_i * T_w[frame_count] + delta_t_i;
	Matrix3d R = b2c_R.transpose() * R_w_n.transpose();
	Vector3d t = -R * T_w_n - b2c_R.transpose() * b2c_T;
	double sum_error=0;
	for (size_t i = 0; i < matches.size(); i++)
	{
		line3d l3d=matches[i].line3dt;
		pairsmatch match(matches[i].index, matches[i].line2dt, l3d.transform3D(R,t));
		auto start_pt2d = match.line2dt.ptstart;
		out2d << start_pt2d.x() << " " << start_pt2d.y() << " ";
		auto end_pt2d = match.line2dt.ptend;
		out2d << end_pt2d.x() << " " << end_pt2d.y() << " ";
		out2d << "\n";

		line2d p_l2d = match.line3dt.project3D(K);
		start_pt2d = p_l2d.ptstart;
		out3d << start_pt2d.x() << " " << start_pt2d.y() << " ";
		end_pt2d = p_l2d.ptend;
		out3d << end_pt2d.x() << " " << end_pt2d.y() << " ";
		out3d << match.distance.x() << " " << match.distance.y() << " " << match.distance.z();
		out3d << "\n";

		sum_error+=(match.calcEulerDist(K, 0.2)).x();
	}

	out2d.close();
	out3d.close();
	if(optimized)
	{
		ROS_INFO("after sum: %f, mean: %f", sum_error, sum_error / matches.size());
		ROS_INFO("------------------------------------------");
	}
	else
	{
		ROS_INFO("before sum: %f, mean: %f", sum_error, sum_error/matches.size());
	}
}

//save 2d and 3d lines for a local frame and camera pose, and image frames, the last frame in slidingwindow
void estimator::savelines_2d3d(const bool &save) 
{
	if (save)
	{
		//save camera poses
		Eigen::Matrix3d R = b2c_R.transpose() * R_w[frame_count].transpose();
		Eigen::Vector3d T = -R * T_w[frame_count] - b2c_R.transpose() * b2c_T;
		Eigen::Quaterniond quat(R);
		
		string pose_file = "estimator_poses.csv";
		ofstream outpose;
		if (index==0)
			outpose.open(pose_file);
		else
			outpose.open(pose_file, ios::app);
		if (outpose)
		{
			outpose << quat.w() << "," << quat.x() << "," << quat.y() << "," << quat.z() << "," << T.x() << "," << T.y() << "," << T.z() << "\n";
		}
		outpose.close();
		//save 2d lines
		char filename2d[100];
		sprintf(filename2d, "lines2d_%d.txt", index);
		ofstream out2d(filename2d);
		for (size_t i = 0; i < undist_lines2d[frame_count].size(); i++)
		{
			auto start_pt2d = undist_lines2d[frame_count][i].ptstart;
			out2d << start_pt2d.x() << " " << start_pt2d.y() << " ";
			auto end_pt2d = undist_lines2d[frame_count][i].ptend;
			out2d << end_pt2d.x() << " " << end_pt2d.y();
			out2d << "\n";
		}
		out2d.close();

		//save 3d lines
		char filename3d[100];
		sprintf(filename3d, "lines3d_%d.txt", index);
		ofstream out3d(filename3d);
		for (size_t i = 0; i < lines3d[frame_count].size(); i++)
		{
			auto start_pt3d = lines3d[frame_count][i].ptstart;
			auto tf_start_pt = K * (R * start_pt3d + T);
			out3d << tf_start_pt.x() / tf_start_pt.z() << " " << tf_start_pt.y() / tf_start_pt.z() << " " << 1 << " ";
			auto end_pt3d = lines3d[frame_count][i].ptend;
			auto tf_end_pt = K * (R * end_pt3d + T);
			out3d << tf_end_pt.x() / tf_end_pt.z() << " " << tf_end_pt.y() / tf_end_pt.z() << " " << 1;
			out3d << "\n";
		}
		out3d.close();

		//save images
		char imgname[100];
		sprintf(imgname, "img_%d.jpg", index);
		cv::imwrite(imgname, image[frame_count]);
		if (index < 0)
			ROS_INFO("%d image time=%lf, lines3d size: %d, lines2d size: %d", index, time_stamp[frame_count], int(lines3d[frame_count].size()), int(lines2d[frame_count].size()));
	}
}
