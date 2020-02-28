#include <cstdio>
#include <vector>
#include <ros/ros.h>
#include <mutex>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_broadcaster.h>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "opencv2/core/utility.hpp"
#include <cv_bridge/cv_bridge.h>
using namespace std;
using namespace Eigen;
using namespace cv;
std::string IMAGE_TOPIC;
std::mutex m_buf;
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
queue<sensor_msgs::ImageConstPtr> image_buf;
queue<nav_msgs::Odometry::ConstPtr> pose_buf;
ros::Publisher pub_3dmap;
ros::Publisher pub_basepose;
nav_msgs::Path path;

bool initialization_done=false, map_pubed=false;
Matrix3d baseRgt, c2b_R;
Vector3d baseTgt, c2b_T;

Mat org;
int n = 0;
// 3D model points.
vector<Point2d> capturePoint;
void on_mouse(int event, int x, int y, int flags, void *ustc) //event鼠标事件代号，x,y鼠标坐标，flags拖拽和键盘操作的代号
{
    Point pt; //坐标点;
    char coordinateName[16];

    if (event == CV_EVENT_LBUTTONDOWN) //左键按下，读取坐标，并在图像上该点处划圆
    {
        pt = Point2d(x, y);
        capturePoint.push_back(pt);
        cout << capturePoint[n].x << " " << capturePoint[n].y << endl;
        cout << "n=" << n <<endl<<"------"<< endl;
        n++;
        circle(org, pt, 5, Scalar(0, 0, 255, 0)); //划圆
        sprintf(coordinateName, "(%d,%d)", x, y);
        putText(org, coordinateName, pt, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0, 255), 1, 8); //在窗口上显示坐标
        imshow("org", org);
        if (n >= 6)
        {
            imshow("org", org);
            cvDestroyAllWindows();
        }
    }
}
void pub3dmap(const nav_msgs::OdometryConstPtr &odom_msg)
{
    if (initialization_done&&!map_pubed)
    {
        ROS_INFO("publish 3D map");
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>());
        transformed_cloud->header.frame_id = "world";
        Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
        transform.block<3, 3>(0, 0) = baseRgt;
        transform.block<3, 1>(0, 3) = baseTgt;
        pcl::transformPointCloud(*cloud, *transformed_cloud, transform);
        pcl_conversions::toPCL(odom_msg->header, transformed_cloud->header);
        pub_3dmap.publish (transformed_cloud);
        map_pubed=true;
    }
}
void odom_callback(const nav_msgs::OdometryConstPtr &odom_msg)
{
    pub3dmap(odom_msg);
    m_buf.lock();
    pose_buf.push(odom_msg);
    m_buf.unlock();
}

void image_callback(const sensor_msgs::ImageConstPtr &image_msg)
{
    m_buf.lock();
    image_buf.push(image_msg);
    m_buf.unlock();
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "benchmark_publisher");
    ros::NodeHandle n("~");

    string cloud_name;
    n.param("cloud_name", cloud_name, std::string(""));
    std::cout << "load point clouds " << cloud_name << std::endl;
    pcl::io::loadPLYFile(cloud_name, *cloud);
    if (cloud->points.size() == 0)
    {
        ROS_WARN("can't load point clouds; wrong path");
        return 0;
    }
    else
    {
        ROS_INFO("Point cloud data: %d", (int)cloud->points.size());
    }
    
    std::string config_file;
    n.getParam("config_file", config_file);
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    fsSettings["image_topic"] >> IMAGE_TOPIC;  
    cv::FileNode ns = fsSettings["distortion_parameters"];
    double m_k1 = static_cast<double>(ns["k1"]);
    double m_k2 = static_cast<double>(ns["k2"]);
    double m_p1 = static_cast<double>(ns["p1"]);
    double m_p2 = static_cast<double>(ns["p2"]);
    cv::Mat dist_coeffs = (cv::Mat_<double>(4,1) << m_k1, m_k2, m_p1, m_p2);
    ns = fsSettings["projection_parameters"];
    double m_fx = static_cast<double>(ns["fx"]);
    double m_fy = static_cast<double>(ns["fy"]);
    double m_cx = static_cast<double>(ns["cx"]);
    double m_cy = static_cast<double>(ns["cy"]);
    cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << m_fx, 0, m_cx, 0 , m_fy, m_cy, 0, 0, 1);
    //camera frame to body frame
    cv::Mat cv_c2b_R, cv_c2b_T;
    fsSettings["extrinsicRotation"] >> cv_c2b_R;
    fsSettings["extrinsicTranslation"] >> cv_c2b_T;
    cv::cv2eigen(cv_c2b_R, c2b_R);
    cv::cv2eigen(cv_c2b_T, c2b_T);
    //initial bias
    cv::Mat cv_bias_R, cv_bias_T;
    fsSettings["initialRotation"] >> cv_bias_R;
    fsSettings["initialTranslation"] >> cv_bias_T;
    cv::cv2eigen(cv_bias_R, baseRgt);
    cv::cv2eigen(cv_bias_T, baseTgt);
    std::vector<cv::Point3d> model_points;
    // nsh wall
    model_points.push_back(cv::Point3d(10.1319, -8.6280, 311.5690));
    model_points.push_back(cv::Point3d(5.8984, -12.0807, 311.5740));
    model_points.push_back(cv::Point3d(3.4637, -11.8742, 303.9010));
    model_points.push_back(cv::Point3d(9.24275, 0.2024, 303.8120));
    model_points.push_back(cv::Point3d(6.5389, -1.4134, 303.9270));
    model_points.push_back(cv::Point3d(5.0779, -13.4357, 306.7890));
    // nsh floor2
    // model_points.push_back(cv::Point3d(0.323834, -1.47807, 248.7590));
    // model_points.push_back(cv::Point3d(2.5406, -2.0500, 248.7320));
    // model_points.push_back(cv::Point3d(3.9421, -4.7723,248.7400));
    // model_points.push_back(cv::Point3d(3.9695,-4.8148, 250.8440));
    // model_points.push_back(cv::Point3d(2.4435,-1.6717,248.7610));
    // model_points.push_back(cv::Point3d(3.0735,-1.3492,248.7490));
    // nsh floor2_long
    // model_points.push_back(cv::Point3d(24.7378, 19.0196, 248.8060));
    // model_points.push_back(cv::Point3d(24.4447, 18.6818, 248.8060));
    // model_points.push_back(cv::Point3d(21.7915, 16.2203, 250.9190));
    // model_points.push_back(cv::Point3d(21.7938, 16.2288, 248.7820));
    // model_points.push_back(cv::Point3d(21.2046, 15.5220, 250.9130));
    // model_points.push_back(cv::Point3d(21.1996, 14.8184, 248.7830));
    
    // smith
    // model_points.push_back(cv::Point3d(-30.0273,2.8234,324.9770));
    // model_points.push_back(cv::Point3d(-30.1617,2.7611,326.1289));
    // model_points.push_back(cv::Point3d(-31.6389,1.1733,328.4540));
    // model_points.push_back(cv::Point3d(-34.7965,1.7041,328.4419));
    // model_points.push_back(cv::Point3d(-34.7761,1.7378,326.050));
    // model_points.push_back(cv::Point3d(-32.6928, 1.3859, 326.0384));


    pub_3dmap =  n.advertise<pcl::PointCloud<pcl::PointXYZRGBA>>("/benchmark_publisher/map_clouds",1000);
    pub_basepose= n.advertise<nav_msgs::Odometry>("/benchmark_publisher/base_pose", 1000);
    ros::Subscriber sub_image = n.subscribe(IMAGE_TOPIC, 2000, image_callback);
    ros::Subscriber sub_odom = n.subscribe("estimated_odometry", 1000, odom_callback);

    int count=0;
    ros::Rate loop_rate(20);
    while (ros::ok())
    {
        sensor_msgs::ImageConstPtr image_msg = NULL;
        nav_msgs::Odometry::ConstPtr pose_msg = NULL;

        // find out the messages with same time stamp
        m_buf.lock();
        if (!image_buf.empty() && !pose_buf.empty())
        {
            if (image_buf.front()->header.stamp.toSec() > pose_buf.front()->header.stamp.toSec())
            {
                pose_buf.pop();
                printf("throw pose at beginning\n");
            }
            else if (image_buf.back()->header.stamp.toSec() >= pose_buf.front()->header.stamp.toSec())
            {
                pose_msg = pose_buf.front();
                pose_buf.pop();
                while (!pose_buf.empty()) // clear the pose_buf
                {
                    pose_buf.pop();
                }
                while (image_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())
                    image_buf.pop();
                image_msg = image_buf.front();
                image_buf.pop();
            }
        }
        m_buf.unlock();

        if (pose_msg != NULL && !initialization_done)
        {
            printf(" pose time %f \n", pose_msg->header.stamp.toSec());
            printf(" image time %f \n", image_msg->header.stamp.toSec());
            cv_bridge::CvImageConstPtr ptr;
            if (image_msg->encoding == "8UC1") //gray img
            {
                sensor_msgs::Image img;
                img.header = image_msg->header;
                img.height = image_msg->height;
                img.width = image_msg->width;
                img.is_bigendian = image_msg->is_bigendian;
                img.step = image_msg->step;
                img.data = image_msg->data;
                img.encoding = "mono8";
                ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
            }
            else //color img
                ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);

            cv::Mat image = ptr->image; // captured first frame image and camera pose in global body frame
            Vector3d body_T = Vector3d(pose_msg->pose.pose.position.x,
                                       pose_msg->pose.pose.position.y,
                                       pose_msg->pose.pose.position.z);
            Quaterniond Q=Quaterniond(pose_msg->pose.pose.orientation.w,
                                          pose_msg->pose.pose.orientation.x,
                                          pose_msg->pose.pose.orientation.y,
                                          pose_msg->pose.pose.orientation.z);
            Matrix3d body_R = Q.normalized().toRotationMatrix(); 
            // Vector3d body_T = Vector3d(0,0,0);
            // Matrix3d body_R = MatrixXd::Identity(3,3);

            cout<<"camera R:"<<Q.w()<<","<<Q.x()<<","<<Q.y()<<","<<Q.z()<<endl<<"camera T:"<<body_T<<endl;
            org=image;
            namedWindow("org", 1);
            setMouseCallback("org", on_mouse, 0);
            imshow("org", org);
            waitKey(0);

            cv::Mat cv_rotation;
            cv::Mat cv_translation;

            cv::solvePnP(model_points, capturePoint, camera_matrix, dist_coeffs, cv_rotation, cv_translation,false, CV_EPNP);
            cv::Mat cv_rot_mat;
            Rodrigues(cv_rotation, cv_rot_mat);
            Eigen::Vector3d trans;
            Eigen::Matrix3d rot;
            cv::cv2eigen(cv_rot_mat, rot);
            cv::cv2eigen(cv_translation, trans);

            baseRgt=body_R*c2b_R*rot; // T^vio_b*T^b_c*T^c_w*P_0
            baseTgt=body_R*c2b_R*trans+body_R*c2b_T+body_T;
            cout<<"est R="<<endl<<baseRgt<<endl;
            cout<<"est T="<<endl<<baseTgt<<endl;
            // Eigen::Vector3d pt3d(model_points[0].x, model_points[0].y, model_points[0].z);
            // cout<<"3D points:"<<pt3d<<endl;
            // Eigen::Matrix3d K;
            // cv::cv2eigen(camera_matrix, K);
            // Eigen::Vector3d point2d=K*(rot*pt3d+trans);
            // cout<<"2D points:"<< point2d.x()/point2d.z()<<","<<point2d.y()/point2d.z()<<endl;
            initialization_done = true;
        }

        if (pose_msg != NULL && initialization_done)
        {
            count++;
            
            nav_msgs::Odometry base_pose;
            // base_pose.header.stamp = pose_msg->header.stamp; //some times NULL
            base_pose.header.frame_id = "world";
            base_pose.pose.pose.position.x = baseTgt.x();
            base_pose.pose.pose.position.y = baseTgt.y();
            base_pose.pose.pose.position.z = baseTgt.z();
            Quaterniond baseR_quat(baseRgt);
            base_pose.pose.pose.orientation.w = baseR_quat.w();
            base_pose.pose.pose.orientation.x = baseR_quat.x();
            base_pose.pose.pose.orientation.y = baseR_quat.y();
            base_pose.pose.pose.orientation.z = baseR_quat.z();
            pub_basepose.publish(base_pose);

        }
        ros::spinOnce();
        loop_rate.sleep();
    }
}
