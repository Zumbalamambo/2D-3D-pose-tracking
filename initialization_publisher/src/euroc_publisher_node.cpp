#include <cstdio>
#include <vector>
#include <ros/ros.h>
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
#include "afm/lines2d.h"
using namespace std;
using namespace Eigen;

const int SKIP = 2;  
string benchmark_output_path;
string estimate_output_path;
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

struct Data
{
    Data(FILE *f)
    {
        if (fscanf(f, " %lf,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f", &t,
               &px, &py, &pz,
               &qw, &qx, &qy, &qz,
               &vx, &vy, &vz,
               &wx, &wy, &wz,
               &ax, &ay, &az) != EOF)
        {
            t /= 1e9;
        }
    }
    double t;
    float px, py, pz;
    float qw, qx, qy, qz;
    float vx, vy, vz;
    float wx, wy, wz;
    float ax, ay, az;
};
int idx = 1, copy_idx=1;
vector<Data> benchmark;
ros::Publisher pub_3dmap;
ros::Publisher pub_odom;
ros::Publisher pub_path;
ros::Publisher pub_basepose;
nav_msgs::Path path;

int init = 0;
Quaterniond baseRgt;
Vector3d baseTgt;
Eigen::Matrix3d bias_R;
Eigen::Vector3d bias_T;
tf::Transform trans;

void pub3dmap(const std_msgs::Header &header)
{
    if (init==SKIP+2)
    {
        ROS_INFO("publish 3D map");
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
        transform.block<3, 3>(0, 0) = baseRgt.normalized().toRotationMatrix();
        transform.block<3, 1>(0, 3) = baseTgt;
        pcl::transformPointCloud(*cloud, *transformed_cloud, transform);
        pcl_conversions::toPCL(header, transformed_cloud->header);
        transformed_cloud->header.frame_id = "world";
        pub_3dmap.publish (transformed_cloud);
    }
}

void track_pose_callback(const nav_msgs::OdometryConstPtr &odom_msg)
{
    //ROS_INFO("odom callback!"); callback after tracking work, so no need for wait at the start.
    if (odom_msg->header.stamp.toSec() > benchmark.back().t)
      return;

    for (; idx < static_cast<int>(benchmark.size()) && benchmark[copy_idx].t <= odom_msg->header.stamp.toSec(); copy_idx++)
        ;
    // bais transfrom between ground truth pose coordinate with point cloud frame
    Eigen::Matrix3d R = bias_R * Quaterniond(benchmark[idx - 1].qw,
                                             benchmark[idx - 1].qx,
                                             benchmark[idx - 1].qy,
                                             benchmark[idx - 1].qz)
                                     .toRotationMatrix();
    Eigen::Vector3d T = bias_R * Vector3d{benchmark[idx - 1].px, benchmark[idx - 1].py, benchmark[idx - 1].pz} + bias_T;
    
    nav_msgs::Odometry odometry;
    odometry.header.stamp = ros::Time(benchmark[copy_idx - 1].t);
    odometry.header.frame_id = "world";
    odometry.child_frame_id = "world";

    Vector3d tmp_T = baseTgt + baseRgt * T;
    odometry.pose.pose.position.x = tmp_T.x();
    odometry.pose.pose.position.y = tmp_T.y();
    odometry.pose.pose.position.z = tmp_T.z();

    Quaterniond tmp_R = (baseRgt * Quaterniond(R)).normalized();
    odometry.pose.pose.orientation.w = tmp_R.w();
    odometry.pose.pose.orientation.x = tmp_R.x();
    odometry.pose.pose.orientation.y = tmp_R.y();
    odometry.pose.pose.orientation.z = tmp_R.z();


    Vector3d tmp_V = baseRgt * bias_R * Vector3d{benchmark[copy_idx - 1].vx, benchmark[copy_idx - 1].vy, benchmark[copy_idx - 1].vz};
    odometry.twist.twist.linear.x = tmp_V.x();
    odometry.twist.twist.linear.y = tmp_V.y();
    odometry.twist.twist.linear.z = tmp_V.z();
    pub_odom.publish(odometry);

    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header = odometry.header;
    pose_stamped.pose = odometry.pose.pose;
    path.header = odometry.header;
    path.poses.push_back(pose_stamped);
    pub_path.publish(path);
}

void odom_callback(const nav_msgs::OdometryConstPtr &odom_msg)
{
    //ROS_INFO("odom callback!");
    if (odom_msg->header.stamp.toSec() > benchmark.back().t)
      return;
  
    for (; idx < static_cast<int>(benchmark.size()) && benchmark[idx].t <= odom_msg->header.stamp.toSec(); idx++)
        ;

    // bais transfrom between ground truth pose coordinate with point cloud frame
    Eigen::Matrix3d R = bias_R * Quaterniond(benchmark[idx - 1].qw,
                                             benchmark[idx - 1].qx,
                                             benchmark[idx - 1].qy,
                                             benchmark[idx - 1].qz)
                                     .toRotationMatrix();
    Eigen::Vector3d T = bias_R * Vector3d{benchmark[idx - 1].px, benchmark[idx - 1].py, benchmark[idx - 1].pz} + bias_T;
    if (init++ < SKIP)
    {
        baseRgt = Quaterniond(odom_msg->pose.pose.orientation.w,
                              odom_msg->pose.pose.orientation.x,
                              odom_msg->pose.pose.orientation.y,
                              odom_msg->pose.pose.orientation.z).toRotationMatrix() * R.transpose();
        baseTgt = Vector3d{odom_msg->pose.pose.position.x,
                           odom_msg->pose.pose.position.y,
                           odom_msg->pose.pose.position.z} -
                  baseRgt * T;
        return;
    }
    pub3dmap(odom_msg->header);
    
    //publish the first start pose in body frame
    nav_msgs::Odometry base_pose;
    base_pose.header.stamp = ros::Time(benchmark[idx - 1].t);
    base_pose.header.frame_id = "world";
    base_pose.child_frame_id = "world";
    base_pose.pose.pose.position.x = baseTgt.x();
    base_pose.pose.pose.position.y = baseTgt.y();
    base_pose.pose.pose.position.z = baseTgt.z();
    base_pose.pose.pose.orientation.w = baseRgt.w();
    base_pose.pose.pose.orientation.x = baseRgt.x();
    base_pose.pose.pose.orientation.y = baseRgt.y();
    base_pose.pose.pose.orientation.z = baseRgt.z();
    pub_basepose.publish(base_pose);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "benchmark_publisher");
    ros::NodeHandle n("~");

    string csv_file = readParam<string>(n, "data_name");
    std::cout << "load ground truth " << csv_file << std::endl;
    FILE *f = fopen(csv_file.c_str(), "r");
    if (f==NULL)
    {
      ROS_WARN("can't load ground truth; wrong path");
      //std::cerr << "can't load ground truth; wrong path " << csv_file << std::endl;
      return 0;
    }
    char tmp[10000];
    if (fgets(tmp, 10000, f) == NULL)
    {
        ROS_WARN("can't load ground truth; no data available");
    }
    while (!feof(f))
        benchmark.emplace_back(f);
    fclose(f);
    benchmark.pop_back();
    ROS_INFO("Data loaded: %d", (int)benchmark.size());

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

    //initial transform from map to camera
    std::string config_file;
    n.getParam("config_file", config_file);
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    cv::Mat cv_bias_R, cv_bias_T;
    fsSettings["initialRotation"] >> cv_bias_R;
    fsSettings["initialTranslation"] >> cv_bias_T;
    cv::cv2eigen(cv_bias_R, bias_R);
    cv::cv2eigen(cv_bias_T, bias_T);

    pub_odom = n.advertise<nav_msgs::Odometry>("/benchmark_publisher/odometry", 1000);
    pub_path = n.advertise<nav_msgs::Path>("/benchmark_publisher/path", 1000);
    pub_basepose= n.advertise<nav_msgs::Odometry>("/benchmark_publisher/base_pose", 1000);
    pub_3dmap =  n.advertise<pcl::PointCloud<pcl::PointXYZ>>("/benchmark_publisher/map_clouds",1000);

    ros::Subscriber sub_odom = n.subscribe("estimated_odometry", 1000, odom_callback);  //vio initialization

    ros::Subscriber sub_trackpose=n.subscribe("/tracking_node/global_odometry", 1000, track_pose_callback);
    ros::Rate r(20);
    ros::spin();
}
