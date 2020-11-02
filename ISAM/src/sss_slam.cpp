#include <ros/ros.h>
#include <sstream>
#include <dirent.h>

// Message libraries ----------------------
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Pose.h>
#include <sensor_msgs/PointCloud.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <std_msgs/String.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

// ISAM libraries -------------------------
#include <isam/isam.h>
#include "isam/Anchor.h"
#include <iostream>
#include <sstream>
#include <bits/stdc++.h>

#define pi 3.141592653589

using namespace std;
using namespace isam;
using namespace Eigen;

ros::Publisher  pub_slam_pose, pub_slam_feature;

Slam slam;
vector<Pose2d_Node*> pose_nodes, gps_nodes;
vector<Point2d_Node*> feature_nodes;

vector<array<double, 3> > pose_xy;
vector<array<double, 2> > features_xy;
vector<int> feature_indx;

const int isam::Pose2d::dim;

int getNumFiles(const char* directory, const char* ext) {
  DIR *dir;
  struct dirent *ent;

  int count = 0;

  if ((dir = opendir (directory)) != NULL) {
    // print all the files and directories within directory
    while ((ent = readdir (dir)) != NULL) {
      if( strstr(ent->d_name, ext)) count++;
      }
    closedir (dir);
  }

  else {    // could not open directory
    perror ("");
      return 0;
  }

  return count;
}


Matrix2d Rot_x(double theta){
  Matrix2d R;
  R <<cos(theta) , -sin(theta), sin(theta), cos(theta);
  return R;
}


Vector2d Get_Dist(double dx, double dy, double theta){

  Matrix2d R;
  R << cos(theta) , -sin(theta), sin(theta), cos(theta);

  Vector2d a(dx, dy);

  Vector2d x = R.colPivHouseholderQr().solve(a);

//  cout << "X: \n" << x << endl << endl;

  return x;
}

// Calback Functions ---------------------------------------------------------------------------------------
void gpsLandmark_callback(const nav_msgs::Odometry& gps_msg){

  // get heading from pose Quaternion
  tf::Quaternion q(gps_msg.pose.pose.orientation.x, gps_msg.pose.pose.orientation.y, gps_msg.pose.pose.orientation.z, gps_msg.pose.pose.orientation.w);
  tf::Matrix3x3 m(q);
  double roll, pitch, yaw;
  m.getRPY(roll, pitch, yaw);

  Noise noNoise3 = Covariance( 0.5 * eye(3));                                  // Create a low uncertainty reading
  Pose2d origin(gps_msg.pose.pose.position.x, gps_msg.pose.pose.position.y, yaw); // Assing gps node as the origon
  Pose2d_Node* gps_node = new Pose2d_Node();                                    // Create a first pose (a node)
  Pose2d_Factor* prior = new Pose2d_Factor(gps_node, origin, noNoise3);         // Add to slam Graph with low uncertainty

  slam.add_node(gps_node);                                                      // add it to the graph
  slam.add_factor(prior);

  pose_nodes.push_back(gps_node);                                               // Remember pose localy
  gps_nodes.push_back(gps_node);
  pose_xy.push_back({gps_msg.pose.pose.position.x, gps_msg.pose.pose.position.y, yaw});

  cout << "GPS Point--> X: " << gps_msg.pose.pose.position.x << " Y: " << gps_msg.pose.pose.position.y << " Heading: " << yaw << endl;
/*
  // printing the complete graph
  cout << endl << "Graph:" << endl;
  slam.write(cout);
*/
  }


void newLandmark_callback(const sensor_msgs::PointCloud& lm_msg, const nav_msgs::Odometry& odom_msg ){

  // next pose --------------------------------------------------------------------------------------------------------
  // get heading from pose Quaternion
  tf::Quaternion q(odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w);
  tf::Matrix3x3 m(q);
  double roll, pitch, yaw;
  m.getRPY(roll, pitch, yaw);

  // connect to previous with odometry measurement
  int ii = pose_xy.size();                                                        // Calculate odometry between poses
  double dX = odom_msg.pose.pose.position.x - pose_xy[ii-1][0];
  double dY = odom_msg.pose.pose.position.y - pose_xy[ii-1][1];
  double heading = pose_xy[ii-1][2];

  Vector2d d_xy = Get_Dist(dX, dY, heading);

  Pose2d odometry(d_xy(0), d_xy(1), yaw - heading); // x,y,theta

  pose_xy.push_back({odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, yaw});

  MatrixXd cov = eye(3);
  cov(0,0) = odom_msg.pose.covariance[0];
  cov(0,1) = odom_msg.pose.covariance[1];
  cov(0,2) = odom_msg.pose.covariance[2];

  cov(1,0) = odom_msg.pose.covariance[3];
  cov(1,1) = odom_msg.pose.covariance[3];
  cov(1,2) = odom_msg.pose.covariance[5];

  cov(2,0) = odom_msg.pose.covariance[6];
  cov(2,1) = odom_msg.pose.covariance[7];
  cov(2,2) = odom_msg.pose.covariance[8];

  Noise  noise3 = Covariance( cov );

//  cout <<"\nCov:\n" << cov << endl;

  Pose2d_Node* new_pose_node = new Pose2d_Node();
  pose_nodes.push_back(new_pose_node);
  slam.add_node(new_pose_node);

  ii = pose_nodes.size();
  Pose2d_Pose2d_Factor* constraint = new Pose2d_Pose2d_Factor(pose_nodes[ii-2], new_pose_node, odometry, noise3);
  slam.add_factor(constraint);

  // create a landmark -----------------------------------------------------------------------------------------------
  for(int jj = 0; jj < lm_msg.points.size(); jj++ ){

    Point2d_Node* new_landmark = new Point2d_Node();
    slam.add_node(new_landmark);

    feature_nodes.push_back(new_landmark);
    features_xy.push_back({lm_msg.points[jj].x, lm_msg.points[jj].y});

    int indx = static_cast<int>(lm_msg.channels[jj].values[0]);
    feature_indx.push_back(indx);

    // Connect the pose and the landmark by a measurement
    dX = lm_msg.points[jj].x - odom_msg.pose.pose.position.x;
    dY = lm_msg.points[jj].y - odom_msg.pose.pose.position.y;

    Vector2d d_xy = Get_Dist(dX, dY, yaw);

    Point2d vehicle_measure(d_xy(0), d_xy(1));                                              // Distance from the vehicle (x,y)

    MatrixXd sonarErr = eye(2);
    sonarErr(0,0) = lm_msg.channels[jj].values[1];
    sonarErr(0,1) = lm_msg.channels[jj].values[2];
    sonarErr(1,0) = lm_msg.channels[jj].values[3];
    sonarErr(1,1) = lm_msg.channels[jj].values[4];

    Noise noise2 = Covariance( sonarErr );

    Pose2d_Point2d_Factor* vehicle_measurement = new Pose2d_Point2d_Factor(new_pose_node, new_landmark, vehicle_measure, noise2);
    slam.add_factor(vehicle_measurement );
    }

// Display Info ----------------------------------------------------------------
    cout << "\n\nNew Landmark" << endl;
    cout << "Odom     --> X: " << odom_msg.pose.pose.position.x << ",  Y: " << odom_msg.pose.pose.position.y << " Heading: " << yaw << endl;
    for(int jj = 0; jj < lm_msg.points.size(); jj++ ){
      cout << "Landmark --> X: " << lm_msg.points[jj].x << ", Y: " << lm_msg.points[jj].y << ", indx: " << static_cast<int>(lm_msg.channels[jj].values[0]) << endl;
    }

//  slam.write(cout);
  slam.print_graph();
  cout << endl;

 }


void closeLoop_callback(const sensor_msgs::PointCloud& lm_msg, const nav_msgs::Odometry& odom_msg ){

  // next pose --------------------------------------------------------------------------------------------------------
  // get heading from pose Quaternion
  tf::Quaternion q(odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w);
  tf::Matrix3x3 m(q);
  double roll, pitch, yaw;
  m.getRPY(roll, pitch, yaw);

  // connect to previous with odometry measurement
  int ii = pose_xy.size();                                                        // Calculate odometry between poses
  double dX = odom_msg.pose.pose.position.x - pose_xy[ii-1][0];
  double dY = odom_msg.pose.pose.position.y - pose_xy[ii-1][1];
  double heading = pose_xy[ii-1][2];

  Vector2d d_xy = Get_Dist(dX, dY, heading);

  Pose2d odometry(d_xy(0), d_xy(1), yaw - heading); // x,y,theta

  pose_xy.push_back({odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, yaw});

  MatrixXd cov = eye(3);
  cov(0,0) = odom_msg.pose.covariance[0];
  cov(0,1) = odom_msg.pose.covariance[1];
  cov(0,2) = odom_msg.pose.covariance[2];

  cov(1,0) = odom_msg.pose.covariance[3];
  cov(1,1) = odom_msg.pose.covariance[3];
  cov(1,2) = odom_msg.pose.covariance[5];

  cov(2,0) = odom_msg.pose.covariance[6];
  cov(2,1) = odom_msg.pose.covariance[7];
  cov(2,2) = odom_msg.pose.covariance[8];

  Noise  noise3 = Covariance( cov );

//  cout <<"\nCov:\n" << cov << endl;

  Pose2d_Node* new_pose_node = new Pose2d_Node();
  pose_nodes.push_back(new_pose_node);
  slam.add_node(new_pose_node);

  ii = pose_nodes.size();
  Pose2d_Pose2d_Factor* constraint = new Pose2d_Pose2d_Factor(pose_nodes[ii-2], new_pose_node, odometry, noise3);
  slam.add_factor(constraint);

  // Connect the Landmarks -----------------------------------------------------------------------------------------------
  for(int jj = 0; jj < lm_msg.points.size(); jj++ ){

    int indx = static_cast<int>(lm_msg.channels[jj].values[0]) - 1;

//    cout << "\nNode Matching: " << static_cast<int>(lm_msg.channels[jj].values[0]) << endl;
//    cout << "Target Node:   " << feature_indx[indx] << endl << endl;

    // Connect the pose and the landmark by a measurement
    dX = lm_msg.points[jj].x - odom_msg.pose.pose.position.x;
    dY = lm_msg.points[jj].y - odom_msg.pose.pose.position.y;

    Vector2d d_xy = Get_Dist(dX, dY, yaw);

    Point2d vehicle_measure(d_xy(0), d_xy(1));

    MatrixXd sonarErr = eye(2);
    sonarErr(0,0) = lm_msg.channels[jj].values[1];
    sonarErr(0,1) = lm_msg.channels[jj].values[2];
    sonarErr(1,0) = lm_msg.channels[jj].values[3];
    sonarErr(1,1) = lm_msg.channels[jj].values[4];

    Noise noise2 = Covariance( sonarErr );
    Pose2d_Point2d_Factor* vehicle_measurement = new Pose2d_Point2d_Factor(pose_nodes[ii-1], feature_nodes[indx], vehicle_measure, noise2);
    slam.add_factor(vehicle_measurement );

//    Point2d_Node* landmark = feature_nodes[indx];
//    cout << "\nTarget Feature:\n" << landmark->value() << endl;

    }


  // Display Data ------------------------------------------------------------------------------------------------------------------------------------------------
  cout << "\n" << endl;
  ROS_INFO("Close Loops");
  cout << "Odom     --> X: " << odom_msg.pose.pose.position.x << ",  Y: " << endl;
  //cout << "Odom Err --> X: " << poseErrX << ", Y: " << poseErrY << endl;
  for(int jj = 0; jj < lm_msg.points.size(); jj++ ){
    cout << "Match    --> X: " << lm_msg.points[jj].x << ", Y: " << lm_msg.points[jj].y << " indx: " << static_cast<int>(lm_msg.channels[jj].values[0]) << endl;
    }

  slam.print_graph();
  cout << endl;

  // Optimize Graph -------------------------------------------------------------------------------------------------------------
  cout << "Optimize SLAM Graph" << endl;
  slam.batch_optimization();

  list<Node*> ids = slam.get_nodes();
  Pose2d_Node* poseNode = dynamic_cast<Pose2d_Node*>(ids.back());
  Pose2d pose = poseNode->value();

  cout << "New Pose: " << pose << endl;

  pose_xy[ii-1][0] = pose.x();
  pose_xy[ii-1][1] = pose.y();
  pose_xy[ii-1][2] = pose.t();

  const Covariances& covariances = slam.covariances().clone();

  Covariances::node_pair_list_t node_pair_list;
  node_pair_list.push_back(make_pair(gps_nodes[0], poseNode));

  list<MatrixXd> cov_entries = covariances.access(node_pair_list);

  MatrixXd new_cov = cov_entries.back();

  nav_msgs::Odometry updatedPose;
  updatedPose.pose.pose.position.x = pose.x();
  updatedPose.pose.pose.position.y = pose.y();

//  q.setRPY(0., 0., pose.t());
//  geometry_msgs::Quaternion odom_quat;
//  tf::quaternionTFToMsg(q, odom_quat);

  for(int ii = 0; ii < new_cov.size(); ii++) updatedPose.pose.covariance[ii] =  new_cov(ii);

  pub_slam_pose.publish(updatedPose);

  cout <<"\nPose Cov:\n" << cov << endl;
  cout << "New Pose: " << pose << endl;
  cout << "New Cov: \n" << new_cov << endl<< endl;


}


void optimize_callback(const nav_msgs::Odometry& odom_msg ){

  cout << "Optimize SLAM Graph????" << endl;
/*
  // next pose --------------------------------------------------------------------------------------------------------
  // get heading from pose Quaternion
  tf::Quaternion q(odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w);
  tf::Matrix3x3 m(q);
  double roll, pitch, yaw;
  m.getRPY(roll, pitch, yaw);

  // connect to previous with odometry measurement
  int ii = pose_xy.size();                                                        // Calculate odometry between poses
  float dX = odom_msg.pose.pose.position.x - pose_xy[ii-1][0];
  float dY = odom_msg.pose.pose.position.y - pose_xy[ii-1][1];

  Pose2d_Node* new_pose_node = new Pose2d_Node();
  slam.add_node(new_pose_node);

  pose_xy.push_back({odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, 0.0});
  pose_nodes.push_back(new_pose_node);

  Pose2d odometry(dX, dY, 0.0); // x,y,theta

  MatrixXd cov = eye(3);
  cov(0,0) = odom_msg.pose.covariance[0];
  cov(0,1) = odom_msg.pose.covariance[1];
  cov(1,0) = odom_msg.pose.covariance[2];
  cov(1,1) = odom_msg.pose.covariance[3];

  Noise noise3 = Covariance( cov );

  ii = pose_nodes.size();

  Pose2d_Pose2d_Factor* constraint = new Pose2d_Pose2d_Factor(pose_nodes[ii-2], new_pose_node, odometry, noise3);
  slam.add_factor(constraint);

  //  slam.print_graph();
  //  cout << endl;

  cout << "Optimize SLAM Graph" << endl;
  slam.batch_optimization();

  list<Node*> ids = slam.get_nodes();
  Pose2d_Node* poseNode = dynamic_cast<Pose2d_Node*>(ids.back());
  Pose2d pose = poseNode->value();

  const Covariances& covariances = slam.covariances().clone();

  Covariances::node_pair_list_t node_pair_list;
  node_pair_list.push_back(make_pair(gps_nodes[0], poseNode));

  list<MatrixXd> cov_entries = covariances.access(node_pair_list);

  MatrixXd new_cov = cov_entries.back();

  nav_msgs::Odometry updatedPose;
  updatedPose.pose.pose.position.x = pose.x();
  updatedPose.pose.pose.position.y = pose.y();

  for(int ii = 0; ii < new_cov.size(); ii++) updatedPose.pose.covariance[ii] =  new_cov(ii);

  pub_slam_pose.publish(updatedPose);


  cout <<"\nPose Cov:\n" << cov << endl;
  cout << "New Pose: " << pose << endl;
  cout << "New Cov: \n" << new_cov << endl<< endl;
*/

}


void clearGraph_callback(const std_msgs::String& msg){

  cout << "\n\n" << endl;

  ROS_INFO("Clear Graph: %s\n", msg.data.c_str() );

  list<Node*> ids = slam.get_nodes();
  for (list<Node*>::iterator iter = ids.begin(); iter != ids.end(); iter++) slam.remove_node(*iter);

  list<Factor*> ids2 = slam.get_factors();
  for (list<Factor*>::iterator iter2 = ids2.begin(); iter2 != ids2.end(); iter2++) slam.remove_factor(*iter2);

  slam.print_graph();
  cout << endl << endl;
  // Clear all of the data vectors
  gps_nodes.clear();
  pose_nodes.clear();
  pose_xy.clear();
  feature_nodes.clear();
  feature_indx.clear();
  features_xy.clear();
}

// Main -----------------------------------------------------------------------------------------------------
int main(int argc, char** argv) {

  cout.precision(17);     	// Set percition on cout to show more digets

  ros::init(argc, argv, "sss_slam");
  ros::NodeHandle n;

  ros::Subscriber sub1 = n.subscribe("/gps_landmark", 5,  gpsLandmark_callback);              // GPS Data Subscriber
  ros::Subscriber sub2 = n.subscribe("/clearGraph", 5,    clearGraph_callback);               // GPS Data Subscriber
  ros::Subscriber sub3 = n.subscribe("/optimizeGraph", 1, optimize_callback);                 // GPS Data Subscriber

  message_filters::Subscriber<nav_msgs::Odometry> odom_sub(n,"/odom", 50);                    // Vehicle Position subscriber
  message_filters::Subscriber<sensor_msgs::PointCloud> landmark_sub(n,"/sss_landmark", 50);   // Landmark Subscriber
  message_filters::Subscriber<sensor_msgs::PointCloud> match_sub(n,"/sss_match", 50);         // Loop Closure Subscriber

  message_filters::TimeSynchronizer<sensor_msgs::PointCloud, nav_msgs::Odometry> sync1(landmark_sub, odom_sub, 50);   // Syncronoze Landmark and odom messages
  message_filters::TimeSynchronizer<sensor_msgs::PointCloud, nav_msgs::Odometry> sync2(match_sub, odom_sub, 50);      // Syncronize loop closure and odom messages

  sync1.registerCallback(&newLandmark_callback);                                // Callback functions
  sync2.registerCallback(&closeLoop_callback);


  // SLAM update Publisher -----------------------------------------------------
  pub_slam_pose = n.advertise<nav_msgs::Odometry>("slam_update_pose", 10);
//  pub_slam_feature = n.advertise<sensor_msgs::PointCloud>("slam_update_features", 10);

  cout << "\n\n\nsss_slam node initilaized\n\nSubscribes to:" << endl;
  cout << "\t/gps_landmark" << endl;
  cout << "\t/sss_landmark" << endl;
  cout << "\t/sss_match" << endl;
  cout << "\t/odom" << endl;
  cout << "\t/optimizeGraph" << endl;
  cout << "\t/clearGraph" << endl;

  cout << "\nPublishes: \n\tslam_update_pose \n\n"<< endl;
  ros::spin();

  return 0;
}



/*  -- Connect to first pose via GPS measurmant??? --
    Point2d gps_measure(dX, dY);    // Distance from the GPS waypoints (x, y)
    Pose2d_Point2d_Factor* gps_measurement = new Pose2d_Point2d_Factor(gps_nodes[0], feature_nodes[indx], gps_measure, noise2);
    slam.add_factor(gps_measurement);
*/

/*  -- Debugging stuff --
    cout << endl << "poseX: " << odom_msg.pose.pose.position.x << endl;
    cout << "msgX:  " << lm_msg.point.x << endl;
    cout << "dX:    " << dX << endl << endl;
    cout << "poseY: " << odom_msg.pose.pose.position.y << endl;
    cout << "msgY:  " << lm_msg.point.y << endl;
    cout << "dY:    " << dY << endl << endl;
*/


/* -- Graph output --
  int numFiles = getNumFiles("/home/jake/catkin_ws", ".graph")/2 + 1;

  char buffer[20];
  ofstream f;
  sprintf(buffer, "before%i.graph", numFiles);
  f.open (buffer); slam.write(f); f.close();

  slam.print_graph();
  cout << endl;

  sprintf(buffer, "after%i.graph",numFiles);
  f.open (buffer); slam.write(f); f.close();
*/
