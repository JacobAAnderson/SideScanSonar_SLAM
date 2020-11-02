
#include <isam/isam.h>
#include "isam/Anchor.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <bits/stdc++.h>

#define pi 3.141592653589

using namespace std;
using namespace isam;
using namespace Eigen;

const int isam::Pose2d::dim;



bool ParsText(const char * fileName, vector<float> &arg1,  vector<float> &arg2, vector<float> &arg3, vector<float> &arg4, vector<float> &arg5,
                                     vector<float> &arg6,  vector<float> &arg7, vector<float> &arg8, vector<float> &arg9, vector<float> &arg10,
                                     vector<float> &arg11, vector<float> &arg12 )
  {

  float a, b, c, d, e, f, g, h, i, j, k, l;


  std::fstream stream;
  stream.open(fileName);

  std::string line;

  int ind = 0;
  while (std::getline(stream, line)) {           // reads "4 4" into `line1`


  	std::stringstream  lineStream(line); // c

  	lineStream >> a >> b >> c >> d >> e >> f >> g >> h >> i >> j >> k >> l;

  	arg1 .push_back(a);
    arg2 .push_back(b);
    arg3 .push_back(c);
    arg4 .push_back(d);
    arg5 .push_back(e);
    arg6 .push_back(f);
    arg7 .push_back(g);
    arg8 .push_back(h);
    arg9 .push_back(i);
    arg10.push_back(j);
    arg11.push_back(k);
    arg12.push_back(l);

/*
  	std::cout  << a << ',';
  	std::cout  << b << ',';
  	std::cout  << c << ',';
  	std::cout  << d << ',';
  	std::cout  << e << ',';
    std::cout  << f << ',';
    std::cout  << g << ',';
    std::cout  << h << ',';
    std::cout  << i << ',';
    std::cout  << j << ',';
    std::cout  << k << ',';
    std::cout  << l << '\n';
*/

  	ind++;

  	}

  stream.close();


  arg1.shrink_to_fit();
  arg2.shrink_to_fit();
  arg3.shrink_to_fit();
  arg4.shrink_to_fit();
  arg5.shrink_to_fit();
  arg6.shrink_to_fit();
  arg7.shrink_to_fit();
  arg8.shrink_to_fit();
  arg9.shrink_to_fit();
  arg10.shrink_to_fit();
  arg11.shrink_to_fit();
  arg12.shrink_to_fit();

  return 1;
}


bool writeTxt(const char* fileName, float arg1[]){

  FILE * myfile;

  myfile = fopen(fileName,"w");


  cout << "\nCounts: " << sizeof(arg1)/sizeof(float) << endl;

  for(int ii=0; ii< sizeof(arg1)/sizeof(float); ii++) {

    fprintf(myfile,"%f \n",arg1[ii]);

    }

  fclose(myfile);

  return 1;
}









int main() {

cout.precision(17);     	// Set percition on cout to show more digets

std::cout << "\n\n\n--------------------------------------------------------------------------------------------------\n\n";

// Get Data ======================================================================================================================
const char dataFile[] = "/home/jake/Desktop/EcomapperData.txt";

vector<float> poseX, poseY, heading, poseErrX, poseErrY, headingErr, bathyX, bathyY, bathErrX, bathErrY, altX, altY;

bool read = ParsText(dataFile, poseX, poseY, heading, poseErrX, poseErrY, headingErr, bathyX, bathyY, bathErrX, bathErrY, altX, altY);

// display_vector(Ygps);

// return 1;

float xOdom[poseX.size()], yOdom[poseX.size()], pose_bathX[poseX.size()], pose_bathY[poseX.size()];


// std::cout << "odometry:\n";

xOdom[0] = 0.;
yOdom[0] = 0.;

pose_bathX[0]  = 0.;
pose_bathY[0]  = 0.;

for( int ii = 1; ii<poseX.size(); ii++ ){

	xOdom[ii] = poseX[ii] - poseX[ii-1];    // Distance between the poses
	yOdom[ii] = poseY[ii] - poseY[ii-1];

  pose_bathX[ii] = altX[ii] - poseX[ii];  // Distance from altimiter reading to the bathymetry feature
  pose_bathY[ii] = altY[ii] - poseY[ii];

  std::cout << xOdom[ii-1] << ' ' << yOdom[ii-1] << '\t\t' << pose_bathX[ii-1] << ' ' << pose_bathX[ii-1] << '\n';

  }

// const char outputFile[] = "/home/jake/Desktop/isamOut.txt";
// bool write = writeTxt(outputFile, xOdom);


// Initialize ISAM ============================================================================================================
cout << "\n\nISAM!!\n\n";

// instance of the main class that manages and optimizes the pose graph
Slam slam;

// locally remember poses
vector<Pose2d_Node*> pose_nodes;

Noise noNoise3 = Information(0.01 * eye(3));
Noise noNoise2 = Information(0.01 * eye(2));
Noise noise3   = Information(0.1 * eye(3));
Noise noise2   = Information(100. * eye(2));


Pose2d_Node* gps_node0 = new Pose2d_Node();   // create a first pose (a node)
slam.add_node(gps_node0);                     // add it to the graph

pose_nodes.push_back(gps_node0);                // also remember it locally


// Pose2d origin(poseX[0], poseY[0], heading[0]);                             // create a prior measurement (a factor)
Pose2d origin(poseX[0], poseY[0], 0.0);
Pose2d_Factor* prior = new Pose2d_Factor(gps_node0, origin, noNoise3);
slam.add_factor(prior);                                                       // add it to the graph


// Create Graph =====================================================================================================-
// std::cout << "Creating Graph" << endl;

for (int ii=1; ii<sizeof(xOdom)/sizeof(float); ii++) {

  // next pose --------------------------------------------------------------------------------------------------------
  Pose2d_Node* new_pose_node = new Pose2d_Node();
  slam.add_node(new_pose_node);
  pose_nodes.push_back(new_pose_node);

  // connect to previous with odometry measurement
  Pose2d odometry(xOdom[ii], yOdom[ii], 0.); // x,y,theta

  noise3 = Information( (poseErrX[ii] * poseErrX[ii] + poseErrY[ii] * poseErrY[ii]) * eye(3));

  Pose2d_Pose2d_Factor* constraint = new Pose2d_Pose2d_Factor(pose_nodes[ii-1], new_pose_node, odometry, noNoise3);
  slam.add_factor(constraint);



  // create a landmark -----------------------------------------------------------------------------------------------
  Point2d_Node* new_landmark = new Point2d_Node();
  slam.add_node(new_landmark);

  // Add Absoulte posiiton of land mark
  Point2d gps_measure(bathyX[ii] - poseX[0], bathyY[ii] - poseY[0]); // x,y
  Pose2d_Point2d_Factor* gps_measurement = new Pose2d_Point2d_Factor(gps_node0, new_landmark, gps_measure, noNoise2);
  slam.add_factor(gps_measurement );


  // Connect the pose and the landmark by a measurement
  Point2d vehicle_measure(pose_bathX[ii], pose_bathY[ii]); // x,y
  noise2 = Information( (bathErrX[ii] * bathErrX[ii] + bathErrY[ii] * bathErrY[ii]) * eye(2));
  Pose2d_Point2d_Factor* vehicle_measurement = new Pose2d_Point2d_Factor(new_pose_node, new_landmark, vehicle_measure, noise2);
  slam.add_factor(vehicle_measurement );


  }


std::cout << endl << "Full Graph Before:" << endl;  // printing the complete graph
slam.write(cout);

std::cout <<"\n\n\n";

slam.batch_optimization();   // optimize the graph


std::cout << endl << "Optimized Graph:" << endl;  // printing the complete graph
slam.write(cout);


std::cout <<"\n\n\n\n";

return 0;
}
