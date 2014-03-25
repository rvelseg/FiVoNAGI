#ifndef INC_DATA_EXPORT_H
#define INC_DATA_EXPORT_H

#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;

// taylor-angle-xy

template <typename TTT>
void dataExport 
(TTT *measure1,
 TTT *u1, TTT *u2, TTT *u3, 
 int *N, int *MDX, int *MDY, 
 TTT *T) {

  int gridBitSize = NX * NY * sizeof(TTT);
  TTT *localgrid = (TTT*)malloc( gridBitSize );
  string dgname = "u1";

  // Export u1 for the whole domain :

  GPUGD_EC( cudaMemcpy( localgrid,
			    u1,
			    gridBitSize, 
			    cudaMemcpyDeviceToHost ) );

  ostringstream Convert;
  Convert << setfill('0') << setw(4);
  Convert << (*N); 
  string Nstr = Convert.str();

  ofstream resfile;
  string filename = deployPath + "/" 
    + dgname + "-" + Nstr + ".dat";
  resfile.open(filename.c_str(), ios::trunc);

  ostringstream ConvertT;
  ConvertT << setprecision(5) << *T;
  string Tstr = ConvertT.str();
  resfile << "# T=" << Tstr << endl;
  resfile << NY << " ";
  for(int j=0; j<=NY-1; j++) {
    float y = static_cast<float>(j+(*MDY)-NY/2)/static_cast<float>(ETA);
    resfile << y << " ";
  }  
  resfile << " \n";
      
  for (int i=0; i<=NX-1; i++) {
    float x = static_cast<float>(i+(*MDX))/static_cast<float>(ETA);
    resfile << x << " ";
    for (int j=0; j<=NY-1; j++) {
      resfile << localgrid[i+j*NX] << " ";
    }
    resfile << " \n";
  }
  resfile.close();

  // Export u1 (numeric and analytic) for the cut :

  GPUGD_EC( cudaMemcpy( localgrid,
			    measure1,
			    gridBitSize, 
			    cudaMemcpyDeviceToHost ) );

  filename = deployPath  
    + "/u1-cut-" + Nstr + ".dat";
  resfile.open(filename.c_str(), ios::trunc);

  resfile << "# T=" << Tstr << endl;
  resfile << "# i xi numeric analitic \n";
  int length = 2*static_cast<int>(5.0*ETA*cos(ISPTHETA));    
  float x, error_tmp1, error_tmp2;
  error_tmp1 = 0;
  error_tmp2 = 0;
  for (int i=0; i<=length-1; i++) {
    x = - 5.0 
      + static_cast<float>(i)/(static_cast<float>(ETA)*cos(ISPTHETA));
    resfile << i << " ";
    resfile << x << " ";
    resfile << localgrid[i+0*NX] << " "; // numeric
    resfile << localgrid[i+1*NX] << " "; // analytic
    resfile << " \n";
    error_tmp1 += pow(localgrid[i+1*NX] - localgrid[i+0*NX],2);
    error_tmp2 += pow(localgrid[i+1*NX],2);
  }
  resfile.close();
  error_tmp1 = pow(error_tmp1,0.5);
  error_tmp2 = pow(error_tmp2,0.5);

  // Export measured errors :

  filename = deployPath + "/u1-error.dat";
  if ((*N)==0) {
    resfile.open(filename.c_str(), ios::trunc);    
    resfile << "# N T error" << endl;
  } else {
    resfile.open(filename.c_str(), ios::app);    
  }
  resfile << (*N) << " ";
  resfile << Tstr << " ";
  resfile << error_tmp1 / error_tmp2 << " ";
  resfile << " \n";
  resfile.close();

  // Export cfl values :

  static float cfl_std_tmp;
  filename = deployPath + "/u1-cfl.dat";
  if ((*N)==0) {
    resfile.open(filename.c_str(), ios::trunc);    
    resfile << "# N T cfl std" << endl;
    cfl_std_tmp = 0;
  } else {
    resfile.open(filename.c_str(), ios::app);    
  }
  cfl_std_tmp += pow(localgrid[0+2*NX] - CFLWHISH,2);
  resfile << (*N) << " ";
  resfile << Tstr << " ";
  resfile << localgrid[0+2*NX] << " ";
  resfile << pow(cfl_std_tmp/((*N)+1),0.5) << " ";
  resfile << " \n";
  resfile.close();
}

#endif  /* INC_DATA_EXPORT_H */
