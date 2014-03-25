#ifndef INC_DATA_EXPORT_H
#define INC_DATA_EXPORT_H

#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;

// hifu-beam

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
  resfile << "# Lines are parallel to y-axis, an columns parallel to x-axis." << endl;
  resfile << "# First line are y values, and first column x values." << endl;
  resfile << "# Left-top corner is the number of columns" << endl;
  // TODO: switch x-y in this file, switch it in the plot scripts too.
  resfile << NY << " ";
  for(int j=0; j<=NY-1; j++) {
    float y = static_cast<float>(j-(*MDY)-NY/2)/static_cast<float>(ISPF);
    resfile << y << " ";
  }  
  resfile << " \n";
      
  for (int i=0; i<=NX-1; i++) {
    float x = static_cast<float>(i+(*MDX))/static_cast<float>(ISPF);
    resfile << x << " ";
    for (int j=0; j<=NY-1; j++) {
      resfile << localgrid[i+j*NX] << " ";
    }
    resfile << " \n";
  }
  resfile.close();

  filename = deployPath + "/" 
    + dgname + "-cut-" + Nstr + ".dat";
  resfile.open(filename.c_str(), ios::trunc);
  
  resfile << "# T=" << Tstr << endl;
  resfile << "# AMPL=" << AMPL << endl;
  resfile << "# x u1(x,y_cut)" << endl;
      
  for (int i=0; i<=NX-1; i++) {
    float x = static_cast<float>(i+(*MDX))/static_cast<float>(ISPF);
    resfile << x << " ";
    int j=JCUT;
    resfile << localgrid[i+j*NX] << " ";
    resfile << " \n";
  }
  resfile.close();

  // Export u1Max for the cut :

  GPUGD_EC( cudaMemcpy( localgrid,
			    measure1,
			    gridBitSize, 
			    cudaMemcpyDeviceToHost ) );

  filename = deployPath + "/u1max.dat";
  resfile.open(filename.c_str(), ios::trunc);
  resfile << "# AMPL=" << AMPL << endl;
  resfile << "# x u1max(x,y_cut)" << endl;
 
  int imax=NX-1+(*MDX);
  for (int i=0; i<=imax; i++) {
    float x = static_cast<float>(i)/static_cast<float>(ISPF);
    resfile << x << " ";
    resfile << localgrid[i+1*NX] << " ";
    resfile << " \n";
  }
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
  cfl_std_tmp += pow(localgrid[0+0*NX] - CFLWHISH,2);
  resfile << (*N) << " ";
  resfile << Tstr << " ";
  resfile << localgrid[0+0*NX] << " ";
  resfile << pow(cfl_std_tmp/((*N)+1),0.5) << " ";
  resfile << " \n";
  resfile.close();
}

#endif  /* INC_DATA_EXPORT_H */
