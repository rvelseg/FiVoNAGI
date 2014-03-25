#ifndef INC_DATA_EXPORT_H
#define INC_DATA_EXPORT_H

#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;

// common

// this file is outdated, it must be updated according to the new structure

template <typename TTT>
void dataExport 
(TTT *devicegrid, 
 string dgname, 
 int *N, 
 int *MDX, int *MDY, 
 TTT *T) {

  int gridBitSize = NX * NY * sizeof(TTT);
  TTT *localgrid = (TTT*)malloc( gridBitSize );

  GPUGD_EC( cudaMemcpy( localgrid,
			    devicegrid,
			    gridBitSize, 
			    cudaMemcpyDeviceToHost ) );

  ostringstream Convert;
  Convert << setfill('0') << setw(4);
  Convert << (*N); 
  string Nstr = Convert.str();

  ofstream resfile;
  string filename = dgname + "-" + Nstr + ".dat";
  resfile.open(filename.c_str(), ios::trunc);

  ostringstream ConvertT;
  ConvertT << setprecision(5) << *T;
  string Tstr = ConvertT.str();
  resfile << "# T=" << Tstr << endl;
  resfile << NY << " ";
  for(int j=0; j<=NY-1; j++) {
    float y = static_cast<float>(j-(*MDY)-NY/2)/static_cast<float>(ETA);
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


  filename = dgname + "-cut-" + Nstr + ".dat";
  resfile.open(filename.c_str(), ios::trunc);
  
  resfile << "# T=" << Tstr << endl;
      
  for (int i=0; i<=NX-1; i++) {
    float x = static_cast<float>(i+(*MDX))/static_cast<float>(ETA);
    resfile << x << " ";
    int j=JCUT;
    resfile << localgrid[i+j*NX] << " ";
    resfile << " \n";
  }
  resfile.close();
}

template <typename TTT>
void u1maxExport 
(TTT *devicegrid, 
 int *MDX, int *MDY, 
 int *N, TTT *T) {

  int gridBitSize = NX * NY * sizeof(TTT);
  TTT *localgrid = (TTT*)malloc( gridBitSize );

  GPUGD_EC( cudaMemcpy( localgrid,
			    devicegrid,
			    gridBitSize, 
			    cudaMemcpyDeviceToHost ) );
  ostringstream Convert;
  Convert << setfill('0') << setw(4);
  Convert << (*N); 
  string Nstr = Convert.str();

  ostringstream ConvertT;
  ConvertT << setprecision(5) << *T;
  string Tstr = ConvertT.str();

  ofstream resfile;
  resfile.open("u1max.dat", ios::trunc);
    
  for (int i=0; i<=NX-1+(*MDX); i++) {
    float x = static_cast<float>(i)/static_cast<float>(ETA);
    resfile << x << " ";
    resfile << localgrid[i+0*NX] << " ";
    resfile << " \n";
  }
  resfile.close();

  static float cfl_std_tmp;
  filename = "u1-cfl.dat";
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
  resfile << localgrid[0+100*NX] << " ";
  resfile << pow(cfl_std_tmp/((*N)+1),0.5) << " ";
  resfile << " \n";
  resfile.close();
}

#endif  /* INC_DATA_EXPORT_H */
