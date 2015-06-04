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
 cufftComplex *spectrum,
 int *frame, int *n, int *MDX, int *MDY, 
 TTT *T, int status) {

  // status variable indicates the contents of the spectum variable as
  // follows:
  // 0 - nothing, before applying the fft
  // 1 - fft of u1 before filtering
  // 2 - fft of u1 after filtering, but before applying the inverse fft
  // 3 - filtered version of u1, that is, after applying the inverse fft

  if ( status == 0 ) {

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
    Convert << (*frame); 
    string frameStr = Convert.str();

    ofstream resfile; 
    string filename = deployPath + "/" 
      + dgname + "-" + frameStr + ".dat";
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
      resfile << "\n";
    }
    resfile.close();

    // Export u1 for the cut :

    filename = deployPath + "/" 
      + dgname + "-cut-" + frameStr + ".dat";
    resfile.open(filename.c_str(), ios::trunc);
  
    resfile << "# T=" << Tstr << endl;
    resfile << "# AMPL=" << AMPL << endl;
    resfile << "# x u1(x,y_cut)" << endl;
      
    for (int i=0; i<=NX-1; i++) {
      float x = static_cast<float>(i+(*MDX))/static_cast<float>(ISPF);
      resfile << x << " ";
      int j=JCUT;
      resfile << localgrid[i+j*NX] << " ";
      resfile << "\n";
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
      resfile << localgrid[i+6*NX];
      resfile << "\n";
    }
    resfile.close();

    // Export microphones

    static int previous_n = 0;
    string i_m_str;
    for (int i_m=1; i_m<=4; i_m++) {
      i_m_str = static_cast<ostringstream*>( &(ostringstream() << i_m) )->str();
      filename = deployPath + "/microphone_" 
	+ i_m_str + ".dat";
      if ((*frame)==0) {
	resfile.open(filename.c_str(), ios::trunc);    
	resfile << "# n T u1" << endl;
      } else {
	resfile.open(filename.c_str(), ios::app);    
      }
      resfile << "## frame = " << (*frame) << "\n";
      resfile << "## T_final = " << Tstr << "\n";
      for (int i_T=0; i_T<(*n)-previous_n; i_T++) {
	resfile << i_T + previous_n + 1;
	resfile << " ";
	resfile << localgrid[i_T+0*NX];
	resfile << " ";
	resfile << localgrid[i_T+(i_m+1)*NX];
	resfile << "\n";
      }
      resfile.close();
    }

    // Export cfl values :

    static float cfl_std_tmp;
    static int first = 1;
    filename = deployPath + "/u1-cfl.dat";
    if ((*frame)==0) {
      resfile.open(filename.c_str(), ios::trunc);    
      resfile << "# n T frame cfl std" << endl;
      cfl_std_tmp = 0;
    } else {
      resfile.open(filename.c_str(), ios::app);    
    }
    for (int i_T=0; i_T<(*n)-previous_n; i_T++) {
      if ( localgrid[i_T+0*NX] > 0 ) {
	if ( first == 0 ) {
	  cfl_std_tmp += pow(localgrid[i_T+1*NX] - CFLWHISH,2); 
	  resfile << i_T + previous_n + 1;
	  resfile << " ";
	  resfile << localgrid[i_T+0*NX];
	  resfile << " ";
	  resfile << (*frame);
	  resfile << " ";
	  resfile << localgrid[i_T+1*NX];
	  resfile << " ";
	  resfile << pow(cfl_std_tmp/((*frame)+1),0.5);
	  resfile << "\n";
	}
	first = 0;
      }
    }
    resfile.close();

    free (localgrid);
    previous_n = (*n);
  }
}

#endif  /* INC_DATA_EXPORT_H */
