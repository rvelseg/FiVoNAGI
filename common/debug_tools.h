#ifndef INC_DEBUG_TOOLS_H
#define INC_DEBUG_TOOLS_H

#ifndef DEBUG
#define DEBUG 0
#endif /* DEBUG */

#if DEBUG >= 1
#define GPUGD_PRINT_INT_TOKEN(token) printf(#token " is %d\n", token)
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define GPUGD_PRINT_STR_TOKEN(token) printf(#token " is %s\n", TOSTRING(token))
#define GPUGD_ADD(inst,p,str) inst.add(p,str)
#define GPUGD_COUT(label,value) cout << label << " : " <<  value << endl
#define GPUGD_VARSFC gd->dev_debug1, gd->dev_debug2
#define GPUGD_PD1(inst,label) inst.print1(gd->dev_debug1,label)
#define GPUGD_DISPLAY(inst,n) inst.display(n)
#define GPUGD_VARSFD TTT *debug1, TTT *debug2
#else /* DEBUG >= 1 */
#define GPUGD_PRINT_INT_TOKEN(token)
#define GPUGD_PRINT_STR_TOKEN(token)
#define GPUGD_ADD(inst,p,str) 
#define GPUGD_COUT(label,value)
#define GPUGD_VARSFC none
static const bool none = true; 
#define GPUGD_PD1(inst,label)
#define GPUGD_DISPLAY(inst,n) 
#define GPUGD_VARSFD bool none
#endif /* DEBUG >= 1 */

#ifndef UNINCLUDE
// cuda
#include "../nv/book.h"

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <cstdio>
#endif /* UNINCLUDE */

using namespace std;

__inline __host__ void gpuErrorCheck
(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUGD_EC: %s %s %d\n", 
	      cudaGetErrorString(code),
	      file, line);
      if (abort) exit( EXIT_FAILURE );
   }
}

#define GPUGD_EC( ans ) ( gpuErrorCheck((ans), __FILE__, __LINE__ ))

template <typename TTT>
class gpuGridDebuger {
  int gridBitSize;
  TTT *debug2;
  vector<vector<vector<TTT> > > debugDisplay;
  int jdmin;
  int jdmax;
  int idmin;
  int idmax;
  int layersd;
  vector<string> dDlabels;
  TTT debug1;
  
 public:

  gpuGridDebuger () {
    gridBitSize = NX * NY * sizeof(TTT);
    debug2 = (TTT*)malloc( gridBitSize );
    jdmin = 8-4;
    jdmax = 8;
    idmin = 0;
    idmax = 8;
    layersd = 0;
    debug1 = Num0;
  }

  ~gpuGridDebuger () {
  free( debug2 );
  }

  void add (TTT *ref,string label) {
    GPUGD_EC( cudaMemcpy( debug2,
    			      ref,
    			      gridBitSize, 
    			      cudaMemcpyDeviceToHost ) );
    debugDisplay.push_back( vector<vector<TTT> >() );
    dDlabels.push_back(label);
    for (int j=0; j<=jdmax-jdmin; j++) {
      debugDisplay[layersd].push_back ( vector<TTT>() );
      for (int i=0; i<=idmax-idmin; i++) {
	debugDisplay[layersd][j].push_back ( debug2[(i+idmin)
						    +(j+jdmin)*NX] );
      }
    }
    layersd++; 
  }

  void print1 (TTT *ref, string label) {
    GPUGD_EC( cudaMemcpy( &debug1,
			      ref,
			      sizeof(TTT),
			      cudaMemcpyDeviceToHost ) );
    cout << label << " : " << debug1 << endl;
  }

  void display (int n) {
    // n is the numeration for the timestep.
    for (int j=0; j<=jdmax-jdmin; j++) {
      for (int layer=0; layer<=layersd-1; layer++) {
	cout << setw(8) << dDlabels[layer] << ":";
	for (int i=0; i<=idmax-idmin; i++) {
	  printf( " %+4.3e", 
		  debugDisplay[layer][j][i]);
	}
	printf("\n");
      }
      if (layersd>=1) {
	cout << setw(8) << "indices" << ":";
	for (int i=0; i<=idmax-idmin; i++) {
	  printf("   %02i,%02i,%02i" , i+idmin,j+jdmin,n); 
	}
	printf("\n");
      }
    }
    debugDisplay.clear();
    layersd = 0;
  }
};

// TODO: use the following code to implement a GPUGD_MEASURE function

      // this is to measure the wave legth, amplitude and mean at the
      // right border
      // {
      // 	TTT u1Min = 1.0;
      // 	TTT u1Max = 1.0;

      // 	int firstCero = -1;
      // 	int secondCero = -1;
      // 	int thirdCero = -1;

      // 	GPUGD_EC( cudaMemcpy( debug2,
      // 				  u1W,
      // 				  gridBitSize, 
      // 				  cudaMemcpyDeviceToHost ) );

      // 	for (int i=NX-1; i>=0; i--) {
      // 	  if ( firstCero == -1 ) {
      // 	    if ( (debug2[i+(JCUT)*NX]-Num1) *
      // 		 (debug2[i+1+(JCUT)*NX]-Num1) < Num0 ) {	     
      // 	      firstCero = i;
      // 	    }
      // 	  } else if ( thirdCero == -1 ) {
      // 	    if ( debug2[i+(JCUT)*NX] < u1Min ) {
      // 	      u1Min = debug2[i+(JCUT)*NX];
      // 	    }
      // 	    if ( debug2[i+(JCUT)*NX] > u1Max ) {
      // 	      u1Max = debug2[i+(JCUT)*NX];
      // 	    }
      // 	    if ( secondCero == -1 ) {
      // 	      if ( (debug2[i+(JCUT)*NX]-Num1) *
      // 		   (debug2[i+1+(JCUT)*NX]-Num1) < Num0 ) {	     
      // 		secondCero = i;
      // 	      }
      // 	    } else if ( thirdCero == -1 &&
      // 			(debug2[i+(JCUT)*NX]-Num1) *
      // 			(debug2[i+1+(JCUT)*NX]-Num1) < Num0 ) {	     
      // 	      thirdCero = i;
      // 	    }
      // 	  } else {
      // 	  break;
      // 	  }
      // 	}
      // 	if ( firstCero != -1 && 
      // 	     secondCero != -1 &&
      // 	     thirdCero != -1 ) {
      // 	  cout << thirdCero << "," << secondCero << "," << firstCero << endl;
      // 	  cout << "Amplitude: " << u1Max - u1Min << endl;
      // 	  cout << "Wavelength: " << -(thirdCero-firstCero)*XMAX/(NX-1) << endl;
      // 	  cout << "Mean: " <<  ( u1Max + u1Min )/ 2.0 << endl;
      // 	}
      // }



#endif /* INC_DEBUG_TOOLS_H */
