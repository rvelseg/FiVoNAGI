//=======================================================================
//
// Name : Finite Volume Nonlinear Acoustics GPU Implementation (FiVoNAGI)
//
// Authors : Roberto Velasco Segura and Pablo L. Rend\'on
//
// License : see licence.txt in the root directory of the repository.
//
//=======================================================================

// use 1 for single precision and 2 for double precision
#ifndef PRECISION
#define PRECISION 1
#endif /* PRECISION */

#if PRECISION == 1 
#define DATATYPEV float
#define DATATYPET float
#elif PRECISION == 2
#define DATATYPEV double
#define DATATYPET int2
#else /* PRECISION value */
# error unresolved PRECISION value
#endif /* PRECISION value */

#ifndef UNINCLUDE
#include "numbers.h"
#include "parameters.h"
#include "data_definitions.h"
#include "init.h"
#include "boundary.h"
#include "draw_float_cut.h"
#include "data_export.h"
#include "data_collect.h"
#include "source.h"
#include "fv.h"

// cuda
#include "cuda.h"
#include "../nv/cpu_anim.h"

// system
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <math.h>
#include <time.h> // just for the sleep function
#include <iomanip>
#endif /* UNINCLUDE */

#ifndef DEBUG
#define DEBUG 0
#endif /* DEBUG */
#include "debug_tools.h"

using namespace std;

// TODO: place the following two functions in a separate file. This is
// needed to include this file as dependence of other h files.
template <typename TTT>
__forceinline__ __device__ TTT getmax(TTT x, TTT y)
{ return (x > y)?x:y; }

template <typename TTT>
__forceinline__ __device__ TTT getmin(TTT x, TTT y)
{ return (x < y)?x:y; }

template <typename TTT>
__global__ void texCpy
(TTT *u1W, TTT *u2W, TTT *u3W, bool dstOut) {
  
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = i + j * blockDim.x * gridDim.x;

  // this is to avoid writting outside the arrays
  if (i > NX-1 || j > NY-1) return;

  if (dstOut) {
    u1W[offset] = texAu1_read(&i,&j); //CB:rho1
    u2W[offset] = texAu2_read(&i,&j);
    u3W[offset] = texAu3_read(&i,&j);
  } else {
    u1W[offset] = texBu1_read(&i,&j); //CB:rho1
    u2W[offset] = texBu2_read(&i,&j);
    u3W[offset] = texBu3_read(&i,&j);
  }
}

template <typename TTT>
__global__ void restore
(TTT *u1W, TTT *u2W, TTT *u3W) {
  
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = i + j * blockDim.x * gridDim.x;

  // this is to avoid writting outside the arrays
  if (i > NX-1 || j > NY-1) return;

  u1W[offset] = texCu1_read(&i,&j); //CB:rho1
  u2W[offset] = texCu2_read(&i,&j);
  u3W[offset] = texCu3_read(&i,&j);
}

template <typename TTT>
__global__ void moveDomain
(TTT *u1W, TTT *u2W, TTT *u3W, 
 TTT *dx,  int *MDX,
 TTT *dy,  int *MDY,
 TTT *T,  GPUGD_VARSFD, bool dstOut) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  // this is to avoid writting outside the arrays
  if (i > NX-1 || j > NY-1) return;

  int offset = i + j * blockDim.x * gridDim.x;

  int MDi, MDj;

  if ( (*T) >= MDT ) {
    // this value is calculated on every thread with the
    // same result, could be calculated just once but that
    // would require an extra device variable MDi, in that
    // case it could be done in the updateMDX kernel
    // before calling this kernel.
    MDi = static_cast<int>(((*T) - MDT)*MDVX/(*dx))-(*MDX);
    MDj = static_cast<int>(((*T) - MDT)*MDVY/(*dy))-(*MDY);
  } else {
    MDi = 0;
    MDj = 0;
  }
  // even when MDi == MDj == 0 you must copy the values from one
  // texture to the other
  int iread = i + MDi;
  int jread = j + MDj;
  if (iread <= NX-1
      && jread <= NY-1) {
    if (dstOut) {
      u1W[offset] = texAu1_read(&iread,&jread); //CB:rho1
      u2W[offset] = texAu2_read(&iread,&jread);
      u3W[offset] = texAu3_read(&iread,&jread);
    } else {
      u1W[offset] = texBu1_read(&iread,&jread); //CB:rho1
      u2W[offset] = texBu2_read(&iread,&jread);
      u3W[offset] = texBu3_read(&iread,&jread);
    }
  } else {
    if (dstOut) {
      u1W[offset] = texAu1_read(&i,&j); //CB:rho1
      u2W[offset] = texAu2_read(&i,&j);
      u3W[offset] = texAu3_read(&i,&j);
    } else {
      u1W[offset] = texBu1_read(&i,&j); //CB:rho1
      u2W[offset] = texBu2_read(&i,&j);
      u3W[offset] = texBu3_read(&i,&j);
    }
  }
}

template <typename TTT>
__global__ void updateMDXY
(int *MDX, TTT *dx, 
 int *MDY, TTT *dy, 
 TTT *T) {
  if ( (*T) >= MDT ) {
    int MDi = static_cast<int>(((*T) - MDT)*MDVX/(*dx))-(*MDX);
    *MDX = (*MDX) + MDi;
    int MDj = static_cast<int>(((*T) - MDT)*MDVY/(*dy))-(*MDY);
    *MDY = (*MDY) + MDj;
  }
}

// this is a way, maybe bad one, to make a reduction over cfl
template <typename TTT>
__global__ void reduceCFL
(TTT *cfl, GPUGD_VARSFD) { 

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int sizered = blockDim.x * gridDim.x;

  if ( i >= 0 && i <= NX * NY - 1 ) {

    cfl[i] = getmax(cfl[i], cfl[i + sizered]);

#if DEBUG == 1
    // if ( cfl[i] > CFLMAX ) { 
    //   *debug1 = cfl[i];
    // }
#endif /* DEBUG */
  }
}

// this is the final step of the reduction.
template <typename TTT>
__global__ void reduceCFL2
(TTT *cfl, TTT *dt, TTT *T, 
 int *simError, GPUGD_VARSFD) {

  TTT cflStep = cfl[0];

  if(cflStep < CFLMAX) {
//    *simError = 0;
  } else {
    *simError = 1;
  }
}

template <typename TTT>
__global__ void update_dt
(TTT *cfl, TTT *dt, 
 GPUGD_VARSFD) {
  TTT cflStep = cfl[0];
  *dt = (*dt) * CFLWHISH / cflStep;
}

template <typename TTT>
__global__ void update_T
(TTT *dt, TTT *T, 
 GPUGD_VARSFD) {
  *T = (*T) + (*dt);
}

void sleepP
(unsigned int mseconds) {
  int ticks = mseconds * CLOCKS_PER_SEC / 1000;
  clock_t goal = ticks + clock();
  while (goal > clock());
}

template <typename TTT>
void calcFrame
( gpu_data<TTT> *gd, int nothing ) {

  GPUGD_EC( cudaEventRecord( gd->start, 0 ) );
  dim3 blocks(NX/16,NY/16);
  dim3 threads(16,16);
  CPUAnimBitmap  *bitmap = gd->bitmap;
  dim3 blocksd((NXW/ZOOM)/16,((NYW/ZOOM)+128)/16);

  float exec_time_frame;
  static bool uOut = true;
  static TTT T = -TIDE;
  static float exec_time_total;
  static int n = 0;
  static int simError = 0;
  static TTT Tprint = T + DTPRINT;
  static int frame = 0;
  
  GPUGD_EC( cudaMemcpy( gd->dev_simError,
			&simError,
			sizeof(int),
			cudaMemcpyHostToDevice ) );

  TTT *u1W;
  TTT *u2W;
  TTT *u3W;

  TTT *u1R;
  TTT *u2R;
  TTT *u3R;

  TTT dt;
  int dummyStop = 1;

#if DEBUG >= 1
  gpuGridDebuger<TTT> DDD;
  TTT debug1 = Num0;
  int gridBitSize = NX * NY * sizeof(TTT);
  TTT *debug2 = (TTT*)malloc( gridBitSize );
#endif /* DEBUG */

  if (initStop && n==0) {
    cout << "Continue? answer 0 or 1 \n"; 
    cin >> dummyStop; 
    if(dummyStop==0) {
      clean_gpu(gd);
      exit(0);
    }
  }

  // For comparison with CLAWPACK see: clawpack-4.6.1/clawpack/2d/lib/claw2.f

  while (T < Tprint) { 
    if (uOut) {
      u1W = gd->dev_SrcBu1;
      u2W = gd->dev_SrcBu2;
      u3W = gd->dev_SrcBu3;
      u1R = gd->dev_SrcAu1;
      u2R = gd->dev_SrcAu2;
      u3R = gd->dev_SrcAu3;
    } else {
      u1W = gd->dev_SrcAu1;
      u2W = gd->dev_SrcAu2;
      u3W = gd->dev_SrcAu3;
      u1R = gd->dev_SrcBu1;
      u2R = gd->dev_SrcBu2;
      u3R = gd->dev_SrcBu3;
    }

    // TODO: place this block at the end of the step calculation, and
    // check calculation is not affected.
    GPUGD_EC( cudaMemcpy( &dt,
    			      gd->dev_dt,
    			      sizeof(TTT),
    			      cudaMemcpyDeviceToHost ) );

    texCpy<<<blocks,threads>>>
      (gd->dev_SrcCu1, gd->dev_SrcCu2, gd->dev_SrcCu3, 
       !uOut);

    boundary<<<blocks,threads>>>
      (gd->dev_MDX, gd->dev_MDY, 
       u1R, u2R, u3R, 
       gd->dev_dx, gd->dev_dy, gd->dev_dt,
       gd->dev_T, 
       GPUGD_VARSFC,
       !uOut);
    // GPUGD_COUT("boundary, !uOut", !uOut);

    // GPUGD_ADD(DDD,u1R,"q1");
    // GPUGD_ADD(DDD,u2R,"q2");
    // GPUGD_ADD(DDD,u3R,"q3");

    // y-FVM

    getWavesSpeedsCFL_y<<<blocks,threads>>>
      (gd->dev_cfl, 
       gd->dev_dy, gd->dev_dt, 
       gd->dev_simError, 
       GPUGD_VARSFC,
       uOut);
    // GPUGD_COUT("getWavesSpeedsCFL_y, uOut", uOut);

    GPUGD_EC( cudaMemcpy( &simError,
    			      gd->dev_simError,
    			      sizeof(int),
    			      cudaMemcpyDeviceToHost ) );
    if (simError == 2) { 
      printf("ERROR: Negative eigenvalue.\n");
      clean_gpu(gd);
      exit(2);
    }

    // GPUGD_COUT("simError", simError);
    // GPUGD_ADD(DDD,gd->dev_s1,"s1");
    // GPUGD_ADD(DDD,gd->dev_s2,"s2");
    // GPUGD_ADD(DDD,gd->dev_s1,"s1");
    // GPUGD_ADD(DDD,gd->dev_s2,"s2");
    // GPUGD_ADD(DDD,gd->dev_W11,"----W11");
    // GPUGD_ADD(DDD,gd->dev_W21,"W21");
    // GPUGD_ADD(DDD,gd->dev_W12,"W12");
    // GPUGD_ADD(DDD,gd->dev_W22,"W22");
    // GPUGD_ADD(DDD,gd->dev_W11,"----W11");
    // GPUGD_ADD(DDD,gd->dev_W21,"W21");
    // GPUGD_ADD(DDD,gd->dev_W12,"W12");
    // GPUGD_ADD(DDD,gd->dev_W22,"W22");
    // GPUGD_ADD(DDD,gd->dev_amdq1,"--amdq1");
    // GPUGD_ADD(DDD,gd->dev_amdq2,"amdq2");
    // GPUGD_ADD(DDD,gd->dev_apdq1,"apdq1");
    // GPUGD_ADD(DDD,gd->dev_apdq2,"apdq2");
    // GPUGD_ADD(DDD,gd->dev_amdq1,"--amdq1");
    // GPUGD_ADD(DDD,gd->dev_amdq2,"amdq2");
    // GPUGD_ADD(DDD,gd->dev_apdq1,"apdq1");
    // GPUGD_ADD(DDD,gd->dev_apdq2,"apdq2");

    step_y<<<blocks,threads>>>
      (u1W, u2W, u3W, 
       gd->dev_dy, gd->dev_dt,
       GPUGD_VARSFC,
       uOut);
    // GPUGD_COUT("step_y, uOut", uOut);
    
    // GPUGD_PD1(DDD,"qadd(5,6)");
    // GPUGD_ADD(DDD,u1W,"q1step");
    // GPUGD_ADD(DDD,u2W,"q2step");
    // GPUGD_ADD(DDD,u3W,"q3step");

    calcLimiters_y<<<blocks,threads>>>
      (GPUGD_VARSFC,
       gd->dev_dy, gd->dev_dt);

    // GPUGD_ADD(DDD,gd->dev_amdq1,"fadd1");
    // GPUGD_ADD(DDD,gd->dev_amdq2,"fadd2");
    // GPUGD_ADD(DDD,gd->dev_amdq1,"fadd1");
    // GPUGD_ADD(DDD,gd->dev_amdq2,"fadd2");

    writeLimiters_y<<<blocks,threads>>>
      (u1R, u2R, u3R, 
       gd->dev_dy, gd->dev_dt,
       GPUGD_VARSFC,
       !uOut);
    // GPUGD_COUT("writeLimiters_y, !uOut", !uOut);

    // GPUGD_ADD(DDD,u1R,"q1new");
    // GPUGD_ADD(DDD,u2R,"q2new");
    // GPUGD_ADD(DDD,u3R,"q3new");

    // always check al this point the last texture write was
    // uXR, otherwise use a texCpy.

    // x-FVM

    getWavesSpeedsCFL_x<<<blocks,threads>>>
      (gd->dev_cfl, 
       gd->dev_dx, gd->dev_dt, 
       gd->dev_simError, 
       GPUGD_VARSFC,
       uOut);
    // GPUGD_COUT("getWavesSpeedsCFL_x, uOut", uOut);

    GPUGD_EC( cudaMemcpy( &simError,
    			      gd->dev_simError,
    			      sizeof(int),
    			      cudaMemcpyDeviceToHost ) );
    if (simError == 2) { 
      printf("ERROR: Negative eigenvalue.\n");
      clean_gpu(gd);
      exit(2);
    }

    // GPUGD_COUT("simError", simError);
    // GPUGD_ADD(DDD,gd->dev_s1,"s1");
    // GPUGD_ADD(DDD,gd->dev_s2,"s2");
    // GPUGD_ADD(DDD,gd->dev_s1,"s1");
    // GPUGD_ADD(DDD,gd->dev_s2,"s2");
    // GPUGD_ADD(DDD,gd->dev_W11,"----W11");
    // GPUGD_ADD(DDD,gd->dev_W21,"W21");
    // GPUGD_ADD(DDD,gd->dev_W12,"W12");
    // GPUGD_ADD(DDD,gd->dev_W22,"W22");
    // GPUGD_ADD(DDD,gd->dev_W11,"----W11");
    // GPUGD_ADD(DDD,gd->dev_W21,"W21");
    // GPUGD_ADD(DDD,gd->dev_W12,"W12");
    // GPUGD_ADD(DDD,gd->dev_W22,"W22");
    // GPUGD_ADD(DDD,gd->dev_amdq1,"--amdq1");
    // GPUGD_ADD(DDD,gd->dev_amdq2,"amdq2");
    // GPUGD_ADD(DDD,gd->dev_apdq1,"apdq1");
    // GPUGD_ADD(DDD,gd->dev_apdq2,"apdq2");
    // GPUGD_ADD(DDD,gd->dev_amdq1,"--amdq1");
    // GPUGD_ADD(DDD,gd->dev_amdq2,"amdq2");
    // GPUGD_ADD(DDD,gd->dev_apdq1,"apdq1");
    // GPUGD_ADD(DDD,gd->dev_apdq2,"apdq2");

    step_x<<<blocks,threads>>>
      (u1W, u2W, u3W, 
       gd->dev_dx, gd->dev_dt,
       GPUGD_VARSFC,
       uOut);
    // GPUGD_COUT("step_x, uOut", uOut);
    
    // GPUGD_PD1(DDD,"qadd(5,6)");
    // GPUGD_ADD(DDD,u1W,"q1step");
    // GPUGD_ADD(DDD,u2W,"q2step");
    // GPUGD_ADD(DDD,u3W,"q3step");

    calcLimiters_x<<<blocks,threads>>>
      (GPUGD_VARSFC,
       gd->dev_dx, gd->dev_dt);

    // GPUGD_ADD(DDD,gd->dev_amdq1,"fadd1");
    // GPUGD_ADD(DDD,gd->dev_amdq2,"fadd2");
    // GPUGD_ADD(DDD,gd->dev_amdq1,"fadd1");
    // GPUGD_ADD(DDD,gd->dev_amdq2,"fadd2");

    writeLimiters_x<<<blocks,threads>>>
      (u1R, u2R, u3R, 
       gd->dev_dx, gd->dev_dt,
       GPUGD_VARSFC,
       !uOut);
    // GPUGD_COUT("writeLimiters_x, !uOut", !uOut);
    
    // GPUGD_ADD(DDD,u1R,"q1new");
    // GPUGD_ADD(DDD,u2R,"q2new");
    // GPUGD_ADD(DDD,u3R,"q3new");

    /////////// reduction (maximum) over gd->dev_cfl
    int sizered = NX * NY;
    while(sizered > 32) { 
      // GPUGD_COUT("sizered :", sizered);
      sizered = sizered/2;
      reduceCFL<<<sizered/16,16>>>
	(gd->dev_cfl,
	 GPUGD_VARSFC);
      // GPUGD_PD1(DDD,"debug1");
    }
    while(sizered > 2) {
      // GPUGD_COUT("sizered :", sizered);
      sizered = sizered/2;
      reduceCFL<<<1,sizered>>>
	(gd->dev_cfl,
	 GPUGD_VARSFC);
      // GPUGD_PD1(DDD,"debug1");
    }
    // GPUGD_COUT("sizered :", sizered);
    reduceCFL2<<<1,1>>>
      (gd->dev_cfl, gd->dev_dt, 
       gd->dev_T, gd->dev_simError,
       GPUGD_VARSFC);
    // GPUGD_PD1(DDD,"cflStep");
    /////////// end of the reduction 

    GPUGD_EC( cudaMemcpy( &simError,
    			      gd->dev_simError,
    			      sizeof(bool),
    			      cudaMemcpyDeviceToHost ) );

    if ( simError == 0 ) {
      n++;

      source<<<blocks,threads>>>
      	(u1W, u2W, u3W, 
      	 gd->dev_dx, gd->dev_dy, gd->dev_dt,
	 gd->dev_MDX, gd->dev_MDY,
      	 GPUGD_VARSFC,
      	 uOut);

      // GPUGD_COUT("source, uOut", uOut);

      // GPUGD_ADD(DDD,u1W,"q1source");
      // GPUGD_ADD(DDD,u2W,"q2source");
      // GPUGD_ADD(DDD,u3W,"q3source");

      // TODO: implement GPUGD_MEASURE 

      update_T<<<1,1>>>
	(gd->dev_dt, gd->dev_T,
	 GPUGD_VARSFC);
      GPUGD_EC( cudaMemcpy( &T,
			    gd->dev_T,
			    sizeof(TTT),
			    cudaMemcpyDeviceToHost ) );

      if(frameExport) { 
	dataCollect<<<blocks,threads>>>
	  (gd->dev_measure1, 
	   gd->dev_T, gd->dev_cfl, 
	   gd->dev_dx, gd->dev_dy,
	   gd->dev_MDX, gd->dev_MDY,
	   !uOut);
      }

      moveDomain<<<blocks,threads>>>
      	(u1R, u2R, u3R, 
      	 gd->dev_dx, gd->dev_MDX,
      	 gd->dev_dy, gd->dev_MDY,
      	 gd->dev_T, GPUGD_VARSFC,
      	 !uOut);

      // GPUGD_PD1(DDD,"MDi");

      updateMDXY<<<1,1>>>
      	(gd->dev_MDX, gd->dev_dx, 
	 gd->dev_MDY, gd->dev_dy, 
	 gd->dev_T);

      // if the last kernel called writes in uXR, X=1,2,3,
      // then call the following kernell to copy values
      // to uXW.
      texCpy<<<blocks,threads>>>
      	(u1W, u2W, u3W, uOut);
    
      // GPUGD_COUT("n", n);
      GPUGD_DISPLAY(DDD,n);
      GPUGD_COUT("dt", dt);
      GPUGD_COUT("------------ good step finished", 0);
      
      uOut = !uOut;
    
      if(stepPause) sleepP(stepPause); 

    } else if ( simError == 1 ) { 
      restore<<<blocks,threads>>>
      	(u1R, u2R, u3R);
      printf( "INFO: dt too large: %20.19f \n", dt);
    } else { 
      printf( "ERROR: something went wrong, simulation error: %d \n",
	      simError );
      clean_gpu(gd);
      exit(simError);
    }
    update_dt<<<1,1>>>
      (gd->dev_cfl, gd->dev_dt, 
       GPUGD_VARSFC);
    simError = 0;
  }
  frame++;
  if(frameStop == 1) {
    cout << "Enter 0 to exit or just hit ENTER to continue: "; 
    dummyStop = 1;
    string input;
    getline( cin, input );
    if ( !input.empty() ) {
      istringstream stream( input );
      stream >> dummyStop;
    }
    if(dummyStop==0) {
      clean_gpu(gd);
      exit(0);
    }
  }
  Tprint = Tprint + DTPRINT;

  draw<<<blocksd,threads>>>
    (gd->dev_MDX, gd->dev_MDY, 
     gd->dev_draw, 
     gd->dev_dx, gd->dev_dy, 
     gd->dev_dt, gd->dev_T, 
     uOut);
  // GPUGD_COUT("draw, uOut", uOut);

  float_to_color<<<blocksd,threads>>>
    (gd->output_bitmap, gd->dev_draw);
  
  GPUGD_EC( cudaMemcpy( bitmap->get_ptr(),
  			    gd->output_bitmap,
  			    bitmap->image_size(),
  			    cudaMemcpyDeviceToHost ) );

  GPUGD_EC( cudaEventRecord( gd->end, 0 ) );
  GPUGD_EC( cudaEventSynchronize( gd->end ) );
  GPUGD_EC( cudaEventElapsedTime( &exec_time_frame,
				  gd->start, gd->end ) );
  exec_time_total += exec_time_frame;

  cout << "Execution time per frame = " << exec_time_total/static_cast<float>(frame) << " ms" << endl;
  cout << "T = " << T << ", n = " << n << ", frame = " << frame << endl;

  int MDX, MDY;
  GPUGD_EC( cudaMemcpy( &MDX,
			gd->dev_MDX,
			sizeof(int),
			cudaMemcpyDeviceToHost ) );
  GPUGD_EC( cudaMemcpy( &MDY,
			gd->dev_MDY,
			sizeof(int),
			cudaMemcpyDeviceToHost ) );

  static int vframe=0;
  if(frameExport) {
    dataExport 
      (gd->dev_measure1, u1W, u2W, u3W, &vframe, &MDX, &MDY, &T);
    vframe++;
  }

  if(finalTime > 0.0 && T > finalTime){
    cout << "final time reached" << endl;
//    cin >> dummyStop;
    clean_gpu(gd);
    exit(0);
  }
    
  GPUGD_PRINT_INT_TOKEN(PRECISION);
  GPUGD_COUT("---------- frame displayed",0);

#if DEBUG >= 1
  free( debug2 );
#endif /* DEBUG */
}

int main 
() {

  cout << "AMPL = " << AMPL << endl;
  cout << "ETA = "  << ETA  << endl;
  printf("NX = %d :: NY = %d \n", NX, NY);

  GPUGD_PRINT_INT_TOKEN(PRECISION);
  GPUGD_PRINT_STR_TOKEN(DATATYPEV);
  GPUGD_PRINT_STR_TOKEN(DATATYPET);
  GPUGD_COUT("sizeof(DATATYPEV)", sizeof(DATATYPEV) );

  cout << "Precision: ";
  if (PRECISION == 1) cout << "single" << endl;
  if (PRECISION == 2) cout << "double" << endl;

  gpu_data<DATATYPEV> gd;
  CPUAnimBitmap bitmap( NXW / ZOOM, (NYW / ZOOM) + 128, &gd );
  gd.bitmap = &bitmap;
  GPUGD_EC( cudaEventCreate( &gd.start ) );
  GPUGD_EC( cudaEventCreate( &gd.end ) );
  
  int imageSize = bitmap.image_size();
  cout << "imageSize = " << imageSize << endl;
  int gridBitSize = NX * NY * sizeof(DATATYPEV);
  cout << "gridBitSize = " << gridBitSize << endl;
  int drawSize = (NXW/ZOOM)*((NYW/ZOOM)+128)*sizeof(float);
  cout << "drawSize = " << drawSize << endl;
  
  DATATYPEV *dt = new DATATYPEV;
  DATATYPEV *dx = new DATATYPEV;
  DATATYPEV *dy = new DATATYPEV;
  DATATYPEV *T  = new DATATYPEV;
  int *MDX  = new int;
  int *MDY  = new int;
  DATATYPEV *debug1 = new DATATYPEV;
  *dt = DTINI;
  *dx = XMAX/(NX-1);
  *dy = YMAX/(NY-1);
  printf("dx = %12.5e :: dy = %12.5e \n", *dx, *dy);
  *T = - TIDE;
  *MDX = 0;
  *MDY = 0;
  
  DATATYPEV *u1 = (DATATYPEV*)malloc( gridBitSize );
  DATATYPEV *u2 = (DATATYPEV*)malloc( gridBitSize );
  DATATYPEV *u3 = (DATATYPEV*)malloc( gridBitSize );

#if DEBUG >= 1
  *debug1 = Num0;
  DATATYPEV *debug2 = (DATATYPEV*)malloc( gridBitSize );
#endif /* DEBUG */

  // initial values in cpu variables
  init
    (u1,u2,u3,
     dx,dy,dt
#if DEBUG >= 1
     ,debug1,debug2
#else
     ,none
#endif /* DEBUG */
     );

  // gpu memory allocation and initial values copy from cpu to gpu
  gpu_init
    (&gd, 
     gridBitSize, imageSize, drawSize,
     dt, dx, dy, 
     T, MDX, MDY, 
     u1, u2, u3
#if DEBUG >= 1
     ,debug1,debug2
#else
     ,none
#endif /* DEBUG */	    
     );
  
  delete dt;
  delete dx;
  delete dy;
  delete T; 
  delete MDX; 
  delete MDY; 

  free( u1 );
  free( u2 );
  free( u3 );

#if DEBUG >= 1
  delete debug1;
  free( debug2 );
#endif /* DEBUG */

  char name[] = "FiVoNAGI";

  bitmap.anim_and_exit( (void (*)(void*,int))calcFrame<DATATYPEV>,
			(void (*)(void*))clean_gpu<DATATYPEV>,
			name );
}

// The following lines are needed because clean_gpu and calcFrame
// functions are called through a pointer, then the compiler doesn't
// know which instance (of the template) to compile, so we are telling
// it.

template void clean_gpu<DATATYPEV>( gpu_data<DATATYPEV> *gd );
template void calcFrame<DATATYPEV>( gpu_data<DATATYPEV> *gd, int nothing );
