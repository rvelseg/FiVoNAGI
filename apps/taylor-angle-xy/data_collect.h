#ifndef INC_DATA_COLLECT_H
#define INC_DATA_COLLECT_H

// taylor-angle-xy

template <typename TTT>
__global__ void dataCollect
(TTT *measure1, 
 TTT *T,  TTT *cfl, 
 int *n,  int *frame, int *n0frame,
 TTT *dx, TTT *dy, 
 int *MDX,  int *MDY,
 int *simError,
 bool dstOut) {

  // since the cfl grid is no longer used, it could be used here
  // instead of measure1, but first read cfl[0]

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = i + j * blockDim.x * gridDim.x;

  // this is to avoid writting outside the arrays
  if (i > NX-1 || j > NY-1) return;

  // this initialization should be in device_var_init function
  if ( *T <= 0 ) {
    measure1[offset] = 0.0;
    return;
  } 

  // collect CFL and time
  // n_f numbers time steps already taken in this frame.
  int n_f = (*n) - 1 - (*n0frame);
  if (n_f < NX) {
    if (i == 0) {
      measure1[n_f + 0*NX] = (*T);
      measure1[n_f + 1*NX] = cfl[0]; // CFL value
    }
  } else { *simError = 4; return; }

  // collect u1 over a cut in the direction of the propagation

  int icenter = static_cast<int>((*T)*MDVX/(*dx))-(*MDX);
  int jcenter = (NY/2) 
    + static_cast<int>(((*T)-MDT)*MDVY/(*dy))
    -(*MDY);

  if( // if i is in the range where the error will be measured
     abs(i-icenter) <= static_cast<int>(5.0*ETA*cos(ISPTHETA))
      ) {
    // interpolate u1 for ja using j and a neighbour
    float ja;
    ja = static_cast<float>(jcenter) 
      + static_cast<float>(i-icenter)
      *((*dx)/(*dy))*tan(ISPTHETA);
    float dj = j - ja;
    TTT measure_tmp;
    if( // j is just over ja
       dj>=0 && dj <= 0.5 ) {
      int jm1 = j-1;
      float djm = jm1 - ja;
      if (dstOut) {
	measure_tmp = -djm*texAu1_read(&i,&j) 
	  + dj*texAu1_read(&i,&jm1);
      } else {
	measure_tmp = -djm*texBu1_read(&i,&j) 
	  + dj*texBu1_read(&i,&jm1);
      }
    } else if ( // j is just below ja
	       dj<0 && -dj < 0.5 ) {
      int jp1 = j+1;
      float djp = jp1 - ja;
      if (dstOut) {
	measure_tmp = -dj*texAu1_read(&i,&jp1)
	  + djp*texAu1_read(&i,&j);
      } else {
	measure_tmp = -dj*texBu1_read(&i,&jp1)
	  + djp*texBu1_read(&i,&j);
      }
    } else {
      return;
    } 
    
    int iwrite = i 
      - (icenter
	 - static_cast<int>(5.0*ETA*cos(ISPTHETA)));

    if(iwrite >= 0) {
      measure1[iwrite + 2*NX] = measure_tmp; // numeric solution
      measure1[iwrite + 3*NX] = ISPRHOA
	* tanh(ISPC*(static_cast<float>(i+(*MDX))*(*dx)*cos(ISPTHETA)
		     + (static_cast<TTT>(j+(*MDY)-NY/2)*(*dy)+MDT*sin(ISPTHETA))
		     *sin(ISPTHETA)
		     - (*T))); // analytic solution
    }
  }
}

#endif  /* INC_DATA_COLLECT_H */
