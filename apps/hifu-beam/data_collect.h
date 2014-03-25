#ifndef INC_DATA_COLLECT_H
#define INC_DATA_COLLECT_H

// hifu-beam

template <typename TTT>
__global__ void dataCollect
(TTT *measure1, 
 TTT *T,  TTT *cfl, 
 TTT *dx, TTT *dy, 
 int *MDX,  int *MDY,
 bool dstOut) {

  // since the cfl grid is no longer used, it could be used here,
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

  // we are measuring just over the beam axis
  if (j != JCUT) return;

  // to restrict ourselves to meassure the central peak
  if ( abs( static_cast<TTT>(i + (*MDX))/ETA
	    - (*T) )
       > 0.5 ) return;

  TTT u1;
  if (dstOut) {
    u1 = texAu1_read(&i,&j); //CB:rho1
  } else {
    u1 = texBu1_read(&i,&j); //CB:rho1
  }
  
  // this could use several lines because of the *MDX value 
  int offsetrw = i + (*MDX) + 1*NX;
  measure1[offsetrw] = getmax(u1, measure1[offsetrw]);

  // write in some place safe, not used above
  measure1[0 + 0*NX] = cfl[0]; // CFL value
}

#endif  /* INC_DATA_COLLECT_H */
