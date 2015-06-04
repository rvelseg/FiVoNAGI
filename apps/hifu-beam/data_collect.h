#ifndef INC_DATA_COLLECT_H
#define INC_DATA_COLLECT_H

// hifu-beam

template <typename TTT>
__global__ void dataCollect
(TTT *measure1, 
 TTT *T,  TTT *cfl, 
 int *n,  int *frame, int *n0frame,
 TTT *dx, TTT *dy, 
 int *MDX,  int *MDY,
 int *simError,
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

  TTT u1;
  if (dstOut) {
    u1 = texAu1_read(&i,&j); //CB:rho1
  } else {
    u1 = texBu1_read(&i,&j); //CB:rho1
  }

  // microphones:
  // pressure for a single point in multiple time values are stored in
  // a single line of meassure1. The number of time steps for a single
  // frame must not exceed NX.

  // n_f numbers time steps already taken in this frame.
  int n_f = (*n) - 1 - (*n0frame);

  if (n_f < NX) {

    int mic1_pos = round(static_cast<float>(ISPF)*0.5f);
    int mic2_pos = round(static_cast<float>(ISPF)*0.9f);
    int mic3_pos = ISPF;
    int mic4_pos = round(static_cast<float>(ISPF)*1.1f);

    if (i + (*MDX) == mic1_pos ) {
      if (n_f == 0) 
      	for(int ii=0; ii<NX; ii++) 
	  measure1[ii + 2*NX] = 0.0; 
      measure1[n_f + 2*NX] = u1;
    } else if (i + (*MDX) == mic2_pos ) {
      if (n_f == 0) 
      	for(int ii=0; ii<NX; ii++)
	  measure1[ii + 3*NX] = 0.0;
      measure1[n_f + 3*NX] = u1;
    } else if (i + (*MDX) == mic3_pos ) {
      if (n_f == 0) 
      	for(int ii=0; ii<NX; ii++)
	  measure1[ii + 4*NX] = 0.0;
      measure1[n_f + 4*NX] = u1;
    } else if (i + (*MDX) == mic4_pos ) {
      if (n_f == 0) 
      	for(int ii=0; ii<NX; ii++)
	  measure1[ii + 5*NX] = 0.0;
      measure1[n_f + 5*NX] = u1;
    }

    if (i == 0) {
      measure1[n_f + 0*NX] = (*T);
      measure1[n_f + 1*NX] = cfl[0]; // CFL value
      if (n_f == 0) {
	if ( mic1_pos - (*MDX) < 0 || mic1_pos - (*MDX) >= NX ) 
	  for(int ii=0; ii<NX; ii++)
	    measure1[ii + 2*NX] = 0.0;
	if ( mic2_pos - (*MDX) < 0 || mic2_pos - (*MDX) >= NX ) 
	  for(int ii=0; ii<NX; ii++)
	    measure1[ii + 3*NX] = 0.0;
	if ( mic3_pos - (*MDX) < 0 || mic3_pos - (*MDX) >= NX ) 
	  for(int ii=0; ii<NX; ii++)
	    measure1[ii + 4*NX] = 0.0;
	if ( mic4_pos - (*MDX) < 0 || mic4_pos - (*MDX) >= NX ) 
	  for(int ii=0; ii<NX; ii++)
	    measure1[ii + 5*NX] = 0.0;
      }
    }
  } else { *simError = 4; return; }

  // to restrict ourselves to meassure the central peak
  if ( abs( static_cast<TTT>(i + (*MDX))/ETA
	    - (*T) )
       > 0.5 ) return;
  
  // this could use several lines because of the *MDX value 
  int offsetrw = i + (*MDX) + 6*NX;
  measure1[offsetrw] = getmax(u1, measure1[offsetrw]); 
}

#endif  /* INC_DATA_COLLECT_H */
