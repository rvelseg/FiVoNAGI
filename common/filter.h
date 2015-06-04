#ifndef INC_FILTER_H
#define INC_FILTER_H

// common

#include <cufft.h>

__global__ void filter
(cufftComplex *spectrum) {

  // To use this file: copy this file to you app directory, remove the
  // following line, uncomment and adjust the code below according to
  // your needs.
  return;

  /* int i = threadIdx.x + blockIdx.x * blockDim.x; */
  /* int j = threadIdx.y + blockIdx.y * blockDim.y; */
  /* int offset_f = 0; */

  /* // this is to avoid writting outside the arrays */
  /* if (i > NX-1 || j > NY-1 ) return; */


  /* if ( i > NX/2 ) { */
  /*   // If the fft is R2C, there is nothing to do here because of the */
  /*   // redundancy; */
  /*   return; */
  /* } else { // i <= NX/2 */
  /*   if ( j > NY/2 ) { */
  /*     offset_f = (NX/2 - i) + (NY/2 + (NY - j) ) * ( (NX/2) + 1 ); */
  /*   } else { // j <= NY/2 */
  /*     offset_f = (NX/2 - i) + (NY/2 - j) * ( (NX/2) + 1 ); */
  /*   } */
  /* } */
  /* float R_f = sqrt( (float) (i-NX/2)*(i-NX/2) + (j-NY/2)*(j-NY/2) ); */
  /* if ( R_f > 17 ) { */
  /*   spectrum[offset_f].x = 0.0; */
  /*   spectrum[offset_f].y = 0.0; */
  /* } */
}

#endif  /* INC_FILTER_H */
