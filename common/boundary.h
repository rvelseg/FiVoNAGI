#ifndef INC_BOUNDARY_H
#define INC_BOUNDARY_H

// common

#include "../common/debug_tools.h"

// For comparison with CLAWPACK see: clawpack-4.6.1/clawpack/2d/lib/bc2.f

template <typename TTT>
__global__ void boundary
(TTT *u1, TTT *u2, TTT *u3, 
 TTT *dx, TTT *dy, TTT *dt,
 TTT *T, 
 GPUGD_VARSFD,
 bool dstOut) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = i + j * blockDim.x * gridDim.x; 

  // this is to avoid writting outside the arrays
  if (i > NX-1 || j > NY-1) return;

  int ir, jr;

  if(i==NX-1 || i == NX-2) {
    ir = NX-3;
    jr = j;
    if (dstOut) {
      u1[offset] = texAu1_read(&ir,&jr);
      u2[offset] = texAu2_read(&ir,&jr);
      u3[offset] = texAu3_read(&ir,&jr);
    } else {
      u1[offset] = texBu1_read(&ir,&jr);
      u2[offset] = texBu2_read(&ir,&jr);
      u3[offset] = texBu3_read(&ir,&jr);
    }
  }

  if(j==0 || j ==1) {
    ir = i;
    jr = 2;
    if (dstOut) {
      u1[offset] = texAu1_read(&ir,&jr);
      u2[offset] = texAu2_read(&ir,&jr);
      u3[offset] = texAu3_read(&ir,&jr);
    } else {
      u1[offset] = texBu1_read(&ir,&jr);
      u2[offset] = texBu2_read(&ir,&jr);
      u3[offset] = texBu3_read(&ir,&jr);
    }
  }

  if(j==NY-1 || j==NY-2) {
    ir = i;
    jr = NY-3;
    if (dstOut) {
      u1[offset] = texAu1_read(&ir,&jr);
      u2[offset] = texAu2_read(&ir,&jr);
      u3[offset] = texAu3_read(&ir,&jr);
    } else {
      u1[offset] = texBu1_read(&ir,&jr);
      u2[offset] = texBu2_read(&ir,&jr);
      u3[offset] = texBu3_read(&ir,&jr);
    }
  }

  if(i==0 || i==1) {
    ir = 2;
    jr = j;
    if (dstOut) {
      u1[offset] = texAu1_read(&ir,&jr);
      u2[offset] = texAu2_read(&ir,&jr);
      u3[offset] = texAu3_read(&ir,&jr);
    } else {
      u1[offset] = texBu1_read(&ir,&jr);
      u2[offset] = texBu2_read(&ir,&jr);
      u3[offset] = texBu3_read(&ir,&jr);
    }
  }
}

#endif  /* INC_BOUNDARY_H */
