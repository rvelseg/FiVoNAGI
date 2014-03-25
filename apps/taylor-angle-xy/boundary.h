#ifndef INC_BOUNDARY_H
#define INC_BOUNDARY_H

// taylor-angle-xy

#include ROOT_PATH(/common/debug_tools.h)

// For comparison with CLAWPACK see: clawpack-4.6.1/clawpack/2d/lib/bc2.f

template <typename TTT>
__global__ void boundary
(int *MDX, int *MDY, 
 TTT *u1, TTT *u2, TTT *u3, 
 TTT *dx, TTT *dy, TTT *dt,
 TTT *T, 
 GPUGD_VARSFD,
 bool dstOut) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = i + j * blockDim.x * gridDim.x; 

  // this is to avoid writting outside the arrays
  if (i > NX-1 || j > NY-1) return;

  // first copy everything
  if (dstOut) {
    u1[offset] = texAu1_read(&i,&j); //CB:rho1
    u2[offset] = texAu2_read(&i,&j);
    u3[offset] = texAu3_read(&i,&j);
  } else {
    u1[offset] = texBu1_read(&i,&j); //CB:rho1
    u2[offset] = texBu2_read(&i,&j);
    u3[offset] = texBu3_read(&i,&j);
  }

  // then rewrite the borders
  if(i==NX-1 || i == NX-2 
     || j==0 || j ==1
     || j==NY-1 || j==NY-2
     || i==0 || i==1) {

    TTT rhop;
    
    rhop = ISPRHOA
      * tanh(ISPC*(static_cast<TTT>(i+(*MDX))*(*dx)
		   *cos(ISPTHETA)
		   + (static_cast<TTT>(j+(*MDY)-NY/2)*(*dy)+MDT*sin(ISPTHETA))
		   *sin(ISPTHETA)
		   - (*T)));

    u1[offset] = rhop; //CB:rho1
    u2[offset] = rhop*cos(ISPTHETA); //CB:rho1
    u3[offset] = rhop*sin(ISPTHETA); //CB:rho1
  }
}


#endif  /* INC_BOUNDARY_H */
