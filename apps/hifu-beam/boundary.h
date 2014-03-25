#ifndef INC_BOUNDARY_H
#define INC_BOUNDARY_H

// hifu-beam

#include ROOT_PATH(/common/debug_tools.h)

// For comparison with CLAWPACK see: clawpack-4.6.1/clawpack/2d/lib/bc2.f

template <typename TTT>
__forceinline__ __device__ TTT packet
( TTT *T, TTT *phi ) {
  TTT opT = 2.0*FREC*((*T) + (*phi))/8.0;
  TTT res = AMPL * sin(ISPW * ((*T) + (*phi)) );
  res = res * exp(-opT*opT*opT*opT*opT*opT*opT*opT*opT*opT);
  return res;
}

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
    int width = static_cast<int>(ISPa/(*dy));
    int jprime = j - 2;
    if( abs(jprime) < width 
	&& jprime >= 0 
	&& (*MDX) == 0) {
      TTT jprime2, nudenom, nux, nuy, phi, pack;
      jprime2 = static_cast<TTT>(jprime*jprime);
      nudenom = sqrt(jprime2 + ISPF*ISPF);
      nux = ISPF/nudenom;
      nuy = -static_cast<TTT>(jprime)/nudenom;
      phi = ((*dx)*jprime2) / (Num2*ISPF); // <--- approximation
      // phi = - (*dx)*(ISPF - sqrt(jprime2 + ISPF*ISPF)); // <--- exact expression
      pack = packet(T, &phi);
      u1[offset] = pack; //CB:rho1
      u2[offset] = nux * pack * (pack + Num1); //CB:rho1
      u3[offset] = nuy * pack * (pack + Num1); //CB:rho1 
    } else {
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
   
  // reflective border
  if (j==0 || j==1) {
    ir = i;
    jr = 4 - j;
    if (dstOut) {
      u1[offset] =  texAu1_read(&ir,&jr);
      u2[offset] =  texAu2_read(&ir,&jr);
      u3[offset] = -texAu3_read(&ir,&jr);
    } else {
      u1[offset] =  texBu1_read(&ir,&jr);
      u2[offset] =  texBu2_read(&ir,&jr);
      u3[offset] = -texBu3_read(&ir,&jr);
    }
  }

  // corners
  if ((   i <= 1    && j <= 1    ) 
      || (i <= 1    && j >= NY-2 )
      || (i >= NX-2 && j <= 1    )
      || (i >= NX-2 && j >= NY-2 )) {
    u1[offset] = Num0;  //CB:rho1
    u2[offset] = Num0;
    u3[offset] = Num0;
    return;
  }
}

#endif  /* INC_BOUNDARY_H */
