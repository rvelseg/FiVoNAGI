#ifndef INC_SOURCE_H
#define INC_SOURCE_H

// common

// For comparison with CLAWPACK see: clawpack-4.6.1/clawpack/2d/lib/src2.f

template <typename TTT>
__global__ void source
(TTT *u1W, TTT *u2W, TTT *u3W, 
 TTT *dx, TTT *dy, TTT *dt, 
 GPUGD_VARSFD, bool dstOut) {
  
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = i + j * blockDim.x * gridDim.x;

  // this is to avoid writting outside the arrays
  if (i > NX-1 || j > NY-1) return;
  
  if(i>=2 && i<=NX-3 && j>=2 && j<=NY-3) {

    TTT ddtx2 = DIFFDELTA * (*dt)/((*dx) * (*dx));
    TTT ddty2 = DIFFDELTA * (*dt)/((*dy) * (*dy)); 

    TTT u1IP0JP0, u1IM1JP0, u1IP0JM1, u1IP1JP0, u1IP0JP1;
    TTT u2IP0JP0, u2IM1JP0, u2IP0JM1, u2IP1JP0, u2IP0JP1;
    TTT u3IP0JP0, u3IM1JP0, u3IP0JM1, u3IP1JP0, u3IP0JP1;
 
    {
      int im1 = i-1;
      int ip1 = i+1;

      int jm1 = j-1;
      int jp1 = j+1;

      if (dstOut) {

	u1IP0JP0 = texAu1_read(&i  ,&j  ) + Num1; //CB:rho1
	u1IM1JP0 = texAu1_read(&im1,&j  ) + Num1; //CB:rho1
	u1IP0JM1 = texAu1_read(&i  ,&jm1) + Num1; //CB:rho1
	u1IP1JP0 = texAu1_read(&ip1,&j  ) + Num1; //CB:rho1
	u1IP0JP1 = texAu1_read(&i  ,&jp1) + Num1; //CB:rho1

	u2IP0JP0 = texAu2_read(&i  ,&j  );
	u2IM1JP0 = texAu2_read(&im1,&j  );
	u2IP0JM1 = texAu2_read(&i  ,&jm1);
	u2IP1JP0 = texAu2_read(&ip1,&j  );
	u2IP0JP1 = texAu2_read(&i  ,&jp1);

	u3IP0JP0 = texAu3_read(&i  ,&j  );
	u3IM1JP0 = texAu3_read(&im1,&j  );
	u3IP0JM1 = texAu3_read(&i  ,&jm1);
	u3IP1JP0 = texAu3_read(&ip1,&j  );
	u3IP0JP1 = texAu3_read(&i  ,&jp1);

      } else {

	u1IP0JP0 = texBu1_read(&i  ,&j  ) + Num1; //CB:rho1
	u1IM1JP0 = texBu1_read(&im1,&j  ) + Num1; //CB:rho1
	u1IP0JM1 = texBu1_read(&i  ,&jm1) + Num1; //CB:rho1
	u1IP1JP0 = texBu1_read(&ip1,&j  ) + Num1; //CB:rho1
	u1IP0JP1 = texBu1_read(&i  ,&jp1) + Num1; //CB:rho1

	u2IP0JP0 = texBu2_read(&i  ,&j  );
	u2IM1JP0 = texBu2_read(&im1,&j  );
	u2IP0JM1 = texBu2_read(&i  ,&jm1);
	u2IP1JP0 = texBu2_read(&ip1,&j  );
	u2IP0JP1 = texBu2_read(&i  ,&jp1);

	u3IP0JP0 = texBu3_read(&i  ,&j  );
	u3IM1JP0 = texBu3_read(&im1,&j  );
	u3IP0JM1 = texBu3_read(&i  ,&jm1);
	u3IP1JP0 = texBu3_read(&ip1,&j  );
	u3IP0JP1 = texBu3_read(&i  ,&jp1);

      }
    }

    // the following line is necesary because we are alternating
    // textures, we must copy the values even when they are not
    // changed.
    u1W[offset] = u1IP0JP0 - Num1; //CB:rho1
  
    u2W[offset] = (u2IP0JP0
		   + ddtx2 
		   * ( ( u2IM1JP0 / u1IM1JP0 )
		       - Num2 * ( u2IP0JP0 / u1IP0JP0 )
		       + ( u2IP1JP0 / u1IP1JP0 ) )
		   + ddty2 
		   * ( ( u2IP0JM1 / u1IP0JM1 )
		       - Num2 * ( u2IP0JP0 / u1IP0JP0 )
		       + ( u2IP0JP1 / u1IP0JP1 ) ) ); 
  
    u3W[offset] = (u3IP0JP0
		   + ddtx2 
		   * ( ( u3IM1JP0 / u1IM1JP0 )
		       - Num2 * ( u3IP0JP0 / u1IP0JP0 )
		       + ( u3IP1JP0 / u1IP1JP0 ) )
		   + ddty2 
		   * ( ( u3IP0JM1 / u1IP0JM1 )
		       - Num2 * ( u3IP0JP0 / u1IP0JP0 )
		       + ( u3IP0JP1 / u1IP0JP1 ) ) ); 

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

#endif  /* INC_SOURCE_H */
