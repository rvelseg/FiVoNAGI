#ifndef INC_FV_H
#define INC_FV_H
// finite volume method

// na

template <typename TTT>
__global__ void getWavesSpeedsCFL_x
(TTT *cfl, 
 TTT *dx, TTT *dt,
 int *simError, 
 GPUGD_VARSFD,
 bool dstOut) {
  
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = i + j * blockDim.x * gridDim.x;

  // this is to avoid writting outside the arrays
  if (i > NX-1 || j > NY-1) return;

  // in this cases you don't have i-1, then return because
  // you cannot run this kernel.
  if (i == 0) {
    cfl[offset] = Num0;
    return;
  }

  TTT *s1 = s1_ptr;
  TTT *s2 = s2_ptr;
  TTT *s3 = s3_ptr;

  TTT *W11 = W11_ptr;
  TTT *W21 = W21_ptr;
  TTT *W31 = W31_ptr;
  TTT *W12 = W12_ptr;
  TTT *W22 = W22_ptr;
  TTT *W32 = W32_ptr;
  TTT *W13 = W13_ptr;
  TTT *W23 = W23_ptr;
  TTT *W33 = W33_ptr;

  TTT *amdq1 = amdq1_ptr;
  TTT *apdq1 = apdq1_ptr;
  TTT *amdq2 = amdq2_ptr;
  TTT *apdq2 = apdq2_ptr;
  TTT *amdq3 = amdq3_ptr;
  TTT *apdq3 = apdq3_ptr;

  TTT dtx = (*dt)/(*dx);

  TTT u1IP0JP0, u1IM1JP0;
  TTT u2IP0JP0, u2IM1JP0;
  TTT u3IP0JP0, u3IM1JP0;
  
  // this line is different to the y case.
  TTT cflMax = cfl[offset];

  {
    int im1 = i-1;
    if (dstOut) {
      u1IP0JP0 = texAu1_read(&i  ,&j  ) + Num1; //CB:rho1
      u1IM1JP0 = texAu1_read(&im1,&j  ) + Num1; //CB:rho1

      u2IP0JP0 = texAu2_read(&i  ,&j  );
      u2IM1JP0 = texAu2_read(&im1,&j  );

      u3IP0JP0 = texAu3_read(&i  ,&j  );
      u3IM1JP0 = texAu3_read(&im1,&j  );
    } else {
      u1IP0JP0 = texBu1_read(&i  ,&j  ) + Num1; //CB:rho1
      u1IM1JP0 = texBu1_read(&im1,&j  ) + Num1; //CB:rho1

      u2IP0JP0 = texBu2_read(&i  ,&j  );
      u2IM1JP0 = texBu2_read(&im1,&j  );

      u3IP0JP0 = texBu3_read(&i  ,&j  );
      u3IM1JP0 = texBu3_read(&im1,&j  );
    }
  }

  TTT a1, a2, a3;
  TTT L1, L2, L3;
  TTT A1, A2;
  TTT tmp1, tmp2, tmp3;
  TTT Z1, Z2, Z3;

  // x axis
  // For comparison with CLAWPACK see: clawpack-4.6.1/apps/euler/1d/rp/rp1eu.f

  Z1 = ( u1IP0JP0 + u1IM1JP0 ) / Num2;
  Z2 = ( u2IP0JP0/u1IP0JP0 + u2IM1JP0/u1IM1JP0 ) / Num2;
  Z3 = ( u3IP0JP0/u1IP0JP0 + u3IM1JP0/u1IM1JP0 ) / Num2;

  // some temporal variables
  tmp1 = Z2/Z1;
  tmp2 = (Z2*Z2)/Z1;
  tmp3 = tmp1*tmp1 
    - Num2 * tmp2 
    + Num1 
    + Num2 * ( BETA - Num1 ) * ( Z1 - Num1 ) ;
  if ( tmp3 < Num0 ) { *simError = 2; return; }

  // eigenvalues
  L1 = tmp1 + sqrt(tmp3);
  L2 = tmp1 - sqrt(tmp3);
  L3 = tmp1;

  // more temporal variables
  A1 = ( L1 * Z3 - Num2 * Z2 * Z3 )
    / ( L1 * Z1 - Z3 );
  A2 = ( L2 * Z3 - Num2 * Z2 * Z3 )
    / ( L2 * Z1 - Z3 );

  // alphas
  a1 = ( u2IP0JP0  - u2IM1JP0 - 
	 ( u1IP0JP0 -  u1IM1JP0 ) * L2 ) / ( L1  -  L2  );
  a2 = u1IP0JP0 - u1IM1JP0 - a1;
  a3 = ((
	 ((L1*A2-L2*A1)
	  * (u1IP0JP0 - u1IM1JP0 ))
	 + ((A1-A2)
	    * (u2IP0JP0 - u2IM1JP0 ))
	 ) / ( L2 - L1 ) )
    + u3IP0JP0 - u3IM1JP0;

  s1[offset] = L1;
  s2[offset] = L2;
  s3[offset] = L3;

  W11[offset] = a1;
  W21[offset] = a1 * L1;
  W31[offset] = a1 * A1;
  W12[offset] = a2;
  W22[offset] = a2 * L2;
  W32[offset] = a2 * A2;
  W13[offset] = Num0;
  W23[offset] = Num0;
  W33[offset] = a3;

  amdq1[offset] = Num0;
  amdq2[offset] = Num0;
  amdq3[offset] = Num0;
  apdq1[offset] = Num0;
  apdq2[offset] = Num0;
  apdq3[offset] = Num0;

  if (L1 < Num0) {
    amdq1[offset] += L1*a1;
    amdq2[offset] += L1*(a1*L1);
    amdq3[offset] += L1*(a1*A1);
    cflMax = getmax(cflMax,-L1*dtx);
  } else if (L1 > Num0) {
    apdq1[offset] += L1*a1;
    apdq2[offset] += L1*(a1*L1);
    apdq3[offset] += L1*(a1*A1);
    cflMax = getmax(cflMax,L1*dtx);
  }
  if (L2 < Num0) {
    amdq1[offset] += L2*a2;
    amdq2[offset] += L2*(a2*L2);
    amdq3[offset] += L2*(a2*A2);
    cflMax = getmax(cflMax,-L2*dtx);
  } else if (L2 > Num0) {
    apdq1[offset] += L2*a2;
    apdq2[offset] += L2*(a2*L2);
    apdq3[offset] += L2*(a2*A2);
    cflMax = getmax(cflMax,L2*dtx);
  }
  if (L3 < Num0) {
    // amdq1[offset] += Num0;
    // amdq2[offset] += Num0;
    amdq3[offset] += L3*a3;
    cflMax = getmax(cflMax,-L3*dtx);
  } else if (L3 > Num0) {
    // apdq1[offset] += Num0;
    // apdq2[offset] += Num0;
    apdq3[offset] += L3*a3;
    cflMax = getmax(cflMax,L3*dtx);
  }

  cfl[offset] = cflMax;

  // The results of this function (getWavesSpeeds_x) are:
  //
  // s1, s2, s3, W11, W21, W31, W12, W22, W32, W13,
  // W23, W33, amdq1, amdq2, amdq3, apdq1, apdq2,
  // apdq3
  //
  // they are indexed (i,j) but they refer to (i-1/2, j)
  //
  // the variable cfl refers to (i,j).
}

template <typename TTT>
__global__ void getWavesSpeedsCFL_y
(TTT *cfl, 
 TTT *dy, TTT *dt,
 int *simError, 
 GPUGD_VARSFD,
 bool dstOut) {
  
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = i + j * blockDim.x * gridDim.x;

  // this is to avoid writting outside the arrays
  if (i > NX-1 || j > NY-1) return;

  // in this cases you don't have j-1, then return because
  // you cannot run this kernel.
  if (j == 0) {
    cfl[offset] = Num0;
    return;
  }

  TTT *s1 = s1_ptr;
  TTT *s2 = s2_ptr;
  TTT *s3 = s3_ptr;

  TTT *W11 = W11_ptr;
  TTT *W21 = W21_ptr;
  TTT *W31 = W31_ptr;
  TTT *W12 = W12_ptr;
  TTT *W22 = W22_ptr;
  TTT *W32 = W32_ptr;
  TTT *W13 = W13_ptr;
  TTT *W23 = W23_ptr;
  TTT *W33 = W33_ptr;

  TTT *amdq1 = amdq1_ptr;
  TTT *apdq1 = apdq1_ptr;
  TTT *amdq2 = amdq2_ptr;
  TTT *apdq2 = apdq2_ptr;
  TTT *amdq3 = amdq3_ptr;
  TTT *apdq3 = apdq3_ptr;

  TTT dty = (*dt)/(*dy);

  TTT u1IP0JP0, u1IP0JM1;
  TTT u2IP0JP0, u2IP0JM1;
  TTT u3IP0JP0, u3IP0JM1;

  // the y case must run first
  TTT cflMax = Num0;

  {
    int jm1 = j-1;
    if (dstOut) {
      u1IP0JP0 = texAu1_read(&i  ,&j  ) + Num1; //CB:rho1
      u1IP0JM1 = texAu1_read(&i  ,&jm1) + Num1; //CB:rho1

      u2IP0JP0 = texAu2_read(&i  ,&j  );
      u2IP0JM1 = texAu2_read(&i  ,&jm1);

      u3IP0JP0 = texAu3_read(&i  ,&j  );
      u3IP0JM1 = texAu3_read(&i  ,&jm1);
    } else {
      u1IP0JP0 = texBu1_read(&i  ,&j  ) + Num1; //CB:rho1
      u1IP0JM1 = texBu1_read(&i  ,&jm1) + Num1; //CB:rho1

      u2IP0JP0 = texBu2_read(&i  ,&j  );
      u2IP0JM1 = texBu2_read(&i  ,&jm1);

      u3IP0JP0 = texBu3_read(&i  ,&j  );
      u3IP0JM1 = texBu3_read(&i  ,&jm1);
    }
  }

  TTT a1, a2, a3;
  TTT L1, L2, L3;
  TTT A1, A2;
  TTT tmp1, tmp2, tmp3;
  TTT Z1, Z2, Z3;

  // y axis
  // For comparison with CLAWPACK see: clawpack-4.6.1/apps/euler/1d/rp/rp1eu.f

  Z1 = ( u1IP0JP0 + u1IP0JM1 ) / Num2;
  Z2 = ( u2IP0JP0/u1IP0JP0 + u2IP0JM1/u1IP0JM1 ) / Num2;
  Z3 = ( u3IP0JP0/u1IP0JP0 + u3IP0JM1/u1IP0JM1 ) / Num2;

  // some temporal variables
  tmp1 = Z3/Z1;
  tmp2 = (Z3*Z3)/Z1;
  tmp3 = tmp1*tmp1 
    - Num2 * tmp2 
    + Num1 
    + Num2 * ( BETA - Num1 ) * ( Z1 - Num1 ) ;
  if ( tmp3 < Num0 ) { *simError = 2; return; }

  // eigenvalues
  L1 = tmp1 + sqrt(tmp3);
  L2 = tmp1 - sqrt(tmp3);
  L3 = tmp1;

  // more temporal variables
  A1 = ( L1 * Z2 - Num2 * Z3 * Z2 )
    / ( L1 * Z1 - Z2 );
  A2 = ( L2 * Z2 - Num2 * Z3 * Z2 )
    / ( L2 * Z1 - Z2 );

  // alphas
  a1 = ( u3IP0JP0  - u3IP0JM1 - 
	 ( u1IP0JP0 -  u1IP0JM1 ) * L2 ) / ( L1  -  L2  );
  a2 = u1IP0JP0 - u1IP0JM1 - a1;
  a3 = ((
	 ((L1*A2-L2*A1)
	  * (u1IP0JP0 - u1IP0JM1 ))
	 + ((A1-A2)
	    * (u3IP0JP0 - u3IP0JM1 ))
	 ) / ( L2 - L1 ))
    + u2IP0JP0 - u2IP0JM1;

  s1[offset] = L1;
  s2[offset] = L2;
  s3[offset] = L3;

  W11[offset] = a1;
  W21[offset] = a1 * L1;
  W31[offset] = a1 * A1;
  W12[offset] = a2;
  W22[offset] = a2 * L2;
  W32[offset] = a2 * A2;
  W13[offset] = Num0;
  W23[offset] = Num0;
  W33[offset] = a3;

  amdq1[offset] = Num0;
  amdq2[offset] = Num0;
  amdq3[offset] = Num0;
  apdq1[offset] = Num0;
  apdq2[offset] = Num0;
  apdq3[offset] = Num0;

  if (L1 < Num0) {
    amdq1[offset] += L1*a1;
    amdq2[offset] += L1*(a1*L1);
    amdq3[offset] += L1*(a1*A1);
    cflMax = getmax(cflMax,-L1*dty);
  } else if (L1 > Num0) {
    apdq1[offset] += L1*a1;
    apdq2[offset] += L1*(a1*L1);
    apdq3[offset] += L1*(a1*A1);
    cflMax = getmax(cflMax,L1*dty);
  }
  if (L2 < Num0) {
    amdq1[offset] += L2*a2;
    amdq2[offset] += L2*(a2*L2);
    amdq3[offset] += L2*(a2*A2);
    cflMax = getmax(cflMax,-L2*dty);
  } else if (L2 > Num0) {
    apdq1[offset] += L2*a2;
    apdq2[offset] += L2*(a2*L2);
    apdq3[offset] += L2*(a2*A2);
    cflMax = getmax(cflMax,L2*dty);
  }
  if (L3 < Num0) {
    // amdq1[offset] += Num0; 
    // amdq2[offset] += Num0; 
    amdq3[offset] += L3*a3;
    cflMax = getmax(cflMax,-L3*dty);
  } else if (L3 > Num0) {
    // apdq1[offset] += Num0;  
    // apdq2[offset] += Num0;  
    apdq3[offset] += L3*a3;
    cflMax = getmax(cflMax,L3*dty);
  }

  cfl[offset] = cflMax;

  // The results of this function (getWavesSpeeds_y) are:
  //
  // s1, s2, s3, W11, W21, W31, W12, W22, W32, W13,
  // W23, W33, amdq1, amdq2, amdq3, apdq1, apdq2,
  // apdq3
  //
  // they are indexed (i,j) but they refer to (i, j-1/2)
  //
  // the variable cfl refers to (i,j).
}

template <typename TTT>
__global__ void step_x
(TTT *u1W, TTT *u2W, TTT *u3W, 
 TTT *dx,  TTT *dt,
 GPUGD_VARSFD, bool dstOut) {
  
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = i + j * blockDim.x * gridDim.x;

  // this is to avoid writting outside the arrays
  if (i > NX-1 || j > NY-1) return;
  
  TTT u1R, u2R, u3R;
  if (dstOut) {
    u1R = texAu1_read(&i,&j); //CB:rho1
    u2R = texAu2_read(&i,&j);
    u3R = texAu3_read(&i,&j);
  } else {
    u1R = texBu1_read(&i,&j); //CB:rho1
    u2R = texBu2_read(&i,&j);
    u3R = texBu3_read(&i,&j);
  }
  
  // here are some unused threads.
  if(i>=2 && i<=NX-3 && j>=2 && j<=NY-3) {

    TTT dtx = (*dt)/(*dx);
  
    TTT apdq1, amdq1, apdq2, amdq2, apdq3, amdq3;

    {
      int ip1 = i+1;

      apdq1 = texapdq1_read(&i  ,&j  );
      amdq1 = texamdq1_read(&ip1,&j  );
      apdq2 = texapdq2_read(&i  ,&j  );
      amdq2 = texamdq2_read(&ip1,&j  );
      apdq3 = texapdq3_read(&i  ,&j  );
      amdq3 = texamdq3_read(&ip1,&j  );
    }

    // For comparison with CLAWPACK see: clawpack-4.6.1/clawpack/2d/lib/flux2.f

    u1W[offset] = (u1R
		   - dtx * (apdq1 + amdq1) ); //CB:rho1

    u2W[offset] = (u2R
		   - dtx * (apdq2 + amdq2) ); 
    
    u3W[offset] = (u3R
		   - dtx * (apdq3 + amdq3) );
  } else { 
    u1W[offset] = u1R; //CB:rho1
    u2W[offset] = u2R;
    u3W[offset] = u3R;
  }
}

template <typename TTT>
__global__ void step_y
(TTT *u1W, TTT *u2W, TTT *u3W, 
 TTT *dy, TTT *dt,
 GPUGD_VARSFD, bool dstOut) {
  
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = i + j * blockDim.x * gridDim.x;

  // this is to avoid writting outside the arrays
  if (i > NX-1 || j > NY-1) return;
  
  TTT u1R, u2R, u3R;
  if (dstOut) {
    u1R = texAu1_read(&i,&j); //CB:rho1
    u2R = texAu2_read(&i,&j);
    u3R = texAu3_read(&i,&j);
  } else {
    u1R = texBu1_read(&i,&j); //CB:rho1
    u2R = texBu2_read(&i,&j);
    u3R = texBu3_read(&i,&j);
  }
  
  // here are some unused threads.
  if(i>=2 && i<=NX-3 && j>=2 && j<=NY-3) {

    TTT dty = (*dt)/(*dy);
  
    TTT apdq1, amdq1, apdq2, amdq2, apdq3, amdq3;

    {
      int jp1 = j+1;

      apdq1 = texapdq1_read(&i  ,&j  );
      amdq1 = texamdq1_read(&i  ,&jp1);
      apdq2 = texapdq2_read(&i  ,&j  );
      amdq2 = texamdq2_read(&i  ,&jp1);
      apdq3 = texapdq3_read(&i  ,&j  );
      amdq3 = texamdq3_read(&i  ,&jp1);
    }

    // For comparison with CLAWPACK see: clawpack-4.6.1/clawpack/2d/lib/flux2.f

    u1W[offset] = (u1R
		   - dty * (apdq1 + amdq1) ); //CB:rho1

    u2W[offset] = (u2R
		   - dty * (apdq3 + amdq3) ); 
    
    u3W[offset] = (u3R
		   - dty * (apdq2 + amdq2) );
  } else { 
    u1W[offset] = u1R; //CB:rho1
    u2W[offset] = u2R;
    u3W[offset] = u3R;
  }
}

template <typename TTT>
__global__ void calcLimiters_x
(GPUGD_VARSFD,
 TTT *dx, TTT *dt) {
  
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = i + j * blockDim.x * gridDim.x;

  if ( i <= 0 || i >= NX-1 ||
       j <= 0 || j >= NY-1 ) return;

  TTT *amdq1 = amdq1_ptr;
  TTT *amdq2 = amdq2_ptr;
  TTT *amdq3 = amdq3_ptr;
  
  // TODO: este codigo esta muy unrolled
  // podria ser optimizado mucho.
  
  TTT dtx = (*dt)/(*dx);

  TTT W11, W21, W31;
  TTT W12, W22, W32;
  TTT W13, W23, W33;
  TTT W11IP1, W21IP1, W31IP1;
  TTT W12IP1, W22IP1, W32IP1;
  TTT W13IP1, W23IP1, W33IP1;
  TTT W11IM1, W21IM1, W31IM1;
  TTT W12IM1, W22IM1, W32IM1;
  TTT W13IM1, W23IM1, W33IM1;
  TTT s1, s2, s3;

  {
    int im1 = i-1;
    int ip1 = i+1;

    W11    = texW11_read(&i  ,&j  );
    W21    = texW21_read(&i  ,&j  );
    W31    = texW31_read(&i  ,&j  );
    W12    = texW12_read(&i  ,&j  );
    W22    = texW22_read(&i  ,&j  );
    W32    = texW32_read(&i  ,&j  );
    W13    = texW13_read(&i  ,&j  );
    W23    = texW23_read(&i  ,&j  );
    W33    = texW33_read(&i  ,&j  );

    W11IP1 = texW11_read(&ip1,&j  );
    W21IP1 = texW21_read(&ip1,&j  );
    W31IP1 = texW31_read(&ip1,&j  );
    W12IP1 = texW12_read(&ip1,&j  );
    W22IP1 = texW22_read(&ip1,&j  );
    W32IP1 = texW32_read(&ip1,&j  );
    W13IP1 = texW13_read(&ip1,&j  );
    W23IP1 = texW23_read(&ip1,&j  );
    W33IP1 = texW33_read(&ip1,&j  );

    W11IM1 = texW11_read(&im1,&j  );
    W21IM1 = texW21_read(&im1,&j  );
    W31IM1 = texW31_read(&im1,&j  );
    W12IM1 = texW12_read(&im1,&j  );
    W22IM1 = texW22_read(&im1,&j  );
    W32IM1 = texW32_read(&im1,&j  );
    W13IM1 = texW13_read(&im1,&j  );
    W23IM1 = texW23_read(&im1,&j  );
    W33IM1 = texW33_read(&im1,&j  );

    s1     = texs1_read(&i  ,&j  );
    s2     = texs2_read(&i  ,&j  );
    s3     = texs3_read(&i  ,&j  );
  }

  TTT wnorm2, dotr, dotl, 
    wlimitr1, wlimitr2, wlimitr3, c, r, r2;
  TTT cqxx1, cqxx2, cqxx3;

  // For comparison with CLAWPACK see: clawpack-4.6.1/clawpack/2d/lib/limiter.f

  // x axis
  wnorm2 = W11 * W11 + W21 * W21 + W31 * W31;
  dotr = W11 * W11IP1 + W21 * W21IP1 + W31 * W31IP1;
  dotl = W11 * W11IM1 + W21 * W21IM1 + W31 * W31IM1;
  // TODO: rename dotl and dotr as dot and place then inside
  // respective if blocks

  // For comparison with CLAWPACK see: clawpack-4.6.1/clawpack/2d/lib/philim.f

  if (wnorm2 != Num0) { 
    if (s1 > Num0) { 
      r = dotl / wnorm2;
      r2 = Num2 * r;
      c = ( Num1 + r ) / Num2;
      wlimitr1 = getmin(c, Num2);
      wlimitr1 = getmin(wlimitr1, r2);
      wlimitr1 = getmax(wlimitr1, Num0);
    } else { 
      r = dotr / wnorm2;
      r2 = Num2 * r;
      c = ( Num1 + r ) / Num2;
      wlimitr1 = getmin(c, Num2);
      wlimitr1 = getmin(wlimitr1, r2);
      wlimitr1 = getmax(wlimitr1, Num0);
    }
  } else { 
    wlimitr1 = Num1;
  }
  
  // The following lines are not used because 
  // the kernel would be reading and writting
  // on the same texture:
  // W11W[offset] = wlimitr1 * W11;
  // W21W[offset] = wlimitr1 * W21;
  // Instead of this the product with wlimitrN
  // have been included in the calculations of 
  // cqxxN.

  wnorm2 = W12 * W12 + W22 * W22 + W32 * W32;
  dotr = W12 * W12IP1 + W22 * W22IP1 + W32 * W32IP1;
  dotl = W12 * W12IM1 + W22 * W22IM1 + W32 * W32IM1;

  if (wnorm2 != Num0) { 
    if (s2 > Num0) { 
      r = dotl / wnorm2;
      r2 = Num2 * r;
      c = ( Num1 + r ) / Num2;
      wlimitr2 = getmin(c, Num2);
      wlimitr2 = getmin(wlimitr2, r2);
      wlimitr2 = getmax(wlimitr2, Num0);
    } else { 
      r = dotr / wnorm2;
      r2 = Num2 * r;
      c = ( Num1 + r ) / Num2;
      wlimitr2 = getmin(c, Num2);
      wlimitr2 = getmin(wlimitr2, r2);
      wlimitr2 = getmax(wlimitr2, Num0);
    }
  } else { 
    wlimitr2 = Num1;
  }

  wnorm2 = W13 * W13 + W23 * W23 + W33 * W33;
  dotr = W13 * W13IP1 + W23 * W23IP1 + W33 * W33IP1;
  dotl = W13 * W13IM1 + W23 * W23IM1 + W33 * W33IM1;

  if (wnorm2 != Num0) { 
    if (s3 > Num0) { 
      r = dotl / wnorm2;
      r2 = Num2 * r;
      c = ( Num1 + r ) / Num2;
      wlimitr3 = getmin(c, Num2);
      wlimitr3 = getmin(wlimitr3, r2);
      wlimitr3 = getmax(wlimitr3, Num0);
    } else { 
      r = dotr / wnorm2;
      r2 = Num2 * r;
      c = ( Num1 + r ) / Num2;
      wlimitr3 = getmin(c, Num2);
      wlimitr3 = getmin(wlimitr3, r2);
      wlimitr3 = getmax(wlimitr3, Num0);
    }
  } else { 
    wlimitr3 = Num1;
  }
  
  // For comparison with CLAWPACK see: clawpack-4.6.1/clawpack/2d/lib/flux2.f

  cqxx1 = fabs(s1) 
    * (Num1 - fabs(s1) * dtx) 
    * W11 * wlimitr1
    + fabs(s2)
    * (Num1 - fabs(s2) * dtx) 
    * W12 * wlimitr2
    + fabs(s3)
    * (Num1 - fabs(s3) * dtx) 
    * W13 * wlimitr3;

  cqxx2 = fabs(s1) 
    * (Num1 - fabs(s1) * dtx) 
    * W21 * wlimitr1
    + fabs(s2) 
    * (Num1 - fabs(s2) * dtx) 
    * W22 * wlimitr2
    + fabs(s3) 
    * (Num1 - fabs(s3) * dtx) 
    * W23 * wlimitr3;

  cqxx3 = fabs(s1) 
    * (Num1 - fabs(s1) * dtx) 
    * W31 * wlimitr1
    + fabs(s2) 
    * (Num1 - fabs(s2) * dtx) 
    * W32 * wlimitr2
    + fabs(s3) 
    * (Num1 - fabs(s3) * dtx) 
    * W33 * wlimitr3;

  // The following lines are a reuse of
  // three textures:
  // amdq1, amdq2, amdq3,
  // I didn't want to define new textures:
  // fadd1, fadd2, fadd3, 
  // The code that should be used is commented
  // above each line actualy used.

  // fadd1[offset] = Num0p5 * cqxx1;
  amdq1[offset] = Num0p5 * cqxx1;

  // fadd2[offset] = Num0p5 * cqxx2;
  amdq2[offset] = Num0p5 * cqxx2;

  // fadd3[offset] = Num0p5 * cqxx3;
  amdq3[offset] = Num0p5 * cqxx3;

  // The results of this function (calcLimiters_x) are:
  //
  // fadd1 written on amdq1
  // fadd2 written on amdq2
  // fadd3 written on amdq3
  //
  // they are indexed (i,j) but they refer to (i-1/2, j)
}

template <typename TTT>
__global__ void calcLimiters_y
(GPUGD_VARSFD,
 TTT *dy, TTT *dt) {
  
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = i + j * blockDim.x * gridDim.x;

  if ( i <= 0 || i >= NX-1 ||
       j <= 0 || j >= NY-1 ) return;

  TTT *amdq1 = amdq1_ptr;
  TTT *amdq2 = amdq2_ptr;
  TTT *amdq3 = amdq3_ptr;
  
  // TODO: este codigo esta muy unrolled
  // podria ser optimizado mucho.
  
  TTT dty = (*dt)/(*dy);

  TTT W11, W21, W31;
  TTT W12, W22, W32;
  TTT W13, W23, W33;
  TTT W11JP1, W21JP1, W31JP1;
  TTT W12JP1, W22JP1, W32JP1;
  TTT W13JP1, W23JP1, W33JP1;
  TTT W11JM1, W21JM1, W31JM1;
  TTT W12JM1, W22JM1, W32JM1;
  TTT W13JM1, W23JM1, W33JM1;
  TTT s1, s2, s3;

  {
    int jm1 = j-1;
    int jp1 = j+1;

    W11    = texW11_read(&i  ,&j  );
    W21    = texW21_read(&i  ,&j  );
    W31    = texW31_read(&i  ,&j  );
    W12    = texW12_read(&i  ,&j  );
    W22    = texW22_read(&i  ,&j  );
    W32    = texW32_read(&i  ,&j  );
    W13    = texW13_read(&i  ,&j  );
    W23    = texW23_read(&i  ,&j  );
    W33    = texW33_read(&i  ,&j  );

    W11JP1 = texW11_read(&i  ,&jp1);
    W21JP1 = texW21_read(&i  ,&jp1);
    W31JP1 = texW31_read(&i  ,&jp1);
    W12JP1 = texW12_read(&i  ,&jp1);
    W22JP1 = texW22_read(&i  ,&jp1);
    W32JP1 = texW32_read(&i  ,&jp1);
    W13JP1 = texW13_read(&i  ,&jp1);
    W23JP1 = texW23_read(&i  ,&jp1);
    W33JP1 = texW33_read(&i  ,&jp1);

    W11JM1 = texW11_read(&i  ,&jm1);
    W21JM1 = texW21_read(&i  ,&jm1);
    W31JM1 = texW31_read(&i  ,&jm1);
    W12JM1 = texW12_read(&i  ,&jm1);
    W22JM1 = texW22_read(&i  ,&jm1);
    W32JM1 = texW32_read(&i  ,&jm1);
    W13JM1 = texW13_read(&i  ,&jm1);
    W23JM1 = texW23_read(&i  ,&jm1);
    W33JM1 = texW33_read(&i  ,&jm1);

    s1     = texs1_read(&i  ,&j  );
    s2     = texs2_read(&i  ,&j  );
    s3     = texs3_read(&i  ,&j  );
  }

  TTT wnorm2, dotr, dotl, 
    wlimitr1, wlimitr2, wlimitr3, c, r, r2;
  TTT cqxx1, cqxx2, cqxx3;

  // For comparison with CLAWPACK see: clawpack-4.6.1/clawpack/2d/lib/limiter.f

  // y axis
  wnorm2 = W11 * W11 + W21 * W21 + W31 * W31;
  dotr = W11 * W11JP1 + W21 * W21JP1 + W31 * W31JP1;
  dotl = W11 * W11JM1 + W21 * W21JM1 + W31 * W31JM1;

  // For comparison with CLAWPACK see: clawpack-4.6.1/clawpack/2d/lib/philim.f

  if (wnorm2 != Num0) { 
    if (s1 > Num0) { 
      r = dotl / wnorm2;
      r2 = Num2 * r;
      c = ( Num1 + r ) / Num2;
      wlimitr1 = getmin(c, Num2);
      wlimitr1 = getmin(wlimitr1, r2);
      wlimitr1 = getmax(wlimitr1, Num0);
    } else { 
      r = dotr / wnorm2;
      r2 = Num2 * r;
      c = ( Num1 + r ) / Num2;
      wlimitr1 = getmin(c, Num2);
      wlimitr1 = getmin(wlimitr1, r2);
      wlimitr1 = getmax(wlimitr1, Num0);
    }
  } else { 
    wlimitr1 = Num1;
  }
  
  wnorm2 = W12 * W12 + W22 * W22 + W32 * W32;
  dotr = W12 * W12JP1 + W22 * W22JP1 + W32 * W32JP1;
  dotl = W12 * W12JM1 + W22 * W22JM1 + W32 * W32JM1;

  if (wnorm2 != Num0) { 
    if (s2 > Num0) { 
      r = dotl / wnorm2;
      r2 = Num2 * r;
      c = ( Num1 + r ) / Num2;
      wlimitr2 = getmin(c, Num2);
      wlimitr2 = getmin(wlimitr2, r2);
      wlimitr2 = getmax(wlimitr2, Num0);
    } else { 
      r = dotr / wnorm2;
      r2 = Num2 * r;
      c = ( Num1 + r ) / Num2;
      wlimitr2 = getmin(c, Num2);
      wlimitr2 = getmin(wlimitr2, r2);
      wlimitr2 = getmax(wlimitr2, Num0);
    }
  } else { 
    wlimitr2 = Num1;
  }

  wnorm2 = W13 * W13 + W23 * W23 + W33 * W33;
  dotr = W13 * W13JP1 + W23 * W23JP1 + W33 * W33JP1;
  dotl = W13 * W13JM1 + W23 * W23JM1 + W33 * W33JM1;

  if (wnorm2 != Num0) { 
    if (s3 > Num0) { 
      r = dotl / wnorm2;
      r2 = Num2 * r;
      c = ( Num1 + r ) / Num2;
      wlimitr3 = getmin(c, Num2);
      wlimitr3 = getmin(wlimitr3, r2);
      wlimitr3 = getmax(wlimitr3, Num0);
    } else { 
      r = dotr / wnorm2;
      r2 = Num2 * r;
      c = ( Num1 + r ) / Num2;
      wlimitr3 = getmin(c, Num2);
      wlimitr3 = getmin(wlimitr3, r2);
      wlimitr3 = getmax(wlimitr3, Num0);
    }
  } else { 
    wlimitr3 = Num1;
  }

  // For comparison with CLAWPACK see: clawpack-4.6.1/clawpack/2d/lib/flux2.f

  cqxx1 = fabs(s1) 
    * (Num1 - fabs(s1) * dty) 
    * W11 * wlimitr1
    + fabs(s2) 
    * (Num1 - fabs(s2) * dty) 
    * W12 * wlimitr2
    + fabs(s3) 
    * (Num1 - fabs(s3) * dty) 
    * W13 * wlimitr3;

  cqxx2 = fabs(s1) 
    * (Num1 - fabs(s1) * dty) 
    * W21 * wlimitr1
    + fabs(s2) 
    * (Num1 - fabs(s2) * dty) 
    * W22 * wlimitr2
    + fabs(s3) 
    * (Num1 - fabs(s3) * dty) 
    * W23 * wlimitr3;

  cqxx3 = fabs(s1) 
    * (Num1 - fabs(s1) * dty) 
    * W31 * wlimitr1
    + fabs(s2) 
    * (Num1 - fabs(s2) * dty) 
    * W32 * wlimitr2
    + fabs(s3) 
    * (Num1 - fabs(s3) * dty) 
    * W33 * wlimitr3;
  
  // The following lines are a reuse of
  // three textures:
  // amdq1, amdq2, amdq3.
  // I didn't want to define new textures:
  // fadd1, fadd2, fadd3.
  // The code that should be used is commented
  // above each line actualy used.

  // fadd1[offset] = Num0p5 * cqxx1;
  amdq1[offset] = Num0p5 * cqxx1;

  // fadd2[offset] = Num0p5 * cqxx2;
  amdq2[offset] = Num0p5 * cqxx2;

  // fadd3[offset] = Num0p5 * cqxx3;
  amdq3[offset] = Num0p5 * cqxx3;

  // The results of this function (calcLimiters_y) are:
  //
  // fadd1 written on amdq1
  // fadd2 written on amdq2
  // fadd3 written on amdq3
  //
  // they are indexed (i,j) but they refer to (i, j-1/2)
}

template <typename TTT>
__global__ void writeLimiters_x
(TTT *u1W, TTT *u2W, TTT *u3W, 
 TTT *dx, TTT *dt,
 GPUGD_VARSFD, bool dstOut) {
  
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = i + j * blockDim.x * gridDim.x;

  // this is to avoid writting outside the arrays
  if (i > NX-1 || j > NY-1) return;

  TTT u1R, u2R, u3R;
  if (dstOut) {
    u1R = texAu1_read(&i,&j); //CB:rho1
    u2R = texAu2_read(&i,&j);
    u3R = texAu3_read(&i,&j);
  } else {
    u1R = texBu1_read(&i,&j); //CB:rho1
    u2R = texBu2_read(&i,&j);
    u3R = texBu3_read(&i,&j);
  }

  if ( i>=2 && i<=NX-3 && j>=2 && j<=NY-3 ) {

    TTT dtx = (*dt)/(*dx);

    TTT fadd1, fadd1IP1;
    TTT fadd2, fadd2IP1;
    TTT fadd3, fadd3IP1;

    {
      int ip1 = i+1;

      fadd1    = texamdq1_read(&i  ,&j  );
      fadd1IP1 = texamdq1_read(&ip1,&j  );

      fadd2    = texamdq2_read(&i  ,&j  );
      fadd2IP1 = texamdq2_read(&ip1,&j  );

      fadd3    = texamdq3_read(&i  ,&j  );
      fadd3IP1 = texamdq3_read(&ip1,&j  );
    }
    
    // For comparison with CLAWPACK see: clawpack-4.6.1/clawpack/2d/lib/step2.f

    u1W[offset] = (u1R 
                   - dtx * (fadd1IP1 - fadd1) ); //CB:rho1

    u2W[offset] = (u2R 
		   - dtx * (fadd2IP1 - fadd2) );

    u3W[offset] = (u3R
		   - dtx * (fadd3IP1 - fadd3) );
  } else { 
    u1W[offset] = u1R; //CB:rho1
    u2W[offset] = u2R;
    u3W[offset] = u3R;
  }
}

template <typename TTT>
__global__ void writeLimiters_y
(TTT *u1W, TTT *u2W, TTT *u3W, 
 TTT *dy, TTT *dt,
 GPUGD_VARSFD, bool dstOut) {
  
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = i + j * blockDim.x * gridDim.x;

  // this is to avoid writting outside the arrays
  if (i > NX-1 || j > NY-1) return;

  TTT u1R, u2R, u3R;
  if (dstOut) {
    u1R = texAu1_read(&i,&j); //CB:rho1
    u2R = texAu2_read(&i,&j);
    u3R = texAu3_read(&i,&j);
  } else {
    u1R = texBu1_read(&i,&j); //CB:rho1
    u2R = texBu2_read(&i,&j);
    u3R = texBu3_read(&i,&j);
  }

  if ( i>=2 && i<=NX-3 && j>=2 && j<=NY-3 ) {

    TTT dty = (*dt)/(*dy);

    TTT fadd1, fadd1JP1;
    TTT fadd2, fadd2JP1;
    TTT fadd3, fadd3JP1;

    {
      int jp1 = j+1;

      fadd1    = texamdq1_read(&i  ,&j  );
      fadd1JP1 = texamdq1_read(&i  ,&jp1);

      fadd2    = texamdq2_read(&i  ,&j  );
      fadd2JP1 = texamdq2_read(&i  ,&jp1);

      fadd3    = texamdq3_read(&i  ,&j  );
      fadd3JP1 = texamdq3_read(&i  ,&jp1);
    }

    // For comparison with CLAWPACK see: clawpack-4.6.1/clawpack/2d/lib/step2.f

    u1W[offset] = (u1R 
		   - dty * (fadd1JP1 - fadd1) ); //CB:rho1

    u2W[offset] = (u2R 
		   - dty * (fadd3JP1 - fadd3) );

    u3W[offset] = (u3R
		   - dty * (fadd2JP1 - fadd2) );
  } else { 
    u1W[offset] = u1R; //CB:rho1
    u2W[offset] = u2R;
    u3W[offset] = u3R;
  }
}

#endif  /* INC_FV_H */
