#ifndef INC_DRAW_FLOAT_CUT_H
#define INC_DRAW_FLOAT_CUT_H

// taylor-angle-xy

template <typename TTT>
__global__ void draw
(int *MDX, int *MDY, 
 float *draw, 
 TTT *dx, TTT *dy, 
 TTT *dt, TTT *T,
 bool dstOut) {
  
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = i + j * blockDim.x * gridDim.x;

  // this is to avoid writting outside the arrays
  if (i > (NXW/ZOOM)-1 || j > (NYW/ZOOM)-1+128) return;

  int iread = (i*ZOOM)+NXWLB;
  int jread = (j*ZOOM)+NYWLB;
  int icenter = static_cast<int>((*T)*MDVX/(*dx))-(*MDX);
  int jcenter = (NY/2) 
    + static_cast<int>(((*T)-MDT)*MDVY/(*dy))
    -(*MDY);
  
  if (i <= (NXW/ZOOM)-1) {
    if (j <= (NYW/ZOOM)-1) {
      // 2D map view plot
      float u1, u2, u3;

      if (dstOut) {
        u1 = static_cast<float>( texAu1_read(&iread,&jread) ); //CB:rho1
        u2 = static_cast<float>( texAu2_read(&iread,&jread) );
        u3 = static_cast<float>( texAu3_read(&iread,&jread) );
      } else {
        u1 = static_cast<float>( texBu1_read(&iread,&jread) ); //CB:rho1
        u2 = static_cast<float>( texBu2_read(&iread,&jread) );
        u3 = static_cast<float>( texBu3_read(&iread,&jread) );
      }

      // Don't rescale here use PLOTSCALE instead
      draw[offset] =  u1/(PLOTSCALE * 2.0f * AMPL) + 0.5f; //CB:rho1
      // draw[offset] =  u2/(PLOTSCALE * 2.0f * AMPL) + 0.5f;
      // draw[offset] =  u3/(PLOTSCALE * 2.0f * AMPL) + 0.5f;
      if(draw[offset] > 1.0f) draw[offset] = 1.0f;
      if(draw[offset] < 0.0f) draw[offset] = 0.0f;
      if(abs(iread-icenter) <
	 static_cast<int>(5.0*ETA*cos(ISPTHETA))
	 &&
	 ZOOM >= abs(static_cast<float>(jread-jcenter) 
		     - static_cast<float>(iread-icenter)
		     *((*dx)/(*dy))*tan(ISPTHETA))
	 ) {
	if(static_cast<int>(static_cast<float>(abs(iread-icenter))/cos(ISPTHETA))
	   %static_cast<int>(2*ETA)
	   <static_cast<int>(ETA)){
	  draw[offset] = 1.0f;
	} else {
	  draw[offset] = 0.0f;
	}

	if(ZOOM >= abs(iread-icenter)
	   && ZOOM >= abs(jread-jcenter)) {
	  draw[offset] = 0.5f;
	}
      }
    } else if (j > (NYW/ZOOM)-1 && j<=(NYW/ZOOM)-1+128) {
      draw[offset] = 0.0f;
      // 1D transverse cut plot

      int h = i - (NXW/ZOOM)/2; // hypotenuse

      iread = (NXW/2) + ZOOM*static_cast<int>
	(static_cast<float>(h) * cos(ISPTHETA));
      jread = (NYW/2) + ZOOM*static_cast<int>
	(static_cast<float>(h) * sin(ISPTHETA));

      float u1, u2, u3;
      if (dstOut) {
      	u1 = static_cast<float>( texAu1_read(&iread,&jread) );
      	u2 = static_cast<float>( texAu2_read(&iread,&jread) );
      	u3 = static_cast<float>( texAu3_read(&iread,&jread) );
      } else {
      	u1 = static_cast<float>( texBu1_read(&iread,&jread) );
      	u2 = static_cast<float>( texBu2_read(&iread,&jread) );
      	u3 = static_cast<float>( texBu3_read(&iread,&jread) );
      }
      
      float draw0; 
      // Don't rescale here use PLOTSCALE instead

      // numeric density
      draw0 =  u1/(PLOTSCALE * 2.0f * AMPL); //CB:rho1
      if ( abs( (float) j 
		- static_cast<float>(NYW/ZOOM) 
		- 64.0f 
		- 32.0f
      		- draw0 * 128.0f )
      	   < 1.0f ) {
	draw[offset] = 1.0f;
      } else if ( j == (NYW/ZOOM) + 64 ) {
      	draw[offset] = 0.5f;
      } 

      // analytic density
      float rhop;
      rhop = ISPRHOA
	* tanh(ISPC*(static_cast<float>(iread+(*MDX))*(*dx)*cos(ISPTHETA)
		     + (static_cast<TTT>(jread+(*MDY)-NY/2)*(*dy)+MDT*sin(ISPTHETA))
		     *sin(ISPTHETA)
		     - (*T)));
 
      draw0 = rhop/(PLOTSCALE * 2.0f * AMPL); //CB:rho1
      if ( abs( (float) j 
		- static_cast<float>(NYW/ZOOM) 
		- 64.0f 
		- 32.0f
      		- draw0 * 128.0f )
      	   < 1.0f ) {
      	draw[offset] = 0.35f;
      }

      // numeric parallel momentum
      float utot = (abs(u2)/u2)*sqrt(u2*u2 + u3*u3);
      draw0 =  utot/(PLOTSCALE * 2.0f * AMPL);
      if ( abs( (float) j 
		- static_cast<float>(NYW/ZOOM) 
		- 32.0f 
      		- draw0 * 128.0f )
      	   < 1.0f ) {
	draw[offset] = 0.75f;
      } 

      // analytic parallel momentum
      draw0 = rhop/(PLOTSCALE * 2.0f * AMPL); //CB:rho1
      if ( abs( (float) j 
		- static_cast<float>(NYW/ZOOM) 
		- 32.0f
      		- draw0 * 128.0f )
      	   < 1.0f ) {
      	draw[offset] = 0.35f;
      } 
    }
  }
}

#endif  /* INC_DRAW_FLOAT_CUT_H */
