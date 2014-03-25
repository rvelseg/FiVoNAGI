#ifndef INC_DRAW_FLOAT_CUT_H
#define INC_DRAW_FLOAT_CUT_H

// hifu-beam

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
  int jcut = JCUT;

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
      if(draw[offset] < 0.0f) draw[offset] = 0.0f; // bug: this line in other draw_float_cut.h files

      if(abs(jread - jcut) <= ZOOM-1) {
	if(iread%40<20){
	  draw[offset] = 1.0f;
	} else {
	  draw[offset] = 0.0f;
	}
      }
      if(abs(iread-static_cast<int>(ISPF-(*MDX)))<=ZOOM-1) {
	if(jread%40<20){
	  draw[offset] = 1.0f;
	} else {
	  draw[offset] = 0.0f;
	}
      }
      if(abs(i - NXW/(2*ZOOM)) <= 1) {
	draw[offset] = 0.7f;
      }
      if(abs(jread - ISPa*ETA) <= 1) {
	draw[offset] = 0.7f;
      }
    } else if (j > (NYW/ZOOM)-1 && j<=(NYW/ZOOM)-1+128) {
      // 1D transverse cut plot
      float u1, u2, u3;
      if (dstOut) {
	u1 = static_cast<float>( texAu1_read(&iread,&jcut) ); //CB:rho1
	u2 = static_cast<float>( texAu2_read(&iread,&jcut) );
	u3 = static_cast<float>( texAu3_read(&iread,&jcut) );
      } else {
	u1 = static_cast<float>( texBu1_read(&iread,&jcut) ); //CB:rho1
	u2 = static_cast<float>( texBu2_read(&iread,&jcut) );
	u3 = static_cast<float>( texBu3_read(&iread,&jcut) );
      }
      
      float draw0;
      // Don't rescale here use PLOTSCALE instead
      draw0 =  u1/(PLOTSCALE * 4.0f * AMPL); //CB:rho1
      // draw0 =  u2/(PLOTSCALE * 4.0f * AMPL);
      // draw0 =  u3/(PLOTSCALE * 4.0f * AMPL);
      if ( abs( (float) j 
		- static_cast<float>(NYW/ZOOM) 
		- 64.0f
		- draw0 * 128.0f ) 
	   < 1.0f ) {
	draw[offset] = 1.0f;
      } else if ( j == (NYW/ZOOM) + 64 ) {
	draw[offset] = 0.5f;
      } else if ( j == (NYW/ZOOM) + 96
		  || j == (NYW/ZOOM) + 32) {
	draw[offset] = 0.25f;
      } else {
	draw[offset] = 0.0f;
      }
    }
  }
}

#endif  /* INC_DRAW_FLOAT_CUT_H */
