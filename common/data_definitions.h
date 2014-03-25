#ifndef INC_DATA_DEFINITIONS_H
#define INC_DATA_DEFINITIONS_H

// common

#include "../nv/cpu_anim.h"
#include "../common/debug_tools.h"

// in this file the variables allocated in the gpu, and some
// operations over them, must be defined.

//////////////////////////////////////////////////////
// variables in the gpu (pointers to these variables)

template <typename TTT>
struct gpu_data {
  unsigned char   *output_bitmap;
  TTT             *dev_SrcAu1;
  TTT             *dev_SrcAu2;
  TTT             *dev_SrcAu3;

  TTT             *dev_SrcBu1;
  TTT             *dev_SrcBu2;
  TTT             *dev_SrcBu3;

  TTT             *dev_SrcCu1;
  TTT             *dev_SrcCu2;
  TTT             *dev_SrcCu3;

  TTT             *dev_s1;
  TTT             *dev_s2;
  TTT             *dev_s3;

  TTT             *dev_W11;
  TTT             *dev_W21;
  TTT             *dev_W31;
  TTT             *dev_W12;
  TTT             *dev_W22;
  TTT             *dev_W32;
  TTT             *dev_W13;
  TTT             *dev_W23;
  TTT             *dev_W33;

  TTT             *dev_amdq1;
  TTT             *dev_apdq1;
  TTT             *dev_amdq2;
  TTT             *dev_apdq2;
  TTT             *dev_amdq3;
  TTT             *dev_apdq3;

  TTT             *dev_cfl;
  float           *dev_draw;
  TTT             *dev_measure1;

#if DEBUG == 1
  TTT             *dev_debug1;
  TTT             *dev_debug2;
#endif /* DEBUG */

  TTT             *dev_dt;
  TTT             *dev_dx;
  TTT             *dev_dy;

  TTT             *dev_T;
  int             *dev_MDX;
  int             *dev_MDY;

  int             *dev_simError;
  // simError == 0 : everything is fine
  // simError == 1 : dt is too large
  // simError == 2 : there is an imaginary eiganvalue
  // simError == 3 : there is a transonic rarefaction
  
  CPUAnimBitmap   *bitmap;
  
  cudaEvent_t     start;
  cudaEvent_t     end;
};

////////////////////////////////////////////////
// textures for the most readed grids in gpu

#ifdef DATATYPET

// Conserved Vars meqn
texture<DATATYPET,2> texAu1;
texture<DATATYPET,2> texAu2;
texture<DATATYPET,2> texAu3;

// Conserved Vars meqn
texture<DATATYPET,2> texBu1;
texture<DATATYPET,2> texBu2;
texture<DATATYPET,2> texBu3;

// Conserved Vars meqn
texture<DATATYPET,2> texCu1;
texture<DATATYPET,2> texCu2;
texture<DATATYPET,2> texCu3;

// Speeds x-axis mwaves
texture<DATATYPET,2> texs1;
texture<DATATYPET,2> texs2;
texture<DATATYPET,2> texs3;

// Waves x-axis meqn-1 mwaves
texture<DATATYPET,2> texW11;
texture<DATATYPET,2> texW21;
texture<DATATYPET,2> texW31;
texture<DATATYPET,2> texW12;
texture<DATATYPET,2> texW22;
texture<DATATYPET,2> texW32;
texture<DATATYPET,2> texW13;
texture<DATATYPET,2> texW23;
texture<DATATYPET,2> texW33;

// fluctuations x-axis pm mwaves
texture<DATATYPET,2> texamdq1;
texture<DATATYPET,2> texapdq1;
texture<DATATYPET,2> texamdq2;
texture<DATATYPET,2> texapdq2;
texture<DATATYPET,2> texamdq3;
texture<DATATYPET,2> texapdq3;

// there are no definitions for y-axis variables because 
// x-axis variables are recycled.

#else
# error unresolved DATATYPET for texture definitions

#endif /* ifdef DATATYPET */

/////////////////////////////////////////////////////////////////////
// the following functions are a trick to use textures with single or
// double precission

#ifdef PRECISION

#if PRECISION == 1 

// Conserved Vars meqn
__device__ float texAu1_read(int *i, int *j) {
  return  tex2D(texAu1,*i,*j);
}
__device__ float texAu2_read(int *i, int *j) {
  return tex2D(texAu2,*i,*j);
}
__device__ float texAu3_read(int *i, int *j) {
  return tex2D(texAu3,*i,*j);
}

// Conserved Vars meqn
__device__ float texBu1_read(int *i, int *j) {
  return tex2D(texBu1,*i,*j);
}
__device__ float texBu2_read(int *i, int *j) {
  return tex2D(texBu2,*i,*j);
}
__device__ float texBu3_read(int *i, int *j) {
  return tex2D(texBu3,*i,*j);
}

// Conserved Vars meqn
__device__ float texCu1_read(int *i, int *j) {
  return tex2D(texCu1,*i,*j);
}
__device__ float texCu2_read(int *i, int *j) {
  return tex2D(texCu2,*i,*j);
}
__device__ float texCu3_read(int *i, int *j) {
  return tex2D(texCu3,*i,*j);
}

// Speeds x-axis mwaves
__device__ float texs1_read(int *i, int *j) {
  return tex2D(texs1,*i,*j);
}
__device__ float texs2_read(int *i, int *j) {
  return tex2D(texs2,*i,*j);
}
__device__ float texs3_read(int *i, int *j) {
  return tex2D(texs3,*i,*j);
}

// Waves x-axis meqn mwaves
__device__ float texW11_read(int *i, int *j) {
  return tex2D(texW11,*i,*j);
}
__device__ float texW21_read(int *i, int *j) {
  return tex2D(texW21,*i,*j);
}
__device__ float texW31_read(int *i, int *j) {
  return tex2D(texW31,*i,*j);
}
__device__ float texW12_read(int *i, int *j) {
  return tex2D(texW12,*i,*j);
}
__device__ float texW22_read(int *i, int *j) {
  return tex2D(texW22,*i,*j);
}
__device__ float texW32_read(int *i, int *j) {
  return tex2D(texW32,*i,*j);
}
__device__ float texW13_read(int *i, int *j) {
  return tex2D(texW13,*i,*j);
}
__device__ float texW23_read(int *i, int *j) {
  return tex2D(texW23,*i,*j);
}
__device__ float texW33_read(int *i, int *j) {
  return tex2D(texW33,*i,*j);
}

// fluctuations x-axis pm mwaves
__device__ float texamdq1_read(int *i, int *j) {
  return tex2D(texamdq1,*i,*j);
}
__device__ float texapdq1_read(int *i, int *j) {
  return tex2D(texapdq1,*i,*j);
}
__device__ float texamdq2_read(int *i, int *j) {
  return tex2D(texamdq2,*i,*j);
}
__device__ float texapdq2_read(int *i, int *j) {
  return tex2D(texapdq2,*i,*j);
}
__device__ float texamdq3_read(int *i, int *j) {
  return tex2D(texamdq3,*i,*j);
}
__device__ float texapdq3_read(int *i, int *j) {
  return tex2D(texapdq3,*i,*j);
}

#elif PRECISION == 2

// Conserved Vars meqn
__device__ double texAu1_read(int *i, int *j) {
  int2 v = tex2D(texAu1,*i,*j);
  return __hiloint2double(v.y, v.x);
}
__device__ double texAu2_read(int *i, int *j) {
  int2 v = tex2D(texAu2,*i,*j);
  return __hiloint2double(v.y, v.x);
}
__device__ double texAu3_read(int *i, int *j) {
  int2 v = tex2D(texAu3,*i,*j);
  return __hiloint2double(v.y, v.x);
}

// Conserved Vars meqn
__device__ double texBu1_read(int *i, int *j) {
  int2 v = tex2D(texBu1,*i,*j);
  return __hiloint2double(v.y, v.x);
}
__device__ double texBu2_read(int *i, int *j) {
  int2 v = tex2D(texBu2,*i,*j);
  return __hiloint2double(v.y, v.x);
}
__device__ double texBu3_read(int *i, int *j) {
  int2 v = tex2D(texBu3,*i,*j);
  return __hiloint2double(v.y, v.x);
}

// Conserved Vars meqn
__device__ double texCu1_read(int *i, int *j) {
  int2 v = tex2D(texCu1,*i,*j);
  return __hiloint2double(v.y, v.x);
}
__device__ double texCu2_read(int *i, int *j) {
  int2 v = tex2D(texCu2,*i,*j);
  return __hiloint2double(v.y, v.x);
}
__device__ double texCu3_read(int *i, int *j) {
  int2 v = tex2D(texCu3,*i,*j);
  return __hiloint2double(v.y, v.x);
}

// Speeds x-axis mwaves
__device__ double texs1_read(int *i, int *j) {
  int2 v = tex2D(texs1,*i,*j);
  return __hiloint2double(v.y, v.x);
}
__device__ double texs2_read(int *i, int *j) {
  int2 v = tex2D(texs2,*i,*j);
  return __hiloint2double(v.y, v.x);
}
__device__ double texs3_read(int *i, int *j) {
  int2 v = tex2D(texs3,*i,*j);
  return __hiloint2double(v.y, v.x);
}

// Waves x-axis meqn mwaves
__device__ double texW11_read(int *i, int *j) {
  int2 v = tex2D(texW11,*i,*j);
  return __hiloint2double(v.y, v.x);
}
__device__ double texW21_read(int *i, int *j) {
  int2 v = tex2D(texW21,*i,*j);
  return __hiloint2double(v.y, v.x);
}
__device__ double texW31_read(int *i, int *j) {
  int2 v = tex2D(texW31,*i,*j);
  return __hiloint2double(v.y, v.x);
}
__device__ double texW12_read(int *i, int *j) {
  int2 v = tex2D(texW12,*i,*j);
  return __hiloint2double(v.y, v.x);
}
__device__ double texW22_read(int *i, int *j) {
  int2 v = tex2D(texW22,*i,*j);
  return __hiloint2double(v.y, v.x);
}
__device__ double texW32_read(int *i, int *j) {
  int2 v = tex2D(texW32,*i,*j);
  return __hiloint2double(v.y, v.x);
}
__device__ double texW13_read(int *i, int *j) {
  int2 v = tex2D(texW13,*i,*j);
  return __hiloint2double(v.y, v.x);
}
__device__ double texW23_read(int *i, int *j) {
  int2 v = tex2D(texW23,*i,*j);
  return __hiloint2double(v.y, v.x);
}
__device__ double texW33_read(int *i, int *j) {
  int2 v = tex2D(texW33,*i,*j);
  return __hiloint2double(v.y, v.x);
}

// fluctuations x-axis pm mwaves
__device__ double texamdq1_read(int *i, int *j) {
  int2 v = tex2D(texamdq1,*i,*j);
  return __hiloint2double(v.y, v.x);
}
__device__ double texapdq1_read(int *i, int *j) {
  int2 v = tex2D(texapdq1,*i,*j);
  return __hiloint2double(v.y, v.x);
}
__device__ double texamdq2_read(int *i, int *j) {
  int2 v = tex2D(texamdq2,*i,*j);
  return __hiloint2double(v.y, v.x);
}
__device__ double texapdq2_read(int *i, int *j) {
  int2 v = tex2D(texapdq2,*i,*j);
  return __hiloint2double(v.y, v.x);
}
__device__ double texamdq3_read(int *i, int *j) {
  int2 v = tex2D(texamdq3,*i,*j);
  return __hiloint2double(v.y, v.x);
}
__device__ double texapdq3_read(int *i, int *j) {
  int2 v = tex2D(texapdq3,*i,*j);
  return __hiloint2double(v.y, v.x);
}

#else /* PRECISION value */
# error unresolved PRECISION value

#endif /* PRECISION value */

#else /* ifdef PRECISION */
# error unresolved PRECISION definition

#endif /* ifdef PRECISION */

/////////////////////////////////////////////////////////////////////
// the following definitions were necessary because there were too
// many parameters to be passed to kernels, it was not possible. This
// kind of definition allows to get this variables inside the kernels
// without passing them, but instead using (inside kernels) clauses
// like
//
//   TTT *s1 = s1_ptr;

#ifdef DATATYPEV

__constant__ DATATYPEV *s1_ptr;
__constant__ DATATYPEV *s2_ptr;
__constant__ DATATYPEV *s3_ptr;

__constant__ DATATYPEV *W11_ptr;
__constant__ DATATYPEV *W21_ptr;
__constant__ DATATYPEV *W31_ptr;
__constant__ DATATYPEV *W12_ptr;
__constant__ DATATYPEV *W22_ptr;
__constant__ DATATYPEV *W32_ptr;
__constant__ DATATYPEV *W13_ptr;
__constant__ DATATYPEV *W23_ptr;
__constant__ DATATYPEV *W33_ptr;

__constant__ DATATYPEV *amdq1_ptr;
__constant__ DATATYPEV *apdq1_ptr;
__constant__ DATATYPEV *amdq2_ptr;
__constant__ DATATYPEV *apdq2_ptr;
__constant__ DATATYPEV *amdq3_ptr;
__constant__ DATATYPEV *apdq3_ptr;

#else
# error unresolved DATATYPEV 

#endif /* ifdef DATATYPEV */

////////////////////////////////////////////////////////
// clean up memory allocated on the GPU
template <typename TTT>
void clean_gpu( gpu_data<TTT> *gd ) {

  cudaUnbindTexture( texAu1 );
  cudaUnbindTexture( texAu2 );  
  cudaUnbindTexture( texAu3 );  

  cudaUnbindTexture( texBu1 );
  cudaUnbindTexture( texBu2 );  
  cudaUnbindTexture( texBu3 );  

  cudaUnbindTexture( texCu1 );
  cudaUnbindTexture( texCu2 );  
  cudaUnbindTexture( texCu3 );  

  cudaUnbindTexture( texs1 );
  cudaUnbindTexture( texs2 );
  cudaUnbindTexture( texs3 );

  cudaUnbindTexture( texW11 );
  cudaUnbindTexture( texW21 );
  cudaUnbindTexture( texW31 );
  cudaUnbindTexture( texW12 );
  cudaUnbindTexture( texW22 );
  cudaUnbindTexture( texW32 );
  cudaUnbindTexture( texW13 );
  cudaUnbindTexture( texW23 );
  cudaUnbindTexture( texW33 );

  cudaUnbindTexture( texamdq1 );
  cudaUnbindTexture( texapdq1 );
  cudaUnbindTexture( texamdq2 );
  cudaUnbindTexture( texapdq2 );
  cudaUnbindTexture( texamdq3 );
  cudaUnbindTexture( texapdq3 );
 
  GPUGD_EC( cudaFree( gd->dev_SrcAu1 ) );
  GPUGD_EC( cudaFree( gd->dev_SrcAu2 ) );
  GPUGD_EC( cudaFree( gd->dev_SrcAu3 ) );

  GPUGD_EC( cudaFree( gd->dev_SrcBu1 ) );
  GPUGD_EC( cudaFree( gd->dev_SrcBu2 ) );
  GPUGD_EC( cudaFree( gd->dev_SrcBu3 ) );

  GPUGD_EC( cudaFree( gd->dev_SrcCu1 ) );
  GPUGD_EC( cudaFree( gd->dev_SrcCu2 ) );
  GPUGD_EC( cudaFree( gd->dev_SrcCu3 ) );

  GPUGD_EC( cudaFree( gd->dev_s1 ) );
  GPUGD_EC( cudaFree( gd->dev_s2 ) );
  GPUGD_EC( cudaFree( gd->dev_s3 ) );

  GPUGD_EC( cudaFree( gd->dev_W11 ) );
  GPUGD_EC( cudaFree( gd->dev_W21 ) );
  GPUGD_EC( cudaFree( gd->dev_W31 ) );
  GPUGD_EC( cudaFree( gd->dev_W12 ) );
  GPUGD_EC( cudaFree( gd->dev_W22 ) );
  GPUGD_EC( cudaFree( gd->dev_W32 ) );
  GPUGD_EC( cudaFree( gd->dev_W13 ) );
  GPUGD_EC( cudaFree( gd->dev_W23 ) );
  GPUGD_EC( cudaFree( gd->dev_W33 ) );

  GPUGD_EC( cudaFree( gd->dev_amdq1 ) );
  GPUGD_EC( cudaFree( gd->dev_apdq1 ) );
  GPUGD_EC( cudaFree( gd->dev_amdq2 ) );
  GPUGD_EC( cudaFree( gd->dev_apdq2 ) );
  GPUGD_EC( cudaFree( gd->dev_amdq3 ) );
  GPUGD_EC( cudaFree( gd->dev_apdq3 ) );

#if DEBUG >= 1
  GPUGD_EC( cudaFree( gd->dev_debug1 ) );
  GPUGD_EC( cudaFree( gd->dev_debug2 ) );
#endif /* DEBUG */

  GPUGD_EC( cudaFree( gd->dev_dt ) );
  GPUGD_EC( cudaFree( gd->dev_dx ) );
  GPUGD_EC( cudaFree( gd->dev_dy ) );

  GPUGD_EC( cudaFree( gd->dev_simError ) );

  GPUGD_EC( cudaFree( gd->dev_cfl ) );
  GPUGD_EC( cudaFree( gd->dev_draw ) );
  GPUGD_EC( cudaFree( gd->dev_measure1 ) );

  GPUGD_EC( cudaEventDestroy( gd->start ) );
  GPUGD_EC( cudaEventDestroy( gd->end ) );
}


////////////////////////////////////////////////////////
// memory allocation and variable initialization in gpu 

template <typename TTT>
void gpu_init
(gpu_data<TTT> *gd, 
 int gridBitSize, int imageSize, int drawSize,
 TTT *dt, TTT *dx, TTT *dy,
 TTT *T, int *MDX, int *MDY,
 TTT *u1, TTT *u2, TTT *u3,
 GPUGD_VARSFD) {

  cudaEvent_t start;
  cudaEvent_t end;
  GPUGD_EC( cudaEventCreate(&start) );
  GPUGD_EC( cudaEventCreate(&end) );
  GPUGD_EC( cudaEventRecord(start,0) );

#if DEBUG >= 1
  GPUGD_EC( cudaMalloc( (void**) &gd->dev_debug1, 
			sizeof(DATATYPEV) ) );
  GPUGD_EC( cudaMalloc( (void**) &gd->dev_debug2, 
			gridBitSize ) );
#endif /* DEBUG */

  GPUGD_EC( cudaMalloc( (void**)&gd->output_bitmap,
			imageSize ) );

  GPUGD_EC( cudaMalloc( (void**) &gd->dev_dt, 
			sizeof(DATATYPEV) ) );
  GPUGD_EC( cudaMalloc( (void**) &gd->dev_dx, 
			sizeof(DATATYPEV) ) );
  GPUGD_EC( cudaMalloc( (void**) &gd->dev_dy, 
			sizeof(DATATYPEV) ) );
  GPUGD_EC( cudaMalloc( (void**) &gd->dev_T, 
			sizeof(DATATYPEV) ) );
  GPUGD_EC( cudaMalloc( (void**) &gd->dev_MDX, 
			sizeof(int) ) );
  GPUGD_EC( cudaMalloc( (void**) &gd->dev_MDY, 
			sizeof(int) ) );

  GPUGD_EC( cudaMalloc( (void**) &gd->dev_simError, 
			sizeof(int) ) );

  GPUGD_EC( cudaMalloc( (void**) &gd->dev_SrcAu1, 
			gridBitSize ) );
  GPUGD_EC( cudaMalloc( (void**) &gd->dev_SrcAu2, 
			gridBitSize ) );
  GPUGD_EC( cudaMalloc( (void**) &gd->dev_SrcAu3, 
			gridBitSize ) );

  GPUGD_EC( cudaMalloc( (void**) &gd->dev_SrcBu1, 
			gridBitSize ) );
  GPUGD_EC( cudaMalloc( (void**) &gd->dev_SrcBu2, 
			gridBitSize ) );
  GPUGD_EC( cudaMalloc( (void**) &gd->dev_SrcBu3, 
			gridBitSize ) );

  GPUGD_EC( cudaMalloc( (void**) &gd->dev_SrcCu1, 
			gridBitSize ) );
  GPUGD_EC( cudaMalloc( (void**) &gd->dev_SrcCu2, 
			gridBitSize ) );
  GPUGD_EC( cudaMalloc( (void**) &gd->dev_SrcCu3, 
			gridBitSize ) );

  GPUGD_EC( cudaMalloc( (void**) &gd->dev_s1, 
			gridBitSize ) );
  GPUGD_EC( cudaMalloc( (void**) &gd->dev_s2, 
			gridBitSize ) );
  GPUGD_EC( cudaMalloc( (void**) &gd->dev_s3, 
			gridBitSize ) );

  GPUGD_EC( cudaMalloc( (void**) &gd->dev_W11, 
			gridBitSize ) );
  GPUGD_EC( cudaMalloc( (void**) &gd->dev_W21, 
			gridBitSize ) );
  GPUGD_EC( cudaMalloc( (void**) &gd->dev_W31, 
			gridBitSize ) );
  GPUGD_EC( cudaMalloc( (void**) &gd->dev_W12, 
			gridBitSize ) );
  GPUGD_EC( cudaMalloc( (void**) &gd->dev_W22, 
			gridBitSize ) );
  GPUGD_EC( cudaMalloc( (void**) &gd->dev_W32, 
			gridBitSize ) );
  GPUGD_EC( cudaMalloc( (void**) &gd->dev_W13, 
			gridBitSize ) );
  GPUGD_EC( cudaMalloc( (void**) &gd->dev_W23, 
			gridBitSize ) );
  GPUGD_EC( cudaMalloc( (void**) &gd->dev_W33, 
			gridBitSize ) );

  GPUGD_EC( cudaMalloc( (void**) &gd->dev_amdq1, 
			gridBitSize ) );
  GPUGD_EC( cudaMalloc( (void**) &gd->dev_apdq1, 
			gridBitSize ) );
  GPUGD_EC( cudaMalloc( (void**) &gd->dev_amdq2, 
			gridBitSize ) );
  GPUGD_EC( cudaMalloc( (void**) &gd->dev_apdq2, 
			gridBitSize ) );
  GPUGD_EC( cudaMalloc( (void**) &gd->dev_amdq3, 
			gridBitSize ) );
  GPUGD_EC( cudaMalloc( (void**) &gd->dev_apdq3, 
			gridBitSize ) );

  GPUGD_EC( cudaMalloc( (void**) &gd->dev_cfl, 
			gridBitSize ) );
  GPUGD_EC( cudaMalloc( (void**) &gd->dev_draw, 
			drawSize ) );
  GPUGD_EC( cudaMalloc( (void**) &gd->dev_measure1, 
			gridBitSize ) );

  cudaChannelFormatDesc desc = cudaCreateChannelDesc<DATATYPET>(); 

  GPUGD_EC( cudaBindTexture2D( NULL, texAu1,
			       gd->dev_SrcAu1,
			       desc, NX, NY,
			       sizeof(DATATYPEV) * NX ) );
  GPUGD_EC( cudaBindTexture2D( NULL, texAu2,
			       gd->dev_SrcAu2,
			       desc, NX, NY,
			       sizeof(DATATYPEV) * NX ) );
  GPUGD_EC( cudaBindTexture2D( NULL, texAu3,
			       gd->dev_SrcAu3,
			       desc, NX, NY,
			       sizeof(DATATYPEV) * NX ) );

  GPUGD_EC( cudaBindTexture2D( NULL, texBu1,
			       gd->dev_SrcBu1,
			       desc, NX, NY,
			       sizeof(DATATYPEV) * NX ) );
  GPUGD_EC( cudaBindTexture2D( NULL, texBu2,
			       gd->dev_SrcBu2,
			       desc, NX, NY,
			       sizeof(DATATYPEV) * NX ) );
  GPUGD_EC( cudaBindTexture2D( NULL, texBu3,
			       gd->dev_SrcBu3,
			       desc, NX, NY,
			       sizeof(DATATYPEV) * NX ) );

  GPUGD_EC( cudaBindTexture2D( NULL, texCu1,
			       gd->dev_SrcCu1,
			       desc, NX, NY,
			       sizeof(DATATYPEV) * NX ) );
  GPUGD_EC( cudaBindTexture2D( NULL, texCu2,
			       gd->dev_SrcCu2,
			       desc, NX, NY,
			       sizeof(DATATYPEV) * NX ) );
  GPUGD_EC( cudaBindTexture2D( NULL, texCu3,
			       gd->dev_SrcCu3,
			       desc, NX, NY,
			       sizeof(DATATYPEV) * NX ) );

  GPUGD_EC( cudaBindTexture2D( NULL, texs1,
			       gd->dev_s1,
			       desc, NX, NY,
			       sizeof(DATATYPEV) * NX ) );
  GPUGD_EC( cudaBindTexture2D( NULL, texs2,
			       gd->dev_s2,
			       desc, NX, NY,
			       sizeof(DATATYPEV) * NX ) );
  GPUGD_EC( cudaBindTexture2D( NULL, texs3,
			       gd->dev_s3,
			       desc, NX, NY,
			       sizeof(DATATYPEV) * NX ) );

  GPUGD_EC( cudaBindTexture2D( NULL, texW11,
			       gd->dev_W11,
			       desc, NX, NY,
			       sizeof(DATATYPEV) * NX ) );
  GPUGD_EC( cudaBindTexture2D( NULL, texW21,
			       gd->dev_W21,
			       desc, NX, NY,
			       sizeof(DATATYPEV) * NX ) );
  GPUGD_EC( cudaBindTexture2D( NULL, texW31,
			       gd->dev_W31,
			       desc, NX, NY,
			       sizeof(DATATYPEV) * NX ) );
  GPUGD_EC( cudaBindTexture2D( NULL, texW12,
			       gd->dev_W12,
			       desc, NX, NY,
			       sizeof(DATATYPEV) * NX ) );
  GPUGD_EC( cudaBindTexture2D( NULL, texW22,
			       gd->dev_W22,
			       desc, NX, NY,
			       sizeof(DATATYPEV) * NX ) );
  GPUGD_EC( cudaBindTexture2D( NULL, texW32,
			       gd->dev_W32,
			       desc, NX, NY,
			       sizeof(DATATYPEV) * NX ) );
  GPUGD_EC( cudaBindTexture2D( NULL, texW13,
			       gd->dev_W13,
			       desc, NX, NY,
			       sizeof(DATATYPEV) * NX ) );
  GPUGD_EC( cudaBindTexture2D( NULL, texW23,
			       gd->dev_W23,
			       desc, NX, NY,
			       sizeof(DATATYPEV) * NX ) );
  GPUGD_EC( cudaBindTexture2D( NULL, texW33,
			       gd->dev_W33,
			       desc, NX, NY,
			       sizeof(DATATYPEV) * NX ) );

  GPUGD_EC( cudaBindTexture2D( NULL, texamdq1,
			       gd->dev_amdq1,
			       desc, NX, NY,
			       sizeof(DATATYPEV) * NX ) );
  GPUGD_EC( cudaBindTexture2D( NULL, texapdq1,
			       gd->dev_apdq1,
			       desc, NX, NY,
			       sizeof(DATATYPEV) * NX ) );
  GPUGD_EC( cudaBindTexture2D( NULL, texamdq2,
			       gd->dev_amdq2,
			       desc, NX, NY,
			       sizeof(DATATYPEV) * NX ) );
  GPUGD_EC( cudaBindTexture2D( NULL, texapdq2,
			       gd->dev_apdq2,
			       desc, NX, NY,
			       sizeof(DATATYPEV) * NX ) );
  GPUGD_EC( cudaBindTexture2D( NULL, texamdq3,
			       gd->dev_amdq3,
			       desc, NX, NY,
			       sizeof(DATATYPEV) * NX ) );
  GPUGD_EC( cudaBindTexture2D( NULL, texapdq3,
			       gd->dev_apdq3,
			       desc, NX, NY,
			       sizeof(DATATYPEV) * NX ) );

#if DEBUG >= 1
  GPUGD_EC( cudaMemcpy( gd->dev_debug1, debug1, 
			sizeof(DATATYPEV), 
			cudaMemcpyHostToDevice ) );	  
  GPUGD_EC( cudaMemcpy( gd->dev_debug2, debug2, 
			gridBitSize, 
			cudaMemcpyHostToDevice ) );	  
#endif /* DEBUG */

  GPUGD_EC( cudaMemcpy( gd->dev_dt, dt, 
			sizeof(DATATYPEV), 
			cudaMemcpyHostToDevice ) );	  
  GPUGD_EC( cudaMemcpy( gd->dev_dx, dx, 
			sizeof(DATATYPEV), 
			cudaMemcpyHostToDevice ) );	  
  GPUGD_EC( cudaMemcpy( gd->dev_dy, dy, 
			sizeof(DATATYPEV), 
			cudaMemcpyHostToDevice ) );	 
  GPUGD_EC( cudaMemcpy( gd->dev_T, T, 
			sizeof(DATATYPEV), 
			cudaMemcpyHostToDevice ) );	  
  GPUGD_EC( cudaMemcpy( gd->dev_MDX, MDX, 
			sizeof(int), 
			cudaMemcpyHostToDevice ) );	  
  GPUGD_EC( cudaMemcpy( gd->dev_MDY, MDY, 
			sizeof(int), 
			cudaMemcpyHostToDevice ) );	  

  GPUGD_EC( cudaMemcpy( gd->dev_SrcAu1, u1, 
			gridBitSize, 
			cudaMemcpyHostToDevice ) );	  
  GPUGD_EC( cudaMemcpy( gd->dev_SrcAu2, u2, 
			gridBitSize, 
			cudaMemcpyHostToDevice ) );	    
  GPUGD_EC( cudaMemcpy( gd->dev_SrcAu3, u3, 
			gridBitSize, 
			cudaMemcpyHostToDevice ) );	  

  GPUGD_EC( cudaMemcpy( gd->dev_SrcBu1, u1, 
			gridBitSize, 
			cudaMemcpyHostToDevice ) );	  
  GPUGD_EC( cudaMemcpy( gd->dev_SrcBu2, u2, 
			gridBitSize, 
			cudaMemcpyHostToDevice ) );	    
  GPUGD_EC( cudaMemcpy( gd->dev_SrcBu3, u3, 
			gridBitSize, 
			cudaMemcpyHostToDevice ) );	  

  GPUGD_EC( cudaMemcpyToSymbol(s1_ptr, 
			       &gd->dev_s1, 
			       sizeof(DATATYPEV *)) );
  GPUGD_EC( cudaMemcpyToSymbol(s2_ptr, 
			       &gd->dev_s2, 
			       sizeof(DATATYPEV *)) );
  GPUGD_EC( cudaMemcpyToSymbol(s3_ptr, 
			       &gd->dev_s3, 
			       sizeof(DATATYPEV *)) );

  GPUGD_EC( cudaMemcpyToSymbol(W11_ptr, 
			       &gd->dev_W11, 
			       sizeof(DATATYPEV *)) );
  GPUGD_EC( cudaMemcpyToSymbol(W21_ptr, 
			       &gd->dev_W21, 
			       sizeof(DATATYPEV *)) );
  GPUGD_EC( cudaMemcpyToSymbol(W31_ptr, 
			       &gd->dev_W31, 
			       sizeof(DATATYPEV *)) );
  GPUGD_EC( cudaMemcpyToSymbol(W12_ptr, 
			       &gd->dev_W12, 
			       sizeof(DATATYPEV *)) );
  GPUGD_EC( cudaMemcpyToSymbol(W22_ptr, 
			       &gd->dev_W22, 
			       sizeof(DATATYPEV *)) );
  GPUGD_EC( cudaMemcpyToSymbol(W32_ptr, 
			       &gd->dev_W32, 
			       sizeof(DATATYPEV *)) );
  GPUGD_EC( cudaMemcpyToSymbol(W13_ptr, 
			       &gd->dev_W13, 
			       sizeof(DATATYPEV *)) );
  GPUGD_EC( cudaMemcpyToSymbol(W23_ptr, 
			       &gd->dev_W23, 
			       sizeof(DATATYPEV *)) );
  GPUGD_EC( cudaMemcpyToSymbol(W33_ptr, 
			       &gd->dev_W33, 
			       sizeof(DATATYPEV *)) );

  GPUGD_EC( cudaMemcpyToSymbol(amdq1_ptr, 
			       &gd->dev_amdq1, 
			       sizeof(DATATYPEV *)) );
  GPUGD_EC( cudaMemcpyToSymbol(apdq1_ptr, 
			       &gd->dev_apdq1, 
			       sizeof(DATATYPEV *)) );
  GPUGD_EC( cudaMemcpyToSymbol(amdq2_ptr, 
			       &gd->dev_amdq2, 
			       sizeof(DATATYPEV *)) );
  GPUGD_EC( cudaMemcpyToSymbol(apdq2_ptr, 
			       &gd->dev_apdq2, 
			       sizeof(DATATYPEV *)) );
  GPUGD_EC( cudaMemcpyToSymbol(amdq3_ptr, 
			       &gd->dev_amdq3, 
			       sizeof(DATATYPEV *)) );
  GPUGD_EC( cudaMemcpyToSymbol(apdq3_ptr, 
			       &gd->dev_apdq3, 
			       sizeof(DATATYPEV *)) );
}

#endif  /* INC_DATA_DEFINITIONS_H */
