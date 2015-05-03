/*
test poisson solver based on a basic geometric multigrid method
this implementation solves a 3D poisson problem Lapace(p) = div(u)
with no-slip boundary condition for a staggered grid

Author :	xinxin zhang
			Master candidate
			Courant institute of Mathematical Science
			NewYork University
*/
#ifndef _MULTIGRID_KERNEL_CU_
#define _MULTIGRID_KERNEL_CU_
#define double_sp

#include <stdio.h>
//#include <cutil_inline.h>
//#include "cutil_math.h"
#include "math_constants.h"
#define MG_MAx_level_tex 10
//constant index buffer for coarsest level
//in-kernel rb gauss sweep. 
typedef unsigned int uint;
typedef unsigned char uchar;
__constant__ uint left[512];
__constant__ uint right[512];
__constant__ uint top[512];
__constant__ uint down[512];
__constant__ uint front[512];
__constant__ uint back[512];


texture<float, 1, cudaReadModeElementType> x_level_tex;
texture<float, 1, cudaReadModeElementType> b_level_tex;
texture<int2, 1, cudaReadModeElementType> x_level_tex_double;
texture<int2, 1, cudaReadModeElementType> b_level_tex_double;
#define omega 1.43f

void setupTexture()
{
		x_level_tex.filterMode	= cudaFilterModePoint;
		b_level_tex.filterMode	= cudaFilterModePoint;
		x_level_tex_double.filterMode	= cudaFilterModePoint;
		b_level_tex_double.filterMode	= cudaFilterModePoint;
}
void bindTexture(size_t *offset,
				 const void * data,
				 const struct textureReference * 	texref,
				 size_t size)
{
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	cudaBindTexture(offset, texref, data, &desc, size);
}
void bindTextureDouble(size_t *offset,
				 const void * data,
				 const struct textureReference * 	texref,
				 size_t size)
{
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<int2>();
	cudaBindTexture(offset, texref, data, &desc, size);
}

#ifndef __hiloint2double
__device__ double __hiloint2double( uint a, uint b )
{
 volatile union {
       double   d;
       signed int i[2];
 } cvt;
 cvt.i[0] = b;
 cvt.i[1] = a;
 return cvt.d;
}
#endif
__inline__ __device__ double tex1Dfetch_double(struct texture<int2, 1, cudaReadModeElementType>
  texref, const int& i)
{
#ifdef double_sp
 int2 v = tex1Dfetch(texref, i);
 return __hiloint2double(v.y, v.x);
#endif
 return 0;
}

void initIdxBuffer(uint dimx, uint dimy, uint dimz)
{
	uint num_cell = dimx*dimy*dimz;
	uint* lefth = new uint[num_cell];
	uint* righth = new uint[num_cell];
	uint* toph = new uint[num_cell];
	uint* downh = new uint[num_cell];
	uint* fronth = new uint[num_cell];
	uint* backh = new uint[num_cell];
	
	for (uint tid=0;tid<num_cell; tid++)
	{
		uint i,j,k;
		i = tid%dimx; j = (tid/dimx)%dimy; k = tid/(dimx*dimy);
		int im1=(int)i-1, ip1=(int)i+1, jm1=(int)j-1, jp1=(int)j+1, km1=(int)k-1, kp1=(int)k+1;
		if(im1<0) im1=dimx-1;
		if(jm1<0) jm1=dimy-1;
		if(km1<0) km1=dimz-1;
		if(ip1>dimx-1) ip1=0;
		if(jp1>dimy-1) jp1=0;
		if(kp1>dimz-1) kp1=0;
		lefth[tid] = k*dimx*dimy+j*dimx+im1;
		righth[tid]=k*dimx*dimy+j*dimx+ip1;
		downh[tid]=k*dimx*dimy+jm1*dimx+i;
		toph[tid]=k*dimx*dimy+jp1*dimx+i;
		fronth[tid]=km1*dimx*dimy+j*dimx+i;
		backh[tid]=kp1*dimx*dimy+dimx*j+i;
	}
	
	cudaMemcpyToSymbol(left, lefth, sizeof(uint)*num_cell);
	cudaMemcpyToSymbol(right, righth, sizeof(uint)*num_cell);
	cudaMemcpyToSymbol(top, toph, sizeof(uint)*num_cell);
	cudaMemcpyToSymbol(down, downh, sizeof(uint)*num_cell);
	cudaMemcpyToSymbol(front, fronth, sizeof(uint)*num_cell);
	cudaMemcpyToSymbol(back, backh, sizeof(uint)*num_cell);

	delete[] lefth;
	delete[] righth;
	delete[] toph;
	delete[] downh;
	delete[] fronth;
	delete[] backh;
}
__global__
void exact_solve_period3D_kernel(float *rhs,
						float *x,
						float *res,
						uint dimx,
						uint dimy,
						uint dimz,
						uint num_ele)
{
	uint tidx = threadIdx.x;
__shared__ float x_n[512];
__shared__ float b[512];
	
	
if(tidx*2<num_ele) {x_n[tidx*2] = x[tidx*2]; b[tidx*2] = rhs[tidx*2]; }
if(tidx*2+1<num_ele) {x_n[tidx*2+1] = x[tidx*2+1]; b[tidx*2+1] = rhs[tidx*2+1]; }
	__syncthreads();
	
	for(uint t=0; t<100; t++)
	{
		float r;
		uint tid = tidx;
		
		int i = tid%dimx, j = (tid/dimx)%dimy, k = tid/(dimx*dimy);
		if(i<dimx && j<dimy &&k<dimz)
		{
			if((i+j+k)%2==0)//for all red cells
			{
				if(tid<num_ele)
				{
					float lv = x_n[left[tid]];
					float rv = x_n[right[tid]];
					float tv = x_n[top[tid]];
					float dv = x_n[down[tid]];
					float fv = x_n[front[tid]];
					float bv = x_n[back[tid]];

					if(i-1<0) lv = 0;
					if(i+1>=dimx) rv = 0;
					if(j+1>=dimy) tv = 0;
					if(j-1<0) dv = 0;
					if(k-1<0) fv = 0;
					if(k+1>=dimz) bv = 0;
					r=0;
					r += lv;
					r += rv;
					r += tv;
					r += dv;
					r += fv;
					r += bv;

					r -= b[tid];
					x_n[tid] = r/6;
					
				}
			}
			__syncthreads();
			if((i+j+k)%2==1)//for all black cells
			{
				if(tid<num_ele)
				{
					float lv = x_n[left[tid]];
					float rv = x_n[right[tid]];
					float tv = x_n[top[tid]];
					float dv = x_n[down[tid]];
					float fv = x_n[front[tid]];
					float bv = x_n[back[tid]];

					if(i-1<0) lv = 0;
					if(i+1>=dimx) rv = 0;
					if(j+1>=dimy) tv = 0;
					if(j-1<0) dv = 0;
					if(k-1<0) fv = 0;
					if(k+1>=dimz) bv = 0;
					r=0;
					r += lv;
					r += rv;
					r += tv;
					r += dv;
					r += fv;
					r += bv;

					r -= b[tid];
					x_n[tid] = r/6;
					
					
				}
			}
			__syncthreads();
			
			if(tidx*2<num_ele) res[tidx*2] = x_n[tidx*2];
			if(tidx*2+1<num_ele) res[tidx*2+1] = x_n[tidx*2+1];
		}
	}
}
__global__
void exact_solve_period3D_jacobi_kernel(float *rhs,
						float *x,
						float *res,
						uint dimx,
						uint dimy,
						uint dimz,
						uint num_ele)
{
	uint tidx = threadIdx.x;
__shared__ float x_n[512];
__shared__ float x2_n[512];
__shared__ float b[512];
	
	
if(tidx*2<num_ele) {x_n[tidx*2] = x[tidx*2]; b[tidx*2] = rhs[tidx*2]; }
if(tidx*2+1<num_ele) {x_n[tidx*2+1] = x[tidx*2+1]; b[tidx*2+1] = rhs[tidx*2+1]; }
	__syncthreads();
	
	for(uint t=0; t<100; t++)
	{
		float r;
		uint tid = tidx;
		
		int i = tid%dimx, j = (tid/dimx)%dimy, k = tid/(dimx*dimy);
		if(i<dimx && j<dimy &&k<dimz)
		{
			//if((i+j+k)%2==0)//for all cells
			{
				if(tid<num_ele)
				{
					
					float lv = x_n[left[tid]];
					float rv = x_n[right[tid]];
					float tv = x_n[top[tid]];
					float dv = x_n[down[tid]];
					float fv = x_n[front[tid]];
					float bv = x_n[back[tid]];

					if(i-1<0) lv = 0;
					if(i+1>=dimx) rv = 0;
					if(j+1>=dimy) tv = 0;
					if(j-1<0) dv = 0;
					if(k-1<0) fv = 0;
					if(k+1>=dimz) bv = 0;
					r=0;
					r += lv;
					r += rv;
					r += tv;
					r += dv;
					r += fv;
					r += bv;

					r -= b[tid];
					x2_n[tid] = r/6;
					
				}
			}
			__syncthreads();
				//if((i+j+k)%2==1)//for all black cells
			{
				if(tid<num_ele)
				{
					float lv = x_n[left[tid]];
					float rv = x_n[right[tid]];
					float tv = x_n[top[tid]];
					float dv = x_n[down[tid]];
					float fv = x_n[front[tid]];
					float bv = x_n[back[tid]];

					if(i-1<0) lv = 0;
					if(i+1>=dimx) rv = 0;
					if(j+1>=dimy) tv = 0;
					if(j-1<0) dv = 0;
					if(k-1<0) fv = 0;
					if(k+1>=dimz) bv = 0;
					r=0;
					r += lv;
					r += rv;
					r += tv;
					r += dv;
					r += fv;
					r += bv;

					r -= b[tid];
					x_n[tid] = r/6;
					
					
				}
			}
			__syncthreads();
			
		}
	}
	if(tidx*2<num_ele) res[tidx*2] = x_n[tidx*2];
	if(tidx*2+1<num_ele) res[tidx*2+1] = x_n[tidx*2+1];
}





__global__
void exact_solve_period3D_kerneld(double *rhs,
						double *x,
						double *res,
						uint dimx,
						uint dimy,
						uint dimz,
						uint num_ele)
{
	uint tidx = threadIdx.x;
__shared__ double x_n[512];
__shared__ double b[512];
	
	
if(tidx*2<num_ele) {x_n[tidx*2] = x[tidx*2]; b[tidx*2] = rhs[tidx*2]; }
if(tidx*2+1<num_ele) {x_n[tidx*2+1] = x[tidx*2+1]; b[tidx*2+1] = rhs[tidx*2+1]; }
	__syncthreads();
	
	for(uint t=0; t<100; t++)
	{
		double r;
		uint tid = tidx;
		
		int i = tid%dimx, j = (tid/dimx)%dimy, k = tid/(dimx*dimy);
		if(i<dimx && j<dimy &&k<dimz)
		{
			if((i+j+k)%2==0)//for all red cells
			{
				if(tid<num_ele)
				{
					
					double lv = x_n[left[tid]];
					double rv = x_n[right[tid]];
					double tv = x_n[top[tid]];
					double dv = x_n[down[tid]];
					double fv = x_n[front[tid]];
					double bv = x_n[back[tid]];

					if(i-1<0) lv = 0;
					if(i+1>=dimx) rv = 0;
					if(j+1>=dimy) tv = 0;
					if(j-1<0) dv = 0;
					if(k-1<0) fv = 0;
					if(k+1>=dimz) bv = 0;
					r=0;
					r += lv;
					r += rv;
					r += tv;
					r += dv;
					r += fv;
					r += bv;

					r -= b[tid];
					x_n[tid] = r/6;
					
				}
			}
			__syncthreads();
			if((i+j+k)%2==1)//for all black cells
			{
				if(tid<num_ele)
				{
					double lv = x_n[left[tid]];
					double rv = x_n[right[tid]];
					double tv = x_n[top[tid]];
					double dv = x_n[down[tid]];
					double fv = x_n[front[tid]];
					double bv = x_n[back[tid]];

					if(i-1<0) lv = 0;
					if(i+1>=dimx) rv = 0;
					if(j+1>=dimy) tv = 0;
					if(j-1<0) dv = 0;
					if(k-1<0) fv = 0;
					if(k+1>=dimz) bv = 0;
					r=0;
					r += lv;
					r += rv;
					r += tv;
					r += dv;
					r += fv;
					r += bv;

					r -= b[tid];
					x_n[tid] = r/6;
					
					
				}
			}
			__syncthreads();
			
		}
	}
	if(tidx*2<num_ele) res[tidx*2] = x_n[tidx*2];
	if(tidx*2+1<num_ele) res[tidx*2+1] = x_n[tidx*2+1];
	
}

__global__
void exact_solve_period3D_jacobi_kerneld(double *rhs,
						double *x,
						double *res,
						uint dimx,
						uint dimy,
						uint dimz,
						uint num_ele)
{
	uint tidx = threadIdx.x;
__shared__ double x_n[512];
__shared__ double x2_n[512];
__shared__ double b[512];
	
	
if(tidx*2<num_ele) {x_n[tidx*2] = x[tidx*2]; b[tidx*2] = rhs[tidx*2]; }
if(tidx*2+1<num_ele) {x_n[tidx*2+1] = x[tidx*2+1]; b[tidx*2+1] = rhs[tidx*2+1]; }
	__syncthreads();
	
	for(uint t=0; t<100; t++)
	{
		double r;
		uint tid = tidx;
		
		int i = tid%dimx, j = (tid/dimx)%dimy, k = tid/(dimx*dimy);
		if(i<dimx && j<dimy &&k<dimz)
		{
			//if((i+j+k)%2==0)//for all cells
			{
				if(tid<num_ele)
				{
					
					double lv = x_n[left[tid]];double rv = x_n[right[tid]];
					double tv = x_n[top[tid]];double dv = x_n[down[tid]];
					double fv = x_n[front[tid]];double bv = x_n[back[tid]];

					if(i-1<0) lv = 0;if(i+1>=dimx) rv = 0;
					if(j+1>=dimy) tv = 0;if(j-1<0) dv = 0;
					if(k-1<0) fv = 0;if(k+1>=dimz) bv = 0;
					r=0;
					r += lv;r += rv;r += tv;
					r += dv;r += fv;r += bv;

					r -= b[tid];
					x2_n[tid] = r/6;
					
				}
			}
			__syncthreads();
				//if((i+j+k)%2==1)//for all cells
			{
				if(tid<num_ele)
				{
					double lv = x_n[left[tid]];double rv = x_n[right[tid]];
					double tv = x_n[top[tid]];double dv = x_n[down[tid]];
					double fv = x_n[front[tid]];double bv = x_n[back[tid]];

					if(i-1<0) lv = 0;if(i+1>=dimx) rv = 0;
					if(j+1>=dimy) tv = 0;if(j-1<0) dv = 0;
					if(k-1<0) fv = 0;if(k+1>=dimz) bv = 0;
					r=0;
					r += lv;r += rv;r += tv;r += dv;r += fv;r += bv;

					r -= b[tid];
					x_n[tid] = r/6;
					
					
				}
			}
			__syncthreads();
			
		}
	}
	if(tidx*2<num_ele) res[tidx*2] = x_n[tidx*2];
	if(tidx*2+1<num_ele) res[tidx*2+1] = x_n[tidx*2+1];
	
}


__global__
void red_sweep_GS_periodic_kernel_type2(float *res,
	uint dimx,uint dimy,uint dimz, float invny, float invnxny, uint slice, uint ystride, 
	uint x_mod, uint y_mod, uint z_mod, float invdiag,
	size_t x_offset,size_t b_offset,uint num_ele)
{
		
		uint i = blockIdx.x*blockDim.x + threadIdx.x;
		uint k = truncf((blockIdx.y*blockDim.y + threadIdx.y)*invny);
		uint j = (blockIdx.y*blockDim.y + threadIdx.y)-__umul24(dimy,k);
		
		i*=2;
		if((k+j+1)%2)
			i++;
		uint tid = k * slice + j * ystride + i;
		if(i<dimx&&j<dimy&&k<dimz)
		{
			float rhs = tex1Dfetch(b_level_tex, tid+b_offset);
			
			uint tidxx = tid + x_offset;
			int left = (i==0)?tidxx + x_mod : tidxx - 1; 
			int right = (i==(dimx-1))?tidxx - x_mod : tidxx + 1; 
			int up = (j==(dimy-1))?tidxx - y_mod: tidxx + ystride; 
			int down = (j==0)?tidxx + y_mod:tidxx - ystride; 
			int front = (k==0)?tidxx + (dimz-1)*slice:tidxx - slice; 
			int back = (k==(dimz-1))?tidxx - (dimz-1)*slice:tidxx + slice;


			float lv = tex1Dfetch(x_level_tex,  left);
			float rv = tex1Dfetch(x_level_tex, right);
			float uv = tex1Dfetch(x_level_tex,    up);
			float dv = tex1Dfetch(x_level_tex,  down);
			float fv = tex1Dfetch(x_level_tex, front);
			float bv = tex1Dfetch(x_level_tex,  back);
			
			
			if(i==0) lv = 0;
			if(i==(dimx-1)) rv = 0;			
			if(j==(dimy-1)) uv = 0;			
			if(j==0) dv = 0;			
			if(k==0) fv=0;			
			if(k==(dimz-1)) bv=0;

			float	r =  lv + rv;
				r += uv + dv;
				r += fv + bv;
			r-=rhs;
			
			res[tid] = (1-omega)*res[tid] + omega * r*invdiag;
		}
		
}
__global__
void black_sweep_GS_periodic_kernel_type2(float *res,
	uint dimx,uint dimy,uint dimz, float invny, float invnxny, uint slice, uint ystride, 
	uint x_mod, uint y_mod, uint z_mod, float invdiag,
	size_t x_offset,size_t b_offset,uint num_ele)
{
		
		uint i = blockIdx.x*blockDim.x + threadIdx.x;
		uint k = truncf((blockIdx.y*blockDim.y + threadIdx.y)*invny);
		uint j = (blockIdx.y*blockDim.y + threadIdx.y)-__umul24(dimy,k);
		
		i*=2;
		if((k+j)%2)
			i++;
		uint tid = k * slice + j * ystride + i;
		if(i<dimx&&j<dimy&&k<dimz)
		{
			float rhs = tex1Dfetch(b_level_tex, tid+b_offset);
			
			uint tidxx = tid + x_offset;
			int left = (i==0)?tidxx + x_mod : tidxx - 1; 
			int right = (i==(dimx-1))?tidxx - x_mod : tidxx + 1; 
			int up = (j==(dimy-1))?tidxx - y_mod: tidxx + ystride; 
			int down = (j==0)?tidxx + y_mod:tidxx - ystride; 
			int front = (k==0)?tidxx + (dimz-1)*slice:tidxx - slice; 
			int back = (k==(dimz-1))?tidxx - (dimz-1)*slice:tidxx + slice;
		
			float lv = tex1Dfetch(x_level_tex,  left);
			float rv = tex1Dfetch(x_level_tex, right);
			float uv = tex1Dfetch(x_level_tex,    up);
			float dv = tex1Dfetch(x_level_tex,  down);
			float fv = tex1Dfetch(x_level_tex, front);
			float bv = tex1Dfetch(x_level_tex,  back);
			if(i==0) lv = 0;			
			if(i==(dimx-1)) rv = 0;			
			if(j==(dimy-1)) uv = 0;			
			if(j==0) dv = 0;			
			if(k==0) fv=0;			
			if(k==(dimz-1)) bv=0;

			float	r =  lv + rv;
				r += uv + dv;
				r += fv + bv;
			r-=rhs;
			
			res[tid] = (1-omega)*res[tid] + omega * r*invdiag;
		}
		
}


__global__
void compute_residual_periodic_kernel(float * x,
									  float * b,
									  float * res,
									  uint dimx,uint dimy,uint dimz, float invny,
									  float invnxny, uint slice, uint ystride, 
	uint x_mod, uint y_mod, uint z_mod, float invdiag,
	size_t x_offset,size_t b_offset,uint num_ele)
{
	uint i = blockIdx.x*blockDim.x + threadIdx.x;
	uint k = truncf((blockIdx.y*blockDim.y + threadIdx.y)*invny);
	uint j = (blockIdx.y*blockDim.y + threadIdx.y)-__umul24(dimy,k);
		
	uint tidx = k * slice + j * ystride + i;
	if(i<dimx&&j<dimy&&k<dimz)
	{
		float center = tex1Dfetch(x_level_tex, tidx + x_offset);
		//__syncthreads();
		float rhs = tex1Dfetch(b_level_tex, tidx + b_offset);
		//__syncthreads();
		uint tidxx = tidx + x_offset;
		int left = (i==0)?tidxx + x_mod : tidxx - 1; 
			int right = (i==(dimx-1))?tidxx - x_mod : tidxx + 1; 
			int up = (j==(dimy-1))?tidxx - y_mod: tidxx + ystride; 
			int down = (j==0)?tidxx + y_mod:tidxx - ystride; 
			int front = (k==0)?tidxx + (dimz-1)*slice:tidxx - slice; 
			int back = (k==(dimz-1))?tidxx - (dimz-1)*slice:tidxx + slice;
		
			float lv = tex1Dfetch(x_level_tex,  left);
			float rv = tex1Dfetch(x_level_tex, right);
			float uv = tex1Dfetch(x_level_tex,    up);
			float dv = tex1Dfetch(x_level_tex,  down);
			float fv = tex1Dfetch(x_level_tex, front);
			float bv = tex1Dfetch(x_level_tex,  back);
			if(i==0) lv = 0;			
			if(i==(dimx-1)) rv = 0;			
			if(j==(dimy-1)) uv = 0;			
			if(j==0) dv = 0;			
			if(k==0) fv=0;			
			if(k==(dimz-1)) bv=0;

			float	r =  lv + rv;
				r += uv + dv;
				r += fv + bv;
		r -= 6*center;
		//__syncthreads();
		res[tidx] =  rhs - r;
	}
}

__global__
void restriction_periodic_kernel(float * next_level,
	uint dimx,uint dimy,uint dimz, float invny, float invnxny, 
	uint slice, uint ystride, 
	uint x_mod, uint y_mod, uint z_mod, float invdiag,
	size_t b_offset,uint num_ele)
{	
	uint i = blockIdx.x*blockDim.x + threadIdx.x;
	uint k = truncf((blockIdx.y*blockDim.y + threadIdx.y)*invny);
	uint j = (blockIdx.y*blockDim.y + threadIdx.y)-__umul24(dimy,k);
		
	uint tidx = k * __umul24(dimx,dimy) + j*dimx + i;
	if(i<dimx&&j<dimy&&k<dimz)
	{
		uint i_cur, j_cur, k_cur;

		i_cur = i*2; j_cur = j*2; k_cur = k*2;
		uint tid2i2j2k = k_cur*slice + __umul24(j_cur,ystride) + i_cur + b_offset;
		uint tid2i2j_12k = tid2i2j2k + ystride;
		uint tid2i2j2k_1 = tid2i2j2k + slice;
		uint tid2i2j_12k_1 = tid2i2j2k_1 + ystride;
		
		float r = 0;
		r += tex1Dfetch(b_level_tex, tid2i2j2k);
		r += tex1Dfetch(b_level_tex, tid2i2j2k + 1);
		r += tex1Dfetch(b_level_tex, tid2i2j_12k );
		r += tex1Dfetch(b_level_tex, tid2i2j_12k + 1);
		r += tex1Dfetch(b_level_tex, tid2i2j2k_1);
		r += tex1Dfetch(b_level_tex, tid2i2j2k_1 + 1);
		r += tex1Dfetch(b_level_tex, tid2i2j_12k_1);
		r += tex1Dfetch(b_level_tex, tid2i2j_12k_1 + 1);
		//__syncthreads();
		r = r*0.5;
		next_level[tidx] = r;

	}
}

__global__
void prolongation_periodic_kernel(float * next_level,
	uint dimx,uint dimy,uint dimz, float invny, float invnxny, 
	uint slice, uint ystride, 
	uint x_mod, uint y_mod, uint z_mod, float invdiag,
	size_t x_offset,uint num_ele)
{

	float r;
	
	uint i = blockIdx.x*blockDim.x + threadIdx.x;
	uint k = truncf((blockIdx.y*blockDim.y + threadIdx.y)*invny);
	uint j = (blockIdx.y*blockDim.y + threadIdx.y)-__umul24(dimy,k);
		
	uint tidx = k * __umul24(dimx,dimy) + j*dimx + i;
	if(i<dimx&&j<dimy&&k<dimz)
	{
		uint i_up, j_up, k_up;

		i_up = i/2; j_up = j/2; k_up = k/2;
		r = tex1Dfetch(x_level_tex, k_up*slice + j_up * ystride + i_up + x_offset);
		//__syncthreads();
		next_level[tidx] += r;
	}
}

////////////////////////////////////////////////////////////////////
///////////////            double version       ////////////////////
////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////
__global__
void red_sweep_GS_periodic_kerneld_type2(double *res,
	uint dimx,uint dimy,uint dimz, double invny, double invnxny, uint slice, uint ystride, 
	uint x_mod, uint y_mod, uint z_mod, double invdiag,
	size_t x_offset,size_t b_offset,uint num_ele)
{
		
		uint i = blockIdx.x*blockDim.x + threadIdx.x;
		uint k = truncf((blockIdx.y*blockDim.y + threadIdx.y)*invny);
		uint j = (blockIdx.y*blockDim.y + threadIdx.y)-__umul24(dimy,k);
		
		i*=2;
		if((k+j+1)%2)
			i++;
		uint tid = k * slice + j * ystride + i;
		if(i<dimx&&j<dimy&&k<dimz)
		{
			double rhs = tex1Dfetch_double(b_level_tex_double, tid+b_offset);
			
			uint tidxx = tid + x_offset;
			int left = (i==0)?tidxx + x_mod : tidxx - 1; 
			int right = (i==(dimx-1))?tidxx - x_mod : tidxx + 1; 
			int up = (j==(dimy-1))?tidxx - y_mod: tidxx + ystride; 
			int down = (j==0)?tidxx + y_mod:tidxx - ystride; 
			int front = (k==0)?tidxx + (dimz-1)*slice:tidxx - slice; 
			int back = (k==(dimz-1))?tidxx - (dimz-1)*slice:tidxx + slice;
			
			double lv = tex1Dfetch_double(x_level_tex_double,  left);
			double rv = tex1Dfetch_double(x_level_tex_double, right);
			double uv = tex1Dfetch_double(x_level_tex_double,    up);
			double dv = tex1Dfetch_double(x_level_tex_double,  down);
			double fv = tex1Dfetch_double(x_level_tex_double, front);
			double bv = tex1Dfetch_double(x_level_tex_double,  back);
			if(i==0) lv = 0;			
			if(i==(dimx-1)) rv = 0;			
			if(j==(dimy-1)) uv = 0;			
			if(j==0) dv = 0;			
			if(k==0) fv=0;			
			if(k==(dimz-1)) bv=0;

			double	r =  lv + rv;
				r += uv + dv;
				r += fv + bv;

			r-=rhs;
			
			res[tid] = (1-omega)*res[tid] + omega * r*invdiag;
		}
		
}
__global__
void black_sweep_GS_periodic_kerneld_type2(double *res,
	uint dimx,uint dimy,uint dimz, double invny, double invnxny, uint slice, uint ystride, 
	uint x_mod, uint y_mod, uint z_mod, double invdiag,
	size_t x_offset,size_t b_offset,uint num_ele)
{
		
		uint i = blockIdx.x*blockDim.x + threadIdx.x;
		uint k = truncf((blockIdx.y*blockDim.y + threadIdx.y)*invny);
		uint j = (blockIdx.y*blockDim.y + threadIdx.y)-__umul24(dimy,k);
		
		i*=2;
		if((k+j)%2)
			i++;
		uint tid = k * slice + j * ystride + i;
		if(i<dimx&&j<dimy&&k<dimz)
		{
			double rhs = tex1Dfetch_double(b_level_tex_double, tid+b_offset);
			
			uint tidxx = tid + x_offset;
			int left = (i==0)?tidxx + x_mod : tidxx - 1; 
			int right = (i==(dimx-1))?tidxx - x_mod : tidxx + 1; 
			int up = (j==(dimy-1))?tidxx - y_mod: tidxx + ystride; 
			int down = (j==0)?tidxx + y_mod:tidxx - ystride; 
			int front = (k==0)?tidxx + (dimz-1)*slice:tidxx - slice; 
			int back = (k==(dimz-1))?tidxx - (dimz-1)*slice:tidxx + slice;
	
			double lv = tex1Dfetch_double(x_level_tex_double,  left);
			double rv = tex1Dfetch_double(x_level_tex_double, right);
			double uv = tex1Dfetch_double(x_level_tex_double,    up);
			double dv = tex1Dfetch_double(x_level_tex_double,  down);
			double fv = tex1Dfetch_double(x_level_tex_double, front);
			double bv = tex1Dfetch_double(x_level_tex_double,  back);
			if(i==0) lv = 0;			
			if(i==(dimx-1)) rv = 0;			
			if(j==(dimy-1)) uv = 0;			
			if(j==0) dv = 0;			
			if(k==0) fv=0;			
			if(k==(dimz-1)) bv=0;

			double	r =  lv + rv;
				r += uv + dv;
				r += fv + bv;
			r-=rhs;
			
			res[tid] = (1-omega)*res[tid] + omega * r*invdiag;
		}
		
}
__global__
void compute_residual_periodic_kernel2(double * x,
									  double * b,
									  double * res,
									  uint dimx,uint dimy,uint dimz,
									  double invny, double invnxny, uint slice, uint ystride, 
	uint x_mod, uint y_mod, uint z_mod, double invdiag,
	size_t x_offset,size_t b_offset,uint num_ele)
{
	uint i = blockIdx.x*blockDim.x + threadIdx.x;
	uint k = truncf((blockIdx.y*blockDim.y + threadIdx.y)*invny);
	uint j = (blockIdx.y*blockDim.y + threadIdx.y)-__umul24(dimy,k);
		
	uint tidx = k * slice + j * ystride + i;
	if(i<dimx&&j<dimy&&k<dimz)
	{
		double center = tex1Dfetch_double(x_level_tex_double, tidx + x_offset);
		//__syncthreads();
		double rhs = tex1Dfetch_double(b_level_tex_double, tidx + b_offset);
		//__syncthreads();
		uint tidxx = tidx + x_offset;
		int left = (i==0)?tidxx + x_mod : tidxx - 1; 
		int right = (i==(dimx-1))?tidxx - x_mod : tidxx + 1; 
		int up = (j==(dimy-1))?tidxx - y_mod: tidxx + ystride; 
		int down = (j==0)?tidxx + y_mod:tidxx - ystride; 
		int front = (k==0)?tidxx + (dimz-1)*slice:tidxx - slice; 
		int back = (k==(dimz-1))?tidxx - (dimz-1)*slice:tidxx + slice;
		
		double lv = tex1Dfetch_double(x_level_tex_double,  left);
			double rv = tex1Dfetch_double(x_level_tex_double, right);
			double uv = tex1Dfetch_double(x_level_tex_double,    up);
			double dv = tex1Dfetch_double(x_level_tex_double,  down);
			double fv = tex1Dfetch_double(x_level_tex_double, front);
			double bv = tex1Dfetch_double(x_level_tex_double,  back);
			if(i==0) lv = 0;			
			if(i==(dimx-1)) rv = 0;			
			if(j==(dimy-1)) uv = 0;			
			if(j==0) dv = 0;			
			if(k==0) fv=0;			
			if(k==(dimz-1)) bv=0;

			double	r =  lv + rv;
				r += uv + dv;
				r += fv + bv;
		
		//__syncthreads();
		r -= 6*center;
		//__syncthreads();
		res[tidx] =  rhs - r;
	}
}

__global__
void restriction_periodic_kernel2(double * next_level,
	uint dimx,uint dimy,uint dimz, double invny, double invnxny, 
	uint slice, uint ystride, 
	uint x_mod, uint y_mod, uint z_mod, double invdiag,
	size_t b_offset,uint num_ele)
{	
	uint i = blockIdx.x*blockDim.x + threadIdx.x;
	uint k = truncf((blockIdx.y*blockDim.y + threadIdx.y)*invny);
	uint j = (blockIdx.y*blockDim.y + threadIdx.y)-__umul24(dimy,k);
		
	uint tidx = k * __umul24(dimx,dimy) + j*dimx + i;
	if(i<dimx&&j<dimy&&k<dimz)
	{
		
		uint i_cur, j_cur, k_cur;

		i_cur = i*2; j_cur = j*2; k_cur = k*2;
		uint tid2i2j2k = k_cur*slice + j_cur*ystride + i_cur + b_offset;
		uint tid2i2j_12k = tid2i2j2k + ystride;
		uint tid2i2j2k_1 = tid2i2j2k + slice;
		uint tid2i2j_12k_1 = tid2i2j2k_1 + ystride;
		
		double r = 0;
		r += tex1Dfetch_double(b_level_tex_double, tid2i2j2k);
		r += tex1Dfetch_double(b_level_tex_double, tid2i2j2k + 1);
		r += tex1Dfetch_double(b_level_tex_double, tid2i2j_12k );
		r += tex1Dfetch_double(b_level_tex_double, tid2i2j_12k + 1);
		r += tex1Dfetch_double(b_level_tex_double, tid2i2j2k_1);
		r += tex1Dfetch_double(b_level_tex_double, tid2i2j2k_1 + 1);
		r += tex1Dfetch_double(b_level_tex_double, tid2i2j_12k_1);
		r += tex1Dfetch_double(b_level_tex_double, tid2i2j_12k_1 + 1);
		//__syncthreads();
		r = r*0.5;
		next_level[tidx] = r;

	}
}

__global__
void prolongation_periodic_kernel2(double * next_level,
	uint dimx,uint dimy,uint dimz, double invny, double invnxny, 
	uint slice, uint ystride, 
	uint x_mod, uint y_mod, uint z_mod, double invdiag,
	size_t x_offset,uint num_ele)
{
	double r;
	uint i = blockIdx.x*blockDim.x + threadIdx.x;
	uint k = truncf((blockIdx.y*blockDim.y + threadIdx.y)*invny);
	uint j = (blockIdx.y*blockDim.y + threadIdx.y)-__umul24(dimy,k);
		
	uint tidx = k * __umul24(dimx,dimy) + j*dimx + i;
	if(i<dimx&&j<dimy&&k<dimz)
	{
		uint i_up, j_up, k_up;

		i_up = i/2; j_up = j/2; k_up = k/2;
		r = tex1Dfetch_double(x_level_tex_double, k_up*slice + j_up * ystride + i_up + x_offset);
		//__syncthreads();
		next_level[tidx] += r;
	}
}

#endif // #ifndef 
