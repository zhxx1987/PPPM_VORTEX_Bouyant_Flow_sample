/*
test poisson solver based on a basic geometric multigrid method
this implementation solves a 3D poisson problem Lapace(p) = div(u)
with no-slip boundary condition for a staggered grid

Author :	xinxin zhang
			Master candidate
			Courant institute of Mathematical Science
			NewYork University
*/
#ifndef _MULTIGRID_2DKERNEL_CU_
#define _MULTIGRID_2DKERNEL_CU_
#define double_sp
#include <stdio.h>

#include "math_constants.h"
#define MG_MAx_level_tex 10
//constant index buffer for coarsest level
//in-kernel rb gauss sweep. 
typedef unsigned int uint;
typedef unsigned char uchar;
__constant__ uint left[64];
__constant__ uint right[64];
__constant__ uint top[64];
__constant__ uint down[64];



texture<float, 1, cudaReadModeElementType> x_level_tex;
texture<float, 1, cudaReadModeElementType> b_level_tex;
texture<int2, 1, cudaReadModeElementType> x_level_tex_double;
texture<int2, 1, cudaReadModeElementType> b_level_tex_double;
#define omega 1.0f

void setupTexture2D()
{
		x_level_tex.filterMode	= cudaFilterModePoint;
		b_level_tex.filterMode	= cudaFilterModePoint;
		x_level_tex_double.filterMode	= cudaFilterModePoint;
		b_level_tex_double.filterMode	= cudaFilterModePoint;
}
void bindTexture2D(size_t *offset,
				 const void * data,
				 const struct textureReference * 	texref,
				 size_t size)
{
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	cudaBindTexture(offset, texref, data, &desc, size);
}
void unbindTexture2D(const struct textureReference * 	texref)
{
	cudaUnbindTexture(texref);
}
void bindTexture2DDouble(size_t *offset,
				 const void * data,
				 const struct textureReference * 	texref,
				 size_t size)
{
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<int2>();
	cudaBindTexture(offset, texref, data, &desc, size);
}


__inline__ __device__ double tex1Dfetch_double(struct texture<int2, 1, cudaReadModeElementType>
  texref, const int& i)
{
#ifdef double_sp
 int2 v = tex1Dfetch(texref, i);
 return __hiloint2double(v.y, v.x);
#endif
 return 0;
}

void initIdxBuffer2D(uint dimx, uint dimy)
{
	uint lefth[64];
	uint righth[64];
	uint toph[64];
	uint downh[64];
	
	
	for (uint j=0;j<dimy;j++)for(uint i=0;i<dimx;i++)
	{
		int tid = j*dimx + i;
		int ip1 = (i+1)%dimx;
		int im1 = (i+dimx-1)%dimx;
		int jp1 = (j+1)%dimy;
		int jm1 = (j+dimy-1)%dimy;
		lefth[tid] = j*dimx+im1;
		righth[tid]= j*dimx+ip1;
		downh[tid] = jm1*dimx+i;
		toph[tid]  = jp1*dimx+i;
	}
	cudaMemcpyToSymbol(left, lefth, sizeof(uint)*64);
	cudaMemcpyToSymbol(right, righth, sizeof(uint)*64);
	cudaMemcpyToSymbol(top, toph, sizeof(uint)*64);
	cudaMemcpyToSymbol(down, downh, sizeof(uint)*64);

}
__global__
void exact_solve_period_2Dkernel(float *rhs,
						float *x,
						float *res,
						uint dimx,
						uint dimy,
						uint num_ele)
{
	uint tidx = threadIdx.x;
	__shared__ float x_n[64];
	__shared__ float b[64];

	if(tidx*2<num_ele) {x_n[tidx*2] = x[tidx*2]; b[tidx*2] = rhs[tidx*2]; }
	if((tidx*2+1)<num_ele) {x_n[tidx*2+1] = x[tidx*2+1]; b[tidx*2+1] = rhs[tidx*2+1]; }

	__syncthreads();

	for(uint t=0; t<100; t++)
	{
		float r;
		uint tid;
		uint i = tidx%dimx, j=tidx/dimx;
		if(i<dimx&&j<dimy)
		{
			tid = tidx;
			if((i+j)%2==0)//for all red cells
			{
				//if(tid<num_ele)
				{
					
					r=0;
					r += x_n[left[tid]];
					r += x_n[right[tid]];
					r += x_n[top[tid]];
					r += x_n[down[tid]];

					r -= b[tid];
					x_n[tid] = r * 0.25;
					
				}
			}
		__syncthreads();
			if((i+j)%2==1)//for all black cells
			{
				//if(tid<num_ele)
				{
					r = 0;
					r += x_n[left[tid]];
					r += x_n[right[tid]];
					r += x_n[top[tid]];
					r += x_n[down[tid]];

					r -= b[tid];
					x_n[tid] = r * 0.25;
					
					
				}
			}
		__syncthreads();
		}
		
		if(tidx*2<num_ele) res[tidx*2] = x_n[tidx*2];
		if(tidx*2+1<num_ele) res[tidx*2+1] = x_n[tidx*2+1];
	}
	
}
__global__
void exact_solve_period_2Dkerneld(double *rhs,
						double *x,
						double *res,
						uint dimx,
						uint dimy,
						uint num_ele)
{
	uint tidx = threadIdx.x;
	__shared__ double x_n[64];
	__shared__ double b[64];

	if(tidx*2<num_ele) {x_n[tidx*2] = x[tidx*2]; b[tidx*2] = rhs[tidx*2]; }
	if((tidx*2+1)<num_ele) {x_n[tidx*2+1] = x[tidx*2+1]; b[tidx*2+1] = rhs[tidx*2+1]; }

	__syncthreads();

	for(uint t=0; t<100; t++)
	{
		double r;
		uint tid;
		uint i = tidx%dimx, j=tidx/dimx;
		if(i<dimx&&j<dimy)
		{
			tid = tidx;
			if((i+j)%2==0)//for all red cells
			{
				//if(tid<num_ele)
				{
					
					r=0;
					r += x_n[left[tid]];
					r += x_n[right[tid]];
					r += x_n[top[tid]];
					r += x_n[down[tid]];

					r -= b[tid];
					x_n[tid] = r * 0.25;
					
				}
			}
		__syncthreads();
			if((i+j)%2==1)//for all black cells
			{
				//if(tid<num_ele)
				{
					r = 0;
					r += x_n[left[tid]];
					r += x_n[right[tid]];
					r += x_n[top[tid]];
					r += x_n[down[tid]];

					r -= b[tid];
					x_n[tid] = r * 0.25;
					
					
				}
			}
		__syncthreads();
		}
		
		if(tidx*2<num_ele) res[tidx*2] = x_n[tidx*2];
		if(tidx*2+1<num_ele) res[tidx*2+1] = x_n[tidx*2+1];
	}
	
}


__global__
void red_sweep_GS_periodic_2Dkernel_type2(float *res,
	uint dimx,uint dimy, uint ystride, 
	uint x_mod, uint y_mod, float invdiag,
	size_t x_offset,size_t b_offset,uint num_ele)
{
		
		uint i = blockIdx.x*blockDim.x + threadIdx.x;
		uint j = (blockIdx.y*blockDim.y + threadIdx.y);
		
		i*=2;
		if((j+1)%2)
			i++;
		uint tid = j * ystride + i;
		if(i<dimx&&j<dimy)
		{
			float rhs = tex1Dfetch(b_level_tex, tid+b_offset);
			
			uint tidxx = tid + x_offset;
			int left = (i==0)? tidxx + x_mod : tidxx - 1; 
			int right = (i==dimx-1)? tidxx - x_mod : tidxx + 1; 
			int up = (j==dimy-1)?tidxx - y_mod : tidxx + ystride; 
			int down = (j==0)?tidxx + y_mod : tidxx - ystride; 
	
			float	r =  (tex1Dfetch(x_level_tex,  left) + tex1Dfetch(x_level_tex, right));
				r += (tex1Dfetch(x_level_tex,    up) + tex1Dfetch(x_level_tex,  down));
			r-=rhs;
			
			res[tid] = (1-omega)*res[tid] + omega * r*invdiag;
		}
		
}
__global__
void black_sweep_GS_periodic_2Dkernel_type2(float *res,
	uint dimx,uint dimy,uint ystride, 
	uint x_mod, uint y_mod,float invdiag,
	size_t x_offset,size_t b_offset,uint num_ele)
{
		
		uint i = blockIdx.x*blockDim.x + threadIdx.x;
		uint j = (blockIdx.y*blockDim.y + threadIdx.y);
		
		i*=2;
		if((j)%2)
			i++;
		uint tid = j * ystride + i;
		if(i<dimx&&j<dimy)
		{
			float rhs = tex1Dfetch(b_level_tex, tid+b_offset);
			
			uint tidxx = tid + x_offset;
			int left = (i==0)? tidxx + x_mod : tidxx - 1; 
			int right = (i==dimx-1)? tidxx - x_mod : tidxx + 1; 
			int up = (j==dimy-1)?tidxx - y_mod : tidxx + ystride; 
			int down = (j==0)?tidxx + y_mod : tidxx - ystride;
			
		
			float	r =  (tex1Dfetch(x_level_tex,  left) + tex1Dfetch(x_level_tex, right));
				r += (tex1Dfetch(x_level_tex,    up) + tex1Dfetch(x_level_tex,  down));
			r-=rhs;
			
			res[tid] = (1-omega)*res[tid] + omega * r*invdiag;
		}
		
}

__global__
void compute_residual_periodic_2Dkernel(float * x,
									  float * b,
									  float * res,
									  uint dimx,uint dimy,
									  uint ystride, 
	uint x_mod, uint y_mod,float invdiag,
	size_t x_offset,size_t b_offset,uint num_ele)
{
	uint i = blockIdx.x*blockDim.x + threadIdx.x;
	uint j = (blockIdx.y*blockDim.y + threadIdx.y);
		
	uint tidx = j * ystride + i;
	if(i<dimx&&j<dimy)
	{
		float center = tex1Dfetch(x_level_tex, tidx + x_offset);
		//__syncthreads();
		float rhs = tex1Dfetch(b_level_tex, tidx + b_offset);
		//__syncthreads();
		uint tidxx = tidx + x_offset;
		int left = (i==0)? tidxx + x_mod : tidxx - 1; 
			int right = (i==dimx-1)? tidxx - x_mod : tidxx + 1; 
			int up = (j==dimy-1)?tidxx - y_mod : tidxx + ystride; 
			int down = (j==0)?tidxx + y_mod : tidxx - ystride;
		
		float	r =  (tex1Dfetch(x_level_tex,  left) + tex1Dfetch(x_level_tex, right));
			r += (tex1Dfetch(x_level_tex,    up) + tex1Dfetch(x_level_tex,  down));
		r -= 4*center;
		//__syncthreads();
		res[tidx] =  rhs - r;
	}
}

__global__
void restriction_periodic_2Dkernel(float * next_level,
	uint dimx,uint dimy,
	uint ystride, 
	uint x_mod, uint y_mod, float invdiag,
	size_t b_offset,uint num_ele)
{	
	uint i = blockIdx.x*blockDim.x + threadIdx.x;
	uint j = (blockIdx.y*blockDim.y + threadIdx.y);
		
	uint tidx = j*dimx + i;
	if(i<dimx&&j<dimy)
	{
		uint i_cur, j_cur;

		i_cur = i*2; j_cur = j*2; 
		uint tid2i2j2k = __umul24(j_cur,ystride) + i_cur + b_offset;
		uint tid2i2j_12k = tid2i2j2k + ystride;
		
		
		float r = 0;
		r += tex1Dfetch(b_level_tex, tid2i2j2k);
		r += tex1Dfetch(b_level_tex, tid2i2j2k + 1);
		r += tex1Dfetch(b_level_tex, tid2i2j_12k );
		r += tex1Dfetch(b_level_tex, tid2i2j_12k + 1);
		//__syncthreads();
		next_level[tidx] = r;

	}
}

__global__
void prolongation_periodic_2Dkernel(float * next_level,
	uint dimx,uint dimy,
	uint ystride, 
	uint x_mod, uint y_mod,float invdiag,
	size_t x_offset,uint num_ele)
{

	float r;
	
	uint i = blockIdx.x*blockDim.x + threadIdx.x;
	uint j = (blockIdx.y*blockDim.y + threadIdx.y);
		
	uint tidx = j*dimx + i;
	if(i<dimx&&j<dimy)
	{
		uint i_up, j_up;

		i_up = i/2; j_up = j/2; 
		r = tex1Dfetch(x_level_tex, j_up * ystride + i_up + x_offset);
		next_level[tidx] += r;
	}
}

////////////////////////////////////////////////////////////////////
///////////////            double version       ////////////////////
////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

__global__
void red_sweep_GS_periodic_2Dkerneld_type2(double *res,
	uint dimx,uint dimy,
	uint ystride, 
	uint x_mod, uint y_mod,double invdiag,
	size_t x_offset,size_t b_offset,uint num_ele)
{
		
		uint i = blockIdx.x*blockDim.x + threadIdx.x;
		uint j = (blockIdx.y*blockDim.y + threadIdx.y);
		
		i*=2;
		if((j+1)%2)
			i++;
		uint tid = j * ystride + i;
		if(i<dimx&&j<dimy)
		{
			double rhs = tex1Dfetch_double(b_level_tex_double, tid+b_offset);
			//__syncthreads();
			uint tidxx = tid + x_offset;
			int left = (i==0)?tidxx + x_mod : tidxx - 1; 
			int right = (i==dimx-1)?tidxx - x_mod:tidxx + 1; 
			int up = (j==dimy-1)?tidxx - y_mod:tidxx + ystride; 
			int down = (j==0)?tidxx + y_mod:tidxx - ystride;
		
			double	r =  (tex1Dfetch_double(x_level_tex_double,  left) + tex1Dfetch_double(x_level_tex_double, right));
				r += (tex1Dfetch_double(x_level_tex_double,    up) + tex1Dfetch_double(x_level_tex_double,  down));
			r-=rhs;
			//__syncthreads();
			res[tid] = (1-omega)*res[tid]+omega*r*invdiag;
		}
		
}
__global__
void black_sweep_GS_periodic_2Dkerneld_type2(double *res,
	uint dimx,uint dimy,uint dimz,
	uint ystride, 
	uint x_mod, uint y_mod,double invdiag,
	size_t x_offset,size_t b_offset,uint num_ele)
{
		
		uint i = blockIdx.x*blockDim.x + threadIdx.x;
		uint j = (blockIdx.y*blockDim.y + threadIdx.y);
		
		i*=2;
		if((j)%2)
			i++;
		uint tid = j * ystride + i;
		if(i<dimx&&j<dimy)
		{
			double rhs = tex1Dfetch_double(b_level_tex_double, tid+b_offset);
			//__syncthreads();
			uint tidxx = tid + x_offset;
			int left = (i==0)?tidxx + x_mod : tidxx - 1; 
			int right = (i==dimx-1)?tidxx - x_mod:tidxx + 1; 
			int up = (j==dimy-1)?tidxx - y_mod:tidxx + ystride; 
			int down = (j==0)?tidxx + y_mod:tidxx - ystride; 
		
			double	r =  (tex1Dfetch_double(x_level_tex_double,  left) + tex1Dfetch_double(x_level_tex_double, right));
				r += (tex1Dfetch_double(x_level_tex_double,    up) + tex1Dfetch_double(x_level_tex_double,  down));
			//__syncthreads();
			r -= rhs;
			res[tid] = (1-omega)*res[tid]+omega*r*invdiag;
		}
		
}
__global__
void compute_residual_periodic_2Dkernel2(double * x,
									  double * b,
									  double * res,
									  uint dimx,uint dimy,
									  uint ystride, 
	uint x_mod, uint y_mod, double invdiag,
	size_t x_offset,size_t b_offset,uint num_ele)
{
	uint i = blockIdx.x*blockDim.x + threadIdx.x;
	uint j = (blockIdx.y*blockDim.y + threadIdx.y);
		
	uint tidx = j * ystride + i;
	if(i<dimx&&j<dimy)
	{
		double center = tex1Dfetch_double(x_level_tex_double, tidx + x_offset);
		//__syncthreads();
		double rhs = tex1Dfetch_double(b_level_tex_double, tidx + b_offset);
		//__syncthreads();
		uint tidxx = tidx + x_offset;
		int left = (i==0)?tidxx + x_mod : tidxx - 1; 
			int right = (i==dimx-1)?tidxx - x_mod:tidxx + 1; 
			int up = (j==dimy-1)?tidxx - y_mod:tidxx + ystride; 
			int down = (j==0)?tidxx + y_mod:tidxx - ystride; 
		
		double	r =  (tex1Dfetch_double(x_level_tex_double,  left) + tex1Dfetch_double(x_level_tex_double, right));
			r += (tex1Dfetch_double(x_level_tex_double,    up) + tex1Dfetch_double(x_level_tex_double,  down));
		
		//__syncthreads();
		r -= 4*center;
		//__syncthreads();
		res[tidx] =  rhs - r;
	}
}

__global__
void restriction_periodic_2Dkernel2(double * next_level,
	uint dimx,uint dimy,
	uint ystride, 
	uint x_mod, uint y_mod, 
	double invdiag,
	size_t b_offset,uint num_ele)
{	
	uint i = blockIdx.x*blockDim.x + threadIdx.x;
	uint j = (blockIdx.y*blockDim.y + threadIdx.y);
		
	uint tidx =  j*dimx + i;
	if(i<dimx&&j<dimy)
	{
		
		uint i_cur, j_cur;

		i_cur = i*2; j_cur = j*2; 
		uint tid2i2j2k = j_cur*ystride + i_cur + b_offset;
		uint tid2i2j_12k = tid2i2j2k + ystride;
		
		
		double r = 0;
		r += tex1Dfetch_double(b_level_tex_double, tid2i2j2k);
		r += tex1Dfetch_double(b_level_tex_double, tid2i2j2k + 1);
		r += tex1Dfetch_double(b_level_tex_double, tid2i2j_12k );
		r += tex1Dfetch_double(b_level_tex_double, tid2i2j_12k + 1);
		//__syncthreads();
		next_level[tidx] = r;

	}
}

__global__
void prolongation_periodic_2Dkernel2(double * next_level,
	uint dimx,uint dimy,
	uint ystride, 
	uint x_mod, uint y_mod, double invdiag,
	size_t x_offset,uint num_ele)
{
	double r;
	uint i = blockIdx.x*blockDim.x + threadIdx.x;
	uint j = (blockIdx.y*blockDim.y + threadIdx.y);
		
	uint tidx = j*dimx + i;
	if(i<dimx&&j<dimy)
	{
		uint i_up, j_up;

		i_up = i/2; j_up = j/2;
		r = tex1Dfetch_double(x_level_tex_double, j_up * ystride + i_up + x_offset);
		//__syncthreads();
		next_level[tidx] += r;
	}
}

#endif // #ifndef 
