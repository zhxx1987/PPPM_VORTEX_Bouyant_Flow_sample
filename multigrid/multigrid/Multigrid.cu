/*
test poisson solver based on a basic geometric multigrid method
this implementation solves a 3D poisson problem Lapace(p) = div(u)
with no-slip boundary condition for a staggered grid

Author :	xinxin zhang
			Master candidate
			Courant institute of Mathematical Science
			NewYork University
*/
//#include <cutil_inline.h>
#include <GL/glew.h>
#include <stdlib.h>

#include <string.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <cuda_gl_interop.h>
#include "MultiGrid_kernel.cu"
#include <helper_cuda.h>

extern "C"
{
	void mg_InitIdxBuffer(int dimx, int dimy, int dimz)
	{
		initIdxBuffer(dimx, dimy, dimz);
	}


	void mg_exact_periodic3D(float * x, 
						   float * b, 
						   float * res, 
						   int dimx,
						   int dimy,
						   int dimz)
	{
		
			exact_solve_period3D_jacobi_kernel<<<1,min(512, dimx*dimy*dimz)>>>(b, x, res, dimx, dimy, dimz, dimx*dimy*dimz);
		getLastCudaError("exact_solve_periodic3D failed!\n");
	}
	void mg_exact_periodic3D2(double * x, 
						   double * b, 
						   double * res, 
						   int dimx,
						   int dimy,
						   int dimz)
	{
		exact_solve_period3D_jacobi_kerneld<<<1,min(512, dimx*dimy*dimz)>>>(b, x, res, dimx, dimy, dimz, dimx*dimy*dimz);
		getLastCudaError("exact_solve_periodic3D failed!\n");
	}
	void mg_zerofy(float * data, size_t size)
	{
		cudaMemset((void *)data, 0, size);
		getLastCudaError("zerofy failed!\n");
	}

	void mg_RBGS_periodic3D(float * x,
							float * b,
							float * res,
							int t,
							int dimx,
							int dimy,
							int dimz)
	{
		for(int i=0 ; i<t; i++)
		{
			size_t x_offset, b_offset;
			bindTexture(&x_offset,
						x,
						&x_level_tex,
						sizeof(float)*dimx*dimy*dimz);
			bindTexture(&b_offset,
						b,
						&b_level_tex,
						sizeof(float)*dimx*dimy*dimz);
			
			
			dim3 threads(32,16);
			dim3 blocks(dimx/32 + (!(dimx%32)?0:1), 
				dimz*dimy/16 + (!(dimz*dimy%16)?0:1));
			
			
			red_sweep_GS_periodic_kernel_type2<<<blocks, threads>>>(x, 
				dimx, dimy, dimz, 1.0/(float)dimy, 1.0/(float)(dimx*dimy),
				dimx*dimy, dimx, dimx-1, (dimy - 1)*dimx, 
			(dimz-1)*dimx*dimy, 1.0/6.0, x_offset, b_offset,dimx*dimy*dimz);

			getLastCudaError("red_sweep failed!\n");
			
			
			black_sweep_GS_periodic_kernel_type2<<<blocks, threads>>>(x, 
				dimx, dimy, dimz, 1.0/(float)dimy, 1.0/(float)(dimx*dimy),
				dimx*dimy, dimx, dimx-1, (dimy - 1)*dimx, 
			(dimz-1)*dimx*dimy, 1.0/6.0, x_offset, b_offset,dimx*dimy*dimz);

			getLastCudaError("black sweep failed!\n");
		}
	}

	void mg_ComputeResidualPeriodic3D(float *x,
									  float *b,
									  float *r,
									  int dimx, 
									  int dimy,
									  int dimz)
	{
		size_t x_offset;
		size_t b_offset;

		bindTexture(&x_offset,
					x,
					&x_level_tex,
					sizeof(float)*dimx*dimy*dimz);
		bindTexture(&b_offset,
					b,
					&b_level_tex,
					sizeof(float)*dimx*dimy*dimz);
		
		dim3 threads(32,16);
		dim3 blocks(dimx/32 + (!(dimx%32)?0:1), dimz*dimy/16 + (!(dimz*dimy%16)?0:1));
		compute_residual_periodic_kernel<<<blocks, threads>>>(x,
														 b,
														 r,
											dimx, dimy, dimz, 
							1.0/(float)dimy, 1.0/(float)(dimx*dimy),
							dimx*dimy, dimx, dimx-1, (dimy - 1)*dimx, 
			(dimz-1)*dimx*dimy, 1.0/6.0, x_offset, b_offset,dimx*dimy*dimz);

		getLastCudaError("get residual failed!\n");
	}
	void mg_RestrictionPeriodic3D(float * next_level,
								  float * curr_level,
								  int next_dimx,
								  int next_dimy,
								  int next_dimz)
	{
		size_t b_offset;
		

		bindTexture(&b_offset,
					curr_level,
					&b_level_tex,
					sizeof(float)*next_dimx*next_dimy*next_dimz*8);
		
		unsigned int block = next_dimx*next_dimy*next_dimz / 256 + (!(next_dimx*next_dimy*next_dimz%256)?0:1) ;
		block = block>32673?32673:block;
		dim3 threads(32,16);
		dim3 blocks(next_dimx/32 + (!(next_dimx%32)?0:1), next_dimz*next_dimy/16 + (!(next_dimz*next_dimy%16)?0:1));

		restriction_periodic_kernel<<<blocks, threads>>>(next_level,
													next_dimx, next_dimy, next_dimz, 
													1.0/(float)next_dimy, 1.0/(float)(next_dimx*next_dimy),
													next_dimx*next_dimy*4, next_dimx*2, 
													next_dimx-1, (next_dimy - 1)*next_dimx,
													(next_dimz-1)*next_dimx*next_dimy, 
													1.0/6.0, b_offset,
													next_dimx*next_dimy*next_dimz);

		getLastCudaError("restriction failed!\n");
	}
	void mg_ProlongationPeriodic3D(float * next_level,
								  float * coaser_level,
								  int next_dimx,
								  int next_dimy,
								  int next_dimz)
	{
		size_t x_offset;
		

		bindTexture(&x_offset,
					coaser_level,
					&x_level_tex,
					sizeof(float)*next_dimx*next_dimy*next_dimz/8);
		
		
		dim3 threads(32,16);
		dim3 blocks(next_dimx/32 + (!(next_dimx%32)?0:1), next_dimz*next_dimy/16 + (!(next_dimz*next_dimy%16)?0:1));

		prolongation_periodic_kernel<<<blocks, threads>>>(next_level,
													next_dimx, next_dimy, next_dimz, 
													1.0/(float)next_dimy, 1.0/(float)(next_dimx*next_dimy),
													next_dimx*next_dimy/4, next_dimx/2, 
													next_dimx-1, (next_dimy - 1)*next_dimx,
													(next_dimz-1)*next_dimx*next_dimy, 
													1.0/6.0, x_offset,
													next_dimx*next_dimy*next_dimz);

		getLastCudaError("prolongation failed!\n");
	}



	////////////////////////////////////////////////////////////////////
	///////////////            double version       ////////////////////
	////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////
	void mg_RBGS_periodic3D2(double * x,
							double * b,
							double * res,
							int t,
							int dimx,
							int dimy,
							int dimz)
	{
		for(int i=0 ; i<t; i++)
		{
			size_t x_offset, b_offset;
			bindTextureDouble(&x_offset,
						x,
						&x_level_tex_double,
						sizeof(double)*dimx*dimy*dimz);
			bindTextureDouble(&b_offset,
						b,
						&b_level_tex_double,
						sizeof(double)*dimx*dimy*dimz);
			dim3 threads(32,16);
			dim3 blocks(dimx/32 + (!(dimx%32)?0:1), dimz*dimy/16 + (!(dimz*dimy%16)?0:1));
			red_sweep_GS_periodic_kerneld_type2<<<blocks, threads>>>(x, 
				dimx, dimy, dimz, 1.0/(double)dimy, 1.0/(double)(dimx*dimy),
				dimx*dimy, dimx, dimx-1, (dimy - 1)*dimx, 
			(dimz-1)*dimx*dimy, 1.0/6.0, x_offset, b_offset,dimx*dimy*dimz);

			getLastCudaError("red_sweep failed!\n");
			black_sweep_GS_periodic_kerneld_type2<<<blocks, threads>>>(x, 
				dimx, dimy, dimz, 1.0/(double)dimy, 1.0/(double)(dimx*dimy),
				dimx*dimy, dimx, dimx-1, (dimy - 1)*dimx, 
			(dimz-1)*dimx*dimy, 1.0/6.0, x_offset, b_offset,dimx*dimy*dimz);

			getLastCudaError("black sweep failed!\n");


			getLastCudaError("black sweep failed!\n");
		}
	}

	void mg_ComputeResidualPeriodic3D2(double *x,
									  double *b,
									  double *r,
									  int dimx, 
									  int dimy,
									  int dimz)
	{
		size_t x_offset;
		size_t b_offset;
		bindTextureDouble(&x_offset,
						x,
						&x_level_tex_double,
						sizeof(double)*dimx*dimy*dimz);
		bindTextureDouble(&b_offset,
						b,
						&b_level_tex_double,
						sizeof(double)*dimx*dimy*dimz);
		dim3 threads(32,16);
		dim3 blocks(dimx/32 + (!(dimx%32)?0:1),
			dimz*dimy/16 + (!(dimz*dimy%16)?0:1));

		
		compute_residual_periodic_kernel2<<<blocks, threads>>>(x,
														 b,
														 r,
														 dimx, dimy, dimz, 
														 1.0/(double)dimy, 
														 1.0/(double)(dimx*dimy),
														 dimx*dimy, dimx, dimx-1, 
														 (dimy - 1)*dimx, 
		(dimz-1)*dimx*dimy, 1.0/6.0, x_offset, b_offset,dimx*dimy*dimz);

		getLastCudaError("get residual failed!\n");
	}
	void mg_RestrictionPeriodic3D2(double * next_level,
								  double * curr_level,
								  int next_dimx,
								  int next_dimy,
								  int next_dimz)
	{
		size_t b_offset;
		

		bindTextureDouble(&b_offset,
					curr_level,
					&b_level_tex_double,
					sizeof(double)*next_dimx*next_dimy*next_dimz*8);
		dim3 threads(32,16);
		dim3 blocks(next_dimx/32 + (!(next_dimx%32)?0:1), next_dimz*next_dimy/16 + (!(next_dimz*next_dimy%16)?0:1));

		
		restriction_periodic_kernel2<<<blocks, threads>>>(next_level,
													next_dimx, next_dimy, next_dimz, 
													1.0/(double)next_dimy, 1.0/(double)(next_dimx*next_dimy),
													next_dimx*next_dimy*4, next_dimx*2, 
													next_dimx-1, (next_dimy - 1)*next_dimx,
													(next_dimz-1)*next_dimx*next_dimy, 
													1.0/6.0, b_offset,
													next_dimx*next_dimy*next_dimz);

		getLastCudaError("restriction failed!\n");
	}
	void mg_ProlongationPeriodic3D2(double * next_level,
								  double * coaser_level,
								  int next_dimx,
								  int next_dimy,
								  int next_dimz)
	{
		size_t x_offset;
		

		bindTextureDouble(&x_offset,
					coaser_level,
					&x_level_tex_double,
					sizeof(double)*next_dimx*next_dimy*next_dimz/8);
		
		dim3 threads(32,16);
		dim3 blocks(next_dimx/32 + (!(next_dimx%32)?0:1), next_dimz*next_dimy/16 + (!(next_dimz*next_dimy%16)?0:1));

		prolongation_periodic_kernel2<<<blocks, threads>>>(next_level,
													next_dimx, next_dimy, next_dimz, 
													1.0/(double)next_dimy, 1.0/(double)(next_dimx*next_dimy),
													next_dimx*next_dimy/4, next_dimx/2, 
													next_dimx-1, (next_dimy - 1)*next_dimx,
													(next_dimz-1)*next_dimx*next_dimy, 
													1.0/6.0, x_offset,
													next_dimx*next_dimy*next_dimz);

		getLastCudaError("prolongation failed!\n");
	}
}