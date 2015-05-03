/*
test poisson solver based on a basic geometric multigrid method
this implementation solves a 2D poisson problem Lapace(p) = div(u)
with no-slip boundary condition for a staggered grid

Author :	xinxin zhang
			Master candidate
			Courant institute of Mathematical Science
			NewYork University
*/
#include <GL/glew.h>
#include <stdlib.h>

#include <string.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <cuda_gl_interop.h>
#include "MultiGrid2D_kernel.cu"
#include <helper_cuda.h> 

extern "C"
{
	void mg_InitIdxBuffer2D(uint dimx, uint dimy )
	{
		initIdxBuffer2D(dimx, dimy );
	}


	void mg_exact_periodic2D(float * x, 
						   float * b, 
						   float * res, 
						   uint dimx,
						   uint dimy)
	{
		
			exact_solve_period_2Dkernel<<<1,256>>>(b, x, res, dimx, dimy, dimx*dimy);
		getLastCudaError("exact_solve_periodic2D failed!\n");
	}
	void mg_exact_periodic2D2(double * x, 
						   double * b, 
						   double * res, 
						   uint dimx,
						   uint dimy)
	{
		exact_solve_period_2Dkerneld<<<1,256>>>(b, x, res, dimx, dimy, dimx*dimy);
		getLastCudaError("exact_solve_periodic2D failed!\n");
	}
	void mg_zerofy2D(float * data, size_t size)
	{
		checkCudaErrors(cudaMemset((void *)data, 0, size));
		getLastCudaError("zerofy failed!\n");
	}

	void mg_RBGS_periodic2D(float * x,
							float * b,
							float * res,
							uint t,
							uint dimx,
							uint dimy)
	{
		for(uint i=0 ; i<t; i++)
		{
			size_t x_offset, b_offset;
			bindTexture2D(&x_offset,
						x,
						&x_level_tex,
						sizeof(float)*dimx*dimy);
			bindTexture2D(&b_offset,
						b,
						&b_level_tex,
						sizeof(float)*dimx*dimy);
			
			
			dim3 threads(32,16);
			dim3 blocks(dimx/32 + (!(dimx%32)?0:1), 
				dimy/16 + (!(dimy%16)?0:1));
			
			
			red_sweep_GS_periodic_2Dkernel_type2<<<blocks, threads>>>(x, 
				dimx, dimy,
				dimx, dimx-1, (dimy - 1)*dimx, 1.0/4.0, x_offset, b_offset,dimx*dimy);

			getLastCudaError("red_sweep2D failed!\n");
			
			
			black_sweep_GS_periodic_2Dkernel_type2<<<blocks, threads>>>(x, 
				dimx, dimy, dimx,
				dimx-1, (dimy - 1)*dimx, 
				1.0/4.0, x_offset, b_offset,dimx*dimy);
			
			unbindTexture2D(&x_level_tex);
			unbindTexture2D(&b_level_tex);
			getLastCudaError("black sweep 2D failed!\n");
		}
	}

	void mg_ComputeResidualPeriodic2D(float *x,
									  float *b,
									  float *r,
									  uint dimx, 
									  uint dimy)
	{
		size_t x_offset;
		size_t b_offset;

		bindTexture2D(&x_offset,
					x,
					&x_level_tex,
					sizeof(float)*dimx*dimy);
		bindTexture2D(&b_offset,
					b,
					&b_level_tex,
					sizeof(float)*dimx*dimy);
		
		dim3 threads(32,16);
		dim3 blocks(dimx/32 + (!(dimx%32)?0:1), dimy/16 + (!(dimy%16)?0:1));
		compute_residual_periodic_2Dkernel<<<blocks, threads>>>(x,
														 b,
														 r, dimx, dimy, dimx, dimx-1, (dimy - 1)*dimx,
														 1.0/4.0, x_offset, b_offset,dimx*dimy);
		

		unbindTexture2D(&x_level_tex);
		unbindTexture2D(&b_level_tex);
		getLastCudaError("get residual failed!\n");
	}
	void mg_RestrictionPeriodic2D(float * next_level,
								  float * curr_level,
								  uint next_dimx,
								  uint next_dimy)
	{
		size_t b_offset;
		

		bindTexture2D(&b_offset,
					curr_level,
					&b_level_tex,
					sizeof(float)*next_dimx*next_dimy*4);
		
		
		dim3 threads(32,16);
		dim3 blocks(next_dimx/32 + (!(next_dimx%32)?0:1), next_dimy/16 + (!(next_dimy%16)?0:1));

		restriction_periodic_2Dkernel<<<blocks, threads>>>(next_level,
													next_dimx, next_dimy,
													next_dimx*2, 
													next_dimx-1, (next_dimy - 1)*next_dimx,
													1.0/4.0, b_offset,
													next_dimx*next_dimy);
		
			unbindTexture2D(&b_level_tex);
		getLastCudaError("restriction failed 2D!\n");
	}
	void mg_ProlongationPeriodic2D(float * next_level,
								  float * coaser_level,
								  uint next_dimx,
								  uint next_dimy)
	{
		size_t x_offset;
		

		bindTexture2D(&x_offset,
					coaser_level,
					&x_level_tex,
					sizeof(float)*next_dimx*next_dimy/4);
		
		
		dim3 threads(32,16);
		dim3 blocks(next_dimx/32 + (!(next_dimx%32)?0:1), next_dimy/16 + (!(next_dimy%16)?0:1));

		prolongation_periodic_2Dkernel<<<blocks, threads>>>(next_level,
													next_dimx, next_dimy,  
													next_dimx/2, 
													next_dimx-1, (next_dimy - 1)*next_dimx, 
													1.0/4.0, x_offset,
													next_dimx*next_dimy);
		unbindTexture2D(&x_level_tex);
		getLastCudaError("prolongation 2D failed!\n");
	}



	////////////////////////////////////////////////////////////////////
	///////////////            double version       ////////////////////
	////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////
	void mg_RBGS_periodic2D2(double * x,
							double * b,
							double * res,
							uint t,
							uint dimx,
							uint dimy)
	{
		for(uint i=0 ; i<t; i++)
		{
			size_t x_offset, b_offset;
			bindTexture2DDouble(&x_offset,
						x,
						&x_level_tex_double,
						sizeof(double)*dimx*dimy);
			bindTexture2DDouble(&b_offset,
						b,
						&b_level_tex_double,
						sizeof(double)*dimx*dimy);
			dim3 threads(32,16);
			dim3 blocks(dimx/32 + (!(dimx%32)?0:1), dimy/16 + (!(dimy%16)?0:1));
			red_sweep_GS_periodic_2Dkerneld_type2<<<blocks, threads>>>(x, 
				dimx, dimy,
				dimx, dimx-1, (dimy - 1)*dimx,
				1.0/4.0, x_offset, b_offset,dimx*dimy);

			getLastCudaError("red_sweep 2D double failed!\n");
			
			black_sweep_GS_periodic_2Dkerneld_type2<<<blocks, threads>>>(x, 
				dimx, dimy, 1.0/(double)dimy,
				dimx, dimx-1, (dimy - 1)*dimx,
				1.0/4.0, x_offset, b_offset,dimx*dimy);

			getLastCudaError("black sweep 2D double failed!\n");


			
		}
	}

	void mg_ComputeResidualPeriodic2D2(double *x,
									  double *b,
									  double *r,
									  uint dimx, 
									  uint dimy)
	{
		size_t x_offset;
		size_t b_offset;
		bindTexture2DDouble(&x_offset,
						x,
						&x_level_tex_double,
						sizeof(double)*dimx*dimy);
		bindTexture2DDouble(&b_offset,
						b,
						&b_level_tex_double,
						sizeof(double)*dimx*dimy);
		dim3 threads(32,16);
		dim3 blocks(dimx/32 + (!(dimx%32)?0:1),
			dimy/16 + (!(dimy%16)?0:1));

		
		compute_residual_periodic_2Dkernel2<<<blocks, threads>>>(x,
														 b,
														 r,
														 dimx, dimy, 
														 dimx, dimx-1, 
														 (dimy - 1)*dimx,
														 1.0/4.0, x_offset, b_offset,
														 dimx*dimy);

		getLastCudaError("get residual 2D d failed!\n");
	}
	void mg_RestrictionPeriodic2D2(double * next_level,
								  double * curr_level,
								  uint next_dimx,
								  uint next_dimy)
	{
		size_t b_offset;
		

		bindTexture2DDouble(&b_offset,
					curr_level,
					&b_level_tex_double,
					sizeof(double)*next_dimx*next_dimy*4);
		dim3 threads(32,16);
		dim3 blocks(next_dimx/32 + (!(next_dimx%32)?0:1), next_dimy/16 + (!(next_dimy%16)?0:1));

		
		restriction_periodic_2Dkernel2<<<blocks, threads>>>(next_level,
													next_dimx, next_dimy, 
													next_dimx*2, 
													next_dimx-1, (next_dimy - 1)*next_dimx,
													1.0/4.0, b_offset,
													next_dimx*next_dimy);

		getLastCudaError("restriction 2D failed!\n");
	}
	void mg_ProlongationPeriodic2D2(double * next_level,
								  double * coaser_level,
								  uint next_dimx,
								  uint next_dimy)
	{
		size_t x_offset;
		

		bindTexture2DDouble(&x_offset,
					coaser_level,
					&x_level_tex_double,
					sizeof(double)*next_dimx*next_dimy/4);
		
		dim3 threads(32,16);
		dim3 blocks(next_dimx/32 + (!(next_dimx%32)?0:1), next_dimy/16 + (!(next_dimy%16)?0:1));

		prolongation_periodic_2Dkernel2<<<blocks, threads>>>(next_level,
													next_dimx, next_dimy,
													next_dimx/2, 
													next_dimx-1, (next_dimy - 1)*next_dimx,
													1.0/4.0, x_offset,
													next_dimx*next_dimy);

		getLastCudaError("prolongation 2D d failed!\n");
	}
}