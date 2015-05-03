/*
test poisson solver based on a basic geometric multigrid method
this implementation solves a 3D poisson problem Lapace(p) = div(u)
with no-slip boundary condition for a staggered grid



Author :	Xinxin Zhang
			Master candidate
			Courant institute of Mathematical Science
			NewYork University
*/
#ifndef __MULTI_GRID2D_CUH__
#define __MULTI_GRID2D_CUH__
typedef unsigned int uint;

extern "C"
{
	void mg_InitIdxBuffer2D(uint dimx, uint dimy);
	void mg_exact_periodic2D(float * x, 
						   float * b, 
						   float * res, 
						   uint dimx,
						   uint dimy );
	

	void mg_zerofy2D(void * data, size_t size);



	void mg_RBGS_periodic2D(float * x,
							float * b,
							float * res,
							uint t,
							uint dimx,
							uint dimy );

	void mg_ComputeResidualPeriodic2D(float *x,
									  float *b,
									  float *r,
									  uint dimx, 
									  uint dimy );

	void mg_RestrictionPeriodic2D(float * next_level,
								  float * curr_level,
								  uint next_dimx,
								  uint next_dimy );

	void mg_ProlongationPeriodic2D(float * next_level,
								  float * coaser_level,
								  uint next_dimx,
								  uint next_dimy );

	/////////////////////////////////////////////////////////////////////
	////////////////////////////double tex version///////////////////////////
	/////////////////////////////////////////////////////////////////////
	void mg_exact_periodic2D2(double * x, 
						   double * b, 
						   double * res, 
						   uint dimx,
						   uint dimy );
	void mg_RBGS_periodic2D2(double * x,
							double * b,
							double * res,
							uint t,
							uint dimx,
							uint dimy );

	void mg_ComputeResidualPeriodic2D2(double *x,
									  double *b,
									  double *r,
									  uint dimx, 
									  uint dimy );

	void mg_RestrictionPeriodic2D2(double * next_level,
								  double * curr_level,
								  uint next_dimx,
								  uint next_dimy );

	void mg_ProlongationPeriodic2D2(double * next_level,
								  double * coaser_level,
								  uint next_dimx,
								  uint next_dimy );
						

}

#endif