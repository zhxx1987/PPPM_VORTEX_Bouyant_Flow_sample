/*
test poisson solver based on a basic geometric multigrid method
this implementation solves a 3D poisson problem Lapace(p) = div(u)
with no-slip boundary condition for a staggered grid



Author :	Xinxin Zhang
			Master candidate
			Courant institute of Mathematical Science
			NewYork University
*/
#ifndef __MULTI_GRID_CUH__
#define __MULTI_GRID_CUH__

extern "C"
{
	void mg_InitIdxBuffer(int dimx, int dimy, int dimz);
	void mg_exact_periodic3D(float * x, 
						   float * b, 
						   float * res, 
						   int dimx,
						   int dimy,
						   int dimz);
	

	void mg_zerofy(void * data, size_t size);



	void mg_RBGS_periodic3D(float * x,
							float * b,
							float * res,
							int t,
							int dimx,
							int dimy,
							int dimz);

	void mg_ComputeResidualPeriodic3D(float *x,
									  float *b,
									  float *r,
									  int dimx, 
									  int dimy,
									  int dimz);

	void mg_RestrictionPeriodic3D(float * next_level,
								  float * curr_level,
								  int next_dimx,
								  int next_dimy,
								  int next_dimz);

	void mg_ProlongationPeriodic3D(float * next_level,
								  float * coaser_level,
								  int next_dimx,
								  int next_dimy,
								  int next_dimz);

	/////////////////////////////////////////////////////////////////////
	////////////////////////////double tex version///////////////////////////
	/////////////////////////////////////////////////////////////////////
	void mg_exact_periodic3D2(double * x, 
						   double * b, 
						   double * res, 
						   int dimx,
						   int dimy,
						   int dimz);
	void mg_RBGS_periodic3D2(double * x,
							double * b,
							double * res,
							int t,
							int dimx,
							int dimy,
							int dimz);

	void mg_ComputeResidualPeriodic3D2(double *x,
									  double *b,
									  double *r,
									  int dimx, 
									  int dimy,
									  int dimz);

	void mg_RestrictionPeriodic3D2(double * next_level,
								  double * curr_level,
								  int next_dimx,
								  int next_dimy,
								  int next_dimz);

	void mg_ProlongationPeriodic3D2(double * next_level,
								  double * coaser_level,
								  int next_dimx,
								  int next_dimy,
								  int next_dimz);
						

}

#endif