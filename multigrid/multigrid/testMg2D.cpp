

#include<GL/glew.h>
#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif


//#include <conio.h>


#include <cuda_gl_interop.h>
#include "MemoryOperation.cuh"
#include "gfGpuArray.h"

using namespace gf;

#include "MultiGrid2D.h"
#include "MultiGridcu2D.h"
#include "MultiGrid.h"
#include "MultiGridcu.h"
#include <stdlib.h>
#include <helper_cuda.h>
#include "helper_timer.h"
#include <stdio.h>
#include <math.h>
StopWatchInterface *timer = NULL;

void computeResidual(float *x, float *b, float *res, int dimx, int dimy)
{
	for(int j=0; j<dimy; j++) for(int i=0; i<dimx; i++)
	{
		float r = 0;
		float center = x[j*dimx+i];
		float left = x[j*dimx+(i+dimx-1)%dimx];
		float right = x[j*dimx+(i+1)%dimx];
		float up = x[((j+1)%dimy)*dimx+i];
		float down = x[((j+dimy-1)%dimy)*dimx+i];


		r += left;
		r += right;
		r += up;
		r += down;
		r -= center;r -= center;r -= center;
		r -= center;

		res[j*dimx+i] = b[j*dimx+i];
		res[j*dimx+i] -= r ;
	}
}
void restriction(float * curr_r, float * next_r, int dimx, int dimy)
{
	for(int j=0; j<dimy; j++) for(int i=0; i<dimx; i++)
	{
		float r;
		int i_cur, j_cur;

		i_cur = i*2; j_cur = j*2; 

		r = 0;
		r += curr_r[j_cur*dimx*2 + i_cur];
		r += curr_r[j_cur*dimx*2 + i_cur + 1];
		r += curr_r[(j_cur + 1)*dimx*2 + i_cur];
		r += curr_r[(j_cur + 1)*dimx*2 + i_cur + 1];



		next_r[j*dimx+i] = r;
	}
}
void prolongation(float * curr_x, float * target_x, int dimx, int dimy)
{
	for(int j=0; j<dimy; j++) for(int i=0; i<dimx; i++)
	{

		int i_up, j_up;

		i_up = i/2; j_up = j/2; 
		float r = curr_x[j_up * dimx/2 + i_up];

		target_x[j*dimx+i] += r;
	}
}
int main(int argc, char** argv)
{
	
	InitCuda(argc, argv);
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);  
	GpuArrayd x1, b1, res1, x2, b2, res2;
	GpuArrayd x1_next, b1_next, x2_next, b2_next;
	int dimx=512, dimy=512;
	if(argc>3)
	{
		sscanf(argv[1], "%d", &dimx);
		sscanf(argv[2], "%d", &dimy);
	}
	MultiGridSolver2DPeriod_double mgs;
	mgs.m_InitialSystem(dimx, dimy);



	x1.alloc(dimx*dimy, false, false, false);
	b1.alloc(dimx*dimy, false, false, false);

	res1.alloc(dimx*dimy, false, false, false);


	memset(x1.getHostPtr(), 0, x1.typeSize()*x1.getSize());
	memset(b1.getHostPtr(), 0, b1.typeSize()*b1.getSize());
	memset(res1.getHostPtr(), 0, res1.typeSize()*res1.getSize());


	//getch();
	for(int i=1;i<dimx-1;i++)
	{
		b1.getHostPtr()[dimx+i] = 1.0f;
	}
	for(int i=1;i<dimx-1;i++)
	{
		b1.getHostPtr()[(dimy-2)*dimx+i] = -1.0f;
	}


	x1.copy(x1.HOST_TO_DEVICE, 0, x1.getSize());
	b1.copy(b1.HOST_TO_DEVICE, 0, b1.getSize());
	res1.copy(res1.HOST_TO_DEVICE, 0, res1.getSize());
	double resd;
	printf("problem size: %dx%d\n", dimx, dimy);
	printf("levels : %d, \n", mgs.m_max_level);
	//for(int i=0; i<4; i++)
	cudaThreadSynchronize();
	sdkStartTimer(&timer);
	for(int i=0; i<3; i++)
		mgs.m_FullMultiGrid(&x1, &b1, 1e-5, resd);

	cudaThreadSynchronize();
	sdkStopTimer(&timer);
	printf("full multi grid done!\n");
	printf("time used : %f\n", sdkGetAverageTimerValue(&timer)/1000);



	x1.copy(x1.DEVICE_TO_HOST, 0, x1.getSize());



	for(int j=0; j<dimy; j++) for(int i=0;i<dimx;i++)
	{
		res1.getHostPtr()[j*dimx+i] = b1.getHostPtr()[j*dimx+i]
		- (-4*x1.getHostPtr()[j*dimx+i]
		+ x1.getHostPtr()[j*dimx+(i+dimx-1)%dimx]
		+ x1.getHostPtr()[j*dimx+(i+1)%dimx]
		+ x1.getHostPtr()[((j+dimy-1)%dimy)*dimx+i]
		+ x1.getHostPtr()[((j+1)%dimy)*dimx+i]);
	}
	double sum=0;
	for(int j=0; j<dimy-0; j++) for(int i=0;i<dimx-0;i++)
	{

		if(b1.getHostPtr()[j*dimx+i]>0.0000001)
			if(fabs(res1.getHostPtr()[j*dimx+i]/b1.getHostPtr()[j*dimx+i])>sum)
				sum = fabs(res1.getHostPtr()[j*dimx+i]/b1.getHostPtr()[j*dimx+i]);
		//printf("%f, ", res.getHostPtr()[k*dimx*dimy+j*dimx+i]);
	}
	printf("%.16f\n",sum);






	//getch();


	return 0;
}