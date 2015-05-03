

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
void computeResidual(float *x, float *b, float *res, int dimx, int dimy, int dimz)
{
	for (int k=0; k<dimz; k++) for(int j=0; j<dimy; j++) for(int i=0; i<dimz; i++)
	{
		float r = 0;
		float center = x[k*dimx*dimy+j*dimx+i];
		float left = x[k*dimx*dimy+j*dimx+(i+dimx-1)%dimx];
		float right = x[k*dimx*dimy+j*dimx+(i+1)%dimx];
		float up = x[k*dimx*dimy+((j+1)%dimy)*dimx+i];
		float down = x[k*dimx*dimy+((j+dimy-1)%dimy)*dimx+i];
		float front = x[((k+dimz-1)%dimz)*dimx*dimy+j*dimx+i];
		float back= x[((k+1)%dimz)*dimx*dimy+dimx*j+i];

		r += left;
		r += right;
		r += up;
		r += down;
		r += front;
		r += back;
		r -= center;r -= center;r -= center;
		r -= center;r -= center;r -= center;

		res[k*dimx*dimy+j*dimx+i] = b[k*dimx*dimy+j*dimx+i];
		res[k*dimx*dimy+j*dimx+i] -= r ;
	}
}
void restriction(float * curr_r, float * next_r, int dimx, int dimy, int dimz)
{
	for (int k=0; k<dimz; k++) for(int j=0; j<dimy; j++) for(int i=0; i<dimz; i++)
	{
		float r;
		int i_cur, j_cur, k_cur;

		i_cur = i*2; j_cur = j*2; k_cur = k*2;

		r = 0;
		r += curr_r[k_cur*dimx*dimy*4 + j_cur*dimx*2 + i_cur];
		r += curr_r[k_cur*dimx*dimy*4 + j_cur*dimx*2 + i_cur + 1];
		r += curr_r[k_cur*dimx*dimy*4 + (j_cur + 1)*dimx*2 + i_cur];
		r += curr_r[k_cur*dimx*dimy*4 + (j_cur + 1)*dimx*2 + i_cur + 1];
		r += curr_r[(k_cur + 1)*dimx*dimy*4 + j_cur*dimx*2 + i_cur];
		r += curr_r[(k_cur + 1)*dimx*dimy*4 + j_cur*dimx*2 + i_cur + 1];
		r += curr_r[(k_cur + 1)*dimx*dimy*4 + (j_cur + 1)*dimx*2 + i_cur];
		r += curr_r[(k_cur + 1)*dimx*dimy*4 + (j_cur + 1)*dimx*2 + i_cur + 1];


		next_r[k*dimx*dimy+j*dimx+i] = r*0.5;
	}
}
void prolongation(float * curr_x, float * target_x, int dimx, int dimy, int dimz)
{
	for (int k=0; k<dimz; k++) for(int j=0; j<dimy; j++) for(int i=0; i<dimz; i++)
	{

		int i_up, j_up, k_up;

		i_up = i/2; j_up = j/2; k_up = k/2;
		float r = curr_x[k_up*dimx*dimy/4 + j_up * dimx/2 + i_up];

		target_x[k*dimx*dimy+j*dimx+i] += r;
	}
}
int main(int argc, char** argv)
{
	//FILE *pFile;

	InitCuda(argc, argv);
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);  
	GpuArrayd x1, b1, res1, x2, b2, res2;
	GpuArrayd x1_next, b1_next, x2_next, b2_next;
	int dimx=256, dimy=256, dimz=256;
	int iter=1;
	if(argc>3)
	{
		sscanf(argv[1], "%d", &dimx);
		sscanf(argv[2], "%d", &dimy);
		sscanf(argv[3], "%d", &dimz);
		sscanf(argv[4], "%d", &iter);
	}
	MultiGridSolver3DPeriod_double mgs;
	mgs.m_InitialSystem(dimx, dimy, dimz);



	x1.alloc(dimx*dimy*dimz, false, false, false);
	b1.alloc(dimx*dimy*dimz, false, false, false);

	res1.alloc(dimx*dimy*dimz, false, false, false);


	memset(x1.getHostPtr(), 0, x1.typeSize()*x1.getSize());
	memset(b1.getHostPtr(), 0, b1.typeSize()*b1.getSize());
	memset(res1.getHostPtr(), 0, res1.typeSize()*res1.getSize());


	//getch();
	for (int k=1; k<dimz-1; k++) for(int i=1;i<dimx-1;i++)
	{
		b1.getHostPtr()[k*dimx*dimy+dimx+i] = 1.0f;
	}
	for (int k=1; k<dimz-1; k++) for(int i=1;i<dimx-1;i++)
	{
		b1.getHostPtr()[k*dimx*dimy+(dimy-2)*dimx+i] = 1.0f;
	}
	/*double sum_of_b1 = 0.0;
	for (int i=0; i<dimx*dimy*dimz;i++)
	{
		sum_of_b1 += b1.getHostPtr()[i];
	}
	double mfactor = sum_of_b1/(double)(dimx*dimy*dimz);
	for (int i=0; i<dimx*dimy*dimz;i++)
	{
		b1.getHostPtr()[i]-=mfactor;
	}*/

	x1.copy(x1.HOST_TO_DEVICE, 0, x1.getSize());
	b1.copy(b1.HOST_TO_DEVICE, 0, b1.getSize());
	res1.copy(res1.HOST_TO_DEVICE, 0, res1.getSize());
	double resd;
	printf("problem size: %dx%dx%d\n", dimx, dimy, dimz);
	printf("levels : %d, \n", mgs.m_max_level);
	//for(int i=0; i<4; i++)
	cudaThreadSynchronize();
	sdkStartTimer(&timer);
	for(int i=0; i<iter; i++)
		mgs.m_FullMultiGrid(&x1, &b1, 1e-5, resd);
	
	cudaThreadSynchronize();
	sdkStopTimer(&timer);
	printf("full multi grid done!\n");
	printf("time used : %f\n", sdkGetAverageTimerValue(&timer)/1000);



	x1.copy(x1.DEVICE_TO_HOST, 0, x1.getSize());


	int modx = dimx-1;
	int mody = (dimy-1)*dimx;
	int modz = (dimz-1)*dimx*dimy;
	for (int k=0; k<dimz; k++) for(int j=0; j<dimy; j++) for(int i=0;i<dimx;i++)
	{
		int center_idx = k*dimx*dimy+j*dimx+i;
		int left_idx = (i==0)? center_idx + modx : center_idx -1;
		int right_idx = (i==(dimx-1))?center_idx - modx: center_idx + 1;
		int up_idx = (j==(dimy-1))?center_idx - mody: center_idx + dimx;
		int down_idx = (j==0)?center_idx + mody: center_idx - dimx;
		int front_idx = (k==0)?center_idx + modz: center_idx - dimx*dimy;
		int back_idx = (k==(dimz-1))?center_idx - modz: center_idx + dimx*dimy;
		double lv = x1.getHostPtr()[left_idx];
		if(i==0) lv = 0;
		double rv = x1.getHostPtr()[right_idx];
		if(i==dimx-1) rv = 0;
		double tv = x1.getHostPtr()[up_idx];
		if(j==dimy-1) tv =0;
		double dv = x1.getHostPtr()[down_idx];
		if(j==0) dv = 0;
		double fv = x1.getHostPtr()[front_idx];
		if(k==0) fv = 0;
		double bv = x1.getHostPtr()[back_idx];
		if(k==dimz-1) bv = 0;

		res1.getHostPtr()[k*dimx*dimy+j*dimx+i] = b1.getHostPtr()[center_idx]
		- (-6*x1.getHostPtr()[center_idx]
		+ lv + rv + tv + dv + fv + bv);
	}
	double sum=0;
	for (int k=0; k<dimz-0; k++) for(int j=0; j<dimy-0; j++) for(int i=0;i<dimx-0;i++)
	{

		//if(b1.getHostPtr()[k*dimx*dimy+j*dimx+i]>0.0000001)
			if(fabs(res1.getHostPtr()[k*dimx*dimy+j*dimx+i])>sum)
				sum = fabs(res1.getHostPtr()[k*dimx*dimy+j*dimx+i]);
		//printf("%f, ", res.getHostPtr()[k*dimx*dimy+j*dimx+i]);
	}
	printf("%e\n",sum);






	//getch();


	return 0;
}