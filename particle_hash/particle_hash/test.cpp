#include <GL/glew.h>
#ifdef UNIX
#include <GL/glxew.h>
#endif
#if defined (_WIN32)
#include <GL/wglew.h>
#endif

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include "SpaceHashClass.h"
#include "MemoryOperation.cuh"


#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include <time.h>

uint numParticles = 65536*16;
ParticleManager g_SpatialHasher(numParticles);
void print_result(float4 * host_pos, float4* pos)
{
	unsigned int *g_Start = new unsigned int[g_SpatialHasher.getNumCells()];
	unsigned int *g_End   = new unsigned int[g_SpatialHasher.getNumCells()];

	MemcpyDevToHst(
		g_SpatialHasher.getStartTable(),
		(void *)g_Start,
		sizeof(unsigned int)*g_SpatialHasher.getNumCells()
		);

	MemcpyDevToHst(
		g_SpatialHasher.getEndTable(),
		(void*)g_End,
		sizeof(unsigned int)*g_SpatialHasher.getNumCells()
		);

	MemcpyDevToHst(pos,(void*)host_pos, sizeof(float4)*numParticles);

	uint count = 0;
	for (uint k=0; k<g_SpatialHasher.getGridSize().z; k++)
		for (uint j=0; j<g_SpatialHasher.getGridSize().y; j++)
			for (uint i=0; i<g_SpatialHasher.getGridSize().x; i++)
			{

				uint idx = k * g_SpatialHasher.getGridSize().x * g_SpatialHasher.getGridSize().y + j*g_SpatialHasher.getGridSize().x + i;

				if (g_Start[idx]<g_End[idx])
				{
					uint flag = 0;
					for(uint p=g_Start[idx]; p<g_End[idx]; p++)
					{
						if (p<numParticles)
						{
							uint ii = (uint)(host_pos[p].x * 64.0);
							uint jj = (uint)(host_pos[p].y * 64.0);
							uint kk = (uint)(host_pos[p].z * 64.0);
							if (ii!=i||jj!=j||kk!=k)
							{
								flag = 1;
							}
						}
					}
					if (flag == 1)
					{
						printf("Cell %u %u %u Start %u End %u: \n", i, j, k, g_Start[idx], g_End[idx]);
						for(uint p=g_Start[idx]; p<g_End[idx]; p++)
						{
							if (p<numParticles)
							{
								printf("\t%f, %f, %f\n", host_pos[p].x,host_pos[p].y,host_pos[p].z);
								count++;
							}
						}
					}
					else
					{
						for(uint p=g_Start[idx]; p<g_End[idx]; p++)
						{
							if (p<numParticles)
							{
								count++;
							}
						}
					}
					
				}

			}
			printf("%u\n", count);
}

int main(int argc, char *argv[])
{

	
	InitCuda(argc, argv);

	g_SpatialHasher.initSpatialHash();
	g_SpatialHasher.setHashParam();
	float4* pos;
	float4* pos_reorder;
	AllocateMemory(sizeof(float4)*numParticles, (void**)&pos);
	AllocateMemory(sizeof(float4)*numParticles, (void**)&pos_reorder);
	float4 *host_pos = (float4 *)malloc(sizeof(float4)*numParticles);

	for (int i=0; i<numParticles; i++)
	{
		host_pos[i].x = ((double)(rand()%RAND_MAX))/((double)RAND_MAX);
		host_pos[i].y = ((double)(rand()%RAND_MAX))/((double)RAND_MAX);
		host_pos[i].z = ((double)(rand()%RAND_MAX))/((double)RAND_MAX);
		host_pos[i].w = 0.0;
	}
	MemcpyHstToDev(pos,host_pos, sizeof(float4)*numParticles);

	clock_t start = clock();
	for (uint i=0;i<1000;i++)
	{
		g_SpatialHasher.doSpatialHash(pos, numParticles);
		g_SpatialHasher.reorderData(numParticles, (void*)pos, (void*)pos_reorder, 4, 1);
		float4* temp;
		temp = pos;
		pos = pos_reorder;
		pos_reorder = temp;
	}
	clock_t end = clock();

	double seconds;
	printf("%lf\n", (double)(end-start)/CLOCKS_PER_SEC/1000.0);


	print_result(host_pos, pos);


	FreeMemory(pos);
	FreeMemory(pos_reorder);
	g_SpatialHasher.endSpatialHash();
	

	return 0;
}