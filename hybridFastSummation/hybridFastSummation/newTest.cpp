#include <stdio.h>
#include <stdlib.h>


#include "SpaceHashClass.h"
#include "ParticleMesh.h"
#include "gfGpuArray.h"
#include "BiotSavartSolver.h"
#include "VortexFluid.h"
#include "VortexSimulationSystem.h"
#include <time.h>
#include <helper_cuda.h>
#include <cstdio>

#include "renderer/render.h"
#include "renderer/shadow.h"
#include "renderer/write_sgi.h"

using namespace gf;

VortexSimulationSystem  *VortexSimulator;

void output(float4* pos,uint N, int frame)
{

	char filename[128];
	int n=sprintf(filename,"Particle_data%04d.bin",frame);
	for (int i=0;i<N;i++)
	{
		pos[i].w = 1.0;
	}
	FILE *data_file = fopen(filename,"wb");
	fwrite(pos,sizeof(float4),N,data_file);
	fclose(data_file);
	printf("timestep %d done\n",frame);

}
int main(int argc, char *argv[])
{

	InitCuda(argc, argv);
	VortexSimulator = new VortexSimulationSystem();
	//VortexSimulator->setVortexRingEmitter(300, 4, 16, 0,0.3,0,0,-1,0,0.01*6,2);
	VortexSimulator->setVortexRingEmitter(200, 2, 16, 0,0,0,0,1,0,0.01*1,0.5);
	VortexSimulator->setHeatEmitter(200, 1, 16, 0,0,0,0,0,0,0.1,0.5);
	VortexSimulator->setTracerEmitter(3000, 1, 16, 0,0,0,0,0,0,100,0.5);
	for (uint i=0; i<200; i++)
	{
		VortexSimulator->solveTimeStep(0.01);
		output(VortexSimulator->getTracerPos(), 
			VortexSimulator->getNumTracer(),
			i);
		
		printf("%d done\n",i);
		printf("%d tracers\n", VortexSimulator->getNumTracer());
	}
	
	return 0;
}