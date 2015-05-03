/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// This file contains C wrappers around the some of the CUDA API and the
// kernel functions so that they can be called from "particleSystem.cpp"
#include<GL\glew.h>
#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <cstdlib>
#include <cstdio>
#include <string.h>

#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"
#include <helper_cuda.h>
#include "particle_hash_kernel.cu"

extern "C"
{
	void setParameters(HashParams *hostParams)
    {
        // copy parameters to constant memory
        checkCudaErrors(cudaMemcpyToSymbol(params, hostParams, sizeof(HashParams)));
    }

    //Round a / b to nearest higher integer value
    uint iDivUp(uint a, uint b)
    {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }

    // compute grid and thread block size for a given number of elements
    void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
    {
        numThreads = min(blockSize, n);
        numBlocks = iDivUp(n, numThreads);
    }

    void calcHash(uint  *gridParticleHash,
                  uint  *gridParticleIndex,
                  float *pos,
                  int    numParticles)
    {
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // execute the kernel
        calcHashD<<< numBlocks, numThreads >>>(gridParticleHash,
                                               gridParticleIndex,
                                               (float4 *) pos,
                                               numParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }

    void FindCellStart(uint  *cellStart,
		uint  *cellEnd,
		uint  *gridParticleHash,
		uint  *gridParticleIndex,
		uint   numParticles,
		uint   numCells)
    {
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // set all cells to empty
        checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));


        uint smemSize = sizeof(uint)*(numThreads+1);
        FindCellStartD<<< numBlocks, numThreads, smemSize>>>(
            cellStart,
            cellEnd,
            gridParticleHash,
            gridParticleIndex,
            numParticles);
        getLastCudaError("Kernel execution failed: FindCellStartD");

    }


    void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles)
    {
        thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
                            thrust::device_ptr<uint>(dGridParticleHash + numParticles),
                            thrust::device_ptr<uint>(dGridParticleIndex));
    }


	void reorder(
		uint * indexTable,
		void *data, 
		uint stride, 
		void *out_data, 
		uint precision, 
		uint num_data)
	{
		
		uint numThreads, numBlocks;
        computeGridSize(num_data, 256, numBlocks, numThreads);
		applyReorder<<<numBlocks, numThreads>>>(
			indexTable,
			(float*)data,(double*)data,
			stride,
			(float*)out_data,(double*)out_data,
			precision,num_data);

		getLastCudaError("Kernel execution failed: reorder");
	}

	void deorder(
		uint * indexTable,
		void *data, 
		uint stride, 
		void *out_data, 
		uint precision, 
		uint num_data)
	{
		
		uint numThreads, numBlocks;
        computeGridSize(num_data, 256, numBlocks, numThreads);
		removeReorder<<<numBlocks, numThreads>>>(
			indexTable,
			(float*)data,(double*)data,
			stride,
			(float*)out_data,(double*)out_data,
			precision,num_data);

		getLastCudaError("Kernel execution failed: reorder");
	}
}   // extern "C"
