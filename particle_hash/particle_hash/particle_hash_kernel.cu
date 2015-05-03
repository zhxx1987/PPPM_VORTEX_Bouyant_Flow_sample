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

/*
 * CUDA particle system kernel code.
 */

#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include "helper_math.h"
#include "math_constants.h"
#include "hash_parameters.h"


// simulation parameters in constant memory
__constant__ HashParams params;



// calculate position in uniform grid
__device__ int3 calcGridPos(float3 p)
{
    int3 gridPos;
    gridPos.x = floor((p.x - params.worldOrigin.x) / params.cellSize.x);
    gridPos.y = floor((p.y - params.worldOrigin.y) / params.cellSize.y);
    gridPos.z = floor((p.z - params.worldOrigin.z) / params.cellSize.z);
    return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash(int3 gridPos)
{
    gridPos.x = gridPos.x % params.gridSize.x;  // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y % params.gridSize.y;
    gridPos.z = gridPos.z % params.gridSize.z;
    return __umul24(__umul24(gridPos.z, params.gridSize.y), params.gridSize.x) + __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
}

// calculate grid hash value for each particle
__global__
void calcHashD(uint   *gridParticleHash,  // output
               uint   *gridParticleIndex, // output
               float4 *pos,               // input: positions
               uint    numParticles)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    volatile float4 p = pos[index];

    // get address in grid
    int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
    uint hash = calcGridHash(gridPos);

    // store grid hash and particle index
    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
void FindCellStartD(uint   *cellStart,        // output: cell start index
                                  uint   *cellEnd,          // output: cell end index
                                  uint   *gridParticleHash, // input: sorted grid hashes
                                  uint   *gridParticleIndex,// input: sorted particle indices
                                  uint    numParticles)
{
    extern __shared__ uint sharedHash[];    // blockSize + 1 elements
    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

    uint hash;

    // handle case when no. of particles not multiple of block size
    if (index < numParticles)
    {
        hash = gridParticleHash[index];

        // Load hash data into shared memory so that we can look
        // at neighboring particle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x+1] = hash;

        if (index > 0 && threadIdx.x == 0)
        {
            // first thread in block must load neighbor particle hash
            sharedHash[0] = gridParticleHash[index-1];
        }
    }

    __syncthreads();

    if (index < numParticles)
    {
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell

        if (index == 0 || hash != sharedHash[threadIdx.x])
        {
            cellStart[hash] = index;

            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
        }

        if (index == numParticles - 1)
        {
            cellEnd[hash] = index + 1;
        }
    }
}

__global__ void
applyReorder(
	uint *indexTable,
	float *dataf,
	double *datad,
	uint stride,
	float *out_dataf,
	double *out_datad,
	uint precision,
	uint num_data)
{
	uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

	if(index <num_data)
	{
		if(precision==1)
		{
			for(uint i=0; i<stride; i++)
				out_dataf[index*stride+i] = dataf[indexTable[index]*stride + i];
		}
		else
		{
			for(uint i=0; i<stride; i++)
				out_datad[index*stride+i] = datad[indexTable[index]*stride + i];
		}
	}
}



__global__ void
removeReorder(
	uint *indexTable,
	float *dataf,
	double *datad,
	uint stride,
	float *out_dataf,
	double *out_datad,
	uint precision,
	uint num_data)
{
	uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

	if(index <num_data)
	{
		if(precision==1)
		{
			for(uint i=0; i<stride; i++)
				out_dataf[indexTable[index]*stride + i] = dataf[index*stride+i];
		}
		else
		{
			for(uint i=0; i<stride; i++)
				out_datad[indexTable[index]*stride + i] = datad[index*stride+i];
		}
	}
}



#endif
