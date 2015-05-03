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

extern "C"
{
    void setParameters(HashParams *hostParams);

    void calcHash(uint  *gridParticleHash,
                  uint  *gridParticleIndex,
                  float *pos,
                  int    numParticles);

    void FindCellStart(
		uint  *cellStart,
		uint  *cellEnd,
		uint  *gridParticleHash,
		uint  *gridParticleIndex,
		uint   numParticles,
		uint   numCells);

	void reorder(
		uint * indexTable,
		void *data, 
		uint stride, 
		void *out_data, 
		uint precision, 
		uint num_data);

	void deorder(
		uint * indexTable,
		void *data, 
		uint stride, 
		void *out_data, 
		uint precision, 
		uint num_data);

    void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles);

}
