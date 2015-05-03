#ifndef HASH_PARAMETERS_H
#define HASH_PARAMETERS_H


#include "vector_types.h"
typedef unsigned int uint;

// simulation parameters
struct HashParams
{


	uint3 gridSize;
	uint numCells;
	float3 worldOrigin;
	float3 cellSize;

	uint maxParticlesPerCell;
};

#endif