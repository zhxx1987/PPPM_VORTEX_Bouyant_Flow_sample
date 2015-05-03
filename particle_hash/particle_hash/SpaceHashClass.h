#ifndef __SPACE_HASH_H__
#define __SPACE_HASH_H__

#include <helper_functions.h>
#include "hash_parameters.h"
#include "vector_functions.h"
#include "SpaceHashHeader.cuh"

// Particle system class
class ParticleManager
{
public:
	ParticleManager()
	{
		m_numParticles = 0;
		m_params = NULL;
		m_dCellStart=m_dCellEnd=m_dGridParticleHash=m_dGridParticleIndex=NULL;
	}
	ParticleManager(uint numParticles);
	~ParticleManager();

	int getNumParticles() const
	{
		return m_numParticles;
	}
	uint3 getGridSize()
	{
		return m_params->gridSize;
	}
	float3 getWorldOrigin()
	{
		return m_params->worldOrigin;
	}
	float3 getCellSize()
	{
		return m_params->cellSize;
	}
	uint getNumCells()
	{
		return m_numGridCells;
	}
	void initSpatialHash(uint numParticles, uint dimx, uint dimy, uint dimz);
	void endSpatialHash();


	void setSpatialHashGrid(uint size0, float h, float3 bbmin, unsigned int numParticles)
	{
		m_params->numCells = size0*size0*size0;
		m_params->gridSize.x = m_params->gridSize.y = m_params->gridSize.z = size0;
		m_params->cellSize.x = m_params->cellSize.y = m_params->cellSize.z = h;
		m_params->worldOrigin = bbmin;
		m_numParticles = numParticles;
		//setParameters(m_params);
	}
	void setHashParam()
	{
		setParameters(m_params);
	}
	void doSpatialHash(float4 * pos, uint numParticles);

	void reorderData(uint numParticles, void *data, 
		void *dataReorder, unsigned int stride, uint precision);

	void deorderData(uint numParticles, void *data, 
		void *dataReorder, unsigned int stride, uint precision);

	uint* getHashTable(){ return m_dGridParticleHash; }
	uint* getStartTable() { return m_dCellStart; }
	uint* getEndTable() { return m_dCellEnd; }

protected: // data
	bool m_bInitialized;
	uint m_numParticles;
	// grid data for sorting method
	uint  *m_dGridParticleHash; // grid hash value for each particle
	uint  *m_dGridParticleIndex;// particle index for each particle
	uint  *m_dCellStart;        // index of start of each cell in sorted list
	uint  *m_dCellEnd;          // index of end of cell
	uint   m_gridSortBits;
	// params
	HashParams* m_params;
	uint3 m_gridSize;
	uint m_numGridCells;
};

#endif // __PARTICLESYSTEM_H__