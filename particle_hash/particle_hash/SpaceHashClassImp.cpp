#include "MemoryOperation.cuh"
#include "SpaceHashClass.h"



ParticleManager::ParticleManager(uint numParticles)
{
	m_numParticles = numParticles;
	float3 min; min.x=0; min.y=0; min.z = 0;
	m_params=(HashParams*)malloc(sizeof(HashParams));
	setSpatialHashGrid(64, 1.0/64.0, min, numParticles);
	m_dCellStart=m_dCellEnd=m_dGridParticleHash=m_dGridParticleIndex=NULL;
	m_numGridCells = m_params->numCells;
}
ParticleManager::~ParticleManager()
{
}
void ParticleManager::doSpatialHash(float4 * pos, uint numParticles)
{
	calcHash(m_dGridParticleHash, m_dGridParticleIndex, (float*)pos, numParticles);
	sortParticles(m_dGridParticleHash, m_dGridParticleIndex, numParticles);
	FindCellStart(m_dCellStart, m_dCellEnd, m_dGridParticleHash, m_dGridParticleIndex, numParticles, m_params->numCells);
}
void ParticleManager::reorderData(uint numParticles, void *data, void *dataReorder, unsigned int stride, uint precision)
{
	reorder(m_dGridParticleIndex, data, stride, dataReorder, precision, numParticles);
	//MemcpyDevToDev(data, dataReorder,sizeof(float)*precision*stride*numParticles);
}

void ParticleManager::deorderData(uint numParticles, void *data, void *dataReorder, unsigned int stride, uint precision)
{
	deorder(m_dGridParticleIndex, data, stride, dataReorder, precision, numParticles);
	//MemcpyDevToDev(data, dataReorder,sizeof(float)*precision*stride*numParticles);
}

void ParticleManager::initSpatialHash(uint numParticles, uint dimx, uint dimy, uint dimz)
{
	m_numParticles = numParticles;
	float3 min; min.x=0; min.y=0; min.z = 0;
	if (m_params!=NULL)
	{
		free(m_params);
		m_params=(HashParams*)malloc(sizeof(HashParams));
	}
	else
	{
		m_params=(HashParams*)malloc(sizeof(HashParams));
	}
	
	setSpatialHashGrid(dimx, 1.0/(double)dimx, min, numParticles);
	m_dCellStart=m_dCellEnd=m_dGridParticleHash=m_dGridParticleIndex=NULL;
	m_numGridCells = m_params->numCells;
	AllocateMemory(sizeof(uint)*m_numParticles, (void**)&m_dGridParticleHash);
	AllocateMemory(sizeof(uint)*m_numParticles, (void**)&m_dGridParticleIndex);
	AllocateMemory(sizeof(uint)*m_numGridCells, (void**)&m_dCellStart);
	AllocateMemory(sizeof(uint)*m_numGridCells, (void**)&m_dCellEnd);

}
void ParticleManager::endSpatialHash()
{
	FreeMemory(m_dGridParticleHash);
	FreeMemory(m_dGridParticleIndex);
	FreeMemory(m_dCellStart);
	FreeMemory(m_dCellEnd);

}