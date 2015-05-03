#ifndef _MULTI_GRID2D_H_
#define _MULTI_GRID2D_H_
#include <vector>
#include"MultiGridcu2D.h"
#include "gfGpuArray.h"
using namespace std;

namespace gf{
typedef unsigned int uint;
struct uniform_grid_descriptor_2D
{
	uint gridx, gridy, system_size;
	uniform_grid_descriptor_2D()
		:gridx(0), gridy(0), system_size(0)
	{}
	uniform_grid_descriptor_2D(uint x, uint y)
		:gridx(x), gridy(y), system_size(x*y)
	{}
	~uniform_grid_descriptor_2D()
	{}
	void set_grid_information(uint x, uint y)
	{
		gridx = x; gridy = y;  
		system_size = x*y;
	}
};
template<class T>
class MultiGrid2D
{
public:
	MultiGrid2D(void){}

	~MultiGrid2D(void){m_FinalMemoryEachLevel();}

	void m_InitialSystem(uint gridx, uint gridy);
	void m_SetGridInformation(uint x_, uint y_);
	void m_AssignMemoryEachLevel();
	void m_FinalMemoryEachLevel();
	void m_ComputeLevels();
	void m_CreateIndexBuffers();
	uniform_grid_descriptor_2D m_system_descriptor;
	uint m_max_level;
	std::vector<gf_GpuArray<T> *> xk;
	std::vector<gf_GpuArray<T> *> bk;
	std::vector<gf_GpuArray<T> *> rk;
	std::vector<uniform_grid_descriptor_2D> systemk;
};

template<class T>
void MultiGrid2D<T>::m_FinalMemoryEachLevel()
{
	for(uint i=1; i<m_max_level; ++i)
	{
		xk[i]->free();
		bk[i]->free();
		rk[i]->free();
	}
	rk[0]->free();
}
template<class T>
void MultiGrid2D<T>::m_AssignMemoryEachLevel()
{
	/*
		if grid has been changed
		as soon as we know what the finest level grid should be
		we can allocate memory for each level's grids
		first compute how many steps it is needed to get to the
		coarest level, a grid whose smallest dimension is 2; 
		suppose its level is 0;
		then, for level=0 to finest;
		memory[level]=alloc corresponding memory;
	*/
	xk.resize(m_max_level);
	bk.resize(m_max_level);
	rk.resize(m_max_level);

	for (uint i=0; i<m_max_level; i++)
	{
		xk[i] = new gf_GpuArray<T>();
		bk[i] = new gf_GpuArray<T>();
		rk[i] = new gf_GpuArray<T>();
	}
	systemk.resize(m_max_level);
	systemk[0].set_grid_information(m_system_descriptor.gridx,m_system_descriptor.gridy);
	for(int i=1; i<m_max_level; ++i)
	{
		systemk[i].set_grid_information((systemk[i-1].gridx/2)==0?1:systemk[i-1].gridx/2,
										(systemk[i-1].gridy/2)==0?1:systemk[i-1].gridy/2);
	}
	for(int i=1; i<m_max_level; ++i)
	{
		xk[i]->alloc(systemk[i].gridx*systemk[i].gridy, false, false, false);
		bk[i]->alloc(systemk[i].gridx*systemk[i].gridy, false, false, false);
		rk[i]->alloc(systemk[i].gridx*systemk[i].gridy, false, false, false);
	}
	rk[0]->alloc(systemk[0].gridx*systemk[0].gridy, false, false, false);
	
}
template<class T>
void MultiGrid2D<T>::m_SetGridInformation(uint x_, uint y_)
{
	m_system_descriptor.set_grid_information(x_, y_);
}
template<class T>
void MultiGrid2D<T>::m_ComputeLevels()
{
	uint level = 0;
	uint x = m_system_descriptor.gridx, y = m_system_descriptor.gridy;
	while (x*y>=64||level<5||x>8||y>8)
	{
		level = level + 1;
		x = x/2; if(x==0) x=1;
		y = y/2; if(y==0) y=1;
	}
	m_max_level = level;
}
template<class T>
void MultiGrid2D<T>::m_InitialSystem(uint gridx, uint gridy)
{
	m_SetGridInformation(gridx, gridy);
	m_ComputeLevels();
	m_AssignMemoryEachLevel();
	mg_InitIdxBuffer2D(systemk[m_max_level-1].gridx,
		systemk[m_max_level-1].gridy);
	m_CreateIndexBuffers();
}
template<class T>
void MultiGrid2D<T>::m_CreateIndexBuffers()
{
	for (uint level = 0; level<m_max_level; level++)
	{
		uint dimx = systemk[level].gridx;
		uint dimy = systemk[level].gridy;
		
		for (unsigned int idx=0; idx<dimx*dimy; idx++)
		{
			int i = idx%dimx; int j = (idx/dimx)%dimy; 
			int left = j*dimx+(i+dimx-1)%dimx;
			int right= j*dimx+(i+1)%dimx;
			int up = ((j+1)%dimy)*dimx+i;
			int down = ((j+dimy-1)%dimy)*dimx+i;
		}

	}
	
}


class MultiGridSolver2DPeriod : public MultiGrid2D<float>
{
public:
	MultiGridSolver2DPeriod(){}
	~MultiGridSolver2DPeriod(){}
	void m_Vcycle(GpuArrayf* x, GpuArrayf* b, float tol, float &residual, int level);
	void m_FullMultiGrid(GpuArrayf* x, GpuArrayf* b, float tol, float &residual);
	

protected:
	//
	//void m_Red_Black_Gauss(uniform_grid_descriptor_2D &system, float* b, float* x, int iter_time, double omega);
	//void m_Restrict(uniform_grid_descriptor_2D &next_level, float* src_level, float* dst_level);
	//void m_ExactSolve();
	//void m_Prolongate(uniform_grid_descriptor_2D &next_level, float* src_level, float* dst_level);
	
	
};



class MultiGridSolver2DPeriod_double : public MultiGrid2D<double>
{
public:
	MultiGridSolver2DPeriod_double()
	{
	}
	~MultiGridSolver2DPeriod_double()
	{

	}
	void m_Vcycle(GpuArrayd* x, GpuArrayd* b, double tol, double &residual, int level);
	void m_FullMultiGrid(GpuArrayd* x, GpuArrayd* b, double tol, double &residual);
	

protected:

	//void m_Red_Black_Gauss(uniform_grid_descriptor_2D &system, double* b, double* x, int iter_time, double omega);
	//void m_Restrict(uniform_grid_descriptor_2D &next_level, double* src_level, double* dst_level);
	//void m_ExactSolve();
	//void m_Prolongate(uniform_grid_descriptor_2D &next_level, double* src_level, double* dst_level);

};


}
#endif