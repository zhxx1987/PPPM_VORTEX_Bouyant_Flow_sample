
#ifndef _MULTI_GRID_H_
#define _MULTI_GRID_H_


#include <vector>
#include"MultiGridcu.h"
#include "gfGpuArray.h"
using namespace std;
namespace gf{

struct uniform_grid_descriptor_3D
{
	int gridx, gridy, gridz, system_size;
	uniform_grid_descriptor_3D()
		:gridx(0), gridy(0), gridz(0), system_size(0)
	{}
	uniform_grid_descriptor_3D(int x, int y, int z)
		:gridx(x), gridy(y), gridz(z), system_size(x*y*z)
	{}
	~uniform_grid_descriptor_3D()
	{}
	void set_grid_information(int x, int y, int z)
	{
		gridx = x; gridy = y;  gridz = z;
		system_size = x*y*z;
	}
};
template<class T>
class MultiGrid
{
public:
	MultiGrid(void){ m_bInitialized = false;}

	~MultiGrid(void){m_FinalMemoryEachLevel();}

	void m_InitialSystem(int gridx, int gridy, int gridz);
	void m_SetGridInformation(int x_, int y_, int z_);
	void m_AssignMemoryEachLevel();
	void m_FinalMemoryEachLevel();
	void m_ComputeLevels();
	void m_CreateIndexBuffers();
	bool m_bInitialized;
	uniform_grid_descriptor_3D m_system_descriptor;
	int m_max_level;
	std::vector<gf_GpuArray<T> *> xk;
	std::vector<gf_GpuArray<T> *> bk;
	std::vector<gf_GpuArray<T> *> rk;
	std::vector<uniform_grid_descriptor_3D> systemk;
};

template<class T>
void MultiGrid<T>::m_FinalMemoryEachLevel()
{
	if(m_bInitialized)
	{
		for(int i=1; i<m_max_level; ++i)
		{
			xk[i]->free();
			bk[i]->free();
			rk[i]->free();
		}
		rk[0]->free();
		m_bInitialized = false;
	}
}
template<class T>
void MultiGrid<T>::m_AssignMemoryEachLevel()
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

	for (int i=0; i<m_max_level; i++)
	{
		xk[i] = new gf_GpuArray<T>();
		bk[i] = new gf_GpuArray<T>();
		rk[i] = new gf_GpuArray<T>();
	}
	systemk.resize(m_max_level);
	systemk[0].set_grid_information(m_system_descriptor.gridx,m_system_descriptor.gridy, m_system_descriptor.gridz);
	for(int i=1; i<m_max_level; ++i)
	{
		systemk[i].set_grid_information((systemk[i-1].gridx/2)==0?1:systemk[i-1].gridx/2,
										(systemk[i-1].gridy/2)==0?1:systemk[i-1].gridy/2, 
										(systemk[i-1].gridz/2)==0?1:systemk[i-1].gridz/2);
	}
	for(int i=1; i<m_max_level; ++i)
	{
		xk[i]->alloc(systemk[i].gridx*systemk[i].gridy*systemk[i].gridz, false, false, false);
		bk[i]->alloc(systemk[i].gridx*systemk[i].gridy*systemk[i].gridz, false, false, false);
		rk[i]->alloc(systemk[i].gridx*systemk[i].gridy*systemk[i].gridz, false, false, false);
	}
	rk[0]->alloc(systemk[0].gridx*systemk[0].gridy*systemk[0].gridz, false, false, false);
	m_bInitialized = true;
	
}
template<class T>
void MultiGrid<T>::m_SetGridInformation(int x_, int y_, int z_)
{
	m_system_descriptor.set_grid_information(x_, y_, z_);
}
template<class T>
void MultiGrid<T>::m_ComputeLevels()
{
	int level = 0;
	int x = m_system_descriptor.gridx, y = m_system_descriptor.gridy, z = m_system_descriptor.gridz;
	while (x*y*z>512)
	{
		level = level + 1;
		x = x/2; if(x==0) x=1;
		y = y/2; if(y==0) y=1;
		z = z/2; if(z==0) z=1;
	}
	m_max_level = level+1;
}
template<class T>
void MultiGrid<T>::m_InitialSystem(int gridx, int gridy, int gridz)
{
	m_FinalMemoryEachLevel();
	m_SetGridInformation(gridx, gridy, gridz);
	m_ComputeLevels();
	m_AssignMemoryEachLevel();
	mg_InitIdxBuffer(systemk[m_max_level-1].gridx,
		systemk[m_max_level-1].gridy,
		systemk[m_max_level-1].gridz);
	m_CreateIndexBuffers();
}
template<class T>
void MultiGrid<T>::m_CreateIndexBuffers()
{
	for (int level = 0; level<m_max_level; level++)
	{
		int dimx = systemk[level].gridx;
		int dimy = systemk[level].gridy;
		int dimz = systemk[level].gridz;
		for (unsigned int idx=0; idx<dimx*dimy*dimz; idx++)
		{
			int i = idx%dimx; int j = (idx/dimx)%dimy; int k = idx/(dimx*dimy);
			int left = k*dimx*dimy+j*dimx+(i+dimx-1)%dimx;
			int right= k*dimx*dimy+j*dimx+(i+1)%dimx;
			int up = k*dimx*dimy+((j+1)%dimy)*dimx+i;
			int down = k*dimx*dimy+((j+dimy-1)%dimy)*dimx+i;
			int front = ((k+dimz-1)%dimz)*dimx*dimy+j*dimx+i;
			int back = ((k+1)%dimz)*dimx*dimy+dimx*j+i;



		}

	}
	
}


class MultiGridSolver3DPeriod : public MultiGrid<float>
{
public:
	MultiGridSolver3DPeriod(){}
	~MultiGridSolver3DPeriod(){}
	void m_Vcycle(GpuArrayf* x, GpuArrayf* b, float tol, float &residual, int level);
	void m_FullMultiGrid(GpuArrayf* x, GpuArrayf* b, float tol, float &residual);
	

protected:
	
	//void m_Red_Black_Gauss(uniform_grid_descriptor_3D &system, float* b, float* x, int iter_time, double omega);
	//void m_Restrict(uniform_grid_descriptor_3D &next_level, float* src_level, float* dst_level);
	//void m_ExactSolve();
	//void m_Prolongate(uniform_grid_descriptor_3D &next_level, float* src_level, float* dst_level);
	
	
};



class MultiGridSolver3DPeriod_double : public MultiGrid<double>
{
public:
	MultiGridSolver3DPeriod_double()
	{
	}
	~MultiGridSolver3DPeriod_double()
	{

	}
	void m_Vcycle(GpuArrayd* x, GpuArrayd* b, double tol, double &residual, int level);
	void m_FullMultiGrid(GpuArrayd* x, GpuArrayd* b, double tol, double &residual);
	

protected:

	//void m_Red_Black_Gauss(uniform_grid_descriptor_3D &system, double* b, double* x, int iter_time, double omega);
	//void m_Restrict(uniform_grid_descriptor_3D &next_level, double* src_level, double* dst_level);
	//void m_ExactSolve();
	//void m_Prolongate(uniform_grid_descriptor_3D &next_level, double* src_level, double* dst_level);

};


}
#endif