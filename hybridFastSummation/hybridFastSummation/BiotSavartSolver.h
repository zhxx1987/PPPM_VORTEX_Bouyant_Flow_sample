#ifndef _VORTEX_BLOB_H_
#define _VORTEX_BLOB_H_
#include "gfGpuArray.h"
#include "MultiGrid.h"
#include "SpaceHashClass.h"
typedef unsigned int uint;
using namespace gf;

enum g_Components{
	COMPONENT_X,
	COMPONENT_Y,
	COMPONENT_Z,
	NUM_COMPONENTS
};

class BiotSavartSolver
{
public:
	BiotSavartSolver();
	
	~BiotSavartSolver();
	bool initializeSolver(uint gdx, uint gdy, uint gdz,bool isVIC, int K, uint M, uint N);
	bool evaluateVelocity(GpuArrayf4 *another_end, uint is_segment);
	GpuArrayd *getU(int i) {return m_particle_U[i]; }
	GpuArrayd *getSortVort(int i) {return m_particle_vort_Reorder[i];}
	GpuArrayd *getVort(int i) { return m_particle_vort[i]; }
	GpuArrayf4 *getSortPos() {return m_p_vortPos_Reorder; }
	uint *getVortStartTable() {return m_SpatialHasher_vort.getStartTable();}
	uint *getVortEndTable() {return m_SpatialHasher_vort.getEndTable(); }
	void sortVort();
	void unsortVort();
	bool setEvalParameter(uint m, GpuArrayf4 * pos);
	bool setVortParameter(uint n, GpuArrayf4 * pos, GpuArrayd *omega[]);
	double getCellh() {return m_SpatialHasher_vort.getCellSize().x; }
	double4 getOrigin() {return m_origin;}
	void computeFarFieldBuffer();
	//void setFarFieldBuffer(double *u, double * v, double * w, )
	bool shutdown();

	

private:
	bool m_isVIC;
	bool setDomain(double4 & origin, float4 * pos, uint num_particle, double & domain_length);
	bool setEvalPos(GpuArrayf4 *pos);
	bool setVortPos(GpuArrayf4 *pos);
	bool setVortStrength(GpuArrayd *omega[]);
	
	bool m_ParticleToMesh();
	
	bool m_SolvePoisson();
	bool m_ComputeCurl();
	bool m_Intepolate();
	bool m_LocalCorrection(GpuArrayf4 *another_end);
	bool m_unsortResult();

	MultiGridSolver3DPeriod_double m_PoissonSolver;
	ParticleManager m_SpatialHasher_vort;
	ParticleManager m_SpatialHasher_eval;
	uint m_M_eval;
	uint m_N_vort;
	double m_cell_h;
	uint m_gridx, m_gridy, m_gridz;
	uint m_hashx, m_hashy, m_hashz;
	bool m_initialized;
	double4 m_origin;
	double m_L;
	int m_K;
	double3 m_center;
	double m_total_vort[NUM_COMPONENTS];

	

	GpuArrayf4 * m_evalPos;
	GpuArrayf4 * m_evalPos_Reorder;
	GpuArrayf4 * m_p_vortPos;
	GpuArrayf4 * m_p_vortPos_Reorder;
	GpuArrayd * m_particle_vort[NUM_COMPONENTS];
	GpuArrayd * m_particle_vort_Reorder[NUM_COMPONENTS];
	GpuArrayd * m_grid_vort[NUM_COMPONENTS];
	GpuArrayd * m_grid_Rhs[NUM_COMPONENTS];
	GpuArrayd * m_grid_Psi[NUM_COMPONENTS];
	GpuArrayd * m_particle_U[NUM_COMPONENTS];
	GpuArrayd * m_particle_U_deorder[NUM_COMPONENTS];
	GpuArrayd * m_grid_U[NUM_COMPONENTS];
	GpuArrayd * m_far_U[NUM_COMPONENTS];
	

};




#endif