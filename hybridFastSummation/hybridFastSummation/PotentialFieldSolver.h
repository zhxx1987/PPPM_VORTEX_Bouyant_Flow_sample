#ifndef __POTENTIAL_SOLVER__
#define __POTENTIAL_SOLVER__
#include "gfGpuArray.h"
#include "MultiGrid.h"
#include "SpaceHashClass.h"
typedef unsigned int uint;
using namespace gf;



class PotentialFieldSolver
{
public:
	PotentialFieldSolver();

	~PotentialFieldSolver();
	bool initializeSolver(uint gdx, uint gdy, uint gdz, uint K, uint M, uint N);
	bool evaluateGradient(bool large_eval);
	bool computeDphiDn();
	GpuArrayd3 *getGradPhi() {return m_particle_gradPhi; }
	GpuArrayd  *getDpDn(){ return m_particle_dphidn; }
	GpuArrayd  *getScalarPhi(){ return m_particle_dphidn; }
	bool setEvalNormals(uint m, GpuArrayd3 * normals);
	bool setEvalAreas(uint m, GpuArrayd * areas);
	bool setEvalParameter(uint m, GpuArrayf4 * pos);
	bool setMassParameter(uint n, GpuArrayf4 * pos, GpuArrayd *mass);
	double getCellh() {return m_SpatialHasher_mass.getCellSize().x; }
	double4 getOrigin() {return m_origin;}
	void computeFarFieldBuffer();
	void computeFarFieldPotentialBuffer();
	void evaluateScalarPotential();
	bool shutdown();



private:
	bool setDomain(double4 & origin, float4 * pos, uint num_particle, double & domain_length);
	bool setEvalPos(GpuArrayf4 *pos);
	bool setMassPos(GpuArrayf4 *pos);
	bool setMassStrength(GpuArrayd *mass);

	bool m_ParticleToMesh();

	bool m_SolvePoisson();
	bool m_ComputeGradient();
	bool m_Intepolate();
	bool m_LocalCorrection();
	bool m_unsortResult();

	MultiGridSolver3DPeriod_double m_PoissonSolver;
	ParticleManager m_SpatialHasher_mass;
	ParticleManager m_SpatialHasher_eval;
	uint m_M_eval;
	uint m_N_mass;
	double m_cell_h;
	uint m_gridx, m_gridy, m_gridz;
	uint m_hashx, m_hashy, m_hashz;
	bool m_initialized;
	double4 m_origin;
	double m_L;
	uint m_K;
	double3 m_center;
	double m_total_mass;



	GpuArrayf4 * m_evalPos;
	GpuArrayd3 * m_evalNormal;
	GpuArrayf4 * m_evalPos_Reorder;
	GpuArrayf4 * m_p_massPos;
	GpuArrayf4 * m_p_massPos_Reorder;
	GpuArrayd * m_particle_mass;
	GpuArrayd * m_particle_mass_Reorder;
	GpuArrayd * m_grid_density;
	GpuArrayd * m_grid_Rhs;
	GpuArrayd * m_grid_phi;
	GpuArrayd * m_particle_dphidn;
	GpuArrayd3 * m_particle_gradPhi;
	GpuArrayd3 * m_particle_gradPhi_deorder;
	GpuArrayd3 * m_grid_gradPhi;
	GpuArrayd3 * m_far_gradPhi;
	GpuArrayd  * m_SLP_area;

	


};




#endif