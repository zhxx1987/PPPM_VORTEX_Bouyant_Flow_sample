#include "PotentialFieldSolver.h"
#include "MultiGrid.h"
#include "ParticleMesh.h"


PotentialFieldSolver::PotentialFieldSolver()
{
	m_M_eval=0;
	m_N_mass=0;
	m_cell_h=0;
	m_gridx=m_gridy=m_gridz=0;
	m_hashx=m_hashy=m_hashz=0;
	m_initialized=false;
	m_origin=make_double4(0,0,0,0);
	m_L=0;
	m_K=0;
	m_center=make_double3(0,0,0);
	m_total_mass=0;



	m_evalPos = new GpuArrayf4;
	m_evalNormal = new GpuArrayd3;
	m_evalPos_Reorder = new GpuArrayf4;
	m_p_massPos = new GpuArrayf4;
	m_p_massPos_Reorder = new GpuArrayf4;
	m_particle_mass = new GpuArrayd;
	m_particle_mass_Reorder = new GpuArrayd;
	m_grid_density = new GpuArrayd;
	m_grid_Rhs = new GpuArrayd;
	m_grid_phi = new GpuArrayd;
	m_particle_gradPhi = new GpuArrayd3;
	m_particle_dphidn = new GpuArrayd;
	m_particle_gradPhi_deorder = new GpuArrayd3;
	m_grid_gradPhi = new GpuArrayd3;
	m_far_gradPhi = new GpuArrayd3;
	m_SLP_area = new GpuArrayd;
}

PotentialFieldSolver::~PotentialFieldSolver()
{

}

bool
PotentialFieldSolver::initializeSolver(uint gdx, uint gdy, uint gdz, uint K, uint M, uint N)
{
	m_K = K;
	if(m_initialized)
	{
		if(gdx==m_gridx && gdy==m_gridy && gdz==m_gridz && M==m_M_eval && N==m_N_mass)
		{
			//zerofy memory




			m_evalPos->memset(make_float4(0,0,0,0));

			m_evalNormal->memset(make_double3(0,0,0));

			m_evalPos_Reorder->memset(make_float4(0,0,0,0));

			m_particle_gradPhi->memset(make_double3(0,0,0));
			
			m_particle_dphidn->memset(0);

			m_particle_gradPhi_deorder->memset(make_double3(0,0,0));

			m_p_massPos->memset(make_float4(0,0,0,0));

			m_p_massPos_Reorder->memset(make_float4(0,0,0,0));


			m_particle_mass->memset(0);


			m_particle_mass_Reorder->memset(0);

			m_grid_density->memset(0);


			m_grid_Rhs->memset(0);


			m_grid_phi->memset(0);


			m_grid_gradPhi->memset(make_double3(0,0,0));


			m_far_gradPhi->memset(make_double3(0,0,0));

			m_SLP_area->memset(0);



		}
		else //just reinitialize everything
		{
			m_gridx = gdx;
			m_gridy = gdy;
			m_gridz = gdz;
			m_M_eval = M;
			m_N_mass = N;


			m_PoissonSolver.m_InitialSystem(m_gridx, m_gridy, m_gridz);
			m_SpatialHasher_eval.endSpatialHash();
			m_SpatialHasher_eval.initSpatialHash(m_M_eval, m_gridx,m_gridy,m_gridz);
			m_SpatialHasher_mass.endSpatialHash();
			m_SpatialHasher_mass.initSpatialHash(m_N_mass, m_gridx,m_gridy,m_gridz);


			m_evalPos->free();
			m_evalPos->alloc(m_M_eval);
			m_evalPos->memset(make_float4(0,0,0,0));

			m_evalPos_Reorder->free();
			m_evalPos_Reorder->alloc(m_M_eval);
			m_evalPos_Reorder->memset(make_float4(0,0,0,0));


			m_evalNormal->free();
			m_evalNormal->alloc(m_M_eval);
			m_evalNormal->memset(make_double3(0,0,0));


			m_particle_gradPhi->free();
			m_particle_gradPhi->alloc(m_M_eval);
			m_particle_gradPhi->memset(make_double3(0,0,0));

			m_particle_gradPhi_deorder->free();
			m_particle_gradPhi_deorder->alloc(m_M_eval);
			m_particle_gradPhi_deorder->memset(make_double3(0,0,0));

			m_particle_dphidn->free();
			m_particle_dphidn->alloc(m_M_eval);
			m_particle_dphidn->memset(0);

			m_p_massPos->free();
			m_p_massPos->alloc(m_N_mass);
			m_p_massPos->memset(make_float4(0,0,0,0));

			m_p_massPos_Reorder->free();
			m_p_massPos_Reorder->alloc(m_N_mass);
			m_p_massPos_Reorder->memset(make_float4(0,0,0,0));

			m_particle_mass->free();
			m_particle_mass->alloc(m_N_mass);
			m_particle_mass->memset(0);

			m_particle_mass_Reorder->free();
			m_particle_mass_Reorder->alloc(m_N_mass);
			m_particle_mass_Reorder->memset(0);
			
			m_grid_density->free();
			m_grid_density->alloc(m_gridx*m_gridy*m_gridz);
			m_grid_density->memset(0);

			m_grid_Rhs->free();
			m_grid_Rhs->alloc(m_gridx*m_gridy*m_gridz);
			m_grid_Rhs->memset(0);

			m_grid_phi->free();
			m_grid_phi->alloc(m_gridx*m_gridy*m_gridz);
			m_grid_phi->memset(0);

			m_grid_gradPhi->free();
			m_grid_gradPhi->alloc(m_gridx*m_gridy*m_gridz);
			m_grid_gradPhi->memset(make_double3(0,0,0));

			m_far_gradPhi->free();
			m_far_gradPhi->alloc(m_gridx*m_gridy*m_gridz);
			m_far_gradPhi->memset(make_double3(0,0,0));

			m_SLP_area->free();
			m_SLP_area->alloc(m_M_eval);
			m_SLP_area->memset(0);
			
		}

	}
	else
	{
		m_gridx = gdx;
		m_gridy = gdy;
		m_gridz = gdz;
		m_M_eval = M;
		m_N_mass = N;


		m_PoissonSolver.m_InitialSystem(m_gridx, m_gridy, m_gridz);
		m_SpatialHasher_eval.initSpatialHash(m_M_eval, m_gridx,m_gridy,m_gridz);
		m_SpatialHasher_mass.initSpatialHash(m_N_mass, m_gridx,m_gridy,m_gridz);



		m_evalPos->alloc(m_M_eval);
		m_evalPos->memset(make_float4(0,0,0,0));


		m_evalPos_Reorder->alloc(m_M_eval);
		m_evalPos_Reorder->memset(make_float4(0,0,0,0));

		m_evalNormal->alloc(m_M_eval);
		m_evalNormal->memset(make_double3(0,0,0));

		m_particle_gradPhi->alloc(m_M_eval);
		m_particle_gradPhi->memset(make_double3(0,0,0));

		m_particle_gradPhi_deorder->alloc(m_M_eval);
		m_particle_gradPhi_deorder->memset(make_double3(0,0,0));

		m_particle_dphidn->alloc(m_M_eval);
		m_particle_dphidn->memset(0);

		m_p_massPos->alloc(m_N_mass);
		m_p_massPos->memset(make_float4(0,0,0,0));

		m_p_massPos_Reorder->alloc(m_N_mass);
		m_p_massPos_Reorder->memset(make_float4(0,0,0,0));

		m_particle_mass->alloc(m_N_mass);
		m_particle_mass->memset(0);

		m_particle_mass_Reorder->alloc(m_N_mass);
		m_particle_mass_Reorder->memset(0);

		m_grid_density->alloc(m_gridx*m_gridy*m_gridz);
		m_grid_density->memset(0);

		m_grid_Rhs->alloc(m_gridx*m_gridy*m_gridz);
		m_grid_Rhs->memset(0);

		m_grid_phi->alloc(m_gridx*m_gridy*m_gridz);
		m_grid_phi->memset(0);

		m_grid_gradPhi->alloc(m_gridx*m_gridy*m_gridz);
		m_grid_gradPhi->memset(make_double3(0,0,0));

		m_far_gradPhi->alloc(m_gridx*m_gridy*m_gridz);
		m_far_gradPhi->memset(make_double3(0,0,0));

		m_SLP_area->alloc(m_M_eval);
		m_SLP_area->memset(0);



		m_initialized = true;

	}
	return true;
}

bool
PotentialFieldSolver::shutdown()
{

	m_PoissonSolver.m_FinalMemoryEachLevel();
	m_SpatialHasher_eval.endSpatialHash();	
	m_SpatialHasher_mass.endSpatialHash();


	m_evalPos->free();

	m_evalPos_Reorder->free();

	m_evalNormal->free();

	m_particle_gradPhi->free();
	m_particle_gradPhi_deorder->free();

	m_particle_dphidn->free();

	m_p_massPos->free();

	m_p_massPos_Reorder->free();

	m_particle_mass->free();

	m_particle_mass_Reorder->free();

	m_grid_density->free();

	m_grid_Rhs->free();

	m_grid_phi->free();

	m_grid_gradPhi->free();

	m_far_gradPhi->free();

	m_SLP_area->free();

	m_initialized = false;
	
	return true;
}


bool
PotentialFieldSolver::setEvalParameter(uint m, GpuArrayf4 * pos)
{
	if(m!=m_M_eval)
	{
		m_M_eval = m;

		m_SpatialHasher_eval.endSpatialHash();
		m_SpatialHasher_eval.initSpatialHash(m_M_eval, m_gridx,m_gridy,m_gridz);

		m_evalPos->free();
		m_evalPos->alloc(m_M_eval);
		m_evalPos->memset(make_float4(0,0,0,0));

		m_evalPos_Reorder->free();
		m_evalPos_Reorder->alloc(m_M_eval);
		m_evalPos_Reorder->memset(make_float4(0,0,0,0));

		m_particle_gradPhi->free();
		m_particle_gradPhi->alloc(m_M_eval);
		m_particle_gradPhi->memset(make_double3(0,0,0));

		m_particle_gradPhi_deorder->free();
		m_particle_gradPhi_deorder->alloc(m_M_eval);
		m_particle_gradPhi_deorder->memset(make_double3(0,0,0));


	}
	




	if( setEvalPos(pos))
	{
		return true;
	}
	else
	{
		return false;
	}
}
bool
PotentialFieldSolver::setEvalPos(GpuArrayf4 *pos)
{
	cudaMemcpy(m_evalPos->getDevicePtr(),pos->getDevicePtr(),m_evalPos->getSize()*m_evalPos->typeSize(),cudaMemcpyDeviceToDevice);
	return true;

}


bool
PotentialFieldSolver::setMassParameter(uint n, GpuArrayf4 * pos, GpuArrayd *mass)
{
	if(m_N_mass!=n)
	{
		m_N_mass = n;
		m_SpatialHasher_mass.endSpatialHash();
		m_SpatialHasher_mass.initSpatialHash(m_N_mass, m_gridx,m_gridy,m_gridz);

		m_p_massPos->free();
		m_p_massPos->alloc(m_N_mass);
		m_p_massPos->memset(make_float4(0,0,0,0));

		m_p_massPos_Reorder->free();
		m_p_massPos_Reorder->alloc(m_N_mass);
		m_p_massPos_Reorder->memset(make_float4(0,0,0,0));

		m_particle_mass->free();
		m_particle_mass->alloc(m_N_mass);
		m_particle_mass->memset(0);

		m_particle_mass_Reorder->free();
		m_particle_mass_Reorder->alloc(m_N_mass);
		m_particle_mass_Reorder->memset(0);


	}
	if (setMassPos(pos) && setMassStrength(mass))
	{
		return true;
	}
	else
	{
		return false;
	}



}
bool
PotentialFieldSolver::setMassPos(GpuArrayf4 *pos)
{

	cudaMemcpy(m_p_massPos->getDevicePtr(),pos->getDevicePtr(),m_p_massPos->getSize()*m_p_massPos->typeSize(),cudaMemcpyDeviceToDevice);
	return true;

}
bool
PotentialFieldSolver::setMassStrength(GpuArrayd *mass)
{

	cudaMemcpy(m_particle_mass->getDevicePtr(),mass->getDevicePtr(),m_particle_mass->getSize()*m_particle_mass->typeSize(),cudaMemcpyDeviceToDevice);

	return true;
}
bool
PotentialFieldSolver::setDomain(double4 & origin, float4 * pos, uint num_particle, double & domain_length)
{
	double maxx,minx, maxy, miny, maxz,minz;
	//find minmax of x y z
	maxx = minx = pos[0].x;
	maxy = miny = pos[0].y;
	maxz = minz = pos[0].z;

	for (uint i=1; i<num_particle; i++)
	{
		if(pos[i].x<minx) minx = pos[i].x;
		else if(pos[i].x>maxx) maxx = pos[i].x;

		if(pos[i].y<miny) miny = pos[i].y;
		else if(pos[i].y>maxy) maxy = pos[i].y;

		if(pos[i].z<minz) minz = pos[i].z;
		else if(pos[i].z>maxz) maxz = pos[i].z;
	}

	domain_length = 2.0*max(max(maxx-minx, maxy-miny),maxz-minz);
	double center_x = minx + 0.5*(maxx-minx);
	double center_y = miny + 0.5*(maxy-miny);
	double center_z = minz + 0.5*(maxz-minz);

	origin.x = center_x - 0.5*domain_length;
	origin.y = center_y - 0.5*domain_length;
	origin.z = center_z - 0.5*domain_length;

	return true;
}

bool PotentialFieldSolver::evaluateGradient( bool large_eval )
{
	double h = m_SpatialHasher_mass.getCellSize().x;
	m_particle_gradPhi->memset(make_double3(0,0,0));


	if(large_eval)
	{
		m_SpatialHasher_eval.endSpatialHash();
		m_SpatialHasher_eval.initSpatialHash(m_M_eval,m_gridx,m_gridy,m_gridz);


		m_SpatialHasher_eval.setSpatialHashGrid(m_gridx, h,
			m_SpatialHasher_mass.getWorldOrigin(),
			m_M_eval);
		m_SpatialHasher_eval.setHashParam();
		m_SpatialHasher_eval.doSpatialHash(m_evalPos->getDevicePtr(),m_M_eval);
		m_SpatialHasher_eval.reorderData(m_M_eval, m_evalPos->getDevicePtr(),m_evalPos_Reorder->getDevicePtr(),4,1);

		m_particle_gradPhi_deorder->memset(make_double3(0,0,0));

		PotentialInterpolateFarField(m_evalPos_Reorder->getDevicePtr(),
			m_far_gradPhi->getDevicePtr(),m_particle_gradPhi_deorder->getDevicePtr(),
			m_SpatialHasher_mass.getCellSize().x,
			m_gridx,m_gridy,m_gridz,m_M_eval,m_origin);

		Potential_PPCorrMN(m_SpatialHasher_mass.getStartTable(),
			m_SpatialHasher_mass.getEndTable(),
			m_evalPos_Reorder->getDevicePtr(),
			m_p_massPos_Reorder->getDevicePtr(),
			m_particle_mass_Reorder->getDevicePtr(),
			m_particle_gradPhi_deorder->getDevicePtr(),
			1.0/h,
			h,
			1.0,
			1.0,
			make_uint3(m_gridx,m_gridy,m_gridz),
			make_uint3(m_gridx,m_gridy,m_gridz),
			make_uint2(m_gridx*m_gridy,m_gridx),
			make_uint2(m_gridx*m_gridy,m_gridx),
			m_K,
			m_M_eval,
			m_N_mass,
			m_origin);
		m_SpatialHasher_eval.deorderData(m_M_eval,m_particle_gradPhi_deorder->getDevicePtr(),m_particle_gradPhi->getDevicePtr(),3,2);
	}
	else
	{
		PotentialInterpolateFarField(m_evalPos->getDevicePtr(),
			m_far_gradPhi->getDevicePtr(),m_particle_gradPhi->getDevicePtr(),
			m_SpatialHasher_mass.getCellSize().x,
			m_gridx,m_gridy,m_gridz,m_M_eval,m_origin);

		Potential_PPCorrMN(m_SpatialHasher_mass.getStartTable(),
			m_SpatialHasher_mass.getEndTable(),
			m_evalPos->getDevicePtr(),
			m_p_massPos_Reorder->getDevicePtr(),
			m_particle_mass_Reorder->getDevicePtr(),
			m_particle_gradPhi->getDevicePtr(),
			1.0/h,
			h,
			1.0,
			1.0,
			make_uint3(m_gridx,m_gridy,m_gridz),
			make_uint3(m_gridx,m_gridy,m_gridz),
			make_uint2(m_gridx*m_gridy,m_gridx),
			make_uint2(m_gridx*m_gridy,m_gridx),
			m_K,
			m_M_eval,
			m_N_mass,
			m_origin);
	}


	

	PotentialComputeGradForOutParticle(m_evalPos->getDevicePtr(),m_total_mass, m_center,
		m_SpatialHasher_mass.getWorldOrigin(), 
		make_float3(m_SpatialHasher_mass.getWorldOrigin().x+m_L,
		m_SpatialHasher_mass.getWorldOrigin().y+m_L,
		m_SpatialHasher_mass.getWorldOrigin().z+m_L),
		1.0,1.0,m_particle_gradPhi->getDevicePtr(),m_M_eval);


	return true;
}
bool
PotentialFieldSolver::m_ParticleToMesh()
{
	m_SpatialHasher_mass.setSpatialHashGrid(m_gridx, m_L/(double)m_gridx,
		make_float3(m_origin.x,m_origin.y,m_origin.z),
		m_N_mass);
	m_SpatialHasher_mass.setHashParam();
	m_SpatialHasher_mass.doSpatialHash(m_p_massPos->getDevicePtr(),m_N_mass);

	m_p_massPos_Reorder->memset(make_float4(0,0,0,0));
	m_SpatialHasher_mass.reorderData(m_N_mass, (void*)(m_p_massPos->getDevicePtr()),
		(void*)(m_p_massPos_Reorder->getDevicePtr()), 4, 1);



	m_particle_mass_Reorder->memset(0);
	m_SpatialHasher_mass.reorderData(m_N_mass, (void*)(m_particle_mass->getDevicePtr()),
			(void*)(m_particle_mass_Reorder->getDevicePtr()), 1, 2);


	m_grid_density->memset(0);
	ParticleToMesh(m_SpatialHasher_mass.getStartTable(),
		m_SpatialHasher_mass.getEndTable(),
		m_p_massPos_Reorder->getDevicePtr(),
		m_particle_mass_Reorder->getDevicePtr(),
		m_SpatialHasher_mass.getCellSize().x,
		m_grid_density->getDevicePtr(),
		make_uint3(m_gridx,m_gridy,m_gridz),
		make_uint3(m_gridx,m_gridy,m_gridz),
		m_N_mass,
		m_origin);
	cudaMemcpy(m_grid_Rhs->getDevicePtr(),
		m_grid_density->getDevicePtr(),
		m_grid_Rhs->getSize()*m_grid_Rhs->typeSize(),
		cudaMemcpyDeviceToDevice);
	ComputeRHS(m_grid_Rhs->getDevicePtr(),
		m_SpatialHasher_mass.getCellSize().x*m_SpatialHasher_mass.getCellSize().x,
		-1.0,
		m_gridx*m_gridy*m_gridz);

	m_p_massPos_Reorder->copy(gf_GpuArray<float4>::DEVICE_TO_HOST);
	m_particle_mass_Reorder->copy(gf_GpuArray<double>::DEVICE_TO_HOST);

	double total_weight = 0;
	double total_mass = 0;
	for(int i=0; i<m_N_mass; i++)
	{
		double *host = m_particle_mass_Reorder->getHostPtr();
		total_weight += fabs(host[i]);
		total_mass += host[i];
	}
	double cx=0, cy=0, cz=0;
	for(int i=0; i<m_N_mass; i++)
	{
		float4 *hpos = m_p_massPos_Reorder->getHostPtr();
		double *hmass = m_particle_mass_Reorder->getHostPtr();
		cx+=hpos[i].x*fabs(hmass[i]);
		cy+=hpos[i].y*fabs(hmass[i]);
		cz+=hpos[i].z*fabs(hmass[i]);
		//printf("%f,%f,%f\n",cx,cy,cz);
	}
	cx=cx/total_weight;
	cy=cy/total_weight;
	cz=cz/total_weight;

	m_center.x = cx;
	m_center.y = cy;
	m_center.z = cz;
	m_total_mass = total_mass;

	applyDirichlet(m_grid_Rhs->getDevicePtr(), 
		make_double4(cx,cy,cz,0),
		m_total_mass,
		make_double4(m_origin.x,m_origin.y,m_origin.z,0),
		m_SpatialHasher_mass.getCellSize().x,
		m_gridx,
		m_gridy,
		m_gridz);

	return true;

}

bool
PotentialFieldSolver::m_SolvePoisson()
{
	m_grid_phi->memset(0);
	double res;
	for (int i=0;i<3;i++)
	{
		m_PoissonSolver.m_FullMultiGrid(m_grid_phi,m_grid_Rhs,1e-7,res);
	}
	return true;
}

bool
PotentialFieldSolver::m_ComputeGradient()
{
	m_grid_gradPhi->memset(make_double3(0,0,0));
	ComputeGradient(m_grid_phi->getDevicePtr(),m_grid_gradPhi->getDevicePtr(),
		m_SpatialHasher_mass.getCellSize().x,1.0,m_gridx,m_gridy,m_gridz);

	return true;
}

bool
PotentialFieldSolver::m_Intepolate()
{
	// we don't need any more
	return true;
}
bool
PotentialFieldSolver::m_LocalCorrection()
{
	//we don't need this any more
	return true;
}

void
PotentialFieldSolver::computeFarFieldBuffer()
{
	m_p_massPos->copy(m_p_massPos->DEVICE_TO_HOST);
	setDomain(m_origin,m_p_massPos->getHostPtr(),m_N_mass,m_L);
	printf("%lf,%lf,%lf,%lf\n",m_L,m_origin.x,m_origin.y,m_origin.z);
	

	m_ParticleToMesh();
	m_SolvePoisson();
	m_ComputeGradient();
	
	PotentialComputeFarField(m_SpatialHasher_mass.getStartTable(),
							 m_SpatialHasher_mass.getEndTable(),
							 m_p_massPos->getDevicePtr(),
							 m_particle_mass_Reorder->getDevicePtr(),
							 m_grid_density->getDevicePtr(),
							 m_grid_phi->getDevicePtr(),
							 m_grid_gradPhi->getDevicePtr(),
							 m_far_gradPhi->getDevicePtr(),
							 m_SpatialHasher_mass.getCellSize().x,
							 1.0,1.0,
							 make_uint3(m_gridx,m_gridy,m_gridz),
							 make_uint3(m_gridx,m_gridy,m_gridz),
							 m_K,
							 m_M_eval,
							 m_origin);
							 
}
bool
PotentialFieldSolver::computeDphiDn()
{
	ComputeDphidn(m_evalNormal->getDevicePtr(),m_particle_gradPhi->getDevicePtr(),m_particle_mass->getDevicePtr(),m_particle_dphidn->getDevicePtr(),m_SLP_area->getDevicePtr(), m_M_eval);
	return true;
}

bool
PotentialFieldSolver::setEvalNormals(uint m, GpuArrayd3 * normals)
{
	cudaMemcpy(m_evalNormal->getDevicePtr(),normals->getDevicePtr(),m*sizeof(double3),cudaMemcpyDeviceToDevice);
	return true;
}

bool
PotentialFieldSolver::setEvalAreas(uint m, GpuArrayd * areas)
{
	cudaMemcpy(m_SLP_area->getDevicePtr(),areas->getDevicePtr(),m*sizeof(double),cudaMemcpyDeviceToDevice);
	return true;
}

void PotentialFieldSolver::computeFarFieldPotentialBuffer()
{
	m_p_massPos->copy(m_p_massPos->DEVICE_TO_HOST);
	setDomain(m_origin,m_p_massPos->getHostPtr(),m_N_mass,m_L);

	m_ParticleToMesh();
	m_SolvePoisson();
	PotentialComputeScalarFarField(m_particle_mass->getDeviceWritePtr(),m_grid_density->getDeviceWritePtr(),m_grid_phi->getDeviceWritePtr(),m_SpatialHasher_mass.getCellSize().x, 1.0,1.0,
		make_uint3(m_gridx,m_gridy,m_gridz),
		make_uint3(m_gridx,m_gridy,m_gridz),
		m_K);
}

void PotentialFieldSolver::evaluateScalarPotential()
{
	double h = m_SpatialHasher_mass.getCellSize().x;
	m_particle_dphidn->memset(0);



	PotentialInterpolateFarFieldScalar(m_evalPos->getDevicePtr(),
		m_grid_phi->getDevicePtr(),m_particle_dphidn->getDevicePtr(),
		m_SpatialHasher_mass.getCellSize().x,
		m_gridx,m_gridy,m_gridz,m_M_eval,m_origin);

	Potential_PPCorrMNScalar(m_SpatialHasher_mass.getStartTable(),
		m_SpatialHasher_mass.getEndTable(),
		m_evalPos->getDevicePtr(),
		m_p_massPos_Reorder->getDevicePtr(),
		m_particle_mass_Reorder->getDevicePtr(),
		m_particle_dphidn->getDevicePtr(),
		1.0/h,
		h,
		1.0,
		1.0,
		make_uint3(m_gridx,m_gridy,m_gridz),
		make_uint3(m_gridx,m_gridy,m_gridz),
		make_uint2(m_gridx*m_gridy,m_gridx),
		make_uint2(m_gridx*m_gridy,m_gridx),
		m_K,
		m_M_eval,
		m_N_mass,
		m_origin);

}
