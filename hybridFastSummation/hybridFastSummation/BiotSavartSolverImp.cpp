#include "BiotSavartSolver.h"
#include "MultiGrid.h"
#include "ParticleMesh.h"


BiotSavartSolver::BiotSavartSolver()
{
	m_N_vort = 0;
	m_M_eval = 0;
	m_gridx = m_gridy = m_gridz = 0;
	m_hashx = m_hashy = m_hashz = 0;
	m_cell_h = 0.0;
	m_origin = make_double4(0,0,0,0);
	m_center = make_double3(0,0,0);
	m_L = 0.0;
	m_initialized = false;
	m_K = 0;
	m_evalPos = new gf_GpuArray<float4>;
	m_evalPos_Reorder = new gf_GpuArray<float4>;
	m_p_vortPos = new gf_GpuArray<float4>;
	m_p_vortPos_Reorder = new GpuArrayf4;
	m_isVIC = false;
	for (int i=0;i<NUM_COMPONENTS; i++)
	{
		m_total_vort[i]=0;
		m_grid_Rhs[i] = new GpuArrayd;
		m_particle_vort[i] = new GpuArrayd;
		m_particle_vort_Reorder[i] = new GpuArrayd;
		m_grid_vort[i] = new GpuArrayd;
		m_grid_Psi[i] = new GpuArrayd;
		m_particle_U[i] = new GpuArrayd;
		m_particle_U_deorder[i] = new GpuArrayd;
		m_grid_U[i] = new GpuArrayd;
		m_far_U[i] = new GpuArrayd;
	}


}
BiotSavartSolver::~BiotSavartSolver()
{

}
bool
BiotSavartSolver::initializeSolver(uint gdx, uint gdy, uint gdz, bool isVIC, int K, uint M, uint N)
{
	m_isVIC = isVIC;
	m_K = K;
	if(m_initialized)
	{
		if(gdx==m_gridx && gdy==m_gridy && gdz==m_gridz && M==m_M_eval && N==m_N_vort)
		{
			//zerofy memory
			m_evalPos->memset(make_float4(0,0,0,0));
			
			m_evalPos_Reorder->memset(make_float4(0,0,0,0));

			
			m_p_vortPos->memset(make_float4(0,0,0,0));
			
			m_p_vortPos_Reorder->memset(make_float4(0,0,0,0));

			for (int i=0;i<NUM_COMPONENTS;i++)
			{
				
				m_grid_Rhs[i]->memset(0);

				
				m_particle_vort[i]->memset(0);

				
				m_particle_vort_Reorder[i]->memset(0);

				
				m_grid_vort[i]->memset(0);

				
				m_grid_Psi[i]->memset(0);

				
				m_particle_U[i]->memset(0);

				
				m_particle_U_deorder[i]->memset(0);

				
				m_grid_U[i]->memset(0);

				if(!m_isVIC)
					m_far_U[i]->memset(0);
			}

			
		}
		else //just reinitialize everything
		{
			m_gridx = gdx;
			m_gridy = gdy;
			m_gridz = gdz;
			m_M_eval = M;
			m_N_vort = N;

			
			m_PoissonSolver.m_InitialSystem(m_gridx, m_gridy, m_gridz);
			if(!m_isVIC){

				m_SpatialHasher_eval.endSpatialHash();
				m_SpatialHasher_eval.initSpatialHash(m_M_eval, m_gridx,m_gridy,m_gridz);
			}
			m_SpatialHasher_vort.endSpatialHash();
			m_SpatialHasher_vort.initSpatialHash(m_N_vort, m_gridx,m_gridy,m_gridz);


			m_evalPos->free();
			m_evalPos->alloc(m_M_eval);
			m_evalPos->memset(make_float4(0,0,0,0));
			m_evalPos_Reorder->free();
			m_evalPos_Reorder->alloc(m_M_eval);
			m_evalPos_Reorder->memset(make_float4(0,0,0,0));

			m_p_vortPos->free();
			m_p_vortPos->alloc(m_N_vort);
			m_p_vortPos->memset(make_float4(0,0,0,0));
			m_p_vortPos_Reorder->free();
			m_p_vortPos_Reorder->alloc(m_N_vort);
			m_p_vortPos_Reorder->memset(make_float4(0,0,0,0));

			for (int i=0;i<NUM_COMPONENTS;i++)
			{
				m_grid_Rhs[i]->free();
				m_grid_Rhs[i]->alloc(m_gridx*m_gridy*m_gridz);
				m_grid_Rhs[i]->memset(0);
				m_particle_vort[i]->free();
				m_particle_vort[i]->alloc(m_N_vort);
				m_particle_vort[i]->memset(0);
				m_particle_vort_Reorder[i]->free();
				m_particle_vort_Reorder[i]->alloc(m_N_vort);
				m_particle_vort_Reorder[i]->memset(0);
				m_grid_vort[i]->free();
				m_grid_vort[i]->alloc(m_gridx*m_gridy*m_gridz);
				m_grid_vort[i]->memset(0);
				m_grid_Psi[i]->free();
				m_grid_Psi[i]->alloc(m_gridx*m_gridy*m_gridz);
				m_grid_Psi[i]->memset(0);
				m_particle_U[i]->free();
				m_particle_U[i]->alloc(m_M_eval);
				m_particle_U[i]->memset(0);
				m_particle_U_deorder[i]->free();
				m_particle_U_deorder[i]->alloc(m_M_eval);
				m_particle_U_deorder[i]->memset(0);
				m_grid_U[i]->free();
				m_grid_U[i]->alloc(m_gridx*m_gridy*m_gridz);
				m_grid_U[i]->memset(0);
				if(!m_isVIC){

					m_far_U[i]->free();
					m_far_U[i]->alloc(m_gridx*m_gridy*m_gridz);
					m_far_U[i]->memset(0);
				}
			}
		}
		
	}
	else
	{
		m_gridx = gdx;
		m_gridy = gdy;
		m_gridz = gdz;
		m_M_eval = M;
		m_N_vort = N;


		m_PoissonSolver.m_InitialSystem(m_gridx, m_gridy, m_gridz);
		if(!m_isVIC){

			m_SpatialHasher_eval.initSpatialHash(m_M_eval, m_gridx,m_gridy,m_gridz);
		}
		m_SpatialHasher_vort.initSpatialHash(m_N_vort, m_gridx,m_gridy,m_gridz);


		m_evalPos->alloc(m_M_eval);
		m_evalPos->memset(make_float4(0,0,0,0));
		m_evalPos_Reorder->alloc(m_M_eval);
		m_evalPos_Reorder->memset(make_float4(0,0,0,0));

		m_p_vortPos->alloc(m_N_vort);
		m_p_vortPos->memset(make_float4(0,0,0,0));
		m_p_vortPos_Reorder->alloc(m_N_vort);
		m_p_vortPos_Reorder->memset(make_float4(0,0,0,0));

		for (int i=0;i<NUM_COMPONENTS;i++)
		{
			m_grid_Rhs[i]->alloc(m_gridx*m_gridy*m_gridz);
			m_grid_Rhs[i]->memset(0);

			m_particle_vort[i]->alloc(m_N_vort);
			m_particle_vort[i]->memset(0);

			m_particle_vort_Reorder[i]->alloc(m_N_vort);
			m_particle_vort_Reorder[i]->memset(0);

			m_grid_vort[i]->alloc(m_gridx*m_gridy*m_gridz);
			m_grid_vort[i]->memset(0);

			m_grid_Psi[i]->alloc(m_gridx*m_gridy*m_gridz);
			m_grid_Psi[i]->memset(0);

			m_particle_U[i]->alloc(m_M_eval);
			m_particle_U[i]->memset(0);

			m_particle_U_deorder[i]->alloc(m_M_eval);
			m_particle_U_deorder[i]->memset(0);

			m_grid_U[i]->alloc(m_gridx*m_gridy*m_gridz);
			m_grid_U[i]->memset(0);
			if(!m_isVIC){

				m_far_U[i]->alloc(m_gridx*m_gridy*m_gridz);
				m_far_U[i]->memset(0);
			}
		}




		m_initialized = true;

	}
	return true;
}
bool
BiotSavartSolver::shutdown()
{

	m_PoissonSolver.m_FinalMemoryEachLevel();
	if(!m_isVIC){
		m_SpatialHasher_eval.endSpatialHash();
	}
	m_SpatialHasher_vort.endSpatialHash();


	m_evalPos->free();
	m_evalPos_Reorder->free();

	m_p_vortPos->free();
	m_p_vortPos_Reorder->free();

	for (int i=0;i<NUM_COMPONENTS;i++)
	{
		m_grid_Rhs[i]->free();
		m_particle_vort[i]->free();
		m_particle_vort_Reorder[i]->free();
		m_grid_vort[i]->free();
		m_grid_Psi[i]->free();
		m_particle_U[i]->free();
		m_particle_U_deorder[i]->free();
		m_grid_U[i]->free();
		if(!m_isVIC){

			m_far_U[i]->free();
		}
	}
	m_initialized = false;
	return true;
}

bool
BiotSavartSolver::setEvalParameter(uint m, GpuArrayf4 * pos)
{
	if (m_M_eval!=m)
	{
		m_M_eval = m;
		if(!m_isVIC){

			m_SpatialHasher_eval.endSpatialHash();
			m_SpatialHasher_eval.initSpatialHash(m_M_eval,m_gridx,m_gridy,m_gridz);
		}

		m_evalPos->free();
		m_evalPos->alloc(m_M_eval);
		m_evalPos->memset(make_float4(0,0,0,0));
		m_evalPos_Reorder->free();
		m_evalPos_Reorder->alloc(m_M_eval);
		m_evalPos_Reorder->memset(make_float4(0,0,0,0));



		for(int i=0;i<NUM_COMPONENTS;i++)
		{

			m_particle_U[i]->free();
			m_particle_U[i]->alloc(m_M_eval);
			m_particle_U[i]->memset(0);
			m_particle_U_deorder[i]->free();
			m_particle_U_deorder[i]->alloc(m_M_eval);
			m_particle_U_deorder[i]->memset(0);
		}
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
BiotSavartSolver::setEvalPos(GpuArrayf4 *pos)
{
	cudaMemcpy(m_evalPos->getDevicePtr(),pos->getDevicePtr(),m_evalPos->getSize()*m_evalPos->typeSize(),cudaMemcpyDeviceToDevice);
	return true;

}


bool
BiotSavartSolver::setVortParameter(uint n, GpuArrayf4 * pos, GpuArrayd *omega[])
{

	if(m_N_vort!=n)
	{
		m_N_vort = n;
		m_SpatialHasher_vort.endSpatialHash();
		m_SpatialHasher_vort.initSpatialHash(m_N_vort, m_gridx,m_gridy,m_gridz);

		m_p_vortPos->free();
		m_p_vortPos->alloc(m_N_vort);
		m_p_vortPos->memset(make_float4(0,0,0,0));
		m_p_vortPos_Reorder->free();
		m_p_vortPos_Reorder->alloc(m_N_vort);
		m_p_vortPos_Reorder->memset(make_float4(0,0,0,0));
		for(int i=0;i<NUM_COMPONENTS;i++)
		{
			m_particle_vort[i]->free();
			m_particle_vort[i]->alloc(m_N_vort);
			m_particle_vort[i]->memset(0);
			m_particle_vort_Reorder[i]->free();
			m_particle_vort_Reorder[i]->alloc(m_N_vort);
			m_particle_vort_Reorder[i]->memset(0);
		}


	}


	m_N_vort = n;
	if (setVortPos(pos) && setVortStrength(omega))
	{
		return true;
	}
	else
	{
		return false;
	}



}
bool
BiotSavartSolver::setVortPos(GpuArrayf4 *pos)
{

	cudaMemcpy(m_p_vortPos->getDevicePtr(),pos->getDevicePtr(),m_p_vortPos->getSize()*m_p_vortPos->typeSize(),cudaMemcpyDeviceToDevice);
	return true;

}
bool
BiotSavartSolver::setVortStrength(GpuArrayd *omega[])
{
	for (int i=0;i<NUM_COMPONENTS;i++)
	{
		cudaMemcpy(m_particle_vort[i]->getDevicePtr(),omega[i]->getDevicePtr(),m_particle_vort[i]->getSize()*m_particle_vort[i]->typeSize(),cudaMemcpyDeviceToDevice);
	}
	return true;
}
bool
BiotSavartSolver::setDomain(double4 & origin, float4 * pos, uint num_particle, double & domain_length)
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

	domain_length = 3*max(max(maxx-minx, maxy-miny),maxz-minz);
	double center_x = minx + 0.5*(maxx-minx);
	double center_y = miny + 0.5*(maxy-miny);
	double center_z = minz + 0.5*(maxz-minz);

	origin.x = center_x - 0.5*domain_length;
	origin.y = center_y - 0.5*domain_length;
	origin.z = center_z - 0.5*domain_length;

	return true;
}
bool BiotSavartSolver::evaluateVelocity( GpuArrayf4 *another_end, uint is_segment )
{

	//m_p_vortPos->copy(gf_GpuArray<float4>::DEVICE_TO_HOST);
	//setDomain(m_origin, m_p_vortPos->getHostPtr(),m_N_vort,m_L);
	////printf("%f,%f,%f,%f,\n",m_origin.x,m_origin.y,m_origin.z,m_L);
	//m_SpatialHasher_eval.setSpatialHashGrid(m_gridx, m_L/(double)m_gridx,
	//	make_float3(m_origin.x,m_origin.y,m_origin.z),
	//	m_M_eval);
	//m_SpatialHasher_eval.setHashParam();
	//m_SpatialHasher_eval.doSpatialHash(m_evalPos->getDevicePtr(),m_M_eval);
	//m_SpatialHasher_eval.reorderData(m_M_eval,m_evalPos->getDevicePtr(),m_evalPos_Reorder->getDevicePtr(),4,1);


	//m_ParticleToMesh();
	//m_SolvePoisson();
	//m_ComputeCurl();
	//m_Intepolate();
	//m_LocalCorrection(another_end);
	//m_unsortResult();
	GpuArrayf4 *temp_pos=new GpuArrayf4;
	temp_pos->alloc(m_M_eval);
	temp_pos->memset(make_float4(0,0,0,0));

	for(int i=0;i<NUM_COMPONENTS;i++)
	{
		m_particle_U[i]->memset(0);
		m_particle_U_deorder[i]->memset(0);
	}
	if(!m_isVIC){
		
		m_SpatialHasher_eval.setSpatialHashGrid(m_gridx, m_L/(double)m_gridx,
			make_float3(m_origin.x,m_origin.y,m_origin.z),
			m_M_eval);
		m_SpatialHasher_eval.setHashParam();
		m_SpatialHasher_eval.doSpatialHash(m_evalPos->getDevicePtr(),m_M_eval);
		m_SpatialHasher_eval.reorderData(m_M_eval, m_evalPos->getDevicePtr(),m_evalPos_Reorder->getDevicePtr(),4,1);
		if(is_segment==1)
		{
			m_SpatialHasher_eval.reorderData(m_M_eval,another_end->getDevicePtr(),temp_pos->getDevicePtr(),4,1);
		}

		BiotSavartInterpolateFarField(m_evalPos_Reorder->getDevicePtr(),
			m_far_U[0]->getDevicePtr(),m_far_U[1]->getDevicePtr(), m_far_U[2]->getDevicePtr(),
			m_particle_U_deorder[0]->getDevicePtr(), m_particle_U_deorder[1]->getDevicePtr(),m_particle_U_deorder[2]->getDevicePtr(),
			m_SpatialHasher_vort.getCellSize().x,
			m_gridx,m_gridy,m_gridz,
			m_M_eval,
			m_origin);
		BiotSavartPPCorrScaleMN(m_SpatialHasher_vort.getStartTable(),
			m_SpatialHasher_vort.getEndTable(),
			m_evalPos_Reorder->getDevicePtr(),
			temp_pos->getDevicePtr(),
			is_segment,
			m_p_vortPos_Reorder->getDevicePtr(),
			m_particle_vort_Reorder[0]->getDevicePtr(),
			m_particle_vort_Reorder[1]->getDevicePtr(),
			m_particle_vort_Reorder[2]->getDevicePtr(),
			m_particle_U_deorder[0]->getDevicePtr(),
			m_particle_U_deorder[1]->getDevicePtr(),
			m_particle_U_deorder[2]->getDevicePtr(),
			m_grid_U[0]->getDevicePtr(),
			m_grid_U[1]->getDevicePtr(),
			m_grid_U[2]->getDevicePtr(),
			m_SpatialHasher_vort.getCellSize().x,
			make_uint3(m_gridx,m_gridy,m_gridz),
			make_uint3(m_gridx,m_gridy,m_gridz),
			m_K,
			m_M_eval,
			m_N_vort,
			m_origin);
		for (int c=0;c<3;c++)
		{
			m_SpatialHasher_eval.deorderData(m_M_eval,m_particle_U_deorder[c]->getDevicePtr(),m_particle_U[c]->getDevicePtr(),1,2);
		}
		
	}
	else
	{
		
		BiotSavartInterpolateFarField(m_evalPos->getDevicePtr(),
			m_grid_U[0]->getDevicePtr(),m_grid_U[1]->getDevicePtr(), m_grid_U[2]->getDevicePtr(),
			m_particle_U[0]->getDevicePtr(), m_particle_U[1]->getDevicePtr(),m_particle_U[2]->getDevicePtr(),
			m_SpatialHasher_vort.getCellSize().x,
			m_gridx,m_gridy,m_gridz,
			m_M_eval,
			m_origin);
		
	}

	//BiotSavartComputeVelocityForOutParticle(m_evalPos->getDevicePtr(),
	//	make_double3(m_total_vort[0],m_total_vort[1],m_total_vort[2]), 
	//	m_center,
	//	m_SpatialHasher_vort.getWorldOrigin(), 
	//	make_float3(m_SpatialHasher_vort.getWorldOrigin().x+m_L,
	//				 m_SpatialHasher_vort.getWorldOrigin().y+m_L,
	//				 m_SpatialHasher_vort.getWorldOrigin().z+m_L),
	//    m_particle_U[0]->getDevicePtr(),
	//	m_particle_U[1]->getDevicePtr(),
	//	m_particle_U[2]->getDevicePtr(),
	//	m_M_eval);

	temp_pos->free();

	return true;

}
bool
BiotSavartSolver::m_ParticleToMesh()
{
	m_SpatialHasher_vort.setSpatialHashGrid(m_gridx, m_L/(double)m_gridx,
		make_float3(m_origin.x,m_origin.y,m_origin.z),
		m_N_vort);
	m_SpatialHasher_vort.setHashParam();
	m_SpatialHasher_vort.doSpatialHash(m_p_vortPos->getDevicePtr(),m_N_vort);

	m_p_vortPos_Reorder->memset(make_float4(0,0,0,0));
	m_SpatialHasher_vort.reorderData(m_N_vort, (void*)(m_p_vortPos->getDevicePtr()),
		(void*)(m_p_vortPos_Reorder->getDevicePtr()), 4, 1);


	for(int i=0;i<NUM_COMPONENTS;i++)
	{
		m_particle_vort_Reorder[i]->memset(0);
		m_SpatialHasher_vort.reorderData(m_N_vort, (void*)(m_particle_vort[i]->getDevicePtr()),
			(void*)(m_particle_vort_Reorder[i]->getDevicePtr()), 1, 2);

	}

	for (int c=0;c<NUM_COMPONENTS;c++)
	{
		m_grid_vort[c]->memset(0);
		ParticleToMesh(m_SpatialHasher_vort.getStartTable(),
			m_SpatialHasher_vort.getEndTable(),
			m_p_vortPos_Reorder->getDevicePtr(),
			m_particle_vort_Reorder[c]->getDevicePtr(),
			m_SpatialHasher_vort.getCellSize().x,
			m_grid_vort[c]->getDevicePtr(),
			make_uint3(m_gridx,m_gridy,m_gridz),
			make_uint3(m_gridx,m_gridy,m_gridz),
			m_N_vort,
			m_origin);
		cudaMemcpy(m_grid_Rhs[c]->getDevicePtr(),
			m_grid_vort[c]->getDevicePtr(),
			m_grid_Rhs[c]->getSize()*m_grid_Rhs[c]->typeSize(),
			cudaMemcpyDeviceToDevice);
		ComputeRHS(m_grid_Rhs[c]->getDevicePtr(),
			m_SpatialHasher_vort.getCellSize().x*m_SpatialHasher_vort.getCellSize().x,
			-1.0,
			m_gridx*m_gridy*m_gridz);
		//m_p_vortPos_Reorder->copy(GpuArrayf4::DEVICE_TO_HOST);
		//m_particle_vort_Reorder[c]->copy(GpuArrayd::DEVICE_TO_HOST);
		//double total_weight = 0;
		//double total_mass = 0;
		//for(int i=0; i<m_N_vort; i++)
		//{
		//	double *host = m_particle_vort_Reorder[c]->getHostPtr();
		//	total_weight += fabs(host[i]);
		//	total_mass += host[i];
		//}
		//double cx=0, cy=0, cz=0;
		//for(int i=0; i<m_N_vort; i++)
		//{
		//	float4 *hpos = m_p_vortPos_Reorder->getHostPtr();
		//	double *hmass = m_particle_vort_Reorder[c]->getHostPtr();
		//	cx+=hpos[i].x*fabs(hmass[i]);
		//	cy+=hpos[i].y*fabs(hmass[i]);
		//	cz+=hpos[i].z*fabs(hmass[i]);
		//	//printf("%f,%f,%f\n",cx,cy,cz);
		//}
		//cx=cx/total_weight;
		//cy=cy/total_weight;
		//cz=cz/total_weight;

		//m_center.x = cx;
		//m_center.y = cy;
		//m_center.z = cz;
		//m_total_vort[c] = total_mass;
		////printf("%f,%f,%f,%f\n",cx,cy,cz,total_mass);
		//applyDirichlet(m_grid_Rhs[c]->getDevicePtr(), 
		//	make_double4(cx,cy,cz,0),
		//	total_mass,
		//	m_origin,
		//	m_SpatialHasher_vort.getCellSize().x,
		//	m_gridx,
		//	m_gridy,
		//	m_gridz);
	}


	return true;

}
bool
BiotSavartSolver::m_SolvePoisson()
{
	double res;
	for (int i=0;i<NUM_COMPONENTS;i++)
	{
		m_grid_Psi[i]->memset(0);
		for (int t=0;t<1;t++)
		{
			double res;
			m_PoissonSolver.m_FullMultiGrid(m_grid_Psi[i],
				m_grid_Rhs[i],
				1e-10, res);
		}

	}
	return true;
}
bool
BiotSavartSolver::m_ComputeCurl()
{
	for (int c=0;c<3;c++)
	{
		m_grid_U[c]->memset(0);
	}
	getCurl(m_grid_Psi[0]->getDevicePtr(),
		m_grid_Psi[1]->getDevicePtr(),
		m_grid_Psi[2]->getDevicePtr(),
		m_grid_U[0]->getDevicePtr(),
		m_grid_U[1]->getDevicePtr(),
		m_grid_U[2]->getDevicePtr(),
		m_gridx,
		m_gridy,
		m_gridz,
		m_SpatialHasher_vort.getCellSize().x);
	return true;
}

bool
BiotSavartSolver::m_Intepolate()
{
	for (int c=0;c<3;c++)
	{
		m_particle_U[c]->memset(0);
		MeshToParticle(m_evalPos->getDevicePtr(),
			m_grid_U[c]->getDevicePtr(),
			m_particle_U[c]->getDevicePtr(),
			m_SpatialHasher_vort.getCellSize().x,
			make_uint3(m_gridx,m_gridy,m_gridz),
			make_uint3(m_gridx,m_gridy,m_gridz),
			m_M_eval,
			m_origin);
	}
	return true;
}

bool
BiotSavartSolver::m_LocalCorrection(GpuArrayf4 *another_end)
{

	//BiotSavartPPCorr(m_SpatialHasher_vort.getStartTable(),
	//	m_SpatialHasher_vort.getEndTable(),
	//	m_p_vortPos_Reorder->getDevicePtr(),
	//	m_particle_vort_Reorder[0]->getDevicePtr(),
	//	m_particle_vort_Reorder[1]->getDevicePtr(),
	//	m_particle_vort_Reorder[2]->getDevicePtr(),
	//	m_particle_U[0]->getDevicePtr(),
	//	m_particle_U[1]->getDevicePtr(),
	//	m_particle_U[2]->getDevicePtr(),
	//	m_SpatialHasher_vort.getCellSize().x,
	//	make_uint3(m_gridx,m_gridy,m_gridz),
	//	make_uint3(m_gridx,m_gridy,m_gridz),
	//	m_K,
	//	m_N_vort,
	//	m_origin);

	//BiotSavartPMCorr(m_SpatialHasher_vort.getStartTable(),
	//	m_SpatialHasher_vort.getEndTable(),
	//	m_evalPos->getDevicePtr(),
	//	m_grid_vort[0]->getDevicePtr(),
	//	m_grid_vort[1]->getDevicePtr(),
	//	m_grid_vort[2]->getDevicePtr(),
	//	m_grid_Psi[0]->getDevicePtr(),
	//	m_grid_Psi[1]->getDevicePtr(),
	//	m_grid_Psi[2]->getDevicePtr(),
	//	m_grid_U[0]->getDevicePtr(),
	//	m_grid_U[1]->getDevicePtr(),
	//	m_grid_U[2]->getDevicePtr(),
	//	m_particle_U[0]->getDevicePtr(),
	//	m_particle_U[1]->getDevicePtr(),
	//	m_particle_U[2]->getDevicePtr(),
	//	m_SpatialHasher_vort.getCellSize().x,
	//	make_uint3(m_gridx,m_gridy,m_gridz),
	//	make_uint3(m_gridx,m_gridy,m_gridz),
	//	m_K,
	//	m_M_eval,
	//	m_origin);
	//ComputeVelocityForOutParticle(m_evalPos->getDevicePtr(),
	//	make_double3(m_total_vort[0],m_total_vort[1],m_total_vort[2]), 
	//	m_center,
	//	m_SpatialHasher_vort.getWorldOrigin(), 
	//	make_float3(m_SpatialHasher_vort.getWorldOrigin().x+m_L,
	//				 m_SpatialHasher_vort.getWorldOrigin().y+m_L,
	//				 m_SpatialHasher_vort.getWorldOrigin().z+m_L),
	//    m_particle_U[0]->getDevicePtr(),
	//	m_particle_U[1]->getDevicePtr(),
	//	m_particle_U[2]->getDevicePtr(),
	//	m_M_eval);

	return true;
}

bool
BiotSavartSolver::m_unsortResult()
{
	if(!m_isVIC){

		for (int c=0;c<NUM_COMPONENTS;c++)
		{
			m_particle_U_deorder[c]->memset(0);
			//m_particle_U[c]->copy(GpuArrayd::DEVICE_TO_HOST);
			m_SpatialHasher_eval.deorderData(m_M_eval,m_particle_U[c]->getDevicePtr(),m_particle_U_deorder[c]->getDevicePtr(),1,2);
			m_particle_U_deorder[c]->copy(GpuArrayd::DEVICE_TO_HOST);
		}
	}


	return true;
}

void
BiotSavartSolver::sortVort()
{
	m_p_vortPos->copy(gf_GpuArray<float4>::DEVICE_TO_HOST);
	setDomain(m_origin, m_p_vortPos->getHostPtr(),m_N_vort,m_L);
	m_SpatialHasher_vort.setSpatialHashGrid(m_gridx, m_L/(double)m_gridx,
		make_float3(m_origin.x,m_origin.y,m_origin.z),
		m_N_vort);
	m_SpatialHasher_vort.setHashParam();
	m_SpatialHasher_vort.doSpatialHash(m_p_vortPos->getDevicePtr(),m_N_vort);

	m_p_vortPos_Reorder->memset(make_float4(0,0,0,0));
	m_SpatialHasher_vort.reorderData(m_N_vort,(void*)(m_p_vortPos->getDevicePtr()),(void*)(m_p_vortPos_Reorder->getDevicePtr()),4,1);
	for(int i=0;i<NUM_COMPONENTS;i++)
	{
		m_particle_vort_Reorder[i]->memset(0);
		m_SpatialHasher_vort.reorderData(m_N_vort, (void*)(m_particle_vort[i]->getDevicePtr()),
			(void*)(m_particle_vort_Reorder[i]->getDevicePtr()), 1, 2);

	}
}

void
BiotSavartSolver::computeFarFieldBuffer()
{
	//in order to do this, we need 
	//grid based velocity,
	//sorted vortex and their start end table
	//a double3 5x5x5xDXxDYxDZ local velocity field 
	m_p_vortPos->copy(gf_GpuArray<float4>::DEVICE_TO_HOST);
	setDomain(m_origin, m_p_vortPos->getHostPtr(),m_N_vort,m_L);
	//printf("%f,%f,%f,%f,\n",m_origin.x,m_origin.y,m_origin.z,m_L);
	if(!m_isVIC){

		m_SpatialHasher_eval.setSpatialHashGrid(m_gridx, m_L/(double)m_gridx,
			make_float3(m_origin.x,m_origin.y,m_origin.z),
			m_M_eval);
		m_SpatialHasher_eval.setHashParam();
		m_SpatialHasher_eval.doSpatialHash(m_evalPos->getDevicePtr(),m_M_eval);
		m_SpatialHasher_eval.reorderData(m_M_eval,m_evalPos->getDevicePtr(),m_evalPos_Reorder->getDevicePtr(),4,1);
	}


	m_ParticleToMesh();
	m_SolvePoisson();
	m_ComputeCurl();
	m_grid_U[0]->copy(m_grid_U[0]->DEVICE_TO_HOST);
	m_grid_U[1]->copy(m_grid_U[1]->DEVICE_TO_HOST);
	m_grid_U[2]->copy(m_grid_U[2]->DEVICE_TO_HOST);
	
	if(!m_isVIC){
		for(int i=0; i<3; i++)
		{
			cudaMemcpy(m_far_U[i]->getDevicePtr(),
				m_grid_U[i]->getDevicePtr(),
				sizeof(double)*m_grid_U[i]->getSize(),
				cudaMemcpyDeviceToDevice);
		}
		BiotSavartComputeFarField(m_SpatialHasher_eval.getStartTable(),m_SpatialHasher_eval.getEndTable(),
			m_evalPos->getDevicePtr(),
			m_grid_vort[0]->getDevicePtr(),
			m_grid_vort[1]->getDevicePtr(),
			m_grid_vort[2]->getDevicePtr(),
			m_grid_Psi[0]->getDevicePtr(),
			m_grid_Psi[1]->getDevicePtr(),
			m_grid_Psi[2]->getDevicePtr(),
			m_grid_U[0]->getDevicePtr(),
			m_grid_U[1]->getDevicePtr(),
			m_grid_U[2]->getDevicePtr(),
			m_far_U[0]->getDevicePtr(),m_far_U[1]->getDevicePtr(),m_far_U[2]->getDevicePtr(),
			m_SpatialHasher_vort.getCellSize().x,
			make_uint3(m_gridx,m_gridy,m_gridz),
			make_uint3(m_gridx,m_gridy,m_gridz),
			m_K,
			m_M_eval,
			m_origin);

	}
	m_grid_U[0]->copy(m_grid_U[0]->HOST_TO_DEVICE);
	m_grid_U[1]->copy(m_grid_U[1]->HOST_TO_DEVICE);
	m_grid_U[2]->copy(m_grid_U[2]->HOST_TO_DEVICE);


}

void BiotSavartSolver::unsortVort()
{
	for (int c=0;c<NUM_COMPONENTS;c++)
	{
		m_particle_vort[c]->memset(0);
		//m_particle_U[c]->copy(GpuArrayd::DEVICE_TO_HOST);
		m_SpatialHasher_vort.deorderData(m_N_vort,m_particle_vort_Reorder[c]->getDevicePtr(),m_particle_vort[c]->getDevicePtr(),1,2);
		//m_particle_vort[c]->copy(GpuArrayd::DEVICE_TO_HOST);
	}
}
