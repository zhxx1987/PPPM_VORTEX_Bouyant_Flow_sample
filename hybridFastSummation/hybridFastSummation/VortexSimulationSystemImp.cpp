#include "VortexSimulationSystem.h"
double frand(double a, double b)
{
	return ((double)(rand()%RAND_MAX))/((double)RAND_MAX)*(b-a) + a;
}
VortexSimulationSystem::VortexSimulationSystem()
{
	g_vortexSolver= new BiotSavartSolver;
	g_potentialFlowSolver = new PotentialFieldSolver;
	g_spacehash = new ParticleManager;
	u_num_boundary_faces = 0;
	u_num_vortices = 0;
	u_num_eval_pos = 0;
	u_dimx_pppm= 0;u_dimy_pppm= 0;u_dimz_pppm= 0;
	u_dimx_brdy= 0;u_dimy_brdy= 0;u_dimz_brdy= 0;
	u_num_per_batch = 163840;
	u_num_shed= 0;
	u_curr_frame= 0;
	i_LCrange= 0;
	d_dt= 0;
	d_nu= 0;
	d_h= 0.1;
	d_cx= 0;
	d_cy= 0;;
	d_cz= 0;;
	d_r= 0.05;
	d_density_domain_length=0;
	d4_density_origin=make_double4(0,0,0,0);
	d_vort_size=0;
	d_beta=0;	
	d_u_solid= 0;;
	d_v_solid= 0;;
	d_w_solid= 0;
	d_shed_coef= 0;
	d_shed_difu= 0;
	b_use_boundary=false;
	b_use_vic=false;
	b_use_shedding=false;
	b_has_tracer=false;
	b_has_heater=false;
	b_has_vortex_tube=false;
	b_autoshed=false;

	
	gf4_vort_enda = new GpuArrayf4;
	gf4_vort_endb = new GpuArrayf4;
	gf4_vort_cent = new GpuArrayf4;
	gf4_shedding_pos = new GpuArrayf4;
	gd4_ua_rotational = new GpuArrayd4;
	gd4_ub_rotational = new GpuArrayd4;
	gd4_uc_rotational = new GpuArrayd4;
	gd4_ua_potential = new GpuArrayd4;
	gd4_ub_potential = new GpuArrayd4;
	gd4_uc_potential = new GpuArrayd4;
	gd_kappa= new GpuArrayd;	
	gf4_boundary_pos = new GpuArrayf4;
	gd3_boundary_normal= new GpuArrayd3;
	gd_boundary_area = new GpuArrayd;
	gd_boundary_mass = new GpuArrayd;
	gd_boundary_b = new GpuArrayd;
	gd_boundary_u = new GpuArrayd;
	gd_boundary_v = new GpuArrayd;
	gd_boundary_w = new GpuArrayd;
	gd_boundary_vortx = new GpuArrayd;
	gd_boundary_vorty = new GpuArrayd;
	gd_boundary_vortz = new GpuArrayd;
	gf_vortex_life = new GpuArrayf;
	gf4_tracer_pos = new GpuArrayf4;
	gd4_tracer_vel = new GpuArrayd4;
	gf_tracer_life = new GpuArrayf;

	for (uint i=0; i<3;i++)
	{
		gd_vort[i]= new GpuArrayd ;
		gd_dvort[i]= new GpuArrayd ;
		gd_shed_vort[i]= new GpuArrayd ;

	}

	cpu_tracer_pos=0;
	cpu_tracer_life=0;
	cpu_tracer_vel=0;

	cpu_heat_pos=0;
	cpu_heat_life=0;
	cpu_heat_vel=0;
	

}
void VortexSimulationSystem::solveTimeStep(double dt)
{
	preTimestep();
	d_dt = dt;
	addBaroclinicForce();
	//construct vortex segments from vortex particles
	convertVortToSeg();


	//step1: compute u_rotational
	computeURotation();

	//step2 : compute u_potential
	computeUPotential();

	
	//step3 : apply the flow to all elements
	applyFlow();


	//step4 : convert vortex segments back to vortex blobs
	convertSegToVort();

	//step5 : compute vortex stretching
	computeStretching();

	//step6 : if auto shed is on, 
	//and there are boundary, do vortex shedding
	computeShedding();
	

	u_curr_frame++;



}

void VortexSimulationSystem::convertVortToSeg()
{
	////compute endb based on vortex strength
	
	
	ComputePosb(gf4_vort_cent->getDevicePtr(),
		gf4_vort_enda->getDevicePtr(), 
		gf4_vort_endb->getDevicePtr(),
		gd_vort[0]->getDevicePtr(),
		gd_vort[1]->getDevicePtr(),
		gd_vort[2]->getDevicePtr(),
		gd_kappa->getDevicePtr(),
		d_h,
		u_num_vortices);
}
void VortexSimulationSystem::computeURotation()
{
	g_vortexSolver->initializeSolver(u_dimx_pppm,
		u_dimy_pppm,
		u_dimz_pppm,
		b_use_vic,
		i_LCrange,
		u_num_vortices,
		u_num_vortices);

	g_vortexSolver->setVortParameter(u_num_vortices,
		gf4_vort_cent ,
		gd_vort);

	g_vortexSolver->computeFarFieldBuffer();

	if(b_use_boundary){
		g_vortexSolver->setEvalParameter(u_num_boundary_faces,
			gf4_boundary_pos );

		g_vortexSolver->evaluateVelocity(gf4_boundary_pos,0);
		gd_boundary_b->memset(0);
		ComputeNormalSlip(g_vortexSolver->getU(0)->getDevicePtr(),
			g_vortexSolver->getU(1)->getDevicePtr(),
			g_vortexSolver->getU(2)->getDevicePtr(),
			d_u_solid,
			d_v_solid,
			d_w_solid,
			gd3_boundary_normal->getDevicePtr(),
			gd_boundary_b->getDevicePtr(),
			u_num_boundary_faces);

		cudaMemcpy(gd_boundary_u->getDevicePtr(),
			g_vortexSolver->getU(0)->getDevicePtr(),
			sizeof(double)*u_num_boundary_faces, 
			cudaMemcpyDeviceToDevice);

		cudaMemcpy(gd_boundary_v->getDevicePtr(),
			g_vortexSolver->getU(1)->getDevicePtr(),
			sizeof(double)*u_num_boundary_faces, 
			cudaMemcpyDeviceToDevice);

		cudaMemcpy(gd_boundary_w->getDevicePtr(),
			g_vortexSolver->getU(2)->getDevicePtr(),
			sizeof(double)*u_num_boundary_faces, 
			cudaMemcpyDeviceToDevice);

		ComputeBoundarySlip(gd_boundary_u->getDevicePtr(),
			gd_boundary_v->getDevicePtr(),
			gd_boundary_w->getDevicePtr(),
			d_u_solid,
			d_v_solid,
			d_w_solid,
			u_num_boundary_faces);
	}

	g_vortexSolver->setEvalParameter(u_num_vortices,
		gf4_vort_endb);

	g_vortexSolver->evaluateVelocity(gf4_vort_cent, 1);
	gd4_ub_rotational->memset(make_double4(0,0,0,0));
	AddURotation(gd4_ub_rotational->getDevicePtr(),
		g_vortexSolver->getU(0)->getDevicePtr(),
		g_vortexSolver->getU(1)->getDevicePtr(),
		g_vortexSolver->getU(2)->getDevicePtr(),
		u_num_vortices);


	g_vortexSolver->setEvalParameter(u_num_vortices, 
		gf4_vort_enda);
	g_vortexSolver->evaluateVelocity(gf4_vort_cent, 1);
	gd4_ua_rotational->memset(make_double4(0,0,0,0));
	AddURotation(gd4_ua_rotational->getDevicePtr(),
		g_vortexSolver->getU(0)->getDevicePtr(),
		g_vortexSolver->getU(1)->getDevicePtr(),
		g_vortexSolver->getU(2)->getDevicePtr(),
		u_num_vortices);
	if(b_has_tracer){

		evaluateBiotSavart(cpu_tracer_pos,cpu_tracer_vel,u_num_eval_pos);
	}
	if(b_has_heater){

		evaluateBiotSavart(cpu_heat_pos, cpu_heat_vel, u_num_heat_pos);
	}
	g_vortexSolver->shutdown();
}
void VortexSimulationSystem::computeUPotential()
{
	if(b_use_boundary){

		BiCGSTAB(gd_boundary_mass,
			gd_boundary_b,
			u_num_boundary_faces,
			2);
		//m_boundary_mass->memset(0);

		g_potentialFlowSolver->initializeSolver(u_dimx_brdy,
			u_dimy_brdy, 
			u_dimz_brdy,
			i_LCrange, 
			u_num_vortices,
			u_num_boundary_faces);

		ComputeSingleLayerPotentialMass(gd_boundary_mass->getDevicePtr(),
			gd_boundary_area->getDevicePtr(),
			gd_boundary_mass->getDevicePtr(),
			u_num_boundary_faces);

		g_potentialFlowSolver->setMassParameter(u_num_boundary_faces,
			gf4_boundary_pos,
			gd_boundary_mass);

		g_potentialFlowSolver->computeFarFieldBuffer();

		g_potentialFlowSolver->setEvalParameter(u_num_vortices,
			gf4_vort_cent);

		g_potentialFlowSolver->evaluateGradient(true);
		gd4_uc_potential->memset(make_double4(0,0,0,0));
		AddUPotential(gd4_uc_potential->getDevicePtr(), 
			g_potentialFlowSolver->getGradPhi()->getDevicePtr(),
			u_num_vortices);

		if(b_has_tracer){

			evaluatePotential(cpu_tracer_pos,cpu_tracer_vel,u_num_eval_pos);
		}
		if(b_has_heater){

			evaluatePotential(cpu_heat_pos, cpu_heat_vel, u_num_heat_pos);
		}
		g_potentialFlowSolver->shutdown();
	}
}
void VortexSimulationSystem::applyFlow()
{
	move_particle(gf4_vort_enda->getDevicePtr(),
		d_dt, 
		gd4_ua_rotational->getDevicePtr(),
		u_num_vortices);

	move_particle(gf4_vort_endb->getDevicePtr(),
		d_dt, 
		gd4_ub_rotational->getDevicePtr(),
		u_num_vortices);




	ComputePosa(gf4_vort_enda->getDevicePtr(), 
		gf4_vort_endb->getDevicePtr(), 
		gf4_vort_cent->getDevicePtr(),
		u_num_vortices);



	if(b_use_boundary){

		move_particle(gf4_vort_cent->getDevicePtr(),
			d_dt,
			gd4_uc_potential->getDevicePtr(),
			u_num_vortices);

		UpdateBoundary(gf4_boundary_pos->getDevicePtr(), 
			d_u_solid,
			d_v_solid,
			d_w_solid,
			d_dt,
			u_num_boundary_faces);

	}
}
void VortexSimulationSystem::convertSegToVort()
{
	for(uint i=0; i<3; i++){

		gd_dvort[i]->memset(0);
	}

	ComputeVortex(gf4_vort_enda->getDevicePtr(),
		gf4_vort_endb->getDevicePtr(),
		gd_vort[0]->getDevicePtr(),
		gd_vort[1]->getDevicePtr(),
		gd_vort[2]->getDevicePtr(),
		gd_dvort[0]->getDevicePtr(),
		gd_dvort[1]->getDevicePtr(),
		gd_dvort[2]->getDevicePtr(),
		gd_kappa->getDevicePtr(),
		u_num_vortices);
}

void
VortexSimulationSystem::
setDensityDomain(double4 & origin, float4 * pos, uint num_particle, double & domain_length)
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

	domain_length = 1.2*max(max(maxx-minx, maxy-miny),maxz-minz);
	double center_x = minx + 0.5*(maxx-minx);
	double center_y = miny + 0.5*(maxy-miny);
	double center_z = minz + 0.5*(maxz-minz);

	origin.x = center_x - 0.5*domain_length;
	origin.y = center_y - 0.5*domain_length;
	origin.z = center_z - 0.5*domain_length;


}


void VortexSimulationSystem::addBaroclinicForce()
{
	//compute the Baroclinic Vorticity generation 
	//based on heat particles

	setDensityDomain(d4_density_origin, cpu_heat_pos, u_num_heat_pos, d_density_domain_length);

	//allocate a hi-res memory for rho and grad(rho)

	double h = 0.2;
	uint density_dimx = floor(d_density_domain_length/h)+1;
	density_dimx = min(max((uint)0,density_dimx), (uint)200);
	h = d_density_domain_length/(double)density_dimx;
	printf("density dimx : %d\n", density_dimx);
	
	GpuArrayf4 *gf4_heat_pos         = new GpuArrayf4;
	GpuArrayf4 *gf4_heat_pos_reorder = new GpuArrayf4;
	GpuArrayf  *gf_heat              = new GpuArrayf;
	GpuArrayf  *gf_heat_reorder      = new GpuArrayf;
	GpuArrayf  *gf_rho               = new GpuArrayf;
	GpuArrayf3 *gf3_gradrho          = new GpuArrayf3;
	gf_rho->alloc(density_dimx*density_dimx*density_dimx);
	gf_rho->memset(0);
	gf3_gradrho->alloc(density_dimx*density_dimx*density_dimx);
	gf3_gradrho->memset(make_float3(0,0,0));
	gf4_heat_pos->alloc(u_num_heat_pos);
	gf4_heat_pos->memset(make_float4(0,0,0,0));
	gf4_heat_pos_reorder->alloc(u_num_heat_pos);
	gf4_heat_pos_reorder->memset(make_float4(0,0,0,0));
	gf_heat->alloc(u_num_heat_pos);
	gf_heat_reorder->alloc(u_num_heat_pos);

	cudaMemcpy(gf4_heat_pos->getDevicePtr(),cpu_heat_pos,sizeof(float4)*u_num_heat_pos,
		cudaMemcpyHostToDevice);
	cudaMemcpy(gf_heat->getDevicePtr(),cpu_heat_life,sizeof(float)*u_num_heat_pos,
		cudaMemcpyHostToDevice);


	g_spacehash->initSpatialHash(u_num_heat_pos,density_dimx,density_dimx,density_dimx);
	g_spacehash->setSpatialHashGrid(density_dimx, d_density_domain_length/(double)density_dimx, make_float3(d4_density_origin.x,d4_density_origin.y,d4_density_origin.z),u_num_heat_pos);
	g_spacehash->setHashParam();
	g_spacehash->doSpatialHash(gf4_heat_pos->getDevicePtr(),u_num_heat_pos);

	g_spacehash->reorderData(u_num_heat_pos,gf4_heat_pos->getDevicePtr(),gf4_heat_pos_reorder->getDevicePtr(),4,1);
	g_spacehash->reorderData(u_num_heat_pos,gf_heat->getDevicePtr(),gf_heat_reorder->getDevicePtr(),1,1);


	AssignDensity(g_spacehash->getStartTable(),
		g_spacehash->getEndTable(),
		gf4_heat_pos_reorder->getDevicePtr(),
		gf_heat_reorder->getDevicePtr(),
		gf_heat_reorder->getDevicePtr(),
		h,
		gf_rho->getDevicePtr(),
		make_uint3(density_dimx,density_dimx,density_dimx),
		make_uint3(density_dimx,density_dimx,density_dimx),
		u_num_heat_pos,d4_density_origin);


	//compute grad(rho)
	GradDensity(gf_rho->getDevicePtr(),
		gf3_gradrho->getDevicePtr(),
		h,
		density_dimx,density_dimx,density_dimx);


	//apply baroclinic force.
	ParticleGetBaroclinic(gf4_vort_cent->getDevicePtr(),
		gd_vort[0]->getDevicePtr(),
		gd_vort[1]->getDevicePtr(),
		gd_vort[2]->getDevicePtr(),
		gf3_gradrho->getDevicePtr(),
		d_dt*5.0, 
		u_num_vortices, 
		density_dimx,density_dimx,density_dimx,
		1.0/g_spacehash->getCellSize().x,
		d4_density_origin);


	g_spacehash->endSpatialHash();


	gf4_heat_pos->free();      
	gf4_heat_pos_reorder->free();
	gf_heat->free();             
	gf_heat_reorder->free();     
	gf_rho->free();              
	gf3_gradrho->free();         
	






	
}
void VortexSimulationSystem::computeStretching()
{
	//compute a smooth omega dot grad(u) 
	for(uint i=0; i<3;i++)
	{

		VectorAdd(gd_vort[i]->getDevicePtr(),
			gd_dvort[i]->getDevicePtr(),
			gd_dvort[i]->getDevicePtr(),
			1,1,u_num_vortices);
	}
	g_vortexSolver->initializeSolver(u_dimx_pppm,
		u_dimy_pppm,
		u_dimz_pppm,
		b_use_vic,
		i_LCrange,
		u_num_vortices,
		u_num_vortices);

	g_vortexSolver->setEvalParameter(u_num_vortices,
		gf4_vort_cent );

	g_vortexSolver->setVortParameter(u_num_vortices,
		gf4_vort_cent ,
		gd_dvort);
	g_vortexSolver->sortVort();

	ParticleGaussianSmooth(g_vortexSolver->getVortStartTable(),
		g_vortexSolver->getVortEndTable(),
		gf4_vort_cent->getDevicePtr(),
		g_vortexSolver->getSortPos()->getDevicePtr(),
		0.1,
		d_dt,
		gd_dvort[0]->getDevicePtr(),
		gd_dvort[1]->getDevicePtr(),
		gd_dvort[2]->getDevicePtr(),
		g_vortexSolver->getSortVort(0)->getDevicePtr(),
		g_vortexSolver->getSortVort(1)->getDevicePtr(),
		g_vortexSolver->getSortVort(2)->getDevicePtr(),
		g_vortexSolver->getCellh(),
		make_uint3(u_dimx_pppm,u_dimx_pppm,u_dimx_pppm),
		u_num_vortices,
		g_vortexSolver->getOrigin());


	for(uint i=0; i<3;i++)
	{

		VectorAdd(gd_vort[i]->getDevicePtr(),
			gd_dvort[i]->getDevicePtr(),
			gd_vort[i]->getDevicePtr(),
			0,1,u_num_vortices);
	}





	g_vortexSolver->shutdown();
}
void VortexSimulationSystem::computeShedding()
{
	if(b_use_boundary&&b_autoshed){


		//compute vortex shedding
		GpuArrayd *vort[3];
		vort[0] = gd_boundary_vortx;
		vort[1] = gd_boundary_vorty;
		vort[2] = gd_boundary_vortz;
		ComputeBoundaryVortex(gd_boundary_u->getDevicePtr(),
			gd_boundary_v->getDevicePtr(),
			gd_boundary_w->getDevicePtr(),
			gd_boundary_vortx->getDevicePtr(),
			gd_boundary_vorty->getDevicePtr(),
			gd_boundary_vortz->getDevicePtr(),
			gf4_boundary_pos->getDevicePtr(),
			gd3_boundary_normal->getDevicePtr(),
			gd_boundary_area->getDevicePtr(),
			d_shed_coef,
			u_num_boundary_faces);

		u_num_shed = u_num_boundary_faces;
		for (int i=0;i<3;i++)
		{
			gd_shed_vort[i]->memset(0);
			cudaMemcpy(gd_shed_vort[i]->getDevicePtr(), 
				vort[i]->getDevicePtr(),
				sizeof(double)*u_num_boundary_faces, 
				cudaMemcpyDeviceToDevice);
		}

		ComputeSheddingPos(gf4_shedding_pos->getDevicePtr(), 
			gf4_boundary_pos->getDevicePtr(),
			gd3_boundary_normal->getDevicePtr(),
			u_num_boundary_faces);

		for (int v=0;v<3;v++)
		{
			gd_dvort[v]->memset(0);
		}

		g_vortexSolver->initializeSolver(u_dimx_pppm, 
			u_dimy_pppm, 
			u_dimz_pppm,
			b_use_vic,
			i_LCrange,
			u_num_vortices,
			u_num_vortices);

		g_vortexSolver->setEvalParameter(u_num_vortices,
			gf4_vort_cent );

		g_vortexSolver->setVortParameter(u_num_vortices,
			gf4_vort_cent ,
			gd_dvort);
		g_vortexSolver->sortVort();




		VortexShedding(g_vortexSolver->getVortStartTable(),
			g_vortexSolver->getVortEndTable(),
			g_vortexSolver->getSortPos()->getDevicePtr(),
			g_vortexSolver->getSortVort(0)->getDevicePtr(),
			g_vortexSolver->getSortVort(1)->getDevicePtr(),
			g_vortexSolver->getSortVort(2)->getDevicePtr(),
			gf4_boundary_pos->getDevicePtr(),
			vort[0]->getDevicePtr(),
			vort[1]->getDevicePtr(),
			vort[2]->getDevicePtr(),
			0.05,//shall be kept small
			d_shed_difu, 
			u_dimx_pppm, 
			u_dimy_pppm, 
			u_dimz_pppm, 
			1.0/g_vortexSolver->getCellh(),
			g_vortexSolver->getOrigin(),
			u_num_boundary_faces,
			u_num_vortices);

		ParticleGaussianSmooth(g_vortexSolver->getVortStartTable(),
			g_vortexSolver->getVortEndTable(),
			gf4_vort_cent->getDevicePtr(),
			g_vortexSolver->getSortPos()->getDevicePtr(),
			0.25,
			d_dt,
			gd_dvort[0]->getDevicePtr(),
			gd_dvort[1]->getDevicePtr(),
			gd_dvort[2]->getDevicePtr(),
			g_vortexSolver->getSortVort(0)->getDevicePtr(),
			g_vortexSolver->getSortVort(1)->getDevicePtr(),
			g_vortexSolver->getSortVort(2)->getDevicePtr(),
			g_vortexSolver->getCellh(),
			make_uint3(u_dimx_pppm, u_dimy_pppm, u_dimz_pppm),
			u_num_vortices,
			g_vortexSolver->getOrigin());

		for (uint i=0; i<3; i++)
		{
			VectorAdd(gd_vort[i]->getDevicePtr(),
				gd_vort[i]->getDevicePtr(),
				gd_vort[i]->getDevicePtr(),
				1,1,u_num_vortices);
		}


		g_vortexSolver->shutdown();
	}
}

void
VortexSimulationSystem::BiCGSTAB(GpuArrayd * x, GpuArrayd * b, uint num_elements, uint iter)
{
	GpuArrayd * r = new GpuArrayd;
	GpuArrayd * r_hat = new GpuArrayd;
	GpuArrayd * v = new GpuArrayd;
	GpuArrayd * p = new GpuArrayd;
	GpuArrayd * s = new GpuArrayd;
	GpuArrayd * t = new GpuArrayd;
	double rho = 1.0;
	double aph = 1.0;
	double omg = 1.0;
	double beta;

	x->memset(0);
	r->alloc(num_elements);
	r_hat->alloc(num_elements);
	v->alloc(num_elements);
	p->alloc(num_elements);
	s->alloc(num_elements);
	t->alloc(num_elements);


	cudaMemcpy(r->getDevicePtr(),b->getDevicePtr(),sizeof(double)*num_elements,cudaMemcpyDeviceToDevice);
	cudaMemcpy(r_hat->getDevicePtr(),r->getDevicePtr(),sizeof(double)*num_elements,cudaMemcpyDeviceToDevice);

	v->memset(0);
	p->memset(0);
	s->memset(0);
	t->memset(0);

	for (int i=0;i<iter;i++)
	{
		double rho_temp = ComputeDot(r_hat->getDevicePtr(),r->getDevicePtr(), num_elements);
		//printf("%f\n",rho_temp);
		beta = (rho_temp/rho)*(aph/omg);
		rho = rho_temp;
		if (fabs(rho)<1e-12)
		{
			break;
		}
		VectorAdd(r->getDevicePtr(),p->getDevicePtr(),p->getDevicePtr(), 1.0,beta,num_elements);
		VectorAdd(p->getDevicePtr(),v->getDevicePtr(),p->getDevicePtr(),1.0,-beta*omg,num_elements);

		computeAx(p,v,num_elements);
		aph = rho/ComputeDot(r_hat->getDevicePtr(),v->getDevicePtr(), num_elements);
		VectorAdd(r->getDevicePtr(),v->getDevicePtr(),s->getDevicePtr(), 1.0,-aph,num_elements);
		computeAx(s,t,num_elements);
		omg = ComputeDot(t->getDevicePtr(),s->getDevicePtr(),num_elements)/ComputeDot(t->getDevicePtr(),t->getDevicePtr(),num_elements);
		VectorAdd(x->getDevicePtr(),p->getDevicePtr(),x->getDevicePtr(), 1.0,aph,num_elements);
		VectorAdd(x->getDevicePtr(),s->getDevicePtr(),x->getDevicePtr(), 1.0,omg,num_elements);


		VectorAdd(s->getDevicePtr(),t->getDevicePtr(),r->getDevicePtr(), 1.0,-omg,num_elements);

	}

	r->free();
	r_hat->free();
	v->free();
	p->free();
	s->free();
	t->free();

}

void
VortexSimulationSystem::computeAx(GpuArrayd * rhs, GpuArrayd * res, uint num_faces)
{
	GpuArrayd * mass = new GpuArrayd;
	mass->alloc(num_faces);
	mass->memset(0);
	ComputeSingleLayerPotentialMass(rhs->getDevicePtr(),
		gd_boundary_area->getDevicePtr(),mass->getDevicePtr(),num_faces);
	g_potentialFlowSolver->initializeSolver(u_dimx_brdy,u_dimy_brdy,u_dimz_brdy,3,num_faces,num_faces);
	g_potentialFlowSolver->setMassParameter(num_faces,gf4_boundary_pos,mass);
	g_potentialFlowSolver->setEvalParameter(num_faces,gf4_boundary_pos);
	g_potentialFlowSolver->setEvalAreas(num_faces,gd_boundary_area);
	g_potentialFlowSolver->setEvalNormals(num_faces,gd3_boundary_normal);
	g_potentialFlowSolver->computeFarFieldBuffer();
	g_potentialFlowSolver->evaluateGradient(false);
	g_potentialFlowSolver->computeDphiDn();
	cudaMemcpy(res->getDevicePtr(),g_potentialFlowSolver->getDpDn()->getDevicePtr(),sizeof(double)*num_faces,cudaMemcpyDeviceToDevice);
	mass->free();
	g_potentialFlowSolver->shutdown();

}

void VortexSimulationSystem::freeComputation()
{
	gf4_vort_enda->free();
	gf4_vort_endb->free();
	gf4_vort_cent->free();
	gf4_shedding_pos->free();
	gd4_ua_rotational->free();
	gd4_ub_rotational->free();
	gd4_uc_rotational->free();
	gd4_ua_potential->free();
	gd4_ub_potential->free();
	gd4_uc_potential->free();
	gd_kappa->free();	
	gf4_boundary_pos->free();
	gd3_boundary_normal->free();
	gd_boundary_area->free();
	gd_boundary_mass->free();
	gd_boundary_b->free();
	gd_boundary_u->free();
	gd_boundary_v->free();
	gd_boundary_w->free();
	gd_boundary_vortx->free();
	gd_boundary_vorty->free();
	gd_boundary_vortz->free();
	gf_vortex_life->free();
	gf4_tracer_pos->free();
	gd4_tracer_vel->free();
	gf_tracer_life->free();

	if (cpu_tracer_pos)
	{
		free(cpu_tracer_pos);
	}
	if(cpu_tracer_life){

		free(cpu_tracer_life);
	}
	if(cpu_tracer_vel){
		free(cpu_tracer_vel);
	}

	if (cpu_heat_pos)
	{
		free(cpu_heat_pos);
	}
	if(cpu_heat_life){

		free(cpu_heat_life);
	}
	if(cpu_heat_vel){
		free(cpu_heat_vel);
	}

	for (uint i=0; i<3;i++)
	{
		gd_vort[i]->free();
		gd_dvort[i]->free();
		gd_shed_vort[i]->free();

	}


}
void VortexSimulationSystem::setupComputation()
{
	
	u_num_vortices = v_vortex_pos.size();
	u_num_eval_pos = v_tracer_pos.size();
	u_num_heat_pos = v_heat_pos.size();
	
	gf4_vort_enda->alloc(u_num_vortices);
	gf4_vort_endb->alloc(u_num_vortices);
	gf4_vort_cent->alloc(u_num_vortices);
	gd4_ua_rotational->alloc(u_num_vortices);
	gd4_ub_rotational->alloc(u_num_vortices);
	gd4_uc_rotational->alloc(u_num_vortices);
	gd4_ua_potential->alloc(u_num_vortices);
	gd4_ub_potential->alloc(u_num_vortices);
	gd4_uc_potential->alloc(u_num_vortices);
	gd_kappa->alloc(u_num_vortices);
	gf_vortex_life->alloc(u_num_vortices);
	for (uint i=0; i<3;i++)
	{
		gd_vort[i]->alloc(u_num_vortices);
		gd_dvort[i]->alloc(u_num_vortices);
		
	}

	if (b_use_boundary)
	{
		gf4_shedding_pos->alloc(u_num_boundary_faces);
		gf4_boundary_pos->alloc(u_num_boundary_faces);
		gd3_boundary_normal->alloc(u_num_boundary_faces);
		gd_boundary_area->alloc(u_num_boundary_faces);
		gd_boundary_mass->alloc(u_num_boundary_faces);
		gd_boundary_b->alloc(u_num_boundary_faces);
		gd_boundary_u->alloc(u_num_boundary_faces);
		gd_boundary_v->alloc(u_num_boundary_faces);
		gd_boundary_w->alloc(u_num_boundary_faces);
		gd_boundary_vortx->alloc(u_num_boundary_faces);
		gd_boundary_vorty->alloc(u_num_boundary_faces);
		gd_boundary_vortz->alloc(u_num_boundary_faces);
		for (uint i=0; i<3; i++)
		{
			gd_shed_vort[i]->alloc(u_num_boundary_faces);
		}
		
	}

	gf4_tracer_pos->alloc(u_num_per_batch);
	gd4_tracer_vel->alloc(u_num_per_batch);
	gf_tracer_life->alloc(u_num_per_batch);


	if(u_num_eval_pos>0){

		cpu_tracer_pos = (float4*)malloc(sizeof(float4)*u_num_eval_pos);
		cpu_tracer_life = (float*)malloc(sizeof(float)*u_num_eval_pos);
		cpu_tracer_vel = (double4*)malloc(sizeof(double4)*u_num_eval_pos);
	}
	if(u_num_heat_pos>0){

		cpu_heat_pos = (float4*)malloc(sizeof(float4)*u_num_heat_pos);
		cpu_heat_life = (float*)malloc(sizeof(float)*u_num_heat_pos);
		cpu_heat_vel = (double4*)malloc(sizeof(double4)*u_num_heat_pos);
	}

	
	for (uint i=0; i<u_num_vortices; i++)
	{
		gf4_vort_cent->getHostPtr()[i].x=v_vortex_pos[i].v[0];
		gf4_vort_cent->getHostPtr()[i].y=v_vortex_pos[i].v[1];
		gf4_vort_cent->getHostPtr()[i].z=v_vortex_pos[i].v[2];
		gf4_vort_cent->getHostPtr()[i].w=0;
		gf_vortex_life->getHostPtr()[i] = v_vortex_life[i];
		gd_vort[0]->getHostPtr()[i] = v_vortex_vortx[i];
		gd_vort[1]->getHostPtr()[i] = v_vortex_vorty[i];
		gd_vort[2]->getHostPtr()[i] = v_vortex_vortz[i];

	}
	gf4_vort_cent->copy(gf4_vort_cent->HOST_TO_DEVICE);
	gf_vortex_life->copy(gf_vortex_life->HOST_TO_DEVICE);
	gd_vort[0]->copy(gd_vort[0]->HOST_TO_DEVICE);
	gd_vort[1]->copy(gd_vort[1]->HOST_TO_DEVICE);
	gd_vort[2]->copy(gd_vort[2]->HOST_TO_DEVICE);
	for (uint i=0; i<u_num_eval_pos; i++)
	{
		cpu_tracer_pos[i].x = v_tracer_pos[i].v[0];
		cpu_tracer_pos[i].y = v_tracer_pos[i].v[1];
		cpu_tracer_pos[i].z = v_tracer_pos[i].v[2];
		cpu_tracer_pos[i].w = 0;
		cpu_tracer_life[i] = v_tracer_life[i];
		cpu_tracer_vel[i] = make_double4(0,0,0,0);
	}

	for (uint i=0; i<u_num_heat_pos; i++)
	{
		cpu_heat_pos[i].x = v_heat_pos[i].v[0];
		cpu_heat_pos[i].y = v_heat_pos[i].v[1];
		cpu_heat_pos[i].z = v_heat_pos[i].v[2];
		cpu_heat_pos[i].w = 0;
		cpu_heat_life[i] = v_heat_life[i];
		cpu_heat_vel[i] = make_double4(0,0,0,0);
	}

	if(u_num_vortices<16384)
	{
		u_dimx_pppm=u_dimy_pppm=u_dimz_pppm=64;
		i_LCrange=1;

	}
	else if(u_num_vortices<16384*8)
	{
		u_dimx_pppm=u_dimy_pppm=u_dimz_pppm=128;
		i_LCrange=1;
	}
	else
	{
		u_dimx_pppm=u_dimy_pppm=u_dimz_pppm=128;
		i_LCrange=2;
	}
	u_dimx_brdy = u_dimy_brdy = u_dimz_brdy = 64;


}
void VortexSimulationSystem::deleteHeat()
{
	v_heat_pos.resize(0);
	v_heat_life.resize(0);
	v_heat_vel.resize(0);

	for (uint i=0; i<u_num_heat_pos; i++)
	{
		if(cpu_heat_life[i]>0.01)
		{
			v_heat_life.push_back(cpu_heat_life[i]/(1+1.0*d_dt));
			Vec4d vel = Vec4d(cpu_heat_vel[i].x,cpu_heat_vel[i].y,cpu_heat_vel[i].z,0);
			v_heat_vel.push_back(vel);
			Vec4f x = Vec4f(cpu_heat_pos[i].x,cpu_heat_pos[i].y, cpu_heat_pos[i].z,0);
			v_heat_pos.push_back(x);
		}
	}
}
void VortexSimulationSystem::deleteVortex()
{
	v_vortex_pos.resize(0);
	v_vortex_vortx.resize(0);
	v_vortex_vorty.resize(0);
	v_vortex_vortz.resize(0);
	v_vortex_life.resize(0);
	//copy data from GPU to CPU
	gf4_vort_cent->copy(gf4_vort_cent->DEVICE_TO_HOST);
	gf_vortex_life->copy(gf_vortex_life->DEVICE_TO_HOST);
	for (uint i=0; i<3; i++)
	{
		gd_vort[i]->copy(gd_vort[i]->DEVICE_TO_HOST);
	}
	float *life = gf_vortex_life->getHostPtr();
	float4 *pos = gf4_vort_cent->getHostPtr();
	double* vort[3];
	for (uint i=0; i<3; i++)
	{
		vort[i]=gd_vort[i]->getHostPtr() ;
	}

	for (uint p=0; p<u_num_vortices; p++)
	{
		if ( life[p]>0 
			|| fabs(vort[0][p])>0.0001
			|| fabs(vort[1][p])>0.0001
			|| fabs(vort[2][p])>0.0001)
		{
			Vec4f x = Vec4f(pos[p].x,pos[p].y, pos[p].z,0);
			v_vortex_pos.push_back(x);
			v_vortex_vortx.push_back(vort[0][p]);
			v_vortex_vorty.push_back(vort[1][p]);
			v_vortex_vortz.push_back(vort[2][p]);
			v_vortex_life.push_back(life[p]);

		}

	}



}
void VortexSimulationSystem::deleteTracer()
{
	v_tracer_pos.resize(0);
	v_tracer_life.resize(0);
	v_tracer_vel.resize(0);

	for (uint i=0; i<u_num_eval_pos; i++)
	{
		if(cpu_tracer_life[i]>0)
		{
			v_tracer_life.push_back(cpu_tracer_life[i]);
			Vec4d vel = Vec4d(cpu_tracer_vel[i].x,cpu_tracer_vel[i].y,cpu_tracer_vel[i].z,0);
			v_tracer_vel.push_back(vel);
			Vec4f x = Vec4f(cpu_tracer_pos[i].x,cpu_tracer_pos[i].y, cpu_tracer_pos[i].z,0);
			v_tracer_pos.push_back(x);
		}
	}
}
void VortexSimulationSystem::emitTracer()
{
	for (uint i=0; i< v_tracerSource.size();i++)
	{
		uint num = 0;
		Vec4f center = Vec4f(v_tracerSource[i].cx,
			                 v_tracerSource[i].cy,
							 v_tracerSource[i].cz,
							 0);
		double r = v_tracerSource[i].r;
		while (num<v_tracerSource[i].N)
		{
			Vec4f pos = Vec4f(frand(-r,r),
							  frand(-r,r),
							  frand(-r,r),
							  0);
			if(mag(pos)<r)
			{
				v_tracer_pos.push_back(pos+center);
				v_tracer_life.push_back(v_tracerSource[i].str);
				num++;
			}
			
		}
		
	}
	v_tracer_vel.resize(v_tracer_life.size());

}



void VortexSimulationSystem::emitHeat()
{
	for (uint i=0; i< v_heatSource.size();i++)
	{
		uint num = 0;
		Vec4f center = Vec4f(v_heatSource[i].cx,
			v_heatSource[i].cy,
			v_heatSource[i].cz,
			0);
		double r = v_heatSource[i].r;
		while (num<v_heatSource[i].N)
		{
			Vec4f pos = Vec4f(frand(-r,r),
				frand(-r,r),
				frand(-r,r),
				0);
			if(mag(pos)<r)
			{
				v_heat_pos.push_back(pos+center);
				v_heat_life.push_back(v_heatSource[i].str);
				num++;
			}

		}

	}
	v_heat_vel.resize(v_heat_life.size());

}


void VortexSimulationSystem::emitVortexRing()
{
	for (uint i=0; i< v_vortexTube.size();i++)
	{
		if((u_curr_frame % v_vortexTube[i].rate)==0)
		{

			Vec3f dir = Vec3f(v_vortexTube[i].dx,
				v_vortexTube[i].dy,
				v_vortexTube[i].dz);
			normalize(dir);
			Vec3f world_up = Vec3f(0,1,0);
			float l = dot(dir, world_up);
			Vec3f disc_x = world_up - l*dir;
			if (mag(disc_x)<0.001)
			{
				disc_x = Vec3f(1,0,0);
			}
			Vec3f disc_y = cross(dir, disc_x);
			normalize(disc_x);
			normalize(disc_y);
			double dtheta = 2*M_PI/((double)v_vortexTube[i].N);
			for (uint p=0; p<v_vortexTube[i].N; p++)
			{
				double theta = dtheta*((double)p);
				float c1 = v_vortexTube[i].r*cos(theta), c2 = v_vortexTube[i].r*sin(theta);
				Vec3f pos = c1*disc_x + c2*disc_y + Vec3f(v_vortexTube[i].cx,v_vortexTube[i].cy,v_vortexTube[i].cz);
				v_vortex_pos.push_back(Vec4f(pos.v[0], pos.v[1], pos.v[2],0));
				v_vortex_life.push_back(v_vortexTube[i].life);
				Vec3f tan = c1*disc_y - c2*disc_x;
				normalize(tan);
				tan = (float)(v_vortexTube[i].str) * tan;
				v_vortex_vortx.push_back(tan.v[0]);
				v_vortex_vorty.push_back(tan.v[1]);
				v_vortex_vortz.push_back(tan.v[2]);


			}
		}
	}
}
void VortexSimulationSystem::setVortexRingEmitter(uint		pN,
												  uint		prate,
												  double	plife,
												  double	pcx,
												  double	pcy,
												  double	pcz,
												  double	pdx,
												  double	pdy,
												  double	pdz,
												  double	pstr,
												  double r)
{
	
	b_has_vortex_tube = true;
	v_vortexTube.push_back(VortexRing(pN,
										prate,
										plife,
										pcx,
										pcy,
										pcz,
										pdx,
										pdy,
										pdz,
										pstr,
										r));
}


void 
VortexSimulationSystem::setTracerEmitter(uint	pN,
					  uint	prate,
					  double	plife,
					  double	pcx,
					  double	pcy,
					  double	pcz,
					  double	pdx,
					  double	pdy,
					  double	pdz,
					  double	pstr,
					  double  r)
{
	b_has_tracer = true;
	v_tracerSource.push_back(VortexRing(pN,
		prate,
		plife,
		pcx,
		pcy,
		pcz,
		pdx,
		pdy,
		pdz,
		pstr,
		r));
}
void 
VortexSimulationSystem::setHeatEmitter(uint	pN,
										 uint	prate,
										 double	plife,
										 double	pcx,
										 double	pcy,
										 double	pcz,
										 double	pdx,
										 double	pdy,
										 double	pdz,
										 double	pstr,
										 double  r)
{
	b_has_heater = true;
	v_heatSource.push_back(VortexRing(pN,
		prate,
		plife,
		pcx,
		pcy,
		pcz,
		pdx,
		pdy,
		pdz,
		pstr,
		r));
}
void VortexSimulationSystem::preTimestep()
{
	//deleteTracer();
	deleteTracer();
	deleteVortex();
	deleteHeat();

	emitTracer();
	emitHeat();
	emitVortexRing();
	
	//addBaroclinicForce();

	freeComputation();
	setupComputation();

}
uint computeNumBatches(uint num, uint num_per_batch)
{
	return num/num_per_batch + ((num%num_per_batch==0)?0:1);
}
void
VortexSimulationSystem::evaluateBiotSavart(float4* pos, double4* vel, uint num)
{
	uint N_batches = computeNumBatches(num, u_num_per_batch);
	for (uint i=0; i<N_batches; i++)
	{
		uint num_particles;
		num_particles = u_num_per_batch;
		if (i==N_batches-1)
		{
			num_particles = num + u_num_per_batch - N_batches * u_num_per_batch;
		}
		memcpy(gf4_tracer_pos->getHostPtr(),
			&pos[i*u_num_per_batch],
			sizeof(float4)*num_particles);

		gf4_tracer_pos->copy(gf4_tracer_pos->HOST_TO_DEVICE);

		memcpy(gd4_tracer_vel->getHostPtr(),
			&vel[i*u_num_per_batch],
			sizeof(double4)*num_particles);
		gd4_tracer_vel->copy(gd4_tracer_vel->HOST_TO_DEVICE);

		g_vortexSolver->setEvalParameter(num_particles,gf4_tracer_pos);
		g_vortexSolver->evaluateVelocity(gf4_tracer_pos,0);
		AddURotation(gd4_tracer_vel->getDevicePtr(),
			g_vortexSolver->getU(0)->getDevicePtr(),
			g_vortexSolver->getU(1)->getDevicePtr(),
			g_vortexSolver->getU(2)->getDevicePtr(),
			num_particles);

		if(!b_use_boundary)
		{
			move_particle(gf4_tracer_pos->getDevicePtr(),
				d_dt,
				gd4_tracer_vel->getDevicePtr(),
				num_particles);
		}

		gf4_tracer_pos->copy(gf4_tracer_pos->DEVICE_TO_HOST);
		gd4_tracer_vel->copy(gd4_tracer_vel->DEVICE_TO_HOST);

		memcpy(&pos[i*u_num_per_batch],
			gf4_tracer_pos->getHostPtr(),
			sizeof(float4)*num_particles);
		memcpy(&vel[i*u_num_per_batch],
			gd4_tracer_vel->getHostPtr(),
			sizeof(double4)*num_particles);
	}

}

void
VortexSimulationSystem::evaluatePotential(float4* pos, double4* vel, uint num)
{
	if(b_use_boundary){

		uint N_batches = computeNumBatches(num, u_num_per_batch);
		for (uint i=0; i<N_batches; i++)
		{
			uint num_particles;
			num_particles = u_num_per_batch;
			if (i==N_batches-1)
			{
				num_particles = num + u_num_per_batch - N_batches * u_num_per_batch;
			}
			memcpy(gf4_tracer_pos->getHostPtr(),
				&pos[i*u_num_per_batch],
				sizeof(float4)*num_particles);

			gf4_tracer_pos->copy(gf4_tracer_pos->HOST_TO_DEVICE);

			memcpy(gd4_tracer_vel->getHostPtr(),
				&vel[i*u_num_per_batch],
				sizeof(double4)*num_particles);
			gd4_tracer_vel->copy(gd4_tracer_vel->HOST_TO_DEVICE);

			g_potentialFlowSolver->setEvalParameter(num_particles,
				gf4_tracer_pos);
			g_potentialFlowSolver->evaluateGradient(true);
			AddUPotential(gd4_tracer_vel->getDevicePtr(),
						 g_potentialFlowSolver->getGradPhi()->getDevicePtr(),
						 num_particles);

			
			move_particle(gf4_tracer_pos->getDevicePtr(),
				d_dt,
				gd4_tracer_vel->getDevicePtr(),
				num_particles);
			

			gf4_tracer_pos->copy(gf4_tracer_pos->DEVICE_TO_HOST);
			gd4_tracer_vel->copy(gd4_tracer_vel->DEVICE_TO_HOST);

			memcpy(&pos[i*u_num_per_batch],
				gf4_tracer_pos->getHostPtr(),
				sizeof(float4)*num_particles);
			memcpy(&vel[i*u_num_per_batch],
				gd4_tracer_vel->getHostPtr(),
				sizeof(double4)*num_particles);
		}
	}

}