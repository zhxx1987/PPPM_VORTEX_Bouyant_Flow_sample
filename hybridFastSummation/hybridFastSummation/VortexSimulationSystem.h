#ifndef _VortexSimulationSystem_
#define _VortexSimulationSystem_

#include "BiotSavartSolver.h"
#include "ParticleMesh.h"
#include "PotentialFieldSolver.h"
#include "SpaceHashClass.h"
#include <vector>
#include "renderer/vec.h"
using namespace std;
struct VortexRing 
{
	uint	N;
	uint	rate;
	double	life;
	double	cx;
	double	cy;
	double	cz;
	double	dx;
	double	dy;
	double	dz;
	double	str;
	double r;
	~VortexRing(){}
	VortexRing()
	{
		N=0;
		rate=0;
		life=16;
		cx=0;
		cy=0;
		cz=0;
		dx=0;
		dy=0;
		dz=0;
		str=0;
		r=0;
	}
	VortexRing(const VortexRing &v)
	{
		N		=v.N		;
		rate	=v.rate		;
		life	=v.life		;
		cx		=v.cx		;
		cy		=v.cy		;
		cz		=v.cz		;
		dx		=v.dx		;
		dy		=v.dy		;
		dz		=v.dz		;
		str		=v.str	;
		r		=v.r;
	}
	VortexRing(uint	pN,
			uint	prate,
			double	plife,
			double	pcx,
			double	pcy,
			double	pcz,
			double	pdx,
			double	pdy,
			double	pdz,
			double	pstr,
			double pr)
	{
		N		=pN;
		rate	=prate;
		life	=plife;
		cx		=pcx;
		cy		=pcy;
		cz		=pcz;
		dx		=pdx;
		dy		=pdy;
		dz		=pdz;
		str		=pstr;
		r = pr;
		
	}

};
class VortexSimulationSystem//a vortex based fluid solver using vortex segments
{
public:
	VortexSimulationSystem();
	~VortexSimulationSystem(){};
	void solveTimeStep(double dt);
	void setHeatEmitter(uint	pN,
		uint	prate,
		double	plife,
		double	pcx,
		double	pcy,
		double	pcz,
		double	pdx,
		double	pdy,
		double	pdz,
		double	pstr,
		double  r);
	void setVortexRingEmitter(uint	pN,
		uint	prate,
		double	plife,
		double	pcx,
		double	pcy,
		double	pcz,
		double	pdx,
		double	pdy,
		double	pdz,
		double	pstr,
		double  r);
	void setTracerEmitter(uint	pN,
		uint	prate,
		double	plife,
		double	pcx,
		double	pcy,
		double	pcz,
		double	pdx,
		double	pdy,
		double	pdz,
		double	pstr,
		double  r);
	uint getNumVortex() {return u_num_vortices; }
	float4 * getVortexPos() {
		gf4_vort_cent->copy(gf4_vort_cent->DEVICE_TO_HOST);
		return gf4_vort_cent->getHostPtr();
	}
	uint getNumTracer(){return u_num_eval_pos;}
	float4* getTracerPos(){return cpu_tracer_pos;}

private:
	
	void deleteHeat();
	void deleteTracer();
	void deleteVortex();

	void emitHeat();
	void emitTracer();
	void emitVortexRing();
	void freeComputation();
	void setupComputation();

	void evaluateBiotSavart(float4* pos, double4* vel, uint num);
	void evaluatePotential(float4* pos, double4* vel, uint num);
	

	void preTimestep();
	void convertVortToSeg();
	void computeURotation();
	void computeUPotential();
	void solveBoundary();
	void applyFlow();
	void convertSegToVort();
	void computeStretching();
	void computeShedding();
	void computeAx(GpuArrayd * rhs, GpuArrayd * res, uint num_faces);
	void BiCGSTAB(GpuArrayd * x, GpuArrayd * b, uint num_elements, uint iter);
	

	void assignTempToGrid();
	void computeCurl();
	double computeVortexVol();
	void addBaroclinicForce();
	void setDensityDomain(double4 & origin, float4 * pos, uint num_particle, double & domain_length);
	
	uint		u_num_boundary_faces;
	uint		u_num_vortices;
	uint		u_num_eval_pos;
	uint		u_num_heat_pos;
	uint		u_dimx_pppm, u_dimy_pppm, u_dimz_pppm;
	uint		u_dimx_brdy, u_dimy_brdy, u_dimz_brdy;
	uint		u_num_shed;
	uint		u_curr_frame;
	uint		u_num_per_batch;
	int			i_LCrange;
	double 		d_dt;
	double 		d_nu;
	double 		d_h;
	double 		d_cx;
	double 		d_cy;
	double 		d_cz;
	double 		d_r;
	double 		d_density_domain_length;
	double4 	d4_density_origin;
	double 		d_vort_size;
	double 		d_beta;	
	double 		d_u_solid;
	double 		d_v_solid;
	double 		d_w_solid;
	double		d_shed_coef;
	double		d_shed_difu;
	bool 		b_use_boundary;
	bool 		b_use_vic;
	bool		b_use_shedding;
	bool		b_has_tracer;
	bool		b_has_heater;
	bool		b_has_vortex_tube;
	bool		b_autoshed;


	BiotSavartSolver *			g_vortexSolver;
	PotentialFieldSolver *		g_potentialFlowSolver;
	ParticleManager *			g_spacehash;


	GpuArrayf4 *				gf4_vort_enda;
	GpuArrayf4 *				gf4_vort_endb;
	GpuArrayf4 *				gf4_vort_cent;
	GpuArrayf4 *				gf4_shedding_pos;
	GpuArrayd4 *				gd4_ua_rotational;
	GpuArrayd4 *				gd4_ub_rotational;
	GpuArrayd4 *				gd4_uc_rotational;
	GpuArrayd4 *				gd4_ua_potential;
	GpuArrayd4 *				gd4_ub_potential;
	GpuArrayd4 *				gd4_uc_potential;

	GpuArrayd * gd_kappa;
	GpuArrayd * gd_vort[NUM_COMPONENTS];
	GpuArrayd * gd_dvort[NUM_COMPONENTS];
	GpuArrayd * gd_shed_vort[NUM_COMPONENTS];
	GpuArrayf4 * gf4_boundary_pos;
	GpuArrayd3 * gd3_boundary_normal;
	GpuArrayd  * gd_boundary_area;
	GpuArrayd  * gd_boundary_mass;
	//GpuArrayd4 * m_boundary_slip;
	GpuArrayd  * gd_boundary_b;
	GpuArrayd  * gd_boundary_u;
	GpuArrayd  * gd_boundary_v;
	GpuArrayd  * gd_boundary_w;
	GpuArrayd  * gd_boundary_vortx;
	GpuArrayd  * gd_boundary_vorty;
	GpuArrayd  * gd_boundary_vortz;

	GpuArrayf * gf_vortex_life;
	GpuArrayf4 * gf4_tracer_pos;
	GpuArrayd4 * gd4_tracer_vel;
	GpuArrayf  * gf_tracer_life;

	//vortex particle
	vector<float> v_vortex_life;
	vector<Vec4f> v_vortex_pos;
	vector<double> v_vortex_vortx;
	vector<double> v_vortex_vorty;
	vector<double> v_vortex_vortz;

	//vortex ring emitter
	vector<VortexRing> v_vortexTube;
	vector<VortexRing> v_heatSource;
	vector<VortexRing> v_tracerSource;



	float4 * cpu_tracer_pos;
	float *  cpu_tracer_life;
	double4 *  cpu_tracer_vel;

	vector<Vec4f> v_tracer_pos;
	vector<float> v_tracer_life;
	vector<Vec4d> v_tracer_vel;


	float4 * cpu_heat_pos;
	float *  cpu_heat_life;
	double4 *  cpu_heat_vel;

	vector<Vec4f> v_heat_pos;
	vector<float> v_heat_life;
	vector<Vec4d> v_heat_vel;
	


	
	
};

#endif