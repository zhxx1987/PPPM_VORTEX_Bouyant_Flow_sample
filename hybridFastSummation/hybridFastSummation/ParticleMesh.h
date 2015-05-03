#ifndef _PARTICLE_MESH_H_
#define _PARTICLE_MESH_H_
typedef unsigned int uint;





void ParticleToMesh(uint * start,
	uint * end,
	float4 * pos,
	double * mass,
	double cell_h,
	double * gridval,
	uint3 griddim,
	uint3 hashdim,
	uint num_particle,
	double4 origin);

void ParticleGetForceFromMesh(float4 * pos,
	const double * mass,
	const double3 * gforce,
	double3 * pforce,
	double cell_h,
	uint3 gdim,
	uint3 hdim,
	uint num_particles,
	double4 origin);

void ParticleGetMassFromMesh(float4 * pos,
	const float * mass,
	const double * grho,
	float * pmass,
	double cell_h,
	uint3 gdim,
	uint3 hdim,
	uint num_particles,
	double4 origin);

void ComputeRHS(double * gridval, double h_sqr, double G, uint num_cell);


void ComputeGradient(double* phi,
	double3 * dphi,
	double cell_h,
	double direction,
	uint dimx, 
	uint dimy, 
	uint dimz);

//void LocalCorrection(uint * start,
//	uint * end,
//	float4 * pos,
//	double * mass,
//	double * gridval,
//	double * phi,
//	double3 * gforce,
//	double3 * pforce,
//	double cell_h,
//	double G,
//	uint3 gdim,
//	uint3 hdim,
//	int K,
//	uint num_particles,
//	double4 origin);
//
//void SubtractGridForce(uint* start,
//	uint * end,
//	float4 * pos,
//	float * mass,
//	double * gridval,
//	double3 * pforce,
//	double cell_h,
//	double G,
//	uint3 gdim,
//	uint3 hdim,
//	uint num_particles,
//	double4 origin);


void Potential_PPCorrMN(uint * start,
						uint * end,
						float4 * pos_eval,
						float4 * pos_mass,
						double * mass,
						double3 * pforce,
						double invh,
						double h,
						double G,
						double direction/*-1 for gravity, 1 for single layer potential*/,
						uint3 gdim,
						uint3 hdim,
						uint2 gstride,
						uint2 hstride,
						int K,
						uint M,
						uint N,
						double4 origin);




void PotentialComputeGradForOutParticle(float4 * pos, double total_mass, double3 mass_center, float3 bbmin, float3 bbmax, double G, double direction/*-1 for gravity, 1for SLP*/,double3* p_force, uint M);

void PotentialInterpolateFarField(float4 * pos,
							 double3 * far_force,
							 double3 * p_force,
							 double cell_h,
							 uint dimx,uint dimy,uint dimz,
							 uint num_particles,
							 double4 origin);

void PotentialComputeFarField(uint * start,
							  uint * end,
							  float4 * pos,
							  double * mass,
							  double * gridval,
							  double * phi,
							  double3 * gforce,
							  double3 * pforce,
							  double cell_h,
							  double G,
							  double direction,//-1 gravity, 1 for single layer potential
							  uint3 gdim,
							  uint3 hdim,
							  int K,
							  uint num_particles,
							  double4 origin);

void ComputeDphidn(double3 * normals,
				   double3 * gradients,
				   double  * mass,
				   double  * dphi_dn,
				   double  * area,
				   uint M);

/////////////////////////////////////////////////////////////////////Biot Savart///////////////////////////////////////////////////////////////

void getCurl(double * Fx,
	double * Fy,
	double * Fz,
	double * u,
	double * v,
	double * w,
	uint dimx,
	uint dimy,
	uint dimz,
	double h);

void MeshToParticle( float4 * pos,
	double * mesh_value,
	double * particle_value,
	double h,
	uint3 gdim,
	uint3 hdim,
	uint num_particles,
	double4 origin);
void
	BiotSavartPPCorr(uint * start,
	uint * end,
	float4 * pos,
	double * vort_x,
	double * vort_y,
	double * vort_z,
	double * u,
	double * v,
	double * w,
	double cellh,
	uint3 gdim,
	uint3 hdim,
	int K,
	uint num_particle,
	double4 origin);

void BiotSavartPPCorrMN(uint * start,
				   uint * end,
				   float4 * evalposa,
				   float4 * evalposb,
				   uint is_segment,
				   float4 * vortpos,
				   double * vort_x,
				   double * vort_y,
				   double * vort_z,
				   double * u,
				   double * v,
				   double * w,
				   double cellh,
				   uint3 gdim,
				   uint3 hdim,
				   int K,
				   uint M,
				   uint N,
				   double4 origin);
void
BiotSavartPPCorrScaleMN(uint * start,
						uint * end,
						float4 * evalposa,
						float4 * evalposb,
						uint is_segment,
						float4 * vortpos,
						double * vort_x,
						double * vort_y,
						double * vort_z,
						double * u,
						double * v,
						double * w,
						double * u_pm,
						double * v_pm,
						double * w_pm,
						double cellh,
						uint3 gdim,
						uint3 hdim,
						int K,
						uint M,
						uint N,
						double4 origin);

//void BiotSavartPMCorr(uint* start,
//	uint * end,
//	float4 * pos,
//	double * vrho_x,
//	double * vrho_y,
//	double * vrho_z,
//	double * psi_x,
//	double * psi_y,
//	double * psi_z,
//	double * g_u,
//	double * g_v,
//	double * g_w,
//	double * u,
//	double * v,
//	double * w,
//	double cell_h,
//	uint3 gdim,
//	uint3 hdim,
//	int K, 
//	uint num_particles, 
//	double4 origin);

//void split_pos(float4 * pos,
//	double * x,double * y,double * z,
//	double * mass, 
//	uint num_particle, uint weighted);

void applyDirichlet(double * RHS,
	double4 center,
	double monopole,
	double4 grid_coner,
	double h,
	uint dimx,
	uint dimy,
	uint dimz);

void move_particle(float4 * pos, double dt, double4 *U, uint num_particles);

void ComputeVortex(float4 * posa,float4 *posb, double* vortx, double * vorty, double * vortz,double * dvortx, double * dvorty, double * dvortz,double * k, uint num_particles);

void ComputePosb(float4 *posa, float4* posc, float4 *posb, double * vortx, double * vorty, double * vortz, double * k, double h, uint num_particles);

void ComputePosa(float4 *posc, float4 *posb, float4* posa, uint num_particles);

void BiotSavartComputeVelocityForOutParticle(float4 * pos, double3 omega, double3 vort_center, float3 bbmin, float3 bbmax, double* u, double* v, double* w, uint M);

void ParticleGaussianSmooth(uint * start,
				  uint * end,
				  float4 * pos,
				  float4 * pos_sort,
				  double nu,
				  double dt,
				  double * o_vortx,
				  double * o_vorty,
				  double * o_vortz,
				  double * i_vortx,
				  double * i_vorty,
				  double * i_vortz,
				  double h,
				  uint3 gdim,
				  uint N, 
				  double4 origin);

void BiotSavartComputeFarField(uint* start,uint * end,float4 * pos,double * vrho_x,double * vrho_y,double * vrho_z,
						  double * psi_x,double * psi_y,double * psi_z,
						  double * g_u,double * g_v,double * g_w,
						  double * u,double * v,double * w,
						  double cell_h,uint3 gdim,uint3 hdim,
						  int K, uint num_particles, double4 origin);

void BiotSavartInterpolateFarField(float4 * pos,
						 double * far_u,double * far_v,double * far_w,
						 double * p_u, double * p_v,double * p_w,
						 double cell_h,
						 uint dimx,uint dimy,uint dimz,
						 uint num_particles,
						 double4 origin);


void VectorAdd(double * A, double * B, double * C, double a, double b, uint num);

double ComputeDot(double * a, double * b, uint num);

void ComputeSingleLayerPotentialMass(double * den, double * area,  double * mass, uint num);

void ComputeNormalSlip(double * u, double * v, double * w, double u_solid, double v_solid, double w_solid, double3 * normal, double *b, uint num);

void ComputeBoundarySlip(double * u, double * v, double * w, double u_solid, double v_solid, double w_solid,   uint num);

void UpdateBoundary(float4 * pos,  double u_solid, double v_solid, double w_solid, double dt, uint num);

void AddURotation(double4 * U, double *u, double *v, double *w, uint num);

void AddUPotential(double4 * U, double3 *u_p, uint num);
//void
//	reduce_summation(int size, int threads, int blocks,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
//	int whichKernel, double *d_idata, double *d_odata);
//
//void
//	reduce_Min(int size, int threads, int blocks,
//	int whichKernel, double *d_idata, double *d_odata);
//
//
//void
//	reduce_Max(int size, int threads, int blocks,
//	int whichKernel, double *d_idata, double *d_odata);
//void
//	reduce_summationabs(int size, int threads, int blocks,
//	int whichKernel, double *d_idata, double *d_odata);

void AssignDensity(uint * start,
				   uint * end,
				   float4 * pos,
				   float * rho,
				   float * vort_size,
				   double cell_h,
				   float * gridval,
				   uint3 griddim,
				   uint3 hashdim,
				   uint num_particle, 
				   double4 origin);

void GradDensity(float* phi,
				 float3 * dphi,
				 double cell_h,
				 uint dimx, 
				 uint dimy, 
				 uint dimz);

void
ParticleGetBaroclinic(float4 * vortex_pos,
					  double * vort_x,
					  double * vort_y,
					  double * vort_z,
					  float3 * grad_rho,
					  double dt_beta,
					  uint num_vort,
					  uint dimx,
					  uint dimy,
					  uint dimz,
					  double inv_h,
					  double4 origin);

void FixBoundary(float4 * pos,
				 double cx,
				 double cy,
				 double cz,
				 double r,
				 uint num);

void VortexShedding(uint *start,
					uint *end,
					float4 *vort_pos,
					double *vort_x,
					double *vort_y,
					double *vort_z,
					float4 *boundary_pos,
					double *boundary_vortx,
					double *boundary_vorty,
					double *boundary_vortz,
					double shedding_r,
					double dt_c,
					uint dimx, 
					uint dimy,
					uint dimz,
					double inv_h,
					double4 origin,
					uint num_boundaryEle,
					uint num_vortexEle);

void ComputeBoundaryVortex(double *u,
						   double *v,
						   double *w,
						   double *vort_x,
						   double *vort_y,
						   double *vort_z,
						   float4 *boundary_pos,
						   double3 * normal,
						   double *area,
						   double dt_c,
						   uint num);




void appendArrayf4(float4 *new_array,
				  float4 *old_array,
				  float4 *append_array,
				  uint n_old,
				  uint n_new);


void appendArrayd(double *new_array,
				  double *old_array,
				  double *append_array,
				  uint n_old,
				  uint n_new);

void appendArrayd4(double4 *new_array,
				  double4 *old_array,
				  double4 *append_array,
				  uint n_old,
				  uint n_new);


void ComputeSheddingPos(float4 * shedding_pos,
						float4 * boundary_pos,
						double3 * boundary_normal,
						uint N);



void PotentialComputeScalarFarField(
									double * mass,
									double * gridval,
									double * phi,
									double cell_h,
									double G,
									double direction,//-1 gravity, 1 for single layer potential
									uint3 gdim,
									uint3 hdim,
									int K);


void Potential_PPCorrMNScalar(uint * start,
							  uint * end,
							  float4 * pos_eval,
							  float4 * pos_mass,
							  double * mass,
							  double * p_phi,
							  double invh,
							  double h,
							  double G,
							  double direction,//-1 for gravity, 1 for single layer potential
							  uint3 gdim,
							  uint3 hdim,
							  uint2 gstride,
							  uint2 hstride,
							  int K,
							  uint M,
							  uint N,
							  double4 origin);


void PotentialInterpolateFarFieldScalar(float4 * pos,
										double * far_force,
										double * p_force,
										double cell_h,
										uint dimx,uint dimy,uint dimz,
										uint num_particles,
										double4 origin);

void ParticleToMeshGaussian(uint * start,
							uint * end,
							float4 * pos,
							double * mass,
							double cell_h,
							double * gridval,
							uint3 griddim,
							uint3 hashdim,
							uint num_particle, 
							double4 origin);

void computeDivergence(double* u, double *v, double * w, 
					   double* f,
					   double cell_h,
					   uint dimx,
					   uint dimy,
					   uint dimz);


#endif