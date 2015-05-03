#ifndef _PPPM_UNUSED_KERNEL_
#define _PPPM_UNUSED_KERNEL_
#include <cuda_runtime.h>
#include <helper_cuda.h>
typedef unsigned int uint;
double cpu_compute_r(double4 & r1, double4 & r2)
{
	double r_sqr = (r1.x - r2.x)*(r1.x - r2.x) + (r1.y - r2.y)*(r1.y - r2.y) + (r1.z - r2.z)*(r1.z - r2.z);
	if(r_sqr>1e-12)
		return 12.5663706144 * (sqrt(r_sqr));
	else
		return 1e+14;
}
__device__
double sample3D(double * f, int x, int y, int z, uint dimx, uint dimy, uint dimz)
{
	int i = min(max(0,x), dimx-1);
	int j = min(max(0,y), dimy-1);
	int k = min(max(0,z), dimz-1);
	return f[k*dimx*dimy+j*dimx+i];
}
__device__ 
double shape_function_mass(double r)
{
	double res = 0;
	
	if(r<0.5&&r>=-0.5)
		res = 1;
	res = res;
	return res;
	
	/*double rr=fabs(r);
	
	double res = 0;
	if(rr<2)
		res = ((5-2*rr)-sqrt(-4*rr*rr+12*rr-7));
	if(rr<1)
		res = ((3-2*rr)+sqrt(1+4*rr-4*rr*rr));
	res = res*0.125;
	return res;*/
}

__device__ 
double shape_function(double r)
{
	double rr=fabs(r);
	
	double res = 0;
	if(rr<2)
		res = ((5-2*rr)-sqrt(-4*rr*rr+12*rr-7));
	if(rr<1)
		res = ((3-2*rr)+sqrt(1+4*rr-4*rr*rr));
	res = res*0.125;
	return res;



}
__device__
double compute_r3(double3 r)
{
	double a = 0.1;
	double r_sqr = (r.x)*(r.x) + (r.y)*(r.y) + (r.z)*(r.z);
	if(r_sqr>a)
		return 12.5663706144 * (r_sqr * sqrt(r_sqr));
	else if(r_sqr>1e-12)
		return 12.5663706144 * sqrt(r_sqr)*a*a;
	else
		return 1e+14;
}
__device__
double compute_r(double4 & r1, double4 & r2)
{
	double r_sqr = (r1.x - r2.x)*(r1.x - r2.x) + (r1.y - r2.y)*(r1.y - r2.y) + (r1.z - r2.z)*(r1.z - r2.z);
	if(r_sqr>1e-12)
		return 12.5663706144 * (sqrt(r_sqr));
	else
		return 1e+12;
}
__device__
double3 cross_uxv(double3 &u, double3 &v)
{
	return make_double3(u.y*v.z-u.z*v.y, u.z*v.x-u.x*v.z, u.x*v.y-u.y*v.x);
}


__global__ void
MeshToParitcleMass_kernel(const float4 * pos,
					  const float * mass,
					  const double * gmass,
					  float * pmass,
					  double invh,
					  uint3 gdim,
					  uint3 hdim,
					  uint2 gstride,
					  uint2 hstride,
					  uint num_particles,
					  double4 origin)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<num_particles)
	{
		double sum = 0.0;
		float4 ppos = pos[idx];
		ppos.x -= origin.x;
		ppos.y -= origin.y;
		ppos.z -= origin.z;
		//compute the grid idx it belongs to
		int i = floor(ppos.x*invh);
		int j = floor(ppos.y*invh);
		int k = floor(ppos.z*invh);
		//loop over the grid around it to cumulate force
		for(int kk=k-2;kk<=k+2;kk++)for(int jj= j-2; jj<=j+2; jj++)for(int ii=i-2; ii<=i+2; ii++)
		{
			if(kk>=0&&kk<gdim.z && jj>=0&&jj<gdim.y && ii>=0&&ii<gdim.x)
			{
				//fetch the grid value
				uint read_idx = kk*gstride.x + jj * gstride.y + ii;
				double f = gmass[read_idx];
				float4 pijk = make_float4(((double)ii+0.5),((double)jj+0.5),((double)kk+0.5),0);
				double r1 = (double)ppos.x * invh - pijk.x;
				double r2 = (double)ppos.y * invh - pijk.y;
				double r3 = (double)ppos.z * invh - pijk.z;
				double w = shape_function(fabs(r1))*shape_function(fabs(r2))*shape_function(fabs(r3));
				sum += f * w;
			}
		}
		pmass[idx] = sum/invh/invh/invh;
		
	}
}
void ParticleGetMassFromMesh(float4 * pos,
					  const float * mass,
					  const double * grho,
					  float * pmass,
					  double cell_h,
					  uint3 gdim,
					  uint3 hdim,
					  uint num_particles,
					  double4 origin)
{
	uint threads = 256;
	uint blocks = num_particles/threads + (!(num_particles%threads)?0:1);
	MeshToParitcleMass_kernel<<<blocks, threads>>>(pos, mass, grho, pmass, 1.0/cell_h, gdim, hdim, 
		make_uint2(gdim.x*gdim.y, gdim.x), 
		make_uint2(hdim.x*hdim.y, hdim.x),
		num_particles,
		origin);
	getLastCudaError("get mass failed!\n");
}


__global__ void 
subtract_grid_force_kernel(const uint * start,
					 const uint * end,
					 const float4 * pos,
					 const double * mass,
					 const double * rho,
					 double3 * pforce,
					 double invh,
					 double h,
					 double G,
					 double gridscale,
					 uint3 gdim,
					  uint3 hdim,
					  uint2 gstride,
					  uint2 hstride,
					  uint ChunckSize,
					  uint ChunckIdx,
					  int K,
					  uint num_particle,
					  double4 origin)
{
	//we have dimx*dimy*dimz blocks, each block has K*K*K threads

	//__shared__ double local_rho[512];
	__shared__ double local_phi[512]; //at most 7x7x7
	__shared__ double3 local_f[128];  //at most 5x5x5

	//step 1 : 
	uint g_blockid = ChunckIdx*ChunckSize + blockIdx.x;

	if(g_blockid<(hdim.x*hdim.y*hdim.z))
	if(end[g_blockid]>start[g_blockid])
	{

		//locate the grid i j k
		int i = g_blockid%gdim.x;int j = (g_blockid/gdim.x)%gdim.y;int k = g_blockid/(gstride.x);

		//we gonna have a 8x8x8 thread
		int ii = threadIdx.x;int jj = threadIdx.y;int kk = threadIdx.z;
		int local_idx = kk*64 + jj*8 + ii;
		local_phi[local_idx] = 0.0f;
		//__syncthreads();
		if(local_idx<128)
		{
			local_f[local_idx].x = 0.0f;local_f[local_idx].y = 0.0f;local_f[local_idx].z = 0.0f;
		}
		__syncthreads();

		uint s2 = K*2+1;
		//step2 : read 5x5x5 grid val
		if(ii<s2 && jj <s2 && kk<s2)
		{
			int iii = i + ii - K;int jjj = j + jj - K;int kkk = k + kk - K;
			if(iii>=0 && iii<gdim.x && jjj>=0 && jjj<gdim.y && kkk>=0 && kkk<gdim.z)
			{
				int local_idx = (kk+3-K)*64 + (jj+3-K)*8 + (ii+3-K);
				int global_idx = kkk*gstride.x + jjj*gstride.y + iii;
				local_phi[local_idx] = rho[global_idx];
			}
		}
		__syncthreads();

		//step3 : convolute using free space green's functoin
		double phi_sum=0.0;
		if(ii<7 && jj<7 && kk<7)
		{
			for(int kkk=0; kkk<7;kkk++)for(int jjj=0;jjj<7;jjj++)for(int iii=0;iii<7;iii++)
			{
				if(iii==ii && jjj==jj &&kkk==kk)
				{
					int read_idx = kkk*64+jjj*8+iii;
					phi_sum +=  G*local_phi[read_idx]*0.250580605913965*h*h;
				}
				else
				{
					int read_idx = kkk*64+jjj*8+iii;
					double  w = compute_r(make_double4((double)ii*h, (double)jj*h, (double)kk*h, 0),
						make_double4((double)iii*h, (double)jjj*h, (double)kkk*h, 0));
					phi_sum += G*local_phi[read_idx]*h*h*h/w;
				}
			}
		}
		__syncthreads();
		local_phi[kk*64+jj*8+ii]=phi_sum;
		__syncthreads();
		//step 4 : compute the local force
		if(ii<5 && jj<5 && kk<5)
		{
			int l = ii - 1, r = ii + 1, u = jj + 1, d = jj - 1, f = kk - 1, b = kk + 1;
			double lv = local_phi[(kk+1)*64 + (jj+1)*8 + l+1],
				rv = local_phi[(kk+1)*64 + (jj+1)*8 + r+1],
				uv = local_phi[(kk+1)*64 + (u+1)*8 + ii+1],
				dv = local_phi[(kk+1)*64 + (d+1)*8 + ii+1],
				fv = local_phi[(f+1)*64 + (jj+1)*8 + ii+1],
				bv = local_phi[(b+1)*64 + (jj+1)*8 + ii+1];
			int write_idx = kk*25 + jj*5 + ii;
			local_f[write_idx].x = - 0.5 * (rv - lv) * invh;
			local_f[write_idx].y = - 0.5 * (uv - dv) * invh;
			local_f[write_idx].z = - 0.5 * (bv - fv) * invh;
		}
		__syncthreads();

		//step 5 : for all particles in this cell

		if(end[g_blockid]>start[g_blockid])
		{
			int local_np = end[g_blockid]-start[g_blockid];
			int round = local_np/512+1;
			for(int t = 0; t<round; t++)
			{
				int tid1D = kk*64 + jj*8 + ii;
				int pidx = start[g_blockid] + t * 512 + tid1D;
				if(pidx<end[g_blockid])
				{
					double3 f_lpm = make_double3(0.0,0.0,0.0);
					for(int kkk=0;kkk<5;kkk++)for(int jjj=0;jjj<5;jjj++)for(int iii=0;iii<5;iii++)
					{
						double3 f = local_f[kkk*25+jjj*5+iii];
						float4 pijk = make_float4((float)i+(float)iii+0.5-2.0,
							(float)j+(float)jjj+0.5-2.0,
							(float)k+(float)kkk+0.5-2.0,
							0.0);
						float4 ppos = pos[pidx];
						ppos.x -= origin.x;
						ppos.y -= origin.y;
						ppos.z -= origin.z;
						double r1 = ppos.x*invh - pijk.x;
						double r2 = ppos.y*invh - pijk.y;
						double r3 = ppos.z*invh - pijk.z;
						double w = shape_function(fabs(r1))*shape_function(fabs(r2))*shape_function(fabs(r3));
						f_lpm.x += w*f.x;
						f_lpm.y += w*f.y;
						f_lpm.z += w*f.z;
					}
					pforce[pidx] = make_double3(pforce[pidx].x-f_lpm.x,
						pforce[pidx].y-f_lpm.y,
						pforce[pidx].z-f_lpm.z);
				}
			}
		}
	}
}


void SubtractGridForce(uint* start,
					 uint * end,
					 float4 * pos,
					 double * mass,
					 double * gridval,
					 double3 * pforce,
					 double cell_h,
					 double G,
					 uint3 gdim,
					 uint3 hdim,
					 int K,
					 uint num_particles,
					 double4 origin)
{
	if(K>=0)
	{

	uint num_gridcells = hdim.x*hdim.y*hdim.z;
	uint chunck_size = 256;
	uint num_chuncks = num_gridcells/chunck_size + 1;
	printf("num chuncks : %d\n", num_chuncks);
	for(uint chunck_id = 0; chunck_id<num_chuncks; chunck_id++)
	{
		//printf("%d done\n", chunck_id);
		dim3 blocks(chunck_size,1,1);
		dim3 threads(8,8,8); 
		subtract_grid_force_kernel<<<blocks, threads>>>(start, 
			end,
			pos,
			mass,
			gridval,
			pforce,
			1.0/cell_h,
			cell_h,
			G,
			cell_h*cell_h*cell_h,
			gdim,
			hdim, 
			make_uint2(gdim.x*gdim.y, gdim.x),
			make_uint2(hdim.x*hdim.y, hdim.x),
			chunck_size,
			chunck_id,
			K,
			num_particles,
			origin);
		getLastCudaError("compute subtract grid val failed\n!");
	}
	}
	

}
__global__ 
void BiotSavartPPCorr_kernel(const uint * start,
					 const uint * end,
					 const float4 * pos,
					 double * vort_x,
					 double * vort_y,
					 double * vort_z,
					 double * u,
					 double * v,
					 double * w,
					 double invh,
					 uint3 gdim,
					 uint3 hdim,
					 uint2 gstride,
					 uint2 hstride,
					 int K,
					 uint num_particle, 
					 double4 origin)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	uint p=0;
	if(idx<num_particle)
	{
		double3 vel_PM = make_double3(u[idx],v[idx],w[idx]);
		float4 ppos = pos[idx];
		ppos.x -= origin.x;
		ppos.y -= origin.y;
		ppos.z -= origin.z;
		//compute the grid idx it belongs to
		int i = floor(ppos.x*invh);
		int j = floor(ppos.y*invh);
		int k = floor(ppos.z*invh);
		
		double3 vel_pp = make_double3(0,0,0);
		for(int kk=k-K;kk<=k+K;kk++)for(int jj= j-K; jj<=j+K; jj++)for(int ii=i-K; ii<=i+K; ii++)
		{
			if(kk>=0&&kk<hdim.z && jj>=0&&jj<hdim.y && ii>=0&&ii<hdim.x)
			{
				uint read_idx = kk*hstride.x + jj*hstride.y + ii;
				for(p = start[read_idx]; p<end[read_idx]; p++)
				{
					if(p<num_particle && p!=idx)
					{
						float4 ppos2 = pos[p];
						ppos2.x -= origin.x;
						ppos2.y -= origin.y;
						ppos2.z -= origin.z;
						double3 omega = make_double3(vort_x[p],vort_y[p], vort_z[p]);

						double3 dir = make_double3(ppos.x-ppos2.x, ppos.y-ppos2.y, ppos.z-ppos2.z);
						double r3 = compute_r3(dir);
						double3 vel_ij = cross_uxv(omega, dir);
						vel_pp.x += vel_ij.x/r3;
						vel_pp.y += vel_ij.y/r3;
						vel_pp.z += vel_ij.z/r3;
					}
				}
			}
		}
		u[idx] = vel_PM.x + vel_pp.x; 
		v[idx] = vel_PM.y + vel_pp.y;
		w[idx] = vel_PM.z + vel_pp.z;
		
	}
}

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
				double4 origin)
{
	if(K>=0)
	{
		uint threads = 256;
		uint blocks = num_particle/threads + (!(num_particle%threads)?0:1);
		BiotSavartPPCorr_kernel<<<blocks, threads>>>(start, end, 
			pos, 
			vort_x, vort_y, vort_z, 
			u, v, w, 
			1.0/cellh,
			gdim, hdim,
			make_uint2(gdim.x*gdim.y, gdim.x),
			make_uint2(hdim.x*hdim.y, hdim.x),
			K,
			num_particle,
			origin);
	}

}

__global__ 
void BiotSavartPMCorr_kernel(const uint * start,
					const uint * end,
					const float4 * pos,
					const double * vrho_x,
					const double * vrho_y,
					const double * vrho_z,
					double * u,
					double * v,
					double * w,
					double invh,
					double h,
					double gridscale,
					uint3 gdim,
					uint3 hdim,
					uint2 gstride,
					uint2 hstride,
					uint ChunckSize,
					uint ChunckIdx,
					int modify_size,
					uint num_particle,
					double4 origin)
{
	//we have dimx*dimy*dimz blocks, each block has K*K*K threads

	
	__shared__ double local_psi_x[512]; //at most 7x7x7
	__shared__ double local_psi_y[512];
	__shared__ double local_psi_z[512];
	__shared__ double3 local_v[128];  //at most 5x5x5

	//step 1 : 
	uint g_blockid = ChunckIdx*ChunckSize + blockIdx.x;

	if(g_blockid<(hdim.x*hdim.y*hdim.z))
	if(end[g_blockid]>start[g_blockid])
	{

		//locate the grid i j k
		int i = g_blockid%gdim.x;int j = (g_blockid/gdim.x)%gdim.y;int k = g_blockid/(gstride.x);

		//we gonna have a 8x8x8 thread
		int ii = threadIdx.x;int jj = threadIdx.y;int kk = threadIdx.z;
		int local_idx = kk*64 + jj*8 + ii;

		local_psi_x[local_idx] = 0.0f;
		local_psi_y[local_idx] = 0.0f;
		local_psi_z[local_idx] = 0.0f;

		//__syncthreads();
		if(local_idx<128)
		{
			local_v[local_idx].x = 0.0f;
			local_v[local_idx].y = 0.0f;
			local_v[local_idx].z = 0.0f;
		}
		__syncthreads();

		//step2 : read 7x7x7 grid val
		uint s2 = modify_size*2+1;
		if(ii<s2 && jj <s2 && kk<s2)
			
		{
			int iii = i + ii - modify_size;int jjj = j + jj - modify_size;int kkk = k + kk - modify_size;
			if(iii>=0 && iii<gdim.x && jjj>=0 && jjj<gdim.y && kkk>=0 && kkk<gdim.z)
			{
				int local_idx = (kk + 3 - modify_size)*64 + (jj + 3 - modify_size)*8 + (ii + 3 - modify_size );
				int global_idx = kkk*gstride.x + jjj*gstride.y + iii;
				local_psi_x[local_idx] = vrho_x[global_idx];
				local_psi_y[local_idx] = vrho_y[global_idx];
				local_psi_z[local_idx] = vrho_z[global_idx];
			}
		}
		__syncthreads();

		//step3 : convolute using free space green's functoin
		double psi_sum_x=0.0;
		double psi_sum_y=0.0;
		double psi_sum_z=0.0;
		if(ii<7 && jj<7 && kk<7)
		{
			for(int kkk=0; kkk<7;kkk++)for(int jjj=0;jjj<7;jjj++)for(int iii=0;iii<7;iii++)
			{
				
				if(iii==ii && jjj==jj &&kkk==kk)
				{
					int read_idx = kkk*64+jjj*8+iii;
					psi_sum_x += local_psi_x[read_idx]*0.250580605913965*h*h;
					psi_sum_y += local_psi_y[read_idx]*0.250580605913965*h*h;
					psi_sum_z += local_psi_z[read_idx]*0.250580605913965*h*h;
				}
				else
				{
					int read_idx = kkk*64+jjj*8+iii;
					double  w = compute_r(make_double4((double)ii*h, (double)jj*h, (double)kk*h, 0),
						make_double4((double)iii*h, (double)jjj*h, (double)kkk*h, 0));
					psi_sum_x += local_psi_x[read_idx]*gridscale/w;
					psi_sum_y += local_psi_y[read_idx]*gridscale/w;
					psi_sum_z += local_psi_z[read_idx]*gridscale/w;
				}
			}
		}
		__syncthreads();
		local_psi_x[kk*64+jj*8+ii]=psi_sum_x;
		local_psi_y[kk*64+jj*8+ii]=psi_sum_y;
		local_psi_z[kk*64+jj*8+ii]=psi_sum_z;

		__syncthreads();
		//step 4 : compute the local velocity
		//which is u = curl(Psi)
		if(ii<5 && jj<5 && kk<5)
		{
			int l = ii - 1, r = ii + 1, u = jj + 1, d = jj - 1, f = kk - 1, b = kk + 1;
			double Fz_up	= sample3D(local_psi_z, ii+1, u+1, kk+1, 8, 8, 8);
			double Fz_down	= sample3D(local_psi_z, ii+1, d+1, kk+1, 8, 8, 8);
			double Fz_left	= sample3D(local_psi_z, l+1, jj+1, kk+1, 8, 8, 8);
			double Fz_right	= sample3D(local_psi_z, r+1, jj+1, kk+1, 8, 8, 8);
			double Fy_front	= sample3D(local_psi_y, ii+1, jj+1, f+1, 8, 8, 8);
			double Fy_back	= sample3D(local_psi_y, ii+1, jj+1, b+1, 8, 8, 8);
			double Fy_left	= sample3D(local_psi_y, l+1, jj+1, kk+1, 8, 8, 8);
			double Fy_right	= sample3D(local_psi_y, r+1, jj+1, kk+1, 8, 8, 8);
			double Fx_front	= sample3D(local_psi_x, ii+1, jj+1, f+1, 8, 8, 8);
			double Fx_back	= sample3D(local_psi_x, ii+1, jj+1, b+1, 8, 8, 8);
			double Fx_up	= sample3D(local_psi_x, ii+1, u+1, kk+1, 8, 8, 8);
			double Fx_down	= sample3D(local_psi_x, ii+1, d+1, kk+1, 8, 8, 8);
			int write_idx = kk*25 + jj*5 + ii;
		
			local_v[write_idx].x = 0.5*invh*(Fz_up-Fz_down + Fy_front-Fy_back);
			local_v[write_idx].y = 0.5*invh*(Fx_back-Fx_front + Fz_left - Fz_right);
			local_v[write_idx].z = 0.5*invh*(Fy_right-Fy_left + Fx_down - Fx_up);
		}
		__syncthreads();

		//step 5 : for all particles in this cell
		//each particle is assigned a thread.
		if(end[g_blockid]>start[g_blockid])
		{
			int local_np = end[g_blockid]-start[g_blockid];
			int round = local_np/512+1;
			for(int t = 0; t<round; t++)
			{
				int tid1D = kk*64 + jj*8 + ii;
				int pidx = start[g_blockid] + t * 512 + tid1D;
				if(pidx<end[g_blockid])
				{
					double3 v_PM = make_double3(0.0,0.0,0.0);
					for(int kkk=0;kkk<5;kkk++)for(int jjj=0;jjj<5;jjj++)for(int iii=0;iii<5;iii++)
					{
						double3 v = local_v[kkk*25+jjj*5+iii];
						float4 pijk = make_float4((float)i+(float)iii+0.5-2.0,
							(float)j+(float)jjj+0.5-2.0,
							(float)k+(float)kkk+0.5-2.0,
							0.0);
						float4 ppos = pos[pidx];
						ppos.x -= origin.x;
						ppos.y -= origin.y;
						ppos.z -= origin.z;
						double r1 = ppos.x*invh - pijk.x;
						double r2 = ppos.y*invh - pijk.y;
						double r3 = ppos.z*invh - pijk.z;
						double w = shape_function(fabs(r1))*shape_function(fabs(r2))*shape_function(fabs(r3));
						v_PM.x += w*v.x;
						v_PM.y += w*v.y;
						v_PM.z += w*v.z;
					}
					u[pidx] = u[pidx]-v_PM.x;
					v[pidx] = v[pidx]-v_PM.y;
					w[pidx] = w[pidx]-v_PM.z;
				}
			}
		}
	}
}


void BiotSavartPMCorr(uint* start,
					 uint * end,
					 float4 * pos,
					 double * vrho_x,
					 double * vrho_y,
					 double * vrho_z,
					 double * psi_x,
					 double * psi_y,
					 double * psi_z,
					 double * g_u,
					 double * g_v,
					 double * g_w,
					 double * u,
					 double * v,
					 double * w,
					 double cell_h,
					 uint3 gdim,
					 uint3 hdim,
					 int K, 
					 uint num_particles, 
					 double4 origin)
{
	//if(K>=0)
	//{
	//	uint num_gridcells = hdim.x*hdim.y*hdim.z;uint chunck_size = 256;uint num_chuncks = num_gridcells/chunck_size + 1;printf("num chuncks : %d\n", num_chuncks);
	//	for(uint chunck_id = 0; chunck_id<num_chuncks; chunck_id++)
	//	{
	//		dim3 blocks(chunck_size,1,1);dim3 threads(8,8,8); 
	//		BiotSavartPMCorr_kernel<<<blocks, threads>>>(start,end,pos,vrho_x,vrho_y,vrho_z,u,v,w,1.0/cell_h,cell_h,cell_h*cell_h*cell_h,
	//			gdim,hdim,make_uint2(gdim.x*gdim.y, gdim.x),make_uint2(hdim.x*hdim.y, hdim.x),chunck_size,chunck_id,K,num_particles,origin);
	//		getLastCudaError("compute biot-savart subtract grid val failed\n!");
	//	}
	//}
	//if(K>=0)
	//{
	//cudaMemset(psi_x,0,sizeof(double)*gdim.x*gdim.y*gdim.z);cudaMemset(psi_y,0,sizeof(double)*gdim.x*gdim.y*gdim.z);cudaMemset(psi_z,0,sizeof(double)*gdim.x*gdim.y*gdim.z);
	//cudaMemset(g_u,0,sizeof(double)*gdim.x*gdim.y*gdim.z);cudaMemset(g_v,0,sizeof(double)*gdim.x*gdim.y*gdim.z);cudaMemset(g_w,0,sizeof(double)*gdim.x*gdim.y*gdim.z);
	//
	//double3 *dir_XX_host = (double3*)malloc(sizeof(double3)*(2*K+1)*(2*K+1)*(2*K+1)*5*5*5);
	//double h = cell_h;
	//for(int kk=-K;kk<=K;kk++)for(int jj= -K; jj<=K; jj++)for(int ii=-K; ii<=K; ii++)
	//{
	//	double3 X_prime = make_double3((double)ii+0.5,(double)jj+0.5,(double)kk+0.5);
	//			X_prime.x*=h; X_prime.y*=h; X_prime.z*=h;
	//	for(int kkk=-2;kkk<=2;kkk++)for(int jjj= -2; jjj<=2; jjj++)for(int iii=-2; iii<=2; iii++)
	//	{
	//		if(!(kkk==kk && jjj==jj && iii==ii))
	//		{
	//			uint idx_i = (kkk+2)*25+(jjj+2)*5+(iii+2);
	//			uint idx_j = (kk+K)*(2*K+1)*(2*K+1)+(jj+K)*(2*K+1)+(ii+K);
	//			uint idx_read = idx_j*125+idx_i;
	//			double3 X=make_double3((double)iii+0.5,(double)jjj+0.5,(double)kkk+0.5);
	//			X.x*=h; X.y*=h; X.z*=h;
	//			double3 dir_XX = make_double3(0,0,0);
	//			dir_XX.x = 0.5*((1.0/cpu_compute_r(make_double4(X.x+h, X.y, X.z, 0), make_double4(X_prime.x, X_prime.y, X_prime.z, 0)))
	//				-(1.0/cpu_compute_r(make_double4(X.x-h, X.y, X.z, 0), make_double4(X_prime.x, X_prime.y, X_prime.z, 0)))
	//				)/h;

	//			dir_XX.y = 0.5*((1.0/cpu_compute_r(make_double4(X.x, X.y+h, X.z, 0), make_double4(X_prime.x, X_prime.y, X_prime.z, 0)))
	//				-(1.0/cpu_compute_r(make_double4(X.x, X.y-h, X.z, 0), make_double4(X_prime.x, X_prime.y, X_prime.z, 0)))
	//				)/h;

	//			dir_XX.z = 0.5*((1.0/cpu_compute_r(make_double4(X.x, X.y, X.z+h, 0), make_double4(X_prime.x, X_prime.y, X_prime.z, 0)))
	//				-(1.0/cpu_compute_r(make_double4(X.x, X.y, X.z-h, 0), make_double4(X_prime.x, X_prime.y, X_prime.z, 0)))
	//				)/h;

	//			dir_XX_host[idx_read]=make_double3(dir_XX.x,dir_XX.y,dir_XX.z);
	//		}
	//	}
	//}
	//double3 *dir_XX_devc;
	//cudaMalloc((void**)&dir_XX_devc,sizeof(double3)*(2*K+1)*(2*K+1)*(2*K+1)*5*5*5);
	//cudaMemcpy(dir_XX_devc,dir_XX_host,sizeof(double3)*(2*K+1)*(2*K+1)*(2*K+1)*5*5*5,cudaMemcpyHostToDevice);

	//uint threads = 256;
	//uint blocks = num_particles/threads + (!(num_particles%threads)?0:1);
	//BiotSavartPMCorr2_kernel<<<blocks, threads>>>(start, end, pos, vrho_x, vrho_y, vrho_z, u,v,w,cell_h, 1.0/cell_h,dir_XX_devc,gdim, hdim,
	//	make_uint2(gdim.x*gdim.y, gdim.x),make_uint2(hdim.x*hdim.y, hdim.x),K,num_particles, origin);
	//ComputeDiagPsi(vrho_x, vrho_y, vrho_z, psi_x, psi_y, psi_z,gdim.x*gdim.y*gdim.z, cell_h*cell_h);
	//getCurl(psi_x,psi_y,psi_z,g_u,g_v,g_w,gdim.x, gdim.y, gdim.z, cell_h);
	//SubtractDiag(pos, g_u, g_v,g_w,u,v,w,cell_h,gdim, hdim,num_particles, origin);

	//free(dir_XX_host);
	//cudaFree(dir_XX_devc);
	//}
	//

}


__global__ void split_pos_kernel(float4 * pos,
						  double * x,
						  double * y,
						  double * z,
						  double * mass,
						  uint num_particle,
						  uint weighted)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<num_particle)
	{
		x[idx]=pos[idx].x;
		y[idx]=pos[idx].y;
		z[idx]=pos[idx].z;
		if(weighted)
		{
			x[idx]=x[idx]*fabs(mass[idx]);
			y[idx]=y[idx]*fabs(mass[idx]);
			z[idx]=z[idx]*fabs(mass[idx]);
		}
	}
}
void split_pos(float4 * pos,double * x,double * y,double * z,double * mass, uint num_particle, uint weighted)
{
	uint blocks = num_particle/512 + 1;
	split_pos_kernel<<<blocks, 512>>>(pos, x, y, z, mass, num_particle, weighted);
}




//modified from nVidia's reduction code example
template <unsigned int blockSize>
__global__ void
reduce_summation_kernel(double *g_idata, double *g_odata, unsigned int n)
{
    extern __shared__ double sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;

    double mySum = (i < n) ? g_idata[i] : 0;

    if (i + blockSize < n)
	{
		mySum += g_idata[i+blockSize];
	}

    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 256];
        }

        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            sdata[tid] = mySum = mySum + sdata[tid +  64];
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile double *smem = sdata;

        if (blockSize >=  64)
        {
            smem[tid] = mySum = mySum + smem[tid + 32];
        }

        if (blockSize >=  32)
        {
            smem[tid] = mySum = mySum + smem[tid + 16];
        }

        if (blockSize >=  16)
        {
            smem[tid] = mySum = mySum + smem[tid +  8];
        }

        if (blockSize >=   8)
        {
            smem[tid] = mySum = mySum + smem[tid +  4];
        }

        if (blockSize >=   4)
        {
            smem[tid] = mySum = mySum + smem[tid +  2];
        }

        if (blockSize >=   2)
        {
            smem[tid] = mySum = mySum + smem[tid +  1];
        }
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}















void
reduce_summation(int size, int threads, int blocks,
       int whichKernel, double *d_idata, double *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);


	switch (threads)
            {
                case 512:
                    reduce_summation_kernel<512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case 256:
                    reduce_summation_kernel<256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case 128:
                    reduce_summation_kernel<128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case 64:
                    reduce_summation_kernel<64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case 32:
                    reduce_summation_kernel<32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case 16:
                    reduce_summation_kernel<16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case  8:
                    reduce_summation_kernel<8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case  4:
                    reduce_summation_kernel<4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case  2:
                    reduce_summation_kernel<2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case  1:
                    reduce_summation_kernel<1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            }
}


template <unsigned int blockSize>
__global__ void
reduce_Max_kernel(double *g_idata, double *g_odata, unsigned int n)
{
    extern __shared__ double sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;

    double mySum = (i < n) ? g_idata[i] : g_idata[n-1];

    if (i + blockSize < n)
	{
		mySum = max(g_idata[i+blockSize], mySum);
	}

    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = mySum = max(mySum, sdata[tid + 256]);
        }

        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] = mySum = max(mySum , sdata[tid + 128]);
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            sdata[tid] = mySum = max(mySum, sdata[tid +  64]);
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile double *smem = sdata;

        if (blockSize >=  64)
        {
            smem[tid] = mySum = max(mySum, smem[tid + 32]);
        }

        if (blockSize >=  32)
        {
            smem[tid] = mySum = max(mySum, smem[tid + 16]);
        }

        if (blockSize >=  16)
        {
            smem[tid] = mySum = max(mySum, smem[tid +  8]);
        }

        if (blockSize >=   8)
        {
            smem[tid] = mySum = max(mySum, smem[tid +  4]);
        }

        if (blockSize >=   4)
        {
            smem[tid] = mySum = max(mySum, smem[tid +  2]);
        }

        if (blockSize >=   2)
        {
            smem[tid] = mySum = max(mySum, smem[tid +  1]);
        }
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

void
reduce_Max(int size, int threads, int blocks,
       int whichKernel, double *d_idata, double *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);


	switch (threads)
            {
                case 512:
                    reduce_Max_kernel<512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case 256:
                    reduce_Max_kernel<256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case 128:
                    reduce_Max_kernel<128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case 64:
                    reduce_Max_kernel<64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case 32:
                    reduce_Max_kernel<32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case 16:
                    reduce_Max_kernel<16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case  8:
                    reduce_Max_kernel<8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case  4:
                    reduce_Max_kernel<4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case  2:
                    reduce_Max_kernel<2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case  1:
                    reduce_Max_kernel<1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            }
}

template <unsigned int blockSize>
__global__ void
reduce_Min_kernel(double *g_idata, double *g_odata, unsigned int n)
{
    extern __shared__ double sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;

    double mySum = (i < n) ? g_idata[i] : g_idata[n-1];

    if (i + blockSize < n)
	{
		mySum = min(g_idata[i+blockSize], mySum);
	}

    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = mySum = min(mySum, sdata[tid + 256]);
        }

        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] = mySum = min(mySum , sdata[tid + 128]);
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            sdata[tid] = mySum = min(mySum, sdata[tid +  64]);
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile double *smem = sdata;

        if (blockSize >=  64)
        {
            smem[tid] = mySum = min(mySum, smem[tid + 32]);
        }

        if (blockSize >=  32)
        {
            smem[tid] = mySum = min(mySum, smem[tid + 16]);
        }

        if (blockSize >=  16)
        {
            smem[tid] = mySum = min(mySum, smem[tid +  8]);
        }

        if (blockSize >=   8)
        {
            smem[tid] = mySum = min(mySum, smem[tid +  4]);
        }

        if (blockSize >=   4)
        {
            smem[tid] = mySum = min(mySum, smem[tid +  2]);
        }

        if (blockSize >=   2)
        {
            smem[tid] = mySum = min(mySum, smem[tid +  1]);
        }
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

void
reduce_Min(int size, int threads, int blocks,
       int whichKernel, double *d_idata, double *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);


	switch (threads)
            {
                case 512:
                    reduce_Min_kernel<512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case 256:
                    reduce_Min_kernel<256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case 128:
                    reduce_Min_kernel<128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case 64:
                    reduce_Min_kernel<64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case 32:
                    reduce_Min_kernel<32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case 16:
                    reduce_Min_kernel<16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case  8:
                    reduce_Min_kernel<8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case  4:
                    reduce_Min_kernel<4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case  2:
                    reduce_Min_kernel<2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case  1:
                    reduce_Min_kernel<1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            }
}





template <unsigned int blockSize>
__global__ void
reduce_summationabs_kernel(double *g_idata, double *g_odata, unsigned int n)
{
    extern __shared__ double sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;

    double mySum = (i < n) ? fabs(g_idata[i]) : 0;

    if (i + blockSize < n)
	{
		mySum += fabs(g_idata[i+blockSize]);
	}

    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = mySum = mySum + fabs(sdata[tid + 256]);
        }

        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] = mySum = mySum + fabs(sdata[tid + 128]);
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            sdata[tid] = mySum = mySum + fabs(sdata[tid +  64]);
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile double *smem = sdata;

        if (blockSize >=  64)
        {
            smem[tid] = mySum = mySum + fabs(smem[tid + 32]);
        }

        if (blockSize >=  32)
        {
            smem[tid] = mySum = mySum + fabs(smem[tid + 16]);
        }

        if (blockSize >=  16)
        {
            smem[tid] = mySum = mySum + fabs(smem[tid +  8]);
        }

        if (blockSize >=   8)
        {
            smem[tid] = mySum = mySum + fabs(smem[tid +  4]);
        }

        if (blockSize >=   4)
        {
            smem[tid] = mySum = mySum + fabs(smem[tid +  2]);
        }

        if (blockSize >=   2)
        {
            smem[tid] = mySum = mySum + fabs(smem[tid +  1]);
        }
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}





void
reduce_summationabs(int size, int threads, int blocks,
       int whichKernel, double *d_idata, double *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);


	switch (threads)
            {
                case 512:
                    reduce_summationabs_kernel<512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case 256:
                    reduce_summationabs_kernel<256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case 128:
                    reduce_summationabs_kernel<128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case 64:
                    reduce_summationabs_kernel<64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case 32:
                    reduce_summationabs_kernel<32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case 16:
                    reduce_summationabs_kernel<16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case  8:
                    reduce_summationabs_kernel<8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case  4:
                    reduce_summationabs_kernel<4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case  2:
                    reduce_summationabs_kernel<2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
                case  1:
                    reduce_summationabs_kernel<1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            }
}



#endif