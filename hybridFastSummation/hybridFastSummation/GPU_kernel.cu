#ifndef _PARTICLEMESH_KERNEL_CU_
#define _PARTICLEMESH_KERNEL_CU_


// CUDA Runtime
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "reduction_kernel.cu"
typedef unsigned int uint;
double cpu_compute_r(double4 & r1, double4 & r2)
{
	double r_sqr = (r1.x - r2.x)*(r1.x - r2.x) + (r1.y - r2.y)*(r1.y - r2.y) + (r1.z - r2.z)*(r1.z - r2.z);
	if(r_sqr>1e-12)
		return 12.5663706144 * (sqrt(r_sqr));
	else
		return 1e+24;
}
__device__
double my_dot(double3 &a, double3 &b)
{
	return a.x*b.x+a.y*b.y+a.z*b.z;
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
double poly_shape(double r)
{
	double rr=fabs(r);
	
	//double res = 0;
	//if(rr<2)
	//	res = ((5-2*rr)-sqrt(-4*rr*rr+12*rr-7));
	//if(rr<1)
	//	res = ((3-2*rr)+sqrt(1+4*rr-4*rr*rr));
	//res = res*0.125;
	//return res;
	//double rr=fabs(r);
	//
	if(rr<1)
		return 1-rr;
	else
		return 0;
	/*double r1 = 1-rr;
	double r2 = r1*r1;
	double r4 = r2*r2;
	if(rr<1)
		return r4;
	else
		return 0;*/
}
__device__
double interpolate_weights(double3 grid, double3 particle)
{
	return poly_shape(grid.x-particle.x)*poly_shape(grid.y-particle.y)*poly_shape(grid.z-particle.z);
	/*double3 dir = make_double3(grid.x-particle.x,grid.y-particle.y,grid.z-particle.z);
	double r = dir.x*dir.x + dir.y*dir.y + dir.z*dir.z;
	return poly_shape(r);*/
}


__device__
double poly_shape2(double r)
{
	double rr=fabs(r);
	
	
	double r1 = 1-rr;
	double r2 = r1*r1;
	double r4 = r2*r2;
	if(rr<1)
		return r4;
	else
		return 0;
}

__device__
double interpolate_weights2(double3 grid, double3 particle)
{
	/*return poly_shape(grid.x-particle.x)*poly_shape(grid.y-particle.y)*poly_shape(grid.z-particle.z);*/
	double3 dir = make_double3(grid.x-particle.x,grid.y-particle.y,grid.z-particle.z);
	double r = dir.x*dir.x + dir.y*dir.y + dir.z*dir.z;
	return poly_shape2(r);
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
__inline__ __device__
double linearInterp(const double & a, const double & b, const double & t)
{
	return (1-t)*a+t*b;
}
__inline__ __device__
void clamp(int3 &idx, int3 dimm)
{
	idx.x = min(max(0,idx.x), dimm.x-1);idx.y = min(max(0,idx.y),dimm.y-1); idx.z = min(max(0,idx.z),dimm.z-1);
}
__inline__ __device__
void getTriCoeff(const double3 & pos, const double & h, int3 &idx000,double3 & coe, uint4 &bidx, uint4 &tidx, uint dimx,uint dimy, uint dimz)
{
	idx000.x = (int)(floor(pos.x/h - 0.5));
	idx000.y = (int)(floor(pos.y/h - 0.5));
	idx000.z = (int)(floor(pos.z/h - 0.5));

	coe.x = pos.x/h - (double)idx000.x - 0.5;
	coe.y = pos.y/h - (double)idx000.y - 0.5;
	coe.z = pos.z/h - (double)idx000.z - 0.5;

	int3 idx;
	int3 dimm = make_int3(dimx, dimy, dimz);
	idx = idx000;
	clamp(idx, dimm);
	bidx.x = idx.z*dimx*dimy + idx.y*dimx + idx.x;
	idx = idx000;idx.x = idx000.x + 1;
	clamp(idx, dimm);
	bidx.y = idx.z*dimx*dimy + idx.y*dimx + idx.x;
	idx = idx000;idx.z = idx000.z + 1;
	clamp(idx, dimm);
	bidx.z = idx.z*dimx*dimy + idx.y*dimx + idx.x;
	idx = idx000;idx.z = idx000.z + 1;idx.x = idx000.x+1;
	clamp(idx, dimm);
	bidx.w = idx.z*dimx*dimy + idx.y*dimx + idx.x;

	idx = idx000;idx.y = idx000.y+1;
	clamp(idx, dimm);
	tidx.x = idx.z*dimx*dimy + idx.y*dimx + idx.x;
	idx = idx000;idx.y = idx000.y+1;idx.x = idx000.x + 1;
	clamp(idx, dimm);
	tidx.y = idx.z*dimx*dimy + idx.y*dimx + idx.x;
	idx = idx000;idx.y = idx000.y+1;idx.z = idx000.z + 1;
	clamp(idx, dimm);
	tidx.z = idx.z*dimx*dimy + idx.y*dimx + idx.x;
	idx = idx000;idx.y = idx000.y+1;idx.z = idx000.z + 1;idx.x = idx000.x+1;
	clamp(idx, dimm);
	tidx.w = idx.z*dimx*dimy + idx.y*dimx + idx.x;


}
__inline__ __device__
void triInterp(double3 &coe,double & out_v, double4 & b, double4 & t)
{
	out_v = linearInterp(
			linearInterp(linearInterp(b.x,b.y,coe.x), linearInterp(b.z,b.w,coe.x),coe.z),
			linearInterp(linearInterp(t.x,t.y,coe.x), linearInterp(t.z,t.w,coe.x),coe.z),
			coe.y);

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
ParticleToMesh_kernel(const uint * start,
					  const uint * end,
					  const float4 * pos,
					  const double * mass,
					  double * gridval,
					  uint3 griddim,
					  uint3 hashdim,
					  double invh,
					  double invny,
					  uint2 gstride,
					  uint2 hstride,
					  uint num_particle,
					  double4 origin)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int k = truncf((blockIdx.y*blockDim.y + threadIdx.y)*invny);
	int j = (blockIdx.y*blockDim.y + threadIdx.y)-__umul24(griddim.y,k);
	
	
	if(i<griddim.x && j<griddim.y && k<griddim.z)
	{
		double sum = 0;
		float4 pijk = make_float4(((double)i+0.5),((double)j+0.5),((double)k+0.5),0);

		for(int kk=k-2;kk<=k+2;kk++)for(int jj= j-2; jj<=j+2; jj++)for(int ii=i-2; ii<=i+2; ii++)
		{
			if(kk>=0&&kk<hashdim.z && jj>=0&&jj<hashdim.y && ii>=0&&ii<hashdim.x)
			{
				uint read_idx = kk*hstride.x + jj*hstride.y + ii;
				for(uint p = start[read_idx]; p<end[read_idx]; p++)
				{
					if(p<num_particle)
					{
						float4 ppos = pos[p];
						ppos.x -= origin.x;
						ppos.y -= origin.y;
						ppos.z -= origin.z;
						double r1 = (double)ppos.x * invh - pijk.x;
						double r2 = (double)ppos.y * invh - pijk.y;
						double r3 = (double)ppos.z * invh - pijk.z;
						double w = shape_function_mass(r1)*shape_function_mass(r2)*shape_function_mass(r3);
						double rho = w * mass[p];
						sum += rho;

					}
				}
			}//end if
		}//end for

		uint write_idx = k*gstride.x + j*gstride.y + i;
		gridval[write_idx] = invh*invh*invh*sum;
	}//end if
}
void ParticleToMesh(uint * start,
					uint * end,
					float4 * pos,
					double * mass,
					double cell_h,
					double * gridval,
					uint3 griddim,
					uint3 hashdim,
					uint num_particle, 
					double4 origin)
{
	dim3 threads(16,16);
	dim3 blocks(griddim.x/16 + (!(griddim.x%16)?0:1), griddim.z*griddim.y/16 + (!(griddim.z*griddim.y%16)?0:1));

	ParticleToMesh_kernel<<<blocks,threads>>>(start, end, pos, mass, gridval, 
		griddim, hashdim, 1.0/cell_h, 1.0/(double)griddim.y, 
		make_uint2(griddim.x*griddim.y, griddim.x), 
		make_uint2(hashdim.x*hashdim.y, hashdim.x), 
		num_particle,
		origin);
	getLastCudaError("get residual failed!\n");

}



__global__ void
ParticleToMeshGaussian_kernel(const uint * start,
					  const uint * end,
					  const float4 * pos,
					  const double * mass,
					  double * gridval,
					  uint3 griddim,
					  uint3 hashdim,
					  double invh,
					  double invny,
					  uint2 gstride,
					  uint2 hstride,
					  uint num_particle,
					  double4 origin)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int k = truncf((blockIdx.y*blockDim.y + threadIdx.y)*invny);
	int j = (blockIdx.y*blockDim.y + threadIdx.y)-__umul24(griddim.y,k);
	
	uint p = 0;
	if(i<griddim.x && j<griddim.y && k<griddim.z)
	{
		double sum = 0;
		float4 pijk = make_float4(((double)i+0.5),((double)j+0.5),((double)k+0.5),0);

		double total_weight = 0.0;
		for(int kk=k-3;kk<=k+3;kk++)for(int jj= j-3; jj<=j+3; jj++)for(int ii=i-3; ii<=i+3; ii++)
		{
			if(kk>=0&&kk<hashdim.z && jj>=0&&jj<hashdim.y && ii>=0&&ii<hashdim.x)
			{
				uint read_idx = kk*hstride.x + jj*hstride.y + ii;
				if(end[read_idx]-start[read_idx]>=0)
				{
					for(uint p = start[read_idx]; p<end[read_idx]; p++)
					{
						if(p<num_particle)
						{
							float4 ppos = pos[p];
							ppos.x -= origin.x;
							ppos.y -= origin.y;
							ppos.z -= origin.z;
							double r1 = (double)ppos.x * invh - pijk.x;
							double r2 = (double)ppos.y * invh - pijk.y;
							double r3 = (double)ppos.z * invh - pijk.z;

							double dir_sqr = r1*r1 + r2*r2 + r3*r3;
							double omg = 0.5;
							double w = exp(-dir_sqr/(2*omg*omg));
							double rho = w * mass[p];
							total_weight += w;
							sum += rho;

						}
					}
				}
				else
				{
					float4 ppos;
					ppos.x = (double)ii+0.5;
					ppos.y = (double)jj+0.5;
					ppos.z = (double)kk+0.5;
					double r1 = (double)ppos.x - pijk.x;
					double r2 = (double)ppos.y - pijk.y;
					double r3 = (double)ppos.z - pijk.z;
					double dir_sqr = r1*r1 + r2*r2 + r3*r3;
					double omg = 0.5;
					double w = exp(-dir_sqr/(2*omg*omg));
					double rho = w * mass[p];
					total_weight += w;
				}
			}//end if
		}//end for
		if(total_weight<1e-12) total_weight = 1e+24;
		uint write_idx = k*gstride.x + j*gstride.y + i;
		gridval[write_idx] = sum/total_weight;
	}//end if
}
void ParticleToMeshGaussian(uint * start,
					uint * end,
					float4 * pos,
					double * mass,
					double cell_h,
					double * gridval,
					uint3 griddim,
					uint3 hashdim,
					uint num_particle, 
					double4 origin)
{
	dim3 threads(16,16);
	dim3 blocks(griddim.x/16 + (!(griddim.x%16)?0:1), griddim.z*griddim.y/16 + (!(griddim.z*griddim.y%16)?0:1));

	ParticleToMeshGaussian_kernel<<<blocks,threads>>>(start, end, pos, mass, gridval, 
		griddim, hashdim, 1.0/cell_h, 1.0/(double)griddim.y, 
		make_uint2(griddim.x*griddim.y, griddim.x), 
		make_uint2(hashdim.x*hashdim.y, hashdim.x), 
		num_particle,
		origin);
	getLastCudaError("get residual failed!\n");

}


__global__ void compute_divergence_kernel(double* u, double *v, double * w, 
										  double* div,
										  double invh,
											double invny,
											uint dimx,
											uint dimy,
											uint dimz)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int k = truncf((blockIdx.y*blockDim.y + threadIdx.y)*invny);
	int j = (blockIdx.y*blockDim.y + threadIdx.y)-__umul24(dimy,k);

	if(i<dimx && j<dimy && k <dimz)
	{
		int im1 = (i+dimx-1)%dimx;
		int ip1 = (i+1)%dimx;
		int jm1 = (j+dimy-1)%dimy;
		int jp1 = (j+1)%dimy;
		int km1 = (k-1+dimz)%dimz;;
		int kp1 = (k+1)%dimz;
		int slice = dimx*dimy;
		int stride = dimx;
		int idx_c = k*slice + j*stride + i;
		int idx_l = k*slice + j*stride + im1;
		int idx_r = k*slice + j*stride + ip1;
		int idx_t = k*slice + jp1*stride + i;
		int idx_d = k*slice + jm1*stride + i;
		int idx_f = km1*slice + j*stride + i;
		int idx_b = kp1*slice + j*stride + i;
		double t = v[idx_t];
		double d = v[idx_d];
		double l = u[idx_l];
		double r = u[idx_r];
		double f = w[idx_f];
		double b = w[idx_b];
		div[idx_c] += -(r - l)*0.5*invh;
		div[idx_c] += -(t - d)*0.5*invh;
		div[idx_c] += -(b - f)*0.5*invh;
	}
}

void computeDivergence(double* u, double *v, double * w, 
										  double* f,
										  double cell_h,
											uint dimx,
											uint dimy,
											uint dimz)
{
	dim3 threads(16,16);
	dim3 blocks(dimx/16 + (!(dimx%16)?0:1), dimz*dimy/16 + (!(dimz*dimy%16)?0:1));
	compute_divergence_kernel<<<blocks, threads>>>(u,v,w,f,1.0/cell_h, 1.0/(double)dimy, dimx, dimy, dimy);
}


__global__ void
MeshToParitcle_kernel(const float4 * pos,
					  const double * mass,
					  const double3 * gforce,
					  double3 * pforce,
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
		double3 sum=make_double3(0,0,0);
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
				double3 f = gforce[read_idx];
				float4 pijk = make_float4(((double)ii+0.5),((double)jj+0.5),((double)kk+0.5),0);
				double r1 = (double)ppos.x * invh - pijk.x;
				double r2 = (double)ppos.y * invh - pijk.y;
				double r3 = (double)ppos.z * invh - pijk.z;
				double w = shape_function(fabs(r1))*shape_function(fabs(r2))*shape_function(fabs(r3));
				sum.x += f.x * w;
				sum.y += f.y * w;
				sum.z += f.z * w;
			}
		}
		pforce[idx] = make_double3(sum.x, sum.y, sum.z);
		
	}
}

void ParticleGetForceFromMesh(float4 * pos,
					  const double * mass,
					  const double3 * gforce,
					  double3 * pforce,
					  double cell_h,
					  uint3 gdim,
					  uint3 hdim,
					  uint num_particles,
					  double4 origin)
{
	uint threads = 256;
	uint blocks = num_particles/threads + (!(num_particles%threads)?0:1);
	MeshToParitcle_kernel<<<blocks, threads>>>(pos, mass, gforce, pforce, 1.0/cell_h, gdim, hdim, 
		make_uint2(gdim.x*gdim.y, gdim.x), 
		make_uint2(hdim.x*hdim.y, hdim.x),
		num_particles,
		origin);
	getLastCudaError("get force failed!\n");


}

__global__ void 
compute_rhs_kernel(double * gridval,
			uint num_cells,
			double h2,
			double G)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<num_cells)
	{
		double v = gridval[idx];
		gridval[idx] = v*h2*G;
	}
}
void ComputeRHS(double * gridval, double h_sqr, double G, uint num_cell)
{
	uint threads = 256;
	uint blocks = num_cell/threads + (!(num_cell%threads)?0:1);
	compute_rhs_kernel<<<blocks, threads>>>(gridval, num_cell, h_sqr, G);
	getLastCudaError("compute RHS failed\n!");
}


__global__ void
compute_gradient_kernel(const double * phi,
						double3 * dphi,
						double direction, 
						double invh,
						double invny,
						uint dimx,
						uint dimy,
						uint dimz)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int k = truncf((blockIdx.y*blockDim.y + threadIdx.y)*invny);
	int j = (blockIdx.y*blockDim.y + threadIdx.y)-__umul24(dimy,k);
	
	if(i<dimx && j<dimy && k <dimz)
	{
		int im1 = (i+dimx-1)%dimx;
		int ip1 = (i+1)%dimx;
		int jm1 = (j+dimy-1)%dimy;
		int jp1 = (j+1)%dimy;
		int km1 = (k-1+dimz)%dimz;;
		int kp1 = (k+1)%dimz;
		int slice = dimx*dimy;
		int stride = dimx;
		int idx_c = k*slice + j*stride + i;
		int idx_l = k*slice + j*stride + im1;
		int idx_r = k*slice + j*stride + ip1;
		int idx_t = k*slice + jp1*stride + i;
		int idx_d = k*slice + jm1*stride + i;
		int idx_f = km1*slice + j*stride + i;
		int idx_b = kp1*slice + j*stride + i;
		double t = phi[idx_t];
		double d = phi[idx_d];
		double l = phi[idx_l];
		double r = phi[idx_r];
		double f = phi[idx_f];
		double b = phi[idx_b];
		dphi[idx_c].x = direction*(r - l)*0.5*invh;
		dphi[idx_c].y = direction*(t - d)*0.5*invh;
		dphi[idx_c].z = direction*(b - f)*0.5*invh;
	}
}

void ComputeGradient(double* phi,
					 double3 * dphi,
					 double cell_h,
					 double direction,
					 uint dimx, 
					 uint dimy, 
					 uint dimz)
{
	dim3 threads(16,16);
	dim3 blocks(dimx/16 + (!(dimx%16)?0:1), dimz*dimy/16 + (!(dimz*dimy%16)?0:1));
	compute_gradient_kernel<<<blocks, threads>>>(phi, dphi, direction, 1.0/cell_h, 1.0/(double)dimy, dimx, dimy, dimy);
	getLastCudaError("compute gradient failed\n!");
}

__global__ void 
local_correct_kernel(const uint * start,
					 const uint * end,
					 const float4 * pos,
					 const double * mass,
					 const double * gridval,
					 double3 * pforce,
					 double invh,
					 double h,
					 double G,
					 double gridscale,
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
		double3 force_PM = pforce[idx];
		float4 ppos = pos[idx];
		ppos.x -= origin.x;
		ppos.y -= origin.y;
		ppos.z -= origin.z;
		//compute the grid idx it belongs to
		int i = floor(ppos.x*invh);
		int j = floor(ppos.y*invh);
		int k = floor(ppos.z*invh);
		
		double3 force_particle = make_double3(0,0,0);
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
						
						double3 dir = make_double3(ppos2.x-ppos.x, ppos2.y-ppos.y, ppos2.z-ppos.z);
						double r3 = compute_r3(dir);
						force_particle.x += G*dir.x/r3*mass[p];
						force_particle.y += G*dir.y/r3*mass[p];
						force_particle.z += G*dir.z/r3*mass[p];
					}
				}
			}
		}
		
		pforce[idx] = make_double3(force_PM.x+force_particle.x , 
			force_PM.y+force_particle.y,
			force_PM.z+force_particle.z );
		
	}
}

__global__ void
subtractPM_kernel2(const uint * start,
					 const uint * end,
					 const float4 * pos,
					 const double * mass,
					 const double * gridval,
					 double3 * pforce,
					 double invh,
					 double h,
					 double G,
					 double gridscale,
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
		double3 force_PM = pforce[idx];
		float4 ppos = pos[idx];
		ppos.x -= origin.x;
		ppos.y -= origin.y;
		ppos.z -= origin.z;
		//compute the grid idx it belongs to
		int i = floor(ppos.x*invh);
		int j = floor(ppos.y*invh);
		int k = floor(ppos.z*invh);
		double3 force_mesh = make_double3(0,0,0);
		for(int kk=k-K;kk<=k+K;kk++)for(int jj= j-K; jj<=j+K; jj++)for(int ii=i-K; ii<=i+K; ii++)
		{
			uint read_idx = kk*gdim.x*gdim.y+jj*gdim.x+ii;
			double3 X_prime = make_double3((double)ii+0.5,(double)jj+0.5,(double)kk+0.5);
			X_prime.x*=h; X_prime.y*=h; X_prime.z*=h;
			double mass_prime = G*gridval[read_idx]*gridscale;
			double3 dir_x=make_double3(0,0,0);
			for(int kkk=k-2;kkk<=k+2;kkk++)for(int jjj= j-2; jjj<=j+2; jjj++)for(int iii=i-2; iii<=i+2; iii++)
			{
				if(!(kkk==kk && jjj==jj && iii==ii))
				{
					double3 X=make_double3((double)iii+0.5,(double)jjj+0.5,(double)kkk+0.5);
					double3 diff = make_double3(ppos.x*invh - X.x, ppos.y*invh - X.y, ppos.z*invh - X.z);					
					double weight = shape_function(fabs(diff.x))*shape_function(fabs(diff.y))*shape_function(fabs(diff.z));

					X.x*=h; X.y*=h; X.z*=h;
					double3 dir_XX = make_double3(X_prime.x-X.x,X_prime.y-X.y,X_prime.z-X.z);
					//double r3 = compute_r3(dir_XX);
					dir_XX.x = 0.5*((1.0/compute_r(make_double4(X.x+h, X.y, X.z, 0), make_double4(X_prime.x, X_prime.y, X_prime.z, 0)))
						-(1.0/compute_r(make_double4(X.x-h, X.y, X.z, 0), make_double4(X_prime.x, X_prime.y, X_prime.z, 0)))
						)*invh;

					dir_XX.y = 0.5*((1.0/compute_r(make_double4(X.x, X.y+h, X.z, 0), make_double4(X_prime.x, X_prime.y, X_prime.z, 0)))
						-(1.0/compute_r(make_double4(X.x, X.y-h, X.z, 0), make_double4(X_prime.x, X_prime.y, X_prime.z, 0)))
						)*invh;

					dir_XX.z = 0.5*((1.0/compute_r(make_double4(X.x, X.y, X.z+h, 0), make_double4(X_prime.x, X_prime.y, X_prime.z, 0)))
						-(1.0/compute_r(make_double4(X.x, X.y, X.z-h, 0), make_double4(X_prime.x, X_prime.y, X_prime.z, 0)))
						)*invh;
					
					dir_x.x += (dir_XX.x)*weight;
					dir_x.y += (dir_XX.y)*weight;
					dir_x.z += (dir_XX.z)*weight;
				}
			}
			force_mesh.x += dir_x.x*mass_prime;
			force_mesh.y += dir_x.y*mass_prime;
			force_mesh.z += dir_x.z*mass_prime;
			
		}
		pforce[idx] = make_double3(force_PM.x - force_mesh.x, 
			force_PM.y - force_mesh.y,
			force_PM.z - force_mesh.z);

	}
}



__global__ void 
Potential_PPCorrMN_kernel(const uint * start,
					 const uint * end,
					 const float4 * pos_eval,
					 const float4 * pos_mass,
					 const double * mass,
					 double3 * pforce,
					 double invh,
					 double h,
					 double G,//-1 for gravity, 1 for single layer potential
					 uint3 gdim,
					  uint3 hdim,
					  uint2 gstride,
					  uint2 hstride,
					  int K,
					  uint M,
					  uint N,
					  double4 origin)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	uint p=0;
	double correct_radius = ((double)K+0.5)/invh;
	if(idx<M)
	{
		double3 force_PM = pforce[idx];
		float4 ppos = pos_eval[idx];
		ppos.x -= origin.x;
		ppos.y -= origin.y;
		ppos.z -= origin.z;
		//compute the grid idx it belongs to
		int i = floor(ppos.x*invh);
		int j = floor(ppos.y*invh);
		int k = floor(ppos.z*invh);
		
		double a = 0.05;
		double3 force_particle = make_double3(0,0,0);
		double3 grid_center = make_double3((double)i+0.5, (double)j+0.5, (double)k+0.5);
		double3 pos=make_double3(ppos.x*invh, ppos.y*invh, ppos.z*invh);
		int min_ii, max_ii, min_jj, max_jj, min_kk, max_kk;
		if(pos.x>grid_center.x)
		{
			min_ii = i-K-1, max_ii = i + K + 1;
		}
		else
		{
			min_ii = i-K-1, max_ii = i + K+ 1;
		}
		if(pos.y>grid_center.y)
		{
			min_jj = j-K-1, max_jj = j + K + 1;
		}
		else
		{
			min_jj = j-K-1, max_jj = j + K+ 1;
		}
		if(pos.z>grid_center.z)
		{
			min_kk = k-K-1, max_kk = k + K + 1;
		}
		else
		{
			min_kk = k-K-1, max_kk = k + K+ 1;
		}
		for(int kk=min_kk;kk<=max_kk;kk++)for(int jj = min_jj; jj<=max_jj; jj++)for(int ii=min_ii; ii<=max_ii; ii++)
		{
			if(kk>=0&&kk<hdim.z && jj>=0&&jj<hdim.y && ii>=0&&ii<hdim.x)
			{
				uint read_idx = kk*hstride.x + jj*hstride.y + ii;
				for(p = start[read_idx]; p<end[read_idx]; p++)
				{
					if(p<N)
					{
						float4 ppos2 = pos_mass[p];
						ppos2.x -= origin.x;
						ppos2.y -= origin.y;
						ppos2.z -= origin.z;
						if(fabs(ppos2.x-ppos.x)<correct_radius
							&&fabs(ppos2.y-ppos.y)<correct_radius
							&&fabs(ppos2.z-ppos.z)<correct_radius
							)
						{
							double3 dir = make_double3(ppos2.x-ppos.x, ppos2.y-ppos.y, ppos2.z-ppos.z);


							double r1 = sqrt(dir.x*dir.x+dir.y*dir.y+dir.z*dir.z);
							double r3 = 0;
							if(r1>a)
								r3 = 12.5663706144 * r1*r1*r1;
							else
							{
								if(r1<1e-12)
								{
									r3 = 1e+14;
								}
								else
								{
									r3 = 12.5663706144 * a * a * a;
								}
							}
							
							force_particle.x += G*dir.x/r3*mass[p];
							force_particle.y += G*dir.y/r3*mass[p];
							force_particle.z += G*dir.z/r3*mass[p];
						}
					}
				}
			}
		}
		
		pforce[idx] = make_double3(force_PM.x+force_particle.x , 
			force_PM.y+force_particle.y,
			force_PM.z+force_particle.z );
		
	}
}





		

void Potential_PPCorrMN(uint * start,
					 uint * end,
					 float4 * pos_eval,
					 float4 * pos_mass,
					 double * mass,
					 double3 * pforce,
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
					  double4 origin)
{
	if(K>=0)
	{
		uint threads = 256;
		uint blocks = M/threads + (!(M%threads)?0:1);
		Potential_PPCorrMN_kernel<<<blocks, threads>>>(start, end, 
			pos_eval,
			pos_mass,
			mass, 
			pforce, 
			1.0/h,
			h,
			direction*G,
			gdim, hdim,
			make_uint2(gdim.x*gdim.y, gdim.x),
			make_uint2(hdim.x*hdim.y, hdim.x),
			K,
			M,
			N,
			origin);
	}
}


__global__ void
subtractPMdiag_kernel(const float4 * pos,
					  const double * mass,
					  const double3 * gforce,
					  double3 * pforce,
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
		double3 sum=make_double3(0,0,0);
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
				double3 f = gforce[read_idx];
				float4 pijk = make_float4(((double)ii+0.5),((double)jj+0.5),((double)kk+0.5),0);
				double r1 = (double)ppos.x * invh - pijk.x;
				double r2 = (double)ppos.y * invh - pijk.y;
				double r3 = (double)ppos.z * invh - pijk.z;
				double w = shape_function(fabs(r1))*shape_function(fabs(r2))*shape_function(fabs(r3));
				sum.x += f.x * w;
				sum.y += f.y * w;
				sum.z += f.z * w;
			}
		}
		pforce[idx].x -= sum.x;
		pforce[idx].y -= sum.y;
		pforce[idx].z -= sum.z;
		
	}
}

__global__ void
compute_diag_phi_kernel(double * gridval,
						double * phi,
			uint num_cells,
			double h2,
			double G)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<num_cells)
	{
		double v = gridval[idx];
		phi[idx] = 0.250580605913965*v*h2*G;
	}
}

void ComputeDiagPhi(double * gridval,double * phi, double h_sqr, double G, uint num_cell)
{
	uint threads = 256;
	uint blocks = num_cell/threads + (!(num_cell%threads)?0:1);
	compute_diag_phi_kernel<<<blocks, threads>>>(gridval,phi, num_cell, h_sqr, G);
	getLastCudaError("compute diagPhi failed\n!");
}


__global__ 
void PotentialComputeFarFieldStep1_kernel(const uint * start,
					 const uint * end,
					 const float4 * pos,
					 double * g_rho,
					 double3 * g_force,
					 double3 * force,
					 double G,//-1 for gravity, 1 for single layer potential
					 double h,
					 double invh,
					 double3 *dir_XX_array,
					 int3 *cell_idx_array,
					 int K,
					 uint dimx,
					 uint dimy,
					 uint dimz,
					 double invny,
					 uint num_particle, 
					 double4 origin)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int k = truncf((blockIdx.y*blockDim.y + threadIdx.y)*invny);
	int j = (blockIdx.y*blockDim.y + threadIdx.y)-__umul24(dimy,k);
	uint p=0;
	if(i<dimx && j<dimy && k<dimz)
	{
		int table_idx = k*dimx*dimy + j*dimx + i;
		
			for(int neighbor_idx = 0; neighbor_idx<1; neighbor_idx++)
			{// compute near field velocity for each of its 6 neighbor cells;
				int3 cell_idx = cell_idx_array[neighbor_idx];
				uint write_idx = table_idx*1 + neighbor_idx;
				int kkk = k+cell_idx.z;
				int jjj = j+cell_idx.y;
				int iii = i+cell_idx.x;
				double3 force_PM = make_double3(0,0,0);
				double3 force_mesh = make_double3(0,0,0);
				if(kkk>=0&&kkk<dimz && jjj>=0&&jjj<dimy && iii>=0&&iii<dimx)
				{
					uint force_read_idx = kkk*dimx*dimy + jjj*dimx + iii;
					force_PM = g_force[force_read_idx];
					for(int kk=k-K;kk<=k+K;kk++)for(int jj= j-K; jj<=j+K; jj++)for(int ii=i-K; ii<=i+K; ii++)
					{
						double mass = 0;
						double3 dir_x= make_double3(0,0,0);
						if(kk>=0&&kk<dimz && jj>=0&&jj<dimy && ii>=0&&ii<dimx)
						{
							uint mass_read_idx = kk*dimx*dimy + jj*dimx + ii;
							double v = h*h*h;
							mass = G*g_rho[mass_read_idx]*v;//read the mass

							
							if(!(kkk==kk && jjj==jj && iii==ii))
							{
								uint idx_i = (kkk-k+2)*25+(jjj-j+2)*5+(iii-i+2);
								uint idx_j = (kk-k+K)*(2*K+1)*(2*K+1)+(jj-j+K)*(2*K+1)+(ii-i+K);
								uint idx_read = idx_j*125+idx_i;
								dir_x=dir_XX_array[idx_read];
							}
						}
						double3 ftemp = make_double3(dir_x.x*mass, dir_x.y*mass, dir_x.z*mass);
						force_mesh.x += ftemp.x;
						force_mesh.y += ftemp.y;
						force_mesh.z += ftemp.z;
					}
				}
				force[write_idx] = make_double3(force_PM.x-force_mesh.x,
												force_PM.y-force_mesh.y,
												force_PM.z-force_mesh.z);
			}
		
	}
}

__global__
void PotentialSubtract_diagonal_grid_kernel(
					 double3 * g_f,
					 double3 * f,
					 int3 *cell_idx_array,
					 uint dimx,
					 uint dimy,
					 uint dimz,
					 double invny)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int k = truncf((blockIdx.y*blockDim.y + threadIdx.y)*invny);
	int j = (blockIdx.y*blockDim.y + threadIdx.y)-__umul24(dimy,k);
	if(k>=0&&k<dimz && j>=0&&j<dimy &&i>=0&&i<dimx)
	{
		int table_idx = k*dimx*dimy + j*dimx + i;
		for(int neighbor_idx = 0; neighbor_idx<1; neighbor_idx++)
		{
			int3 cell_idx = cell_idx_array[neighbor_idx];
			uint write_idx = table_idx*1 + neighbor_idx;
			double3 vel = make_double3(0,0,0);
			int kkk = k+cell_idx.z;
			int jjj = j+cell_idx.y;
			int iii = i+cell_idx.x;
			if(kkk>=0&&kkk<dimz && jjj>=0&&jjj<dimy && iii>=0&&iii<dimx)
			{
				uint vel_idx = kkk*dimx*dimy+jjj*dimy+iii;
				vel = g_f[vel_idx];
			}
			f[write_idx].x -= vel.x;
			f[write_idx].y -= vel.y;
			f[write_idx].z -= vel.z;
		}
	}
}

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
					 double4 origin)
{
	
	if(K>=0)
	{
	
	
	double3 *dir_XX_host = (double3*)malloc(sizeof(double3)*(2*K+1)*(2*K+1)*(2*K+1)*5*5*5);
	double h = cell_h;
	for(int kk=-K;kk<=K;kk++)for(int jj= -K; jj<=K; jj++)for(int ii=-K; ii<=K; ii++)
	{
		double3 X_prime = make_double3((double)ii+0.5,(double)jj+0.5,(double)kk+0.5);
				X_prime.x*=h; X_prime.y*=h; X_prime.z*=h;
		for(int kkk=-2;kkk<=2;kkk++)for(int jjj= -2; jjj<=2; jjj++)for(int iii=-2; iii<=2; iii++)
		{
			if(!(kkk==kk && jjj==jj && iii==ii))
			{
				uint idx_i = (kkk+2)*25+(jjj+2)*5+(iii+2);
				uint idx_j = (kk+K)*(2*K+1)*(2*K+1)+(jj+K)*(2*K+1)+(ii+K);
				uint idx_read = idx_j*125+idx_i;
				double3 X=make_double3((double)iii+0.5,(double)jjj+0.5,(double)kkk+0.5);
				X.x*=h; X.y*=h; X.z*=h;
				double3 dir_XX = make_double3(0,0,0);
				dir_XX.x = 0.5*((1.0/cpu_compute_r(make_double4(X.x+h, X.y, X.z, 0), make_double4(X_prime.x, X_prime.y, X_prime.z, 0)))
					-(1.0/cpu_compute_r(make_double4(X.x-h, X.y, X.z, 0), make_double4(X_prime.x, X_prime.y, X_prime.z, 0)))
					)/h;

				dir_XX.y = 0.5*((1.0/cpu_compute_r(make_double4(X.x, X.y+h, X.z, 0), make_double4(X_prime.x, X_prime.y, X_prime.z, 0)))
					-(1.0/cpu_compute_r(make_double4(X.x, X.y-h, X.z, 0), make_double4(X_prime.x, X_prime.y, X_prime.z, 0)))
					)/h;

				dir_XX.z = 0.5*((1.0/cpu_compute_r(make_double4(X.x, X.y, X.z+h, 0), make_double4(X_prime.x, X_prime.y, X_prime.z, 0)))
					-(1.0/cpu_compute_r(make_double4(X.x, X.y, X.z-h, 0), make_double4(X_prime.x, X_prime.y, X_prime.z, 0)))
					)/h;

				dir_XX_host[idx_read]=make_double3(dir_XX.x,dir_XX.y,dir_XX.z);
			}
		}
	}
	int3 *cell_idx_host = (int3*)malloc(sizeof(int3)*6);
	cell_idx_host[0]=make_int3(0,0,0);
	cell_idx_host[1]=make_int3( 1,0,0);
	cell_idx_host[2]=make_int3(0,-1,0);
	cell_idx_host[3]=make_int3(0, 1,0);
	cell_idx_host[4]=make_int3(0,0,-1);
	cell_idx_host[5]=make_int3(0,0, 1);
	int3 *cell_idx_devc;
	cudaMalloc((void**)&cell_idx_devc, sizeof(int3)*6);
	cudaMemcpy(cell_idx_devc,cell_idx_host,sizeof(int3)*6,cudaMemcpyHostToDevice);
	double3 *dir_XX_devc;
	cudaMalloc((void**)&dir_XX_devc,sizeof(double3)*(2*K+1)*(2*K+1)*(2*K+1)*5*5*5);
	cudaMemcpy(dir_XX_devc,dir_XX_host,sizeof(double3)*(2*K+1)*(2*K+1)*(2*K+1)*5*5*5,cudaMemcpyHostToDevice);
	dim3 threads(16,16);
	dim3 blocks(gdim.x/16 + (!(gdim.x%16)?0:1), gdim.z*gdim.y/16 + (!(gdim.z*gdim.y%16)?0:1));
	PotentialComputeFarFieldStep1_kernel<<<blocks,threads>>>(start,end,
		pos,
		gridval,
		gforce,
		pforce,
		G*direction,
		cell_h,
		1.0/cell_h,
		dir_XX_devc,
		cell_idx_devc,
		K,
		gdim.x,
		gdim.y,
		gdim.z,
		1.0/(double)gdim.y,
		num_particles, 
		origin);
	
	cudaMemset(phi,0,sizeof(double)*gdim.x*gdim.y*gdim.z);
	cudaMemset(gforce,0,sizeof(double3)*gdim.x*gdim.y*gdim.z);
	ComputeDiagPhi(gridval, phi,cell_h*cell_h,G,gdim.x*gdim.y*gdim.z);
	ComputeGradient(phi,gforce,cell_h, direction, gdim.x, gdim.y, gdim.z);
	dim3 threads2(16,16);
	dim3 blocks2(gdim.x/16 + (!(gdim.x%16)?0:1), gdim.z*gdim.y/16 + (!(gdim.z*gdim.y%16)?0:1));
	PotentialSubtract_diagonal_grid_kernel<<<blocks, threads>>>(gforce,pforce,cell_idx_devc,gdim.x,gdim.y,gdim.z,1.0/(double)gdim.y);

	free(dir_XX_host);
	cudaFree(dir_XX_devc);
	free(cell_idx_host);
	cudaFree(cell_idx_devc);
	}
}

__device__ 
double compute_G_ij(int i1, int j1, int k1, int i2, int j2, int k2)
{
	double r=0;
	if(i1==i2 && j1==j2 && k1==k2)
	{
		r = 0.25;
	}
	else
	{
		double x = (double)i2-(double)i1;
		double y = (double)j2-(double)j1;
		double z = (double)k2-(double)k1;
		double d = sqrt( x*x + y*y + z*z);
		r = 1.0/(12.5663706144*d);
	}
	return r;
}

__global__ 
void PotentialComputeFarFieldScalarStep1_kernel(
					 double * g_rho,
					 double * g_phi,
					 double G,//-1 for gravity, 1 for single layer potential
					 double h,
					 double invh,
					 int K,
					 uint dimx,
					 uint dimy,
					 uint dimz,
					 double invny)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int k = truncf((blockIdx.y*blockDim.y + threadIdx.y)*invny);
	int j = (blockIdx.y*blockDim.y + threadIdx.y)-__umul24(dimy,k);
	uint p=0;
	if(i<dimx && j<dimy && k<dimz)
	{
		// compute near field velocity for each of its 6 neighbor cells;
		
		
		
		double summation = 0.0;
		
		uint phi_read_idx = k*dimx*dimy + j*dimx + i;
		double phi = g_phi[phi_read_idx];
		
		for(int kk=k-K;kk<=k+K;kk++)for(int jj= j-K; jj<=j+K; jj++)for(int ii=i-K; ii<=i+K; ii++)
		{
			
			if(kk>=0&&kk<dimz && jj>=0&&jj<dimy && ii>=0&&ii<dimx)
			{
				uint rho_read_idx = kk*dimx*dimy + jj*dimx + ii;
				double v = h*h;
				double mass = G*g_rho[rho_read_idx]*v;//read the mass

				double w = compute_G_ij(i,j,k,ii,jj,kk);
				summation += w * mass;
				
			}

		}
		
		g_phi[phi_read_idx] = phi - summation;
		
		
	}
}

void PotentialComputeScalarFarField(
					 double * mass,
					 double * gridval,
					 double * phi,
					 double cell_h,
					 double G,
					 double direction,//-1 gravity, 1 for single layer potential
					 uint3 gdim,
					 uint3 hdim,
					 int K)
{
	
	if(K>=0)
	{
		dim3 threads(16,16);
		dim3 blocks(gdim.x/16 + (!(gdim.x%16)?0:1), gdim.z*gdim.y/16 + (!(gdim.z*gdim.y%16)?0:1));
		PotentialComputeFarFieldScalarStep1_kernel<<<blocks,threads>>>(
			gridval,
			phi,
			G*direction,
			cell_h,
			1.0/cell_h,
			K,
			gdim.x,
			gdim.y,
			gdim.z,
			1.0/(double)gdim.y);

	}
}


__global__ void 
Potential_PPCorrMNScalar_kernel(const uint * start,
					 const uint * end,
					 const float4 * pos_eval,
					 const float4 * pos_mass,
					 const double * mass,
					 double * p_phi,
					 double invh,
					 double h,
					 double G,//-1 for gravity, 1 for single layer potential
					 uint3 gdim,
					  uint3 hdim,
					  uint2 gstride,
					  uint2 hstride,
					  int K,
					  uint M,
					  uint N,
					  double4 origin)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	uint p=0;
	double correct_radius = ((double)K+0.5)/invh;
	if(idx<M)
	{
		double phi_PM = p_phi[idx];
		float4 ppos = pos_eval[idx];
		ppos.x -= origin.x;
		ppos.y -= origin.y;
		ppos.z -= origin.z;
		//compute the grid idx it belongs to
		int i = floor(ppos.x*invh);
		int j = floor(ppos.y*invh);
		int k = floor(ppos.z*invh);
		
		double a = 0.125;
		double phi_particle = 0;
		double3 grid_center = make_double3((double)i+0.5, (double)j+0.5, (double)k+0.5);
		double3 pos=make_double3(ppos.x*invh, ppos.y*invh, ppos.z*invh);
		int min_ii, max_ii, min_jj, max_jj, min_kk, max_kk;
		if(pos.x>grid_center.x)
		{
			min_ii = i-K, max_ii = i + K + 1;
		}
		else
		{
			min_ii = i-K-1, max_ii = i + K;
		}
		if(pos.y>grid_center.y)
		{
			min_jj = j-K, max_jj = j + K + 1;
		}
		else
		{
			min_jj = j-K-1, max_jj = j + K;
		}
		if(pos.z>grid_center.z)
		{
			min_kk = k-K, max_kk = k + K + 1;
		}
		else
		{
			min_kk = k-K-1, max_kk = k + K;
		}
		for(int kk=min_kk;kk<=max_kk;kk++)for(int jj = min_jj; jj<=max_jj; jj++)for(int ii=min_ii; ii<=max_ii; ii++)
		{
			if(kk>=0&&kk<hdim.z && jj>=0&&jj<hdim.y && ii>=0&&ii<hdim.x)
			{
				uint read_idx = kk*hstride.x + jj*hstride.y + ii;
				for(p = start[read_idx]; p<end[read_idx]; p++)
				{
					if(p<N)
					{
						float4 ppos2 = pos_mass[p];
						ppos2.x -= origin.x;
						ppos2.y -= origin.y;
						ppos2.z -= origin.z;
						if(fabs(ppos2.x-ppos.x)<correct_radius
							&&fabs(ppos2.y-ppos.y)<correct_radius
							&&fabs(ppos2.z-ppos.z)<correct_radius
							)
						{
							double3 dir = make_double3(ppos2.x-ppos.x, ppos2.y-ppos.y, ppos2.z-ppos.z);

							double r1 = sqrt(dir.x*dir.x+dir.y*dir.y+dir.z*dir.z);
							double r2 = 0;
							if(r1>a)
								r2 = 12.5663706144 * r1;
							else
							{
								if(r1<1e-12)
								{
									r2 = 1e+14;
								}
								else
								{
									r2 = 12.5663706144 * a ;
								}
							}
							
			
							
								phi_particle += G/r2 * mass[p];
						}
					}
				}
			}
		}
		
		p_phi[idx] = phi_PM+phi_particle; 

		
	}
}





		

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
					  double4 origin)
{
	if(K>=0)
	{
		uint threads = 256;
		uint blocks = M/threads + (!(M%threads)?0:1);
		Potential_PPCorrMNScalar_kernel<<<blocks, threads>>>(start, end, 
			pos_eval,
			pos_mass,
			mass, 
			p_phi, 
			1.0/h,
			h,
			direction*G,
			gdim, hdim,
			make_uint2(gdim.x*gdim.y, gdim.x),
			make_uint2(hdim.x*hdim.y, hdim.x),
			K,
			M,
			N,
			origin);
	}
}


//__global__ 
//void PotentialInterpolate_far_field_kernel(float4 * pos,
//					  double3 * far_force,
//					  double3 * p_force,
//					  double invh,
//					  uint dimx,
//					  uint dimy,
//					  uint dimz,
//					  uint num_particles,
//					  double4 origin)
//{
//	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
//	if(idx<num_particles)
//	{
//		
//		float4 ppos = pos[idx];
//		ppos.x -= origin.x;
//		ppos.y -= origin.y;
//		ppos.z -= origin.z;
//		//compute the grid idx it belongs to
//		int i = floor(ppos.x*invh);
//		int j = floor(ppos.y*invh);
//		int k = floor(ppos.z*invh);
//
//
//		if(i>=0 && i<dimx && j>=0 && j<dimy && k>=0 && k<dimz)
//		{
//			uint g_idx = k*dimx*dimy + j*dimx + i;
//			double3 vel_L = far_force[g_idx*6  ];
//			double3 vel_R = far_force[g_idx*6+1];
//			double3 vel_D = far_force[g_idx*6+2];
//			double3 vel_U = far_force[g_idx*6+3];
//			double3 vel_F = far_force[g_idx*6+4];
//			double3 vel_B = far_force[g_idx*6+5];
//
//			double cx = ppos.x*invh - (double)i;
//			double cy = ppos.y*invh - (double)j;
//			double cz = ppos.z*invh - (double)k;
//
//
//			double3 vel_x = make_double3((1.0-cx)*vel_L.x+cx*vel_R.x,
//										 (1.0-cx)*vel_L.y+cx*vel_R.y,
//										 (1.0-cx)*vel_L.z+cx*vel_R.z);
//			double3 vel_y = make_double3((1.0-cy)*vel_D.x+cy*vel_U.x,
//										 (1.0-cy)*vel_D.y+cy*vel_U.y,
//										 (1.0-cy)*vel_D.z+cy*vel_U.z);
//			double3 vel_z = make_double3((1.0-cz)*vel_F.x+cz*vel_B.x,
//										 (1.0-cz)*vel_F.y+cz*vel_B.y,
//										 (1.0-cz)*vel_F.z+cz*vel_B.z);
//			p_force[idx].x=(vel_x.x+vel_y.x+vel_z.x)/3.0;
//			p_force[idx].y=(vel_x.y+vel_y.y+vel_z.y)/3.0;
//			p_force[idx].z=(vel_x.z+vel_y.z+vel_z.z)/3.0;
//		}
//		
//	}
//}

__global__ 
void PotentialInterpolate_far_field_kernel2(float4 * pos,
					  double3 * far_force,
					  double3 * p_force,
					  double invh,
					  uint dimx,
					  uint dimy,
					  uint dimz,
					  uint num_particles,
					  double4 origin)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<num_particles)
	{
		
		float4 ppos = pos[idx];
		ppos.x -= origin.x;
		ppos.y -= origin.y;
		ppos.z -= origin.z;
		//compute the grid idx it belongs to
		int i = floor(ppos.x*invh);
		int j = floor(ppos.y*invh);
		int k = floor(ppos.z*invh);


		if(i>=0 && i<dimx && j>=0 && j<dimy && k>=0 && k<dimz)
		{
			uint g_idx = k*dimx*dimy + j*dimx + i;


			double3 sum=make_double3(0,0,0);

			for(int kk=k-2;kk<=k+2;kk++)for(int jj= j-2; jj<=j+2; jj++)for(int ii=i-2; ii<=i+2; ii++)
			{
				if(kk>=0&&kk<dimz && jj>=0&&jj<dimy && ii>=0&&ii<dimx)
				{
					//fetch the grid value
					uint g_idx = kk*dimx*dimy + jj*dimx + ii;
					double3 vel_C = far_force[g_idx];
				
					float4 pijk = make_float4(((double)ii+0.5),((double)jj+0.5),((double)kk+0.5),0);
					double r1 = (double)ppos.x * invh - pijk.x;
					double r2 = (double)ppos.y * invh - pijk.y;
					double r3 = (double)ppos.z * invh - pijk.z;
					double w = interpolate_weights(make_double3(pijk.x,pijk.y,pijk.z), make_double3(ppos.x*invh,ppos.y*invh,ppos.z*invh));//shape_function(fabs(r1))*shape_function(fabs(r2))*shape_function(fabs(r3));
					sum.x += vel_C.x * w;
					sum.y += vel_C.y * w;
					sum.z += vel_C.z * w;
				}
			}

			p_force[idx]=sum;
		}
		
	}
}


void PotentialInterpolateFarField(float4 * pos,
					  double3 * far_force,
					  double3 * p_force,
					  double cell_h,
					  uint dimx,uint dimy,uint dimz,
					  uint num_particles,
					  double4 origin)
{
	uint threads = 256;
	uint blocks = num_particles/threads + (!(num_particles%threads)?0:1);
	PotentialInterpolate_far_field_kernel2<<<blocks, threads>>>(pos,far_force,p_force,1.0/cell_h,dimx,dimy,dimz,num_particles,origin);
}





__global__ 
void PotentialInterpolate_far_fieldScalar_kernel2(float4 * pos,
					  double * far_force,
					  double * p_force,
					  double invh,
					  uint dimx,
					  uint dimy,
					  uint dimz,
					  uint num_particles,
					  double4 origin)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<num_particles)
	{
		
		float4 ppos = pos[idx];
		ppos.x -= origin.x;
		ppos.y -= origin.y;
		ppos.z -= origin.z;
		//compute the grid idx it belongs to
		int i = floor(ppos.x*invh);
		int j = floor(ppos.y*invh);
		int k = floor(ppos.z*invh);


		if(i>=0 && i<dimx && j>=0 && j<dimy && k>=0 && k<dimz)
		{
			uint g_idx = k*dimx*dimy + j*dimx + i;


			double sum=0;

			for(int kk=k-2;kk<=k+2;kk++)for(int jj= j-2; jj<=j+2; jj++)for(int ii=i-2; ii<=i+2; ii++)
			{
				if(kk>=0&&kk<dimz && jj>=0&&jj<dimy && ii>=0&&ii<dimx)
				{
					//fetch the grid value
					uint g_idx = kk*dimx*dimy + jj*dimx + ii;
					double vel_C = far_force[g_idx];
				
					float4 pijk = make_float4(((double)ii+0.5),((double)jj+0.5),((double)kk+0.5),0);
					double r1 = (double)ppos.x * invh - pijk.x;
					double r2 = (double)ppos.y * invh - pijk.y;
					double r3 = (double)ppos.z * invh - pijk.z;
					double w = interpolate_weights(make_double3(pijk.x,pijk.y,pijk.z), make_double3(ppos.x*invh,ppos.y*invh,ppos.z*invh));//shape_function(fabs(r1))*shape_function(fabs(r2))*shape_function(fabs(r3));
					sum += vel_C * w;
				}
			}

			p_force[idx]=sum;
		}
		
	}
}


void PotentialInterpolateFarFieldScalar(float4 * pos,
					  double * far_force,
					  double * p_force,
					  double cell_h,
					  uint dimx,uint dimy,uint dimz,
					  uint num_particles,
					  double4 origin)
{
	uint threads = 256;
	uint blocks = num_particles/threads + (!(num_particles%threads)?0:1);
	PotentialInterpolate_far_fieldScalar_kernel2<<<blocks, threads>>>(pos,far_force,p_force,1.0/cell_h,dimx,dimy,dimz,num_particles,origin);
}















__global__
void Potentialparticle_outside_kernel(float4 * pos,
							 double total_mass,
							 double3 mass_center,
							 float3 bbmin,
							 float3 bbmax,
							 double G,
							 double3* p_force,
							 uint M)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<M)
	{
		double3 ppos= make_double3(pos[idx].x, pos[idx].y, pos[idx].z);
		if(ppos.x<bbmin.x || ppos.x>bbmax.x 
		|| ppos.y<bbmin.y || ppos.y>bbmax.y
		|| ppos.z<bbmin.z || ppos.z>bbmax.z)
		{
			double3 vel = make_double3(0,0,0);
			double3 dir = make_double3(ppos.x-mass_center.x,ppos.y-mass_center.y,ppos.z-mass_center.z);
			double r3 = compute_r3(dir);
			vel = make_double3(total_mass*G*dir.x,total_mass*G*dir.y,total_mass*G*dir.z);
			vel.x = vel.x/r3;
			vel.y = vel.y/r3;
			vel.z = vel.z/r3;

			p_force[idx].x=0;
			p_force[idx].y=0;
			p_force[idx].z=0;
		}
	}
}

void PotentialComputeGradForOutParticle(float4 * pos, double total_mass, double3 mass_center, float3 bbmin, float3 bbmax, double G, double direction,double3* p_force, uint M)
{	
	uint threads = 256;
	uint blocks = M/threads + (!(M%threads)?0:1);
	Potentialparticle_outside_kernel<<<blocks, threads>>>(pos, total_mass, mass_center, bbmin, bbmax,direction*G, p_force, M);

}

__global__ void Potentialcompute_dphi_dn_kernel(double3 * normals,
								  double3 * gradients,
								  double  * mass,
								  double  * dphi_dn,
								  double  * area,
								  uint M)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<M)
	{
		double3 grad = gradients[idx];
		dphi_dn[idx]= -0.5*mass[idx]/area[idx] + my_dot(normals[idx],grad);
	}
}
void ComputeDphidn(double3 * normals,
				   double3 * gradients,
				   double  * mass,
				   double  * dphi_dn,
				   double  * area,
				   uint M)
{
	uint threads = 256;
	uint blocks = M/threads + (!(M%threads)?0:1);
	Potentialcompute_dphi_dn_kernel<<<blocks, threads>>>(normals,gradients, mass,dphi_dn,area, M);
}
		


///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////Biot Savart ////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
__global__ void
compute_curl_kernel(double * Fx,
					double * Fy,
					double * Fz,
					double * u,
					double * v,
					double * w,
					uint dimx,
					uint dimy,
					uint dimz,
					double inv_h,
					double invny)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int k = truncf((blockIdx.y*blockDim.y + threadIdx.y)*invny);
	int j = (blockIdx.y*blockDim.y + threadIdx.y)-__umul24(dimy,k);

	if(i<dimx && j<dimy && k<dimz)
	{
		uint idx = k*dimx*dimy + j*dimx + i;
		double Fz_up	= sample3D(Fz, i, j+1, k, dimx, dimy, dimz);
		double Fz_down	= sample3D(Fz, i, j-1, k, dimx, dimy, dimz);
		double Fz_left	= sample3D(Fz, i-1, j, k, dimx, dimy, dimz);
		double Fz_right	= sample3D(Fz, i+1, j, k, dimx, dimy, dimz);
		double Fy_front	= sample3D(Fy, i, j, k-1, dimx, dimy, dimz);
		double Fy_back	= sample3D(Fy, i, j, k+1, dimx, dimy, dimz);
		double Fy_left	= sample3D(Fy, i-1, j, k, dimx, dimy, dimz);
		double Fy_right	= sample3D(Fy, i+1, j, k, dimx, dimy, dimz);
		double Fx_front	= sample3D(Fx, i, j, k-1, dimx, dimy, dimz);
		double Fx_back	= sample3D(Fx, i, j, k+1, dimx, dimy, dimz);
		double Fx_up	= sample3D(Fx, i, j+1, k, dimx, dimy, dimz);
		double Fx_down	= sample3D(Fx, i, j-1, k, dimx, dimy, dimz);


		u[idx] = 0.5*inv_h*(Fz_up-Fz_down + Fy_front-Fy_back);
		v[idx] = 0.5*inv_h*(Fx_back-Fx_front + Fz_left - Fz_right);
		w[idx] = 0.5*inv_h*(Fy_right-Fy_left + Fx_down - Fx_up);

	}
}
void getCurl(double * Fx,
				double * Fy,
				double * Fz,
				double * u,
				double * v,
				double * w,
				uint dimx,
				uint dimy,
				uint dimz,
				double h)
{
	dim3 threads(16,16);
	dim3 blocks(dimx/16 + (!(dimx%16)?0:1), dimz*dimy/16 + (!(dimz*dimy%16)?0:1));
	compute_curl_kernel<<<blocks, threads>>>(Fx, Fy, Fz, u, v, w, dimx, dimy, dimz, 1.0/h, 1.0/(double)dimy);

}



__global__ void
MeshToParticle_kernel2(const float4 * pos,
					  const double * mesh_value,
					  double * particle_value,
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
		double sum=0;
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
				double f = mesh_value[read_idx];
				float4 pijk = make_float4(((double)ii+0.5),((double)jj+0.5),((double)kk+0.5),0);
				double r1 = (double)ppos.x * invh - pijk.x;
				double r2 = (double)ppos.y * invh - pijk.y;
				double r3 = (double)ppos.z * invh - pijk.z;
				double w = poly_shape(fabs(r1))*poly_shape(fabs(r2))*poly_shape(fabs(r3));
				sum += f * w;
			}
		}
		particle_value[idx] = sum;
		
	}
}

void MeshToParticle( float4 * pos,
					  double * mesh_value,
					  double * particle_value,
					  double h,
					  uint3 gdim,
					  uint3 hdim,
					  uint num_particles,
					  double4 origin)
{
	uint threads = 256;
	uint blocks = num_particles/threads + (!(num_particles%threads)?0:1);
	MeshToParticle_kernel2<<<blocks, threads>>>(pos, mesh_value, particle_value, 1.0/h, gdim, hdim, 
		make_uint2(gdim.x*gdim.y, gdim.x), 
		make_uint2(hdim.x*hdim.y, hdim.x),
		num_particles,
		origin);

}











//
//__global__ 
//void BiotSavartPMCorr2_kernel(const uint * start,
//					 const uint * end,
//					 const float4 * pos,
//					 double * g_vort_x,
//					 double * g_vort_y,
//					 double * g_vort_z,
//					 double * u,
//					 double * v,
//					 double * w,
//					 double h,
//					 double invh,
//					 double3 *dir_XX_array,
//					 uint3 gdim,
//					 uint3 hdim,
//					 uint2 gstride,
//					 uint2 hstride,
//					 int K,
//					 uint num_particle, 
//					 double4 origin)
//{
//	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
//	uint p=0;
//	if(idx<num_particle)
//	{
//		double3 vel_PM = make_double3(u[idx],v[idx],w[idx]);
//		float4 ppos = pos[idx];
//		ppos.x -= origin.x;
//		ppos.y -= origin.y;
//		ppos.z -= origin.z;
//		//compute the grid idx it belongs to
//		int i = floor(ppos.x*invh);
//		int j = floor(ppos.y*invh);
//		int k = floor(ppos.z*invh);
//		
//		double3 vel_mesh = make_double3(0,0,0);
//		for(int kk=k-K;kk<=k+K;kk++)for(int jj= j-K; jj<=j+K; jj++)for(int ii=i-K; ii<=i+K; ii++)
//		{
//			if(kk>=0&&kk<hdim.z && jj>=0&&jj<hdim.y && ii>=0&&ii<hdim.x)
//			{
//				double v = h*h*h;
//				uint read_idx = kk*hstride.x + jj*hstride.y + ii;
//				double3 omega = make_double3(g_vort_x[read_idx]*v,g_vort_y[read_idx]*v,g_vort_z[read_idx]*v);
//				double3 dir_x=make_double3(0,0,0);
//				for(int kkk=k-2;kkk<=k+2;kkk++)for(int jjj= j-2; jjj<=j+2; jjj++)for(int iii=i-2; iii<=i+2; iii++)
//				{
//					if(!(kkk==kk && jjj==jj && iii==ii))
//					{
//						uint idx_i = (kkk-k+2)*25+(jjj-j+2)*5+(iii-i+2);
//						uint idx_j = (kk-k+K)*(2*K+1)*(2*K+1)+(jj-j+K)*(2*K+1)+(ii-i+K);
//						uint idx_read = idx_j*125+idx_i;
//						double3 X=make_double3((double)iii+0.5,(double)jjj+0.5,(double)kkk+0.5);
//						double3 diff = make_double3(ppos.x*invh - X.x, ppos.y*invh - X.y, ppos.z*invh - X.z);					
//						double weight = shape_function(fabs(diff.x))*shape_function(fabs(diff.y))*shape_function(fabs(diff.z));
//
//						double3 dir_XX = make_double3(0,0,0);
//						/*dir_XX.x = 0.5*((1.0/compute_r(make_double4(X.x+h, X.y, X.z, 0), make_double4(X_prime.x, X_prime.y, X_prime.z, 0)))
//							-(1.0/compute_r(make_double4(X.x-h, X.y, X.z, 0), make_double4(X_prime.x, X_prime.y, X_prime.z, 0)))
//							)*invh;
//
//						dir_XX.y = 0.5*((1.0/compute_r(make_double4(X.x, X.y+h, X.z, 0), make_double4(X_prime.x, X_prime.y, X_prime.z, 0)))
//							-(1.0/compute_r(make_double4(X.x, X.y-h, X.z, 0), make_double4(X_prime.x, X_prime.y, X_prime.z, 0)))
//							)*invh;
//
//						dir_XX.z = 0.5*((1.0/compute_r(make_double4(X.x, X.y, X.z+h, 0), make_double4(X_prime.x, X_prime.y, X_prime.z, 0)))
//							-(1.0/compute_r(make_double4(X.x, X.y, X.z-h, 0), make_double4(X_prime.x, X_prime.y, X_prime.z, 0)))
//							)*invh;*/
//						dir_XX=dir_XX_array[idx_read];
//
//						dir_x.x += -(dir_XX.x)*weight;
//						dir_x.y += -(dir_XX.y)*weight;
//						dir_x.z += -(dir_XX.z)*weight;
//
//					}
//				}
//				double3 vtemp=cross_uxv(omega, dir_x);
//				vel_mesh.x += vtemp.x;
//				vel_mesh.y += vtemp.y;
//				vel_mesh.z += vtemp.z;
//			}
//		}
//		u[idx] = vel_PM.x - vel_mesh.x; 
//		v[idx] = vel_PM.y - vel_mesh.y;
//		w[idx] = vel_PM.z - vel_mesh.z;
//		
//	}
//}





__global__
void compute_diag_Psi_kernel(double * g_vort_x,
					 double * g_vort_y,
					 double * g_vort_z,
					 double * psi_x,
					 double * psi_y,
					 double * psi_z,
					 uint num_cells,
					 double h2)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<num_cells)
	{
		psi_x[idx] = 0.250580605913965*g_vort_x[idx]*h2;
		psi_y[idx] = 0.250580605913965*g_vort_y[idx]*h2;
		psi_z[idx] = 0.250580605913965*g_vort_z[idx]*h2;
	}
}

void ComputeDiagPsi(double * g_vort_x, double * g_vort_y, double * g_vort_z, double * psi_x, double * psi_y, double * psi_z, uint num_cell, double h_sqr)
{

	uint threads = 256;
	uint blocks = num_cell/threads + (!(num_cell%threads)?0:1);
	compute_diag_Psi_kernel<<<blocks, threads>>>(g_vort_x,g_vort_y,g_vort_z,psi_x,psi_y,psi_z,num_cell,h_sqr);
}

__global__
void subtract_diag_kernel(const float4 * pos,
					  const double * mesh_value,
					  double * particle_value,
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
		double sum=0;
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
				double f = mesh_value[read_idx];
				float4 pijk = make_float4(((double)ii+0.5),((double)jj+0.5),((double)kk+0.5),0);
				double r1 = (double)ppos.x * invh - pijk.x;
				double r2 = (double)ppos.y * invh - pijk.y;
				double r3 = (double)ppos.z * invh - pijk.z;
				double w = shape_function(fabs(r1))*shape_function(fabs(r2))*shape_function(fabs(r3));
				sum += f * w;
			}
		}
		particle_value[idx] -= sum;
		
	}
}
void SubtractDiag(float4 * pos, double * g_u, double * g_v, double* g_w, double* p_u, double* p_v, double* p_w, double h, uint3 gdim, uint3 hdim, uint num_particles, double4 origin)
{
	uint threads = 256;
	uint blocks = num_particles/threads + (!(num_particles%threads)?0:1);
	subtract_diag_kernel<<<blocks, threads>>>(pos, g_u, p_u,1.0/h,gdim, hdim, make_uint2(gdim.x*gdim.y,gdim.x),make_uint2(hdim.x*hdim.y,hdim.x),num_particles, origin);
	subtract_diag_kernel<<<blocks, threads>>>(pos, g_v, p_v,1.0/h,gdim, hdim, make_uint2(gdim.x*gdim.y,gdim.x),make_uint2(hdim.x*hdim.y,hdim.x),num_particles, origin);
	subtract_diag_kernel<<<blocks, threads>>>(pos, g_w, p_w,1.0/h,gdim, hdim, make_uint2(gdim.x*gdim.y,gdim.x),make_uint2(hdim.x*hdim.y,hdim.x),num_particles, origin);


}


__global__ 
void BiotSavartPPCorrMN_kernel(const uint * start,
					 const uint * end,
					 const float4 * evalposa,
					 const float4 * evalposb,
					 uint segment,
					 const float4 * vortpos,
					 const double * vort_x,
					 const double * vort_y,
					 const double * vort_z,
					 double * u,
					 double * v,
					 double * w,
					 double invh,
					 uint3 gdim,
					 uint3 hdim,
					 uint2 gstride,
					 uint2 hstride,
					 int K,
					 uint M,
					 uint N, 
					 double4 origin)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	uint p=0;
	double correct_radius = ((double)K+0.5)/invh;
	if(idx<M)
	{
		double3 vel_PM = make_double3(u[idx],v[idx],w[idx]);
		float4 pposa = evalposa[idx];
		float4 pposb = evalposb[idx];
		pposa.x -= origin.x;
		pposa.y -= origin.y;
		pposa.z -= origin.z;
		pposb.x -= origin.x;
		pposb.y -= origin.y;
		pposb.z -= origin.z;
		//compute the grid idx it belongs to
		int i = floor(pposa.x*invh);
		int j = floor(pposa.y*invh);
		int k = floor(pposa.z*invh);
		double a = 0.025;
		double3 vel_pp = make_double3(0,0,0);
		double3 grid_center = make_double3((double)i+0.5, (double)j+0.5, (double)k+0.5);
		double3 pos=make_double3(pposa.x*invh, pposa.y*invh, pposa.z*invh);
		int min_ii, max_ii, min_jj, max_jj, min_kk, max_kk;
		if(pos.x>grid_center.x)
		{
			min_ii = i-K-1, max_ii = i + K + 1;
		}
		else
		{
			min_ii = i-K-1, max_ii = i + K + 1;
		}
		if(pos.y>grid_center.y)
		{
			min_jj = j-K-1, max_jj = j + K + 1;
		}
		else
		{
			min_jj = j-K-1, max_jj = j + K + 1;
		}
		if(pos.z>grid_center.z)
		{
			min_kk = k-K-1, max_kk = k + K + 1;
		}
		else
		{
			min_kk = k-K-1, max_kk = k + K + 1;
		}

		for(int kk=min_kk;kk<=max_kk;kk++)for(int jj = min_jj; jj<=max_jj; jj++)for(int ii=min_ii; ii<=max_ii; ii++)
		{
			if(kk>=0&&kk<hdim.z && jj>=0&&jj<hdim.y && ii>=0&&ii<hdim.x )
			{
				uint read_idx = kk*hstride.x + jj*hstride.y + ii;
				for(p = start[read_idx]; p<end[read_idx]; p++)
				{
					if(p<N)
					{
						float4 ppos2 = vortpos[p];
						ppos2.x -= origin.x;
						ppos2.y -= origin.y;
						ppos2.z -= origin.z;
						if(fabs(ppos2.x-pposa.x)<correct_radius
							&&fabs(ppos2.y-pposa.y)<correct_radius
							&&fabs(ppos2.z-pposa.z)<correct_radius
							)
						{
							double3 omega = make_double3(vort_x[p],vort_y[p], vort_z[p]);

							double3 dira = make_double3(pposa.x-ppos2.x, pposa.y-ppos2.y, pposa.z-ppos2.z);
							double3 dirb = make_double3(pposb.x-ppos2.x, pposb.y-ppos2.y, pposb.z-ppos2.z);
							double r1 = sqrt(dira.x*dira.x+dira.y*dira.y+dira.z*dira.z);
							
							double r3 = 0;
							double coeff;
							double r_inva = r1/a;
							double r3a3 = 0;
							double r3a3_sqr = 0;

							if(r_inva>=0.01)
							{
								coeff = (1-exp(-r_inva*r_inva*r_inva))/(12.5663706144 * r1*r1*r1);
							}
							else
							{
								r3a3 = r_inva*r_inva*r_inva;
								r3a3_sqr = r3a3*r3a3;
								coeff = 1/(12.5663706144 * a * a * a)*(1-0.5*r3a3 + 1/6*r3a3_sqr - 1/24*r3a3_sqr*r3a3);
							}
							if(r1<1e-12)
							{
								coeff = 0;
							}
							if(segment==1)
							{
								double r2 = sqrt(dirb.x*dirb.x+dirb.y*dirb.y+dirb.z*dirb.z);
								if(r2<1e-12)
									coeff = 0;
							}
							
							double3 vel_ij = cross_uxv(omega, dira);
							vel_pp.x += vel_ij.x*coeff;
							vel_pp.y += vel_ij.y*coeff;
							vel_pp.z += vel_ij.z*coeff;
						}
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
BiotSavartPPCorrMN(uint * start,
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
				double4 origin)
{
	if(K>=0)
	{
		uint threads = 256;
		uint blocks = M/threads + (!(M%threads)?0:1);
		BiotSavartPPCorrMN_kernel<<<blocks, threads>>>(start, end, 
			evalposa,
			evalposb,
			is_segment,
			vortpos,
			vort_x, vort_y, vort_z, 
			u, v, w, 
			1.0/cellh,
			gdim, hdim,
			make_uint2(gdim.x*gdim.y, gdim.x),
			make_uint2(hdim.x*hdim.y, hdim.x),
			K,
			M,
			N,
			origin);
	}

} 

__device__
double3 proj_a(double3 a, double3 b)//return b's projection on a
{
	double len = sqrt(a.x*a.x+a.y*a.y+a.z*a.z);
	if(len<1e-6) return b;
	else
	{
		double3 vec = make_double3(a.x/len, a.y/len, a.z/len);
		double dot_ab = vec.x*b.x + vec.y*b.y + vec.z*b.z;
		return make_double3(vec.x*dot_ab, vec.y*dot_ab, vec.z*dot_ab);
	}
}

__device__
double3 orth_a(double3 a, double3 b)//return b's orth of a
{
	double3 c = proj_a(a,b);
	return make_double3(b.x - c.x, b.y-c.y, b.z-c.z);
}

__global__ 
void BiotSavartPPCorrScaleMN_kernel(const uint * start,
					 const uint * end,
					 const float4 * evalposa,
					 const float4 * evalposb,
					 uint segment,
					 const float4 * vortpos,
					 const double * vort_x,
					 const double * vort_y,
					 const double * vort_z,
					 double * u,
					 double * v,
					 double * w,
					 double * u_pm,
					 double * v_pm,
					 double * w_pm,
					 double invh,
					 uint3 gdim,
					 uint3 hdim,
					 uint2 gstride,
					 uint2 hstride,
					 int K,
					 uint M,
					 uint N, 
					 double4 origin)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	uint p=0;
	double correct_radius = ((double)K+0.5)/invh;
	if(idx<M)
	{
		double3 vel_PM = make_double3(u[idx],v[idx],w[idx]);
		float4 pposa = evalposa[idx];
		float4 pposb = evalposb[idx];
		pposa.x -= origin.x;
		pposa.y -= origin.y;
		pposa.z -= origin.z;
		pposb.x -= origin.x;
		pposb.y -= origin.y;
		pposb.z -= origin.z;
		//compute the grid idx it belongs to
		int i = floor(pposa.x*invh);
		int j = floor(pposa.y*invh);
		int k = floor(pposa.z*invh);
		double a = 0.08;
		double3 vel_pp = make_double3(0,0,0);
		double3 grid_center = make_double3((double)i+0.5, (double)j+0.5, (double)k+0.5);
		double3 pos=make_double3(pposa.x*invh, pposa.y*invh, pposa.z*invh);
		int min_ii, max_ii, min_jj, max_jj, min_kk, max_kk;
		if(pos.x>grid_center.x)
		{
			min_ii = i-K-1, max_ii = i + K + 1;
		}
		else
		{
			min_ii = i-K-1, max_ii = i + K + 1;
		}
		if(pos.y>grid_center.y)
		{
			min_jj = j-K-1, max_jj = j + K + 1;
		}
		else
		{
			min_jj = j-K-1, max_jj = j + K + 1;
		}
		if(pos.z>grid_center.z)
		{
			min_kk = k-K-1, max_kk = k + K + 1;
		}
		else
		{
			min_kk = k-K-1, max_kk = k + K + 1;
		}

		//double3 grid_induced_vel = make_double3(0,0,0);
		//double3 sum=make_double3(0,0,0);
		//for(int kkk=k-2;kkk<=k+2;kkk++)for(int jjj= j-2; jjj<=j+2; jjj++)for(int iii=i-2; iii<=i+2; iii++)
		//{
		//	if(kkk>=0&&kkk<gdim.z && jjj>=0&&jjj<gdim.y && iii>=0&&iii<gdim.x)
		//	{
		//		//fetch the grid value
		//		uint g_idx = kkk*hstride.x + jjj*hstride.y + iii;
		//		double3 vel_C = make_double3(u_pm[g_idx],v_pm[g_idx],w_pm[g_idx]);
		//	
		//		float4 pijk = make_float4(((double)iii+0.5),((double)jjj+0.5),((double)kkk+0.5),0);
		//		double r1 = (double)pposa.x * invh - pijk.x;
		//		double r2 = (double)pposa.y * invh - pijk.y;
		//		double r3 = (double)pposa.z * invh - pijk.z;
		//		double weight = interpolate_weights(make_double3(pijk.x,pijk.y,pijk.z), make_double3(pposa.x*invh,pposa.y*invh,pposa.z*invh));//shape_function(fabs(r1))*shape_function(fabs(r2))*shape_function(fabs(r3));
		//		sum.x += vel_C.x * weight;
		//		sum.y += vel_C.y * weight;
		//		sum.z += vel_C.z * weight;
		//	}
		//}
		//grid_induced_vel.x = sum.x - u[idx];
		//grid_induced_vel.y = sum.y - v[idx] ;
		//grid_induced_vel.z = sum.z - w[idx] ;
		
		//double grid_energy = grid_induced_vel.x*grid_induced_vel.x + grid_induced_vel.y*grid_induced_vel.y + grid_induced_vel.z*grid_induced_vel.z;
		

		for(int kk=min_kk;kk<=max_kk;kk++)for(int jj = min_jj; jj<=max_jj; jj++)for(int ii=min_ii; ii<=max_ii; ii++)
		{
			if(kk>=0&&kk<hdim.z && jj>=0&&jj<hdim.y && ii>=0&&ii<hdim.x )
			{
				uint read_idx = kk*hstride.x + jj*hstride.y + ii;
				for(p = start[read_idx]; p<end[read_idx]; p++)
				{
					if(p<N)
					{
						float4 ppos2 = vortpos[p];
						ppos2.x -= origin.x;
						ppos2.y -= origin.y;
						ppos2.z -= origin.z;
						if(fabs(ppos2.x-pposa.x)<correct_radius
							&&fabs(ppos2.y-pposa.y)<correct_radius
							&&fabs(ppos2.z-pposa.z)<correct_radius
							)
						{
							double3 omega = make_double3(vort_x[p],vort_y[p], vort_z[p]);

							double3 dira = make_double3(pposa.x-ppos2.x, pposa.y-ppos2.y, pposa.z-ppos2.z);
							double3 dirb = make_double3(pposb.x-ppos2.x, pposb.y-ppos2.y, pposb.z-ppos2.z);
							double r1 = sqrt(dira.x*dira.x+dira.y*dira.y+dira.z*dira.z);
							
							double r3 = 0;
							double coeff;
							double r_inva = r1/a;
							double r3a3 = 0;
							double r3a3_sqr = 0;

							if(r_inva>=0.01)
							{
								coeff = (1-exp(-r_inva*r_inva*r_inva))/(12.5663706144 * r1*r1*r1);
							}
							else
							{
								r3a3 = r_inva*r_inva*r_inva;
								r3a3_sqr = r3a3*r3a3;
								coeff = 1/(12.5663706144 * a * a * a)*(1-0.5*r3a3 + 1/6*r3a3_sqr - 1/24*r3a3_sqr*r3a3);
							}
							if(r1<1e-12)
							{
								coeff = 0;
							}
							if(segment==1)
							{
								double r2 = sqrt(dirb.x*dirb.x+dirb.y*dirb.y+dirb.z*dirb.z);
								if(r2<1e-12)
									coeff = 0;
							}
							
							double3 vel_ij = cross_uxv(omega, dira);
							vel_pp.x += vel_ij.x*coeff;
							vel_pp.y += vel_ij.y*coeff;
							vel_pp.z += vel_ij.z*coeff;
						}
					}
				}
			}
		}

		//double vort_energy = vel_pp.x*vel_pp.x + vel_pp.y*vel_pp.y + vel_pp.z*vel_pp.z;
		//double scale = 1.0;
		//if(vort_energy>grid_energy) scale = sqrt(grid_energy/vort_energy);
		//double3 u_far = make_double3(u[idx], v[idx], w[idx]);
		//double3 vec1 = proj_a(u_far, grid_induced_vel);
		//double3 vec2 = orth_a(u_far, vel_pp);

		u[idx] = vel_PM.x + vel_pp.x;// + vec2.x; 
		v[idx] = vel_PM.y + vel_pp.y;// + vec2.y;
		w[idx] = vel_PM.z + vel_pp.z;// + vec2.z;
		
	}
}


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
				double4 origin)
{
	if(K>=0)
	{
		uint threads = 256;
		uint blocks = M/threads + (!(M%threads)?0:1);
		BiotSavartPPCorrScaleMN_kernel<<<blocks, threads>>>(start, end, 
			evalposa,
			evalposb,
			is_segment,
			vortpos,
			vort_x, vort_y, vort_z, 
			u, v, w, 
			u_pm, v_pm, w_pm,
			1.0/cellh,
			gdim, hdim,
			make_uint2(gdim.x*gdim.y, gdim.x),
			make_uint2(hdim.x*hdim.y, hdim.x),
			K,
			M,
			N,
			origin);
	}

} 

__global__ 
void BiotSavartComputeFarFieldStep1_kernel(const uint * start,
					 const uint * end,
					 const float4 * pos,
					 double * g_vort_x,
					 double * g_vort_y,
					 double * g_vort_z,
					 double * g_u,
					 double * g_v,
					 double * g_w,
					 double * u,
					 double * v,
					 double * w,
					 double h,
					 double invh,
					 double3 *dir_XX_array,
					 int3 *cell_idx_array,
					 int K,
					 uint dimx,
					 uint dimy,
					 uint dimz,
					 double invny,
					 uint num_particle, 
					 double4 origin)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int k = truncf((blockIdx.y*blockDim.y + threadIdx.y)*invny);
	int j = (blockIdx.y*blockDim.y + threadIdx.y)-__umul24(dimy,k);
	uint p=0;
	if(i<dimx && j<dimy && k<dimz)
	{
		int table_idx = k*dimx*dimy + j*dimx + i;
		
			for(int neighbor_idx = 0; neighbor_idx<1; neighbor_idx++)
			{// compute near field velocity for each of its 1 neighbor cells;
				int3 cell_idx = cell_idx_array[neighbor_idx];
				uint write_idx = table_idx*1 + neighbor_idx;
				int kkk = k+cell_idx.z;
				int jjj = j+cell_idx.y;
				int iii = i+cell_idx.x;
				double3 vel_PM = make_double3(0,0,0);
				double3 vel_mesh = make_double3(0,0,0);
				if(kkk>=0&&kkk<dimz && jjj>=0&&jjj<dimy && iii>=0&&iii<dimx)
				{
					uint vel_read_idx = kkk*dimx*dimy + jjj*dimx + iii;
					vel_PM = make_double3(g_u[vel_read_idx],g_v[vel_read_idx],g_w[vel_read_idx]);
					for(int kk=k-K;kk<=k+K;kk++)for(int jj= j-K; jj<=j+K; jj++)for(int ii=i-K; ii<=i+K; ii++)
					{
						double3 omega= make_double3(0,0,0);
						double3 dir_x= make_double3(0,0,0);
						if(kk>=0&&kk<dimz && jj>=0&&jj<dimy && ii>=0&&ii<dimx)
						{
							uint vort_read_idx = kk*dimx*dimy + jj*dimx + ii;
							double v = h*h*h;
							omega = make_double3(g_vort_x[vort_read_idx]*v,g_vort_y[vort_read_idx]*v,g_vort_z[vort_read_idx]*v);//read the vorticity

							
							if(!(kkk==kk && jjj==jj && iii==ii))
							{
								uint idx_i = (kkk-k+2)*25+(jjj-j+2)*5+(iii-i+2);
								uint idx_j = (kk-k+K)*(2*K+1)*(2*K+1)+(jj-j+K)*(2*K+1)+(ii-i+K);
								uint idx_read = idx_j*125+idx_i;
								dir_x=dir_XX_array[idx_read];
							}
						}
						double3 vtemp=cross_uxv(dir_x,omega);
						vel_mesh.x += vtemp.x;
						vel_mesh.y += vtemp.y;
						vel_mesh.z += vtemp.z;
					}
				}
				u[write_idx] = vel_PM.x - vel_mesh.x; 
				v[write_idx] = vel_PM.y - vel_mesh.y;
				w[write_idx] = vel_PM.z - vel_mesh.z;
			}
		
	}
}

__global__
void subtract_diagonal_grid_kernel(
					 double * g_u,
					 double * g_v,
					 double * g_w,
					 double * u,
					 double * v,
					 double * w,
					 uint g_K, 
					 int3 *cell_idx_array,
					 uint dimx,
					 uint dimy,
					 uint dimz,
					 double invny)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int k = truncf((blockIdx.y*blockDim.y + threadIdx.y)*invny);
	int j = (blockIdx.y*blockDim.y + threadIdx.y)-__umul24(dimy,k);
	if(k>=0&&k<dimz && j>=0&&j<dimy &&i>=0&&i<dimx)
	{
		if(g_K>0)
		{

			int table_idx = k*dimx*dimy + j*dimx + i;
			for(int neighbor_idx = 0; neighbor_idx<1; neighbor_idx++)
			{
				int3 cell_idx = cell_idx_array[neighbor_idx];
				uint write_idx = table_idx*1 + neighbor_idx;
				double3 vel = make_double3(0,0,0);
				int kkk = k+cell_idx.z;
				int jjj = j+cell_idx.y;
				int iii = i+cell_idx.x;
				if(kkk>=0&&kkk<dimz && jjj>=0&&jjj<dimy && iii>=0&&iii<dimx)
				{
					uint vel_idx = kkk*dimx*dimy+jjj*dimy+iii;
					vel = make_double3(g_u[vel_idx],g_v[vel_idx],g_w[vel_idx]);
				}
				u[write_idx] -= vel.x;
				v[write_idx] -= vel.y;
				w[write_idx] -= vel.z;
			}
		}
		if(g_K==0)
		{
			int table_idx = k*dimx*dimy + j*dimx + i;
			for(int neighbor_idx = 0; neighbor_idx<1; neighbor_idx++)
			{
				int3 cell_idx = cell_idx_array[neighbor_idx];
				uint write_idx = table_idx*1 + neighbor_idx;
				double3 vel = make_double3(0,0,0);
				int kkk = k+cell_idx.z;
				int jjj = j+cell_idx.y;
				int iii = i+cell_idx.x;
				if(kkk>=0&&kkk<dimz && jjj>=0&&jjj<dimy && iii>=0&&iii<dimx)
				{
					uint vel_idx = kkk*dimx*dimy+jjj*dimy+iii;
					vel = make_double3(g_u[vel_idx],g_v[vel_idx],g_w[vel_idx]);
				}
				u[write_idx] -= vel.x;
				v[write_idx] -= vel.y;
				w[write_idx] -= vel.z;
			}
		}
	}
}


		
void BiotSavartComputeFarField(uint* start,uint * end,float4 * pos,double * vrho_x,double * vrho_y,double * vrho_z,
							   double * psi_x,double * psi_y,double * psi_z,
							   double * g_u,double * g_v,double * g_w,
							   double * u,double * v,double * w,
							   double cell_h,uint3 gdim,uint3 hdim,
							   int K, uint num_particles, double4 origin)
{
	
	if(K>=0)
	{
	
	
	double3 *dir_XX_host = (double3*)malloc(sizeof(double3)*(2*K+1)*(2*K+1)*(2*K+1)*5*5*5);
	double h = cell_h;
	for(int kk=-K;kk<=K;kk++)for(int jj= -K; jj<=K; jj++)for(int ii=-K; ii<=K; ii++)
	{
		double3 X_prime = make_double3((double)ii+0.5,(double)jj+0.5,(double)kk+0.5);
				X_prime.x*=h; X_prime.y*=h; X_prime.z*=h;
		for(int kkk=-2;kkk<=2;kkk++)for(int jjj= -2; jjj<=2; jjj++)for(int iii=-2; iii<=2; iii++)
		{
			if(!(kkk==kk && jjj==jj && iii==ii))
			{
				uint idx_i = (kkk+2)*25+(jjj+2)*5+(iii+2);
				uint idx_j = (kk+K)*(2*K+1)*(2*K+1)+(jj+K)*(2*K+1)+(ii+K);
				uint idx_read = idx_j*125+idx_i;
				double3 X=make_double3((double)iii+0.5,(double)jjj+0.5,(double)kkk+0.5);
				X.x*=h; X.y*=h; X.z*=h;
				double3 dir_XX = make_double3(0,0,0);
				dir_XX.x = 0.5*((1.0/cpu_compute_r(make_double4(X.x+h, X.y, X.z, 0), make_double4(X_prime.x, X_prime.y, X_prime.z, 0)))
					-(1.0/cpu_compute_r(make_double4(X.x-h, X.y, X.z, 0), make_double4(X_prime.x, X_prime.y, X_prime.z, 0)))
					)/h;

				dir_XX.y = 0.5*((1.0/cpu_compute_r(make_double4(X.x, X.y+h, X.z, 0), make_double4(X_prime.x, X_prime.y, X_prime.z, 0)))
					-(1.0/cpu_compute_r(make_double4(X.x, X.y-h, X.z, 0), make_double4(X_prime.x, X_prime.y, X_prime.z, 0)))
					)/h;

				dir_XX.z = 0.5*((1.0/cpu_compute_r(make_double4(X.x, X.y, X.z+h, 0), make_double4(X_prime.x, X_prime.y, X_prime.z, 0)))
					-(1.0/cpu_compute_r(make_double4(X.x, X.y, X.z-h, 0), make_double4(X_prime.x, X_prime.y, X_prime.z, 0)))
					)/h;

				dir_XX_host[idx_read]=make_double3(dir_XX.x,dir_XX.y,dir_XX.z);
			}
		}
	}
	int3 *cell_idx_host = (int3*)malloc(sizeof(int3)*8);
	cell_idx_host[0]=make_int3(0,0,0);
	cell_idx_host[1]=make_int3( 1,0,0);
	cell_idx_host[2]=make_int3(0,-1,0);
	cell_idx_host[3]=make_int3(0, 1,0);
	cell_idx_host[4]=make_int3(0,0,-1);
	cell_idx_host[5]=make_int3(0,0, 1);
	cell_idx_host[6]=make_int3(0,0,0);
	cell_idx_host[7]=make_int3(0,0,0);
	int3 *cell_idx_devc;
	cudaMalloc((void**)&cell_idx_devc, sizeof(int3)*8);
	cudaMemcpy(cell_idx_devc,cell_idx_host,sizeof(int3)*8,cudaMemcpyHostToDevice);
	double3 *dir_XX_devc;
	cudaMalloc((void**)&dir_XX_devc,sizeof(double3)*(2*K+1)*(2*K+1)*(2*K+1)*5*5*5);
	cudaMemcpy(dir_XX_devc,dir_XX_host,sizeof(double3)*(2*K+1)*(2*K+1)*(2*K+1)*5*5*5,cudaMemcpyHostToDevice);
	dim3 threads(16,16);
	dim3 blocks(gdim.x/16 + (!(gdim.x%16)?0:1), gdim.z*gdim.y/16 + (!(gdim.z*gdim.y%16)?0:1));
	BiotSavartComputeFarFieldStep1_kernel<<<blocks,threads>>>(start,end,
		pos,
		vrho_x,
		vrho_y,
		vrho_z,
		g_u,
		g_v,
		g_w,
		u,
		v,
		w,
		cell_h,
		1.0/cell_h,
		dir_XX_devc,
		cell_idx_devc,
		K,
		gdim.x,
		gdim.y,
		gdim.z,
		1.0/(double)gdim.y,
		num_particles, 
		origin);
	
	cudaMemset(psi_x,0,sizeof(double)*gdim.x*gdim.y*gdim.z);cudaMemset(psi_y,0,sizeof(double)*gdim.x*gdim.y*gdim.z);cudaMemset(psi_z,0,sizeof(double)*gdim.x*gdim.y*gdim.z);
	cudaMemset(g_u,0,sizeof(double)*gdim.x*gdim.y*gdim.z);cudaMemset(g_v,0,sizeof(double)*gdim.x*gdim.y*gdim.z);cudaMemset(g_w,0,sizeof(double)*gdim.x*gdim.y*gdim.z);
	ComputeDiagPsi(vrho_x, vrho_y, vrho_z, psi_x, psi_y, psi_z,gdim.x*gdim.y*gdim.z, cell_h*cell_h);
	getCurl(psi_x,psi_y,psi_z,g_u,g_v,g_w,gdim.x, gdim.y, gdim.z, cell_h);
	dim3 threads2(16,16);
	dim3 blocks2(gdim.x/16 + (!(gdim.x%16)?0:1), gdim.z*gdim.y/16 + (!(gdim.z*gdim.y%16)?0:1));
	subtract_diagonal_grid_kernel<<<blocks, threads>>>(g_u,g_v,g_w,u,v,w,K, cell_idx_devc,gdim.x,gdim.y,gdim.z,1.0/(double)gdim.y);

	free(dir_XX_host);
	cudaFree(dir_XX_devc);
	free(cell_idx_host);
	cudaFree(cell_idx_devc);
	}
	
	

}


//__gloabl__
//void compute_scale_factor(float * scale_factor,
//						  double * g_u,double * g_v, double * g_w,
//						  double * g_u_far, double * g_v_far, double * g_w_far,
//						  uint * start, uint *end,
//						  float4 * sort_pos,
//						  double * vort_x,
//						  double * vort_y,
//						  double * vort_z,
//						  double cell_h,
//						  uint K,
//						  uint num_particles,
//						  uint dimx,
//						  uint dimy,
//						  uint dimz,
//						  double invny, 
//						  double4 origin)
//{
//	int i = blockIdx.x*blockDim.x + threadIdx.x;
//	int k = truncf((blockIdx.y*blockDim.y + threadIdx.y)*invny);
//	int j = (blockIdx.y*blockDim.y + threadIdx.y)-__umul24(dimy,k);
//	
//	if(k>=0&&k<dimz && j>=0&&j<dimy &&i>=0&&i<dimx)
//	{
//		uint idx = k*dimx*dimy + j*dimx + i;
//		double3 grid_induced_vel = make_double3(g_u[idx]
//		//compute vortex particle induced velocity:
//		double3 vort_induced_vel = make_double3(0,0,0);
//
//
//	}
//	
//	
//
//}
//__global__ 
//void BiotSavartInterpolate_far_field_kernel(float4 * pos,
//					  double * far_u,
//					  double * far_v,
//					  double * far_w,
//					  double * p_u,
//					  double * p_v,
//					  double * p_w,
//					  double invh,
//					  uint dimx,
//					  uint dimy,
//					  uint dimz,
//					  uint num_particles,
//					  double4 origin)
//{
//	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
//	if(idx<num_particles)
//	{
//		
//		float4 ppos = pos[idx];
//		ppos.x -= origin.x;
//		ppos.y -= origin.y;
//		ppos.z -= origin.z;
//		//compute the grid idx it belongs to
//		int i = floor(ppos.x*invh);
//		int j = floor(ppos.y*invh);
//		int k = floor(ppos.z*invh);
//
//
//		if(i>=0 && i<dimx && j>=0 && j<dimy && k>=0 && k<dimz)
//		{
//			uint g_idx = k*dimx*dimy + j*dimx + i;
//			//double3 vel_L = make_double3(far_u[g_idx*8  ],far_v[g_idx*8  ],far_w[g_idx*8  ]);
//			//double3 vel_R = make_double3(far_u[g_idx*8+1],far_v[g_idx*8+1],far_w[g_idx*8+1]);
//			//double3 vel_D = make_double3(far_u[g_idx*8+2],far_v[g_idx*8+2],far_w[g_idx*8+2]);
//			//double3 vel_U = make_double3(far_u[g_idx*8+3],far_v[g_idx*8+3],far_w[g_idx*8+3]);
//			//double3 vel_F = make_double3(far_u[g_idx*8+4],far_v[g_idx*8+4],far_w[g_idx*8+4]);
//			//double3 vel_B = make_double3(far_u[g_idx*8+5],far_v[g_idx*8+5],far_w[g_idx*8+5]);
//			//double3 vel_C = make_double3(far_u[g_idx*8+6],far_v[g_idx*8+6],far_w[g_idx*8+6]);
//			double3 vel_C = make_double3(far_u[g_idx],far_v[g_idx],far_w[g_idx]);
//
//			//double cx = ppos.x*invh - (double)i;
//			//double cy = ppos.y*invh - (double)j;
//			//double cz = ppos.z*invh - (double)k;
//
//
//			//double3 vel_x = make_double3((1.0-cx)*vel_L.x+cx*vel_R.x,
//			//							 (1.0-cx)*vel_L.y+cx*vel_R.y,
//			//							 (1.0-cx)*vel_L.z+cx*vel_R.z);
//			//double3 vel_y = make_double3((1.0-cy)*vel_D.x+cy*vel_U.x,
//			//							 (1.0-cy)*vel_D.y+cy*vel_U.y,
//			//							 (1.0-cy)*vel_D.z+cy*vel_U.z);
//			//double3 vel_z = make_double3((1.0-cz)*vel_F.x+cz*vel_B.x,
//			//							 (1.0-cz)*vel_F.y+cz*vel_B.y,
//			//							 (1.0-cz)*vel_F.z+cz*vel_B.z);
//			//p_u[idx]=(vel_x.x+vel_y.x+vel_z.x)/3.0;
//			//p_v[idx]=(vel_x.y+vel_y.y+vel_z.y)/3.0;
//			//p_w[idx]=(vel_x.z+vel_y.z+vel_z.z)/3.0;
//			double3 dX = make_double3(ppos.x - ((double)i + 0.5)/invh,
//									  ppos.y - ((double)j + 0.5)/invh,
//									  ppos.z - ((double)k + 0.5)/invh);
//
//			double3 dU = make_double3(0,0,0);
//			double3 dV = make_double3(0,0,0);
//			double3 dW = make_double3(0,0,0);
//
//			if(dX.x>=0)
//			{
//				dU.x=invh * (vel_R.x - vel_C.x);
//				dV.x=invh * (vel_R.y - vel_C.y);
//				dW.x=invh * (vel_R.z - vel_C.z);
//			}
//			else
//			{
//				dU.x=invh * (vel_C.x - vel_L.x);
//				dV.x=invh * (vel_C.y - vel_L.y);
//				dW.x=invh * (vel_C.z - vel_L.z);
//			}
//			if(dX.y>=0)
//			{
//				dU.y=invh * (vel_U.x - vel_C.x);
//				dV.y=invh * (vel_U.y - vel_C.y);
//				dW.y=invh * (vel_U.z - vel_C.z);
//			}
//			else
//			{
//				dU.y=invh * (vel_C.x - vel_D.x);
//				dV.y=invh * (vel_C.y - vel_D.y);
//				dW.y=invh * (vel_C.z - vel_D.z);
//			}
//			if(dX.z>=0)
//			{
//				dU.z=invh * (vel_B.x - vel_C.x);
//				dV.z=invh * (vel_B.y - vel_C.y);
//				dW.z=invh * (vel_B.z - vel_C.z);
//			}
//			else
//			{
//				dU.z=invh * (vel_C.x - vel_F.x);
//				dV.z=invh * (vel_C.y - vel_F.y);
//				dW.z=invh * (vel_C.z - vel_F.z);
//			}
//
//
//			//taylor expansion
//			p_u[idx] = vel_C.x + dX.x*dU.x + dX.y*dU.y + dX.z*dU.z;
//			p_v[idx] = vel_C.y + dX.x*dV.x + dX.y*dV.y + dX.z*dV.z;
//			p_w[idx] = vel_C.z + dX.x*dW.x + dX.y*dW.y + dX.z*dW.z;
//			
//
//		}
//		
//	}
//}
__global__ 
void BiotSavartInterpolate_far_field_kernel2(float4 * pos,
					  double * far_u,
					  double * far_v,
					  double * far_w,
					  double * p_u,
					  double * p_v,
					  double * p_w,
					  double invh,
					  uint dimx,
					  uint dimy,
					  uint dimz,
					  uint num_particles,
					  double4 origin)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<num_particles)
	{
		
		float4 ppos = pos[idx];
		ppos.x -= origin.x;
		ppos.y -= origin.y;
		ppos.z -= origin.z;
		//compute the grid idx it belongs to
		int i = floor(ppos.x*invh);
		int j = floor(ppos.y*invh);
		int k = floor(ppos.z*invh);


		
		if(i>=0 && i<dimx && j>=0 && j<dimy && k>=0 && k<dimz)
		{
			double3 sum=make_double3(0,0,0);
			for(int kk=k-2;kk<=k+2;kk++)for(int jj= j-2; jj<=j+2; jj++)for(int ii=i-2; ii<=i+2; ii++)
			{
				if(kk>=0&&kk<dimz && jj>=0&&jj<dimy && ii>=0&&ii<dimx)
				{
					//fetch the grid value
					uint g_idx = kk*dimx*dimy + jj*dimx + ii;
					double3 vel_C = make_double3(far_u[g_idx],far_v[g_idx],far_w[g_idx]);
				
					float4 pijk = make_float4(((double)ii+0.5),((double)jj+0.5),((double)kk+0.5),0);
					double r1 = (double)ppos.x * invh - pijk.x;
					double r2 = (double)ppos.y * invh - pijk.y;
					double r3 = (double)ppos.z * invh - pijk.z;
					double w = interpolate_weights(make_double3(pijk.x,pijk.y,pijk.z), make_double3(ppos.x*invh,ppos.y*invh,ppos.z*invh));//shape_function(fabs(r1))*shape_function(fabs(r2))*shape_function(fabs(r3));
					sum.x += vel_C.x * w;
					sum.y += vel_C.y * w;
					sum.z += vel_C.z * w;
				}
			}
			//int3 idx000;
			//double3 coeff;
			//uint4 BIdx, TIdx;
			//getTriCoeff(make_double3(ppos.x,ppos.y,ppos.z),1.0/invh, idx000,coeff,BIdx, TIdx,dimx, dimy, dimz);
			//double4 b,t;
			//b = make_double4(far_u[BIdx.x],far_u[BIdx.y],far_u[BIdx.z],far_u[BIdx.w]);
			//t = make_double4(far_u[TIdx.x],far_u[TIdx.y],far_u[TIdx.z],far_u[TIdx.w]);
			//triInterp(coeff, sum.x,b,t);

			//b = make_double4(far_v[BIdx.x],far_v[BIdx.y],far_v[BIdx.z],far_v[BIdx.w]);
			//t = make_double4(far_v[TIdx.x],far_v[TIdx.y],far_v[TIdx.z],far_v[TIdx.w]);
			//triInterp(coeff, sum.y,b,t);

			//b = make_double4(far_w[BIdx.x],far_w[BIdx.y],far_w[BIdx.z],far_w[BIdx.w]);
			//t = make_double4(far_w[TIdx.x],far_w[TIdx.y],far_w[TIdx.z],far_w[TIdx.w]);
			//triInterp(coeff, sum.z,b,t);


			//taylor expansion
			p_u[idx] = sum.x;
			p_v[idx] = sum.y;
			p_w[idx] = sum.z;
			

		}
		
	}
}
void BiotSavartInterpolateFarField(float4 * pos,
					  double * far_u,double * far_v,double * far_w,
					  double * p_u, double * p_v,double * p_w,
					  double cell_h,
					  uint dimx,uint dimy,uint dimz,
					  uint num_particles,
					  double4 origin)
{
	uint threads = 256;
	uint blocks = num_particles/threads + (!(num_particles%threads)?0:1);
	BiotSavartInterpolate_far_field_kernel2<<<blocks, threads>>>(pos,far_u,far_v,far_w,p_u,p_v,p_w,1.0/cell_h,dimx,dimy,dimz,num_particles,origin);
}
__global__ void apply_Dirichlet_kernel(
					double * RHS,
					double4 center,
					double monopole,
					double4 grid_coner,
					double h,
					double h_sqr,
					uint dimx,
					uint dimy,
					uint dimz,
					double inv_h,
					double invny)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int k = truncf((blockIdx.y*blockDim.y + threadIdx.y)*invny);
	int j = (blockIdx.y*blockDim.y + threadIdx.y)-__umul24(dimy,k);

	if(i<dimx && j<dimy && k<dimz)
	{
		if(i==0)//left boundary
		{
			double4 ghoast_cell = make_double4((double)i-1.0+0.5, (double)j+0.5, (double)k+0.5 , 0);
			ghoast_cell.x *= h;
			ghoast_cell.y *= h;
			ghoast_cell.z *= h;
			double4 world_pos = ghoast_cell;
			world_pos.x += grid_coner.x;
			world_pos.y += grid_coner.y;
			world_pos.z += grid_coner.z;

			double w = compute_r(world_pos, center);
			double phi = monopole/w;

			uint gidx = k*dimx*dimy + j*dimx + i;
			RHS[gidx] -= phi;

		}
		if(i==dimx-1)//right boundary
		{
			double4 ghoast_cell = make_double4((double)i+1.0+0.5, (double)j+0.5, (double)k+0.5 , 0);
			ghoast_cell.x *= h;
			ghoast_cell.y *= h;
			ghoast_cell.z *= h;
			double4 world_pos = ghoast_cell;
			world_pos.x += grid_coner.x;
			world_pos.y += grid_coner.y;
			world_pos.z += grid_coner.z;

			double w = compute_r(world_pos, center);
			double phi = monopole/w;

			uint gidx = k*dimx*dimy + j*dimx + i;
			RHS[gidx] -= phi;
		}
		if(j==0)//bottom boundary
		{
			double4 ghoast_cell = make_double4((double)i+0.5, (double)j-1.0+0.5, (double)k+0.5 , 0);
			ghoast_cell.x *= h;
			ghoast_cell.y *= h;
			ghoast_cell.z *= h;
			double4 world_pos = ghoast_cell;
			world_pos.x += grid_coner.x;
			world_pos.y += grid_coner.y;
			world_pos.z += grid_coner.z;

			double w = compute_r(world_pos, center);
			double phi = monopole/w;

			uint gidx = k*dimx*dimy + j*dimx + i;
			RHS[gidx] -= phi;
		}
		if(j==dimy-1)//top boundary
		{
			double4 ghoast_cell = make_double4((double)i+0.5, (double)j+1.0+0.5, (double)k+0.5 , 0);
			ghoast_cell.x *= h;
			ghoast_cell.y *= h;
			ghoast_cell.z *= h;
			double4 world_pos = ghoast_cell;
			world_pos.x += grid_coner.x;
			world_pos.y += grid_coner.y;
			world_pos.z += grid_coner.z;

			double w = compute_r(world_pos, center);
			double phi = monopole/w;

			uint gidx = k*dimx*dimy + j*dimx + i;
			RHS[gidx] -= phi;
		}
		if(k==0)//front boundary
		{
			double4 ghoast_cell = make_double4((double)i+0.5, (double)j+0.5, (double)k-1.0+0.5 , 0);
			ghoast_cell.x *= h;
			ghoast_cell.y *= h;
			ghoast_cell.z *= h;
			double4 world_pos = ghoast_cell;
			world_pos.x += grid_coner.x;
			world_pos.y += grid_coner.y;
			world_pos.z += grid_coner.z;

			double w = compute_r(world_pos, center);
			double phi = monopole/w;

			uint gidx = k*dimx*dimy + j*dimx + i;
			RHS[gidx] -= phi;
		}
		if(k==dimz-1)//back boundary
		{
			double4 ghoast_cell = make_double4((double)i+0.5, (double)j+0.5, (double)k+1.0+0.5 , 0);
			ghoast_cell.x *= h;
			ghoast_cell.y *= h;
			ghoast_cell.z *= h;
			double4 world_pos = ghoast_cell;
			world_pos.x += grid_coner.x;
			world_pos.y += grid_coner.y;
			world_pos.z += grid_coner.z;

			double w = compute_r(world_pos, center);
			double phi = monopole/w;

			uint gidx = k*dimx*dimy + j*dimx + i;
			RHS[gidx] -= phi;
		}
	}

}

void applyDirichlet(double * RHS,
					double4 center,
					double monopole,
					double4 grid_coner,
					double h,
					uint dimx,
					uint dimy,
					uint dimz)
{
	dim3 threads(16,16);
	dim3 blocks(dimx/16 + (!(dimx%16)?0:1), dimz*dimy/16 + (!(dimz*dimy%16)?0:1));
	apply_Dirichlet_kernel<<<blocks, threads>>>(RHS, center, monopole, grid_coner, h, h*h, dimx, dimy, dimz, 1.0/h, 1.0/(double)dimy);
}


__global__
void move_particle_kernel(float4 * vortpos,
						  double4 * u,
						  double dt,
						  uint N)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<N)
	{
		double4 pos = make_double4(vortpos[idx].x,vortpos[idx].y,vortpos[idx].z,0);
		pos.x += u[idx].x*dt;
		pos.y += u[idx].y*dt;
		pos.z += u[idx].z*dt;
		vortpos[idx] = make_float4(pos.x,pos.y,pos.z,0);
	}
}

void move_particle(float4 * pos, double dt, double4 *u, uint num_particles)
{
	uint threads = 512;
	uint blocks = num_particles/threads + (!(num_particles%threads)?0:1);
	move_particle_kernel<<<blocks, threads>>>(pos, u,dt,num_particles);
}


__global__
void compute_vort_kernel(float4 * posa,
						 float4 * posb,
						 double * vortx,
						 double * vorty,
						 double * vortz,
						 double * dvortx,
						 double * dvorty,
						 double * dvortz,
						 double * k,
						 uint N)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<N)
	{
		double4 s = make_double4(posb[idx].x-posa[idx].x,
								 posb[idx].y-posa[idx].y,
								 posb[idx].z-posa[idx].z,
								 0);

		double vx = vortx[idx];
		double vy = vorty[idx];
		double vz = vortz[idx];
		dvortx[idx] = k[idx]*s.x - vx;
		dvorty[idx] = k[idx]*s.y - vy;
		dvortz[idx] = k[idx]*s.z - vz;
		//vortx[idx] = k[idx]*s.x;
		//vorty[idx] = k[idx]*s.y;
		//vortz[idx] = k[idx]*s.z;
	}
}

void ComputeVortex(float4 * posa,float4 *posb, double* vortx, double * vorty, double * vortz,double * dvortx, double * dvorty, double * dvortz,double * k, uint num_particles)
{
	uint threads = 256;
	uint blocks = num_particles/threads + (!(num_particles%threads)?0:1);
	compute_vort_kernel<<<blocks, threads>>>(posa, posb, vortx, vorty,vortz,dvortx,dvorty,dvortz,k,num_particles);
}

__global__
void compute_posb_kernel(float4 *posa,
						 float4 *posc,
					  float4 *posb,
					  double * vortx,
					  double * vorty,
					  double * vortz,
					  double * k,
					  double h,
					  uint N)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<N)
	{
		double4 s = make_double4(vortx[idx],
								 vorty[idx],
								 vortz[idx],
								 0);
		
		double l = sqrt(s.x*s.x+s.y*s.y+s.z*s.z);
		
		if(l<h)
		{
			k[idx]=1;
			posb[idx].x = posa[idx].x + 0.5*s.x;
			posb[idx].y = posa[idx].y + 0.5*s.y;
			posb[idx].z = posa[idx].z + 0.5*s.z;
			posc[idx].x = posa[idx].x - 0.5*s.x;
			posc[idx].y = posa[idx].y - 0.5*s.y;
			posc[idx].z = posa[idx].z - 0.5*s.z;

		}
		else
		{
			k[idx] = l/h;
			posb[idx].x = posa[idx].x + 0.5*s.x/l*h;
			posb[idx].y = posa[idx].y + 0.5*s.y/l*h;
			posb[idx].z = posa[idx].z + 0.5*s.z/l*h;
			posc[idx].x = posa[idx].x - 0.5*s.x/l*h;
			posc[idx].y = posa[idx].y - 0.5*s.y/l*h;
			posc[idx].z = posa[idx].z - 0.5*s.z/l*h;
		}


	}
}

void ComputePosb(float4 *posa, float4 *posc, float4 *posb, double * vortx, double * vorty, double * vortz, double * k, double h, uint num_particles)
{
	uint threads = 256;
	uint blocks = num_particles/threads + (!(num_particles%threads)?0:1);
	compute_posb_kernel<<<blocks, threads>>>(posa, posc, posb, vortx, vorty,vortz,k,h,num_particles);
}


__global__ void
compute_posa_kernel(float4 *posc, float4 *posb, float4* posa, uint num_particles)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<num_particles)
	{
		posa[idx].x = 0.5*(posb[idx].x + posc[idx].x);
		posa[idx].y = 0.5*(posb[idx].y + posc[idx].y);
		posa[idx].z = 0.5*(posb[idx].z + posc[idx].z);
	}
}

void ComputePosa(float4 *posc, float4 *posb, float4* posa, uint num_particles)
{
	uint threads = 256;
	uint blocks = blocks = num_particles/threads + (!(num_particles%threads)?0:1);
	compute_posa_kernel<<<blocks, threads>>>(posc,posb,posa,num_particles);
}

__global__
void BiotSavartparticle_outside_kernel(float4 * pos,
							 double3 omega,
							 double3 vort_center,
							 float3 bbmin,
							 float3 bbmax,
							 double* u,
							 double* v,
							 double* w,
							 uint M)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<M)
	{
		double3 ppos= make_double3(pos[idx].x, pos[idx].y, pos[idx].z);
		if(ppos.x<bbmin.x || ppos.x>bbmax.x 
		|| ppos.y<bbmin.y || ppos.y>bbmax.y
		|| ppos.z<bbmin.z || ppos.z>bbmax.z)
		{
			double3 vel = make_double3(0,0,0);
			double3 dir = make_double3(ppos.x-vort_center.x,ppos.y-vort_center.y,ppos.z-vort_center.z);
			double r3 = compute_r3(dir);
			vel = cross_uxv(omega, dir);
			vel.x = vel.x/r3;
			vel.y = vel.y/r3;
			vel.z = vel.z/r3;

			u[idx]=0;
			v[idx]=0;
			w[idx]=0;
		}
	}
}

void BiotSavartComputeVelocityForOutParticle(float4 * pos, double3 omega, double3 vort_center, float3 bbmin, float3 bbmax, double* u, double* v, double* w, uint M)
{	
	uint threads = 256;
	uint blocks = M/threads + (!(M%threads)?0:1);
	BiotSavartparticle_outside_kernel<<<blocks, threads>>>(pos, omega, vort_center, bbmin, bbmax, u, v, w, M);

}


__global__
void compute_sph_stretching_kernel(const uint * start,
							const uint * end,
							float4 *pos,
							float4 *pos_sort,
							double *vort_x,
							double *vort_y,
							double *vort_z,
							double *u_sort,
							double *v_sort,
							double *w_sort,
							double invh,
							uint3 gdim,
							uint3 hdim,
							uint2 gstride,
							uint2 hstride,
							uint N, 
							double4 origin)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	uint p=0;
	if(idx<0)
	{

	}
}


__global__ 
void particle_gaussian_kernel(const uint * start,
											 const uint * end,
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
											 double invh,
											 uint3 gdim,
											 uint3 hdim,
											 uint2 gstride,
											 uint2 hstride,
											 uint N, 
											 double4 origin)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	uint p=0;
	if(idx<N)
	{
		
		float4 pposa = pos[idx];
		
		pposa.x -= origin.x;
		pposa.y -= origin.y;
		pposa.z -= origin.z;
		
		double3 omega1 = make_double3(o_vortx[idx],o_vorty[idx], o_vortz[idx]);
		//compute the grid idx it belongs to
		int i = floor(pposa.x*invh);
		int j = floor(pposa.y*invh);
		int k = floor(pposa.z*invh);
		
		double3 dwdt = make_double3(0,0,0);
		double total_weight = 0.0;
		for(int kk=k-2;kk<=k+2;kk++)for(int jj= j-2; jj<=j+2; jj++)for(int ii=i-2; ii<=i+2; ii++)
		{
			if(kk>=0&&kk<hdim.z && jj>=0&&jj<hdim.y && ii>=0&&ii<hdim.x)
			{
				uint read_idx = kk*hstride.x + jj*hstride.y + ii;
				for(p = start[read_idx]; p<end[read_idx]; p++)
				{
					if(p<N)
					{
						
				
						double3 omega = make_double3(i_vortx[p],i_vorty[p], i_vortz[p]);
						//dwdt.x += omega.x-omega1.x;
						//dwdt.y += omega.y-omega1.y;
						//dwdt.z += omega.z-omega1.z;
						float4 pposb = pos_sort[p];
						pposb.x -= origin.x;
						pposb.y -= origin.y;
						pposb.z -= origin.z;
						//double blur_r = 0.075;
						double omg = nu;
						//double scale = 2*omg/blur_r;
						double3 dir = make_double3(pposb.x-pposa.x, pposb.y-pposa.y, pposb.z-pposa.z);
						//dir.x*=scale; dir.y*=scale; dir.z*=scale;
						double dir_sqr = dir.x*dir.x + dir.y*dir.y + dir.z*dir.z;
						
						
						double weight = exp(-dir_sqr/(2*omg*omg));
						total_weight += weight;
						dwdt.x += weight * omega.x;
						dwdt.y += weight * omega.y;
						dwdt.z += weight * omega.z;
					}
				}
			}
		}
		if(total_weight<1e-12) total_weight = 1e+24;
		o_vortx[idx] = dwdt.x/total_weight;
		o_vorty[idx] = dwdt.y/total_weight;
		o_vortz[idx] = dwdt.z/total_weight;
		//o_vortx[idx] += nu*dt*dwdt.x;
		//o_vorty[idx] += nu*dt*dwdt.y;
		//o_vortz[idx] += nu*dt*dwdt.z;
	}
		
}

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
				  double4 origin)
{
	uint threads = 256;
	uint blocks = N/threads + (!(N%threads)?0:1);
	particle_gaussian_kernel<<<blocks, threads>>>(start, end,
		pos, pos_sort, nu, dt, o_vortx, o_vorty, o_vortz, 
		i_vortx, i_vorty, i_vortz,1.0/h, gdim, gdim,
		make_uint2(gdim.x*gdim.y,gdim.x), make_uint2(gdim.x*gdim.y,gdim.x),
		N, origin);
}



__global__
void vectorAdd_kernel(double * A, double * B, double * C, double a, double b, uint num)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx<num)
	{
		C[idx]=a*A[idx]+b*B[idx];
	}
}
void VectorAdd(double * A, double * B, double * C, double a, double b, uint num)
{

	uint threads = 256;
	uint blocks = num/threads + (!(num%threads)?0:1);
	vectorAdd_kernel<<<blocks, threads>>>(A, B, C, a, b, num);
	getLastCudaError("vector add failed!\n");
}




__global__
void AdotB_kernel(double * a, double * b, double * c, uint num)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<num)
	{
		c[idx] = a[idx]*b[idx];
	}
}

void AdotB(double *a, double*b,double*c,uint num)
{
	uint threads = 256;
	uint blocks = num/threads + (!(num%threads)?0:1);
	AdotB_kernel<<<blocks, threads>>>(a,b,c,num);
	getLastCudaError("AdotB failed!\n");
}

double ComputeDot(double * a, double * b, uint num)
{

	double * c;
	double * sum;
	double * sum_hst = (double*)malloc(sizeof(double)*32);
	cudaMalloc((void**)&c,num*sizeof(double));
	cudaMalloc((void**)&sum, 32*sizeof(double));
	AdotB(a,b,c,num);
	
	reduce<double>(num, 256, 32,5, c, sum);
	//getLastCudaError("dot failed!\n");
	cudaMemcpy(sum_hst,sum,32*sizeof(double),cudaMemcpyDeviceToHost);

	double res = 0;

	for(int i=0;i<32;i++)
	{
		res += sum_hst[i];
	}


	cudaFree(sum);
	cudaFree(c);
	free(sum_hst);

	return res;
}

__global__ 
void compute_SingleLayerPotential_mass_kernel(double * den, double * area, double * mass, uint num)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<num)
	{
		mass[idx] = den[idx]*area[idx];
	}
}
void ComputeSingleLayerPotentialMass(double * den, double * area, double * mass, uint num)
{
	uint threads = 256;
	uint blocks = num/threads + (!(num%threads)?0:1);
	compute_SingleLayerPotential_mass_kernel<<<blocks, threads>>>(den, area, mass, num);
	getLastCudaError("compute single layer mass!\n");
}


__global__
void compute_NormalSlip_kernel(double * u, double * v, double * w, double u_solid, double v_solid, double w_solid, double3 * normal, double *b, uint num)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<num)
	{
		double u_n = (u_solid-u[idx])*normal[idx].x + (v_solid-v[idx])*normal[idx].y + (w_solid-w[idx])*normal[idx].z;
		b[idx] = u_n;
	}
}

void 
ComputeNormalSlip(double * u, double * v, double * w, double u_solid, double v_solid, double w_solid, double3 * normal, double *b, uint num)
{
	uint threads = 256;
	uint blocks = num/threads + (!(num%threads)?0:1);
	compute_NormalSlip_kernel<<<blocks,threads>>>(u, v, w, u_solid, v_solid, w_solid,normal, b, num);
}

__global__
void compute_BoundarySlip_kernel(double * u, double * v, double * w, double u_solid, double v_solid, double w_solid, uint num)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<num)
	{
		u[idx] = u[idx] - u_solid;
		v[idx] = v[idx] - v_solid;
		w[idx] = w[idx] - w_solid;
	}
}

void 
ComputeBoundarySlip(double * u, double * v, double * w, double u_solid, double v_solid, double w_solid, uint num)
{
	uint threads = 256;
	uint blocks = num/threads + (!(num%threads)?0:1);
	compute_BoundarySlip_kernel<<<blocks,threads>>>(u, v, w, u_solid, v_solid, w_solid, num);
}

__global__
void update_boundary_kernel(float4 * pos,  double u_solid, double v_solid, double w_solid, double dt, uint num)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<num)
	{
		pos[idx].x = pos[idx].x + dt * u_solid;
		pos[idx].y = pos[idx].y + dt * v_solid;
		pos[idx].z = pos[idx].z + dt * w_solid;
	}
}

void 
UpdateBoundary(float4 * pos,  double u_solid, double v_solid, double w_solid, double dt, uint num)
{
	uint threads = 256;
	uint blocks = num/threads + (!(num%threads)?0:1);
	update_boundary_kernel<<<blocks,threads>>>(pos,   u_solid,  v_solid,  w_solid,  dt,   num);
}


__global__
void add_u_rotation_kernel(double4 * U, double *u, double *v, double *w, uint num)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<num)
	{
		U[idx].x += u[idx];
		U[idx].y += v[idx];
		U[idx].z += w[idx];
	}
}

void AddURotation(double4 * U, double *u, double *v, double *w, uint num)
{
	uint threads = 256;
	uint blocks = num/threads + (!(num%threads)?0:1);
	add_u_rotation_kernel<<<blocks,threads>>>(U,u,v,w,num);
}

__global__
void add_u_potential_kernel(double4 * U, double3 *u, uint num)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<num)
	{
		U[idx].x += u[idx].x;
		U[idx].y += u[idx].y;
		U[idx].z += u[idx].z;
	}
}

void AddUPotential(double4 * U, double3 *u, uint num)
{
	uint threads = 256;
	uint blocks = num/threads + (!(num%threads)?0:1);
	add_u_potential_kernel<<<blocks,threads>>>(U,u,num);
}


__global__ 
void compute_shedding_pos(float4 * shedding_pos,
								float4 * boundary_pos,
								double3 * boundary_normal,
								uint N)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<N)
	{
		shedding_pos[idx] =  make_float4(boundary_pos[idx].x + 0.01*boundary_normal[idx].x,
			boundary_pos[idx].y + 0.01*boundary_normal[idx].y,
			boundary_pos[idx].z + 0.01*boundary_normal[idx].z,
			1.0);
	}
}

void ComputeSheddingPos(float4 * shedding_pos,
								float4 * boundary_pos,
								double3 * boundary_normal,
								uint N)

{
	uint threads = 256;
	uint blocks = N/threads + (!(N%threads)?0:1);
	compute_shedding_pos<<<blocks, threads>>>(shedding_pos, boundary_pos, boundary_normal, N);
}

__global__
void assign_density_kernel(const uint * start,
					  const uint * end,
					  const float4 * pos,
					  const float * rho,
					  const float * vort_size,
					  float * gridval,
					  uint3 griddim,
					  uint3 hashdim,
					  double invh,
					  double invny,
					  uint2 gstride,
					  uint2 hstride,
					  uint num_particle,
					  double4 origin)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int k = truncf((blockIdx.y*blockDim.y + threadIdx.y)*invny);
	int j = (blockIdx.y*blockDim.y + threadIdx.y)-__umul24(griddim.y,k);
	
	
	if(i<griddim.x && j<griddim.y && k<griddim.z)
	{
		double sum = 0;
		double total_weight = 0;
		double3 pijk = make_double3(((double)i+0.5),((double)j+0.5),((double)k+0.5));

		for(int kk=k-2;kk<=k+2;kk++)for(int jj= j-2; jj<=j+2; jj++)for(int ii=i-2; ii<=i+2; ii++)
		{
			if(kk>=0&&kk<hashdim.z && jj>=0&&jj<hashdim.y && ii>=0&&ii<hashdim.x)
			{
				uint read_idx = kk*hstride.x + jj*hstride.y + ii;
				if(end[read_idx]>=start[read_idx])
				{
					for(uint p = start[read_idx]; p<end[read_idx]; p++)
					{
						if(p<num_particle)
						{
							float4 ppos = pos[p];
							ppos.x -= origin.x;
							ppos.y -= origin.y;
							ppos.z -= origin.z;
							//double r1 = (double)ppos.x * invh - pijk.x;
							//double r2 = (double)ppos.y * invh - pijk.y;
							//double r3 = (double)ppos.z * invh - pijk.z;
							
							double w = interpolate_weights2(pijk, make_double3(ppos.x*invh, ppos.y*invh, ppos.z*invh));
							double rho_temp = w * rho[p];
							sum += rho_temp;
							total_weight += w;

						}
					}
				}
				else
				{
					double3 ppos = make_double3(((double)ii+0.5),((double)jj+0.5),((double)kk+0.5)); 
					double w = interpolate_weights2(pijk, ppos);
					double rho_temp = 0;
					sum += rho_temp;
					total_weight += w;
				}

			}//end if
		}//end for

		if(total_weight<1e-12) total_weight = 1e+24;
		uint write_idx = k*gstride.x + j*gstride.y + i;
		gridval[write_idx] = sum/total_weight;
	}//end if
}

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
					double4 origin)
{
	dim3 threads(16,16);
	dim3 blocks(griddim.x/16 + (!(griddim.x%16)?0:1), griddim.z*griddim.y/16 + (!(griddim.z*griddim.y%16)?0:1));

	assign_density_kernel<<<blocks,threads>>>(start, end, pos, rho,vort_size, gridval, 
		griddim, hashdim, 1.0/cell_h, 1.0/(double)griddim.y, 
		make_uint2(griddim.x*griddim.y, griddim.x), 
		make_uint2(hashdim.x*hashdim.y, hashdim.x), 
		num_particle,
		origin);
	getLastCudaError("Assign Density failed!\n");

}

__global__
void grad_density_kernel(const float * phi,
						float3 * dphi,
						double invh,
						double invny,
						uint dimx,
						uint dimy,
						uint dimz)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int k = truncf((blockIdx.y*blockDim.y + threadIdx.y)*invny);
	int j = (blockIdx.y*blockDim.y + threadIdx.y)-__umul24(dimy,k);
	
	if(i<dimx && j<dimy && k <dimz)
	{
		int im1 = (i+dimx-1)%dimx;
		int ip1 = (i+1)%dimx;
		int jm1 = (j+dimy-1)%dimy;
		int jp1 = (j+1)%dimy;
		int km1 = (k-1+dimz)%dimz;;
		int kp1 = (k+1)%dimz;
		int slice = dimx*dimy;
		int stride = dimx;
		int idx_c = k*slice + j*stride + i;
		int idx_l = k*slice + j*stride + im1;
		int idx_r = k*slice + j*stride + ip1;
		int idx_t = k*slice + jp1*stride + i;
		int idx_d = k*slice + jm1*stride + i;
		int idx_f = km1*slice + j*stride + i;
		int idx_b = kp1*slice + j*stride + i;
		double t = phi[idx_t];
		double d = phi[idx_d];
		double l = phi[idx_l];
		double r = phi[idx_r];
		double f = phi[idx_f];
		double b = phi[idx_b];
		dphi[idx_c].x = (r - l)*0.5*invh;
		dphi[idx_c].y = (t - d)*0.5*invh;
		dphi[idx_c].z = (b - f)*0.5*invh;
	}
}
void GradDensity(float* phi,
					 float3 * dphi,
					 double cell_h,
					 uint dimx, 
					 uint dimy, 
					 uint dimz)
{
	dim3 threads(16,16);
	dim3 blocks(dimx/16 + (!(dimx%16)?0:1), dimz*dimy/16 + (!(dimz*dimy%16)?0:1));
	grad_density_kernel<<<blocks, threads>>>(phi, dphi,1.0/cell_h, 1.0/(double)dimy, dimx, dimy, dimy);
	getLastCudaError("GradDensity failed\n!");
}





__global__
void baroclinic_kernel(float4 * vortex_pos,
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
					   double4 origin
					   )

{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<num_vort)
	{
		double3 sum=make_double3(0,0,0);
		float4 ppos = vortex_pos[idx];
		ppos.x -= origin.x;
		ppos.y -= origin.y;
		ppos.z -= origin.z;
		//compute the grid idx it belongs to
		int i = floor(ppos.x*inv_h);
		int j = floor(ppos.y*inv_h);
		int k = floor(ppos.z*inv_h);
		//loop over the grid around it to cumulate force
		for(int kk=k-2;kk<=k+2;kk++)for(int jj= j-2; jj<=j+2; jj++)for(int ii=i-2; ii<=i+2; ii++)
		{
			if(kk>=0&&kk<dimz && jj>=0&&jj<dimy && ii>=0&&ii<dimx)
			{
				//fetch the grid value
				uint read_idx = kk*dimx*dimy + jj * dimx + ii;
				float3 f = grad_rho[read_idx];
				double3 pijk = make_double3(((double)ii+0.5),((double)jj+0.5),((double)kk+0.5));
				//double r1 = (double)ppos.x * invh - pijk.x;
				//double r2 = (double)ppos.y * invh - pijk.y;
				//double r3 = (double)ppos.z * invh - pijk.z;
				double w = interpolate_weights(pijk, make_double3(ppos.x*inv_h, ppos.y*inv_h, ppos.z*inv_h));
				
				sum.x += f.x * w;
				sum.y += f.y * w;
				sum.z += f.z * w;
			}
		}
		//int3 idx000;
		//double3 coeff;
		//uint4 BIdx, TIdx;
		//getTriCoeff(make_double3(ppos.x,ppos.y,ppos.z),1.0/inv_h, idx000,coeff,BIdx, TIdx,dimx, dimy, dimz);
		//double4 b,t;
		//b = make_double4(grad_rho[BIdx.x].x,grad_rho[BIdx.y].x,grad_rho[BIdx.z].x,grad_rho[BIdx.w].x);
		//t = make_double4(grad_rho[TIdx.x].x,grad_rho[TIdx.y].x,grad_rho[TIdx.z].x,grad_rho[TIdx.w].x);
		//triInterp(coeff, sum.x,b,t);

		//b = make_double4(grad_rho[BIdx.x].y,grad_rho[BIdx.y].y,grad_rho[BIdx.z].y,grad_rho[BIdx.w].y);
		//t = make_double4(grad_rho[TIdx.x].y,grad_rho[TIdx.y].y,grad_rho[TIdx.z].y,grad_rho[TIdx.w].y);
		//triInterp(coeff, sum.y,b,t);

		//b = make_double4(grad_rho[BIdx.x].z,grad_rho[BIdx.y].z,grad_rho[BIdx.z].z,grad_rho[BIdx.w].z);
		//t = make_double4(grad_rho[TIdx.x].z,grad_rho[TIdx.y].z,grad_rho[TIdx.z].z,grad_rho[TIdx.w].z);
		//triInterp(coeff, sum.z,b,t);
		double3 baroclinic = cross_uxv(sum, make_double3(0,10,0));
		vort_x[idx] += dt_beta * 0.1 * baroclinic.x;
		vort_y[idx] += dt_beta * 0.1 * baroclinic.y;
		vort_z[idx] += dt_beta * 0.1 * baroclinic.z;

		
	}
}

void ParticleGetBaroclinic(float4 * vortex_pos,
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
					   double4 origin)
{
	uint threads = 256;
	uint blocks = num_vort/threads + (!(num_vort%threads)?0:1);
	baroclinic_kernel<<<threads, blocks>>>(vortex_pos, vort_x, vort_y, vort_z,grad_rho, dt_beta, num_vort, dimx, dimy, dimz, inv_h, origin);
}


__global__
void fix_boundary_kernel(float4 * pos,
						 double cx,
						 double cy,
						 double cz,
						 double r,
						 uint num)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<num)
	{
		float4 ppos = pos[idx];
		double3 to_sphere = make_double3(ppos.x-cx, ppos.y-cy, ppos.z-cz);
		double d = sqrt(to_sphere.x*to_sphere.x+to_sphere.y*to_sphere.y+to_sphere.z*to_sphere.z);

		if(d<r)//push it out
		{
			double3 unit_vec = make_double3(to_sphere.x/d, to_sphere.y/d, to_sphere.z/d);
			double  l = r*1.01;
			pos[idx] = make_float4(cx + l*unit_vec.x, cy + l*unit_vec.y, cz + l*unit_vec.z,0);
		}
	}
}

void FixBoundary(float4 * pos,
						 double cx,
						 double cy,
						 double cz,
						 double r,
						 uint num)
{
	//uint threads = 256;
	//uint blocks = num/threads + (!(num%threads)?0:1);
	//fix_boundary_kernel<<<blocks, threads>>>(pos, cx, cy, cz, r, num);

}


__global__
void arrayAppendf4(float4 *new_array,
				   float4 *old_array,
				   float4 *append_array,
				   uint n_old,
				   uint n_new)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<n_new)
	{
		if(idx<n_old)
		{
			new_array[idx] = old_array[idx];
		}
		else
		{
			new_array[idx] = append_array[idx-n_old];
		}
	}
}

void appendArrayf4(float4 *new_array,
				   float4 *old_array,
				   float4 *append_array,
				   uint n_old,
				   uint n_new)
{
	uint threads = 256;
	uint blocks = n_new/threads + (!(n_new%threads)?0:1);
	arrayAppendf4<<<blocks, threads>>>(new_array, old_array, append_array, n_old, n_new);
}

__global__
void arrayAppendd4(double4 *new_array,
				   double4 *old_array,
				   double4 *append_array,
				   uint n_old,
				   uint n_new)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<n_new)
	{
		if(idx<n_old)
		{
			new_array[idx] = old_array[idx];
		}
		else
		{
			new_array[idx] = append_array[idx-n_old];
		}
	}
}

void appendArrayd4(double4 *new_array,
				   double4 *old_array,
				   double4 *append_array,
				   uint n_old,
				   uint n_new)
{
	uint threads = 256;
	uint blocks = n_new/threads + (!(n_new%threads)?0:1);
	arrayAppendd4<<<blocks, threads>>>(new_array, old_array, append_array, n_old, n_new);
}

__global__
void arrayAppendd(double *new_array,
				   double *old_array,
				   double *append_array,
				   uint n_old,
				   uint n_new)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<n_new)
	{
		if(idx<n_old)
		{
			new_array[idx] = old_array[idx];
		}
		else
		{
			new_array[idx] = append_array[idx-n_old];
		}
	}
}

void appendArrayd(double *new_array,
				   double *old_array,
				   double *append_array,
				   uint n_old,
				   uint n_new)
{
	uint threads = 256;
	uint blocks = n_new/threads + (!(n_new%threads)?0:1);
	arrayAppendd<<<blocks, threads>>>(new_array, old_array, append_array, n_old, n_new);
}

__global__
void compute_boundary_vortex_kernel(double *u,
									double *v,
									double *w,
									double *vort_x,
									double *vort_y,
									double *vort_z,
									float4 *boundary_pos,
									double3 * normal,
									double *area,
									double dt_c,
									uint num)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<num)
	{
		double normal_component = my_dot(normal[idx], make_double3(u[idx],v[idx],w[idx]));
		double3 v_tangent;
		v_tangent.x = u[idx] - normal_component*normal[idx].x;
		v_tangent.y = v[idx] - normal_component*normal[idx].y;
		v_tangent.z = w[idx] - normal_component*normal[idx].z;
		double3 intensity = cross_uxv(normal[idx], v_tangent);
		intensity.x = intensity.x;
		intensity.y = intensity.y;
		intensity.z = intensity.z;
		vort_x[idx] = dt_c * area[idx] * intensity.x;
		vort_y[idx] = dt_c * area[idx] * intensity.y;
		vort_z[idx] = dt_c * area[idx] * intensity.z;
	}
}
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
									uint num)
{
	uint threads = 256;
	uint blocks = num/threads + (!(num%threads)?0:1);
	compute_boundary_vortex_kernel<<<blocks, threads>>>(u,
									v,
									w,
									vort_x,
									vort_y,
									vort_z,
									boundary_pos,
									normal,
									area,
									dt_c,
									num);
}

__global__
void vortex_shedding_kernel(uint *start,
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
							uint num_vortexEle)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	uint p=0;
	
	if(idx<num_boundaryEle)
	{
		float4 pposa = boundary_pos[idx];
		
		pposa.x -= origin.x;
		pposa.y -= origin.y;
		pposa.z -= origin.z;
		

		//compute the grid idx it belongs to
		int i = floor(pposa.x*inv_h);
		int j = floor(pposa.y*inv_h);
		int k = floor(pposa.z*inv_h);

		//double total_weight = 0.0;
		//double3 dwdt = make_double3(0,0,0);
		
		//find the closest vortex blob
		double close_r = 0.1;
		int close_p = -1;
		for(int kk=k-1;kk<=k+1;kk++)for(int jj= j-1; jj<=j+1; jj++)for(int ii=i-1; ii<=i+1; ii++)
		{
			if(kk>=0&&kk<dimz && jj>=0&&jj<dimy && ii>=0&&ii<dimx)
			{
				uint read_idx = kk*dimx*dimy + jj*dimx + ii;
				for(p = start[read_idx]; p<end[read_idx]; p++)
				{
					if(p<num_vortexEle)
					{

						float4 pposb = vort_pos[p];
						pposb.x -= origin.x;
						pposb.y -= origin.y;
						pposb.z -= origin.z;
						double3 dir = make_double3(pposb.x-pposa.x, pposb.y-pposa.y, pposb.z-pposa.z);
						double r = sqrt(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
						if(r<close_r)
						{
							close_r = r;
							close_p = p;
						}
						//double3 omega = make_double3(boundary_vortx[p]-vort_x[idx], boundary_vorty[p]-vort_y[idx], boundary_vortz[p]-vort_z[idx]);
						//float4 pposb = boundary_pos[p];
						//pposb.x -= origin.x;
						//pposb.y -= origin.y;
						//pposb.z -= origin.z;
						//double omg = shedding_r;
						//double3 dir = make_double3(pposb.x-pposa.x, pposb.y-pposa.y, pposb.z-pposa.z);
						//dir.x/=shedding_r; dir.y/=shedding_r; dir.z/=shedding_r;

						//double dir_sqr = dir.x*dir.x + dir.y*dir.y + dir.z*dir.z;
						//double weight = 0;
						//if(dir_sqr<=4.0) weight = 0.269/(1+dir_sqr); 
						//
						//
						////total_weight += weight;
						//dwdt.x += weight * omega.x;
						//dwdt.y += weight * omega.y;
						//dwdt.z += weight * omega.z;
					}
				}
			}
		}//end for
		__syncthreads();
		if(close_p>=0)
		{
			vort_x[close_p] += dt_c * boundary_vortx[idx];
			vort_y[close_p] += dt_c * boundary_vorty[idx];
			vort_z[close_p] += dt_c * boundary_vortz[idx];
		}

		////if(total_weight<1e-12) total_weight = 1e+24;
		//double invr2=1.0/shedding_r/shedding_r;
		//vort_x[idx] += dt_c*dwdt.x;
		//vort_y[idx] += dt_c*dwdt.y;
		//vort_z[idx] += dt_c*dwdt.z;	
	}
}



__global__
void vortex_shedding_kernel2(uint *start,
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
							uint num_vortexEle)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	uint p=0;
	
	if(idx<num_vortexEle)
	{
		float4 pposa = boundary_pos[idx];
		
		pposa.x -= origin.x;
		pposa.y -= origin.y;
		pposa.z -= origin.z;
		

		//compute the grid idx it belongs to
		int i = floor(pposa.x*inv_h);
		int j = floor(pposa.y*inv_h);
		int k = floor(pposa.z*inv_h);

		//double total_weight = 0.0;
		double3 dwdt = make_double3(0,0,0);
		
		//find the closest vortex blob
		for(int kk=k-1;kk<=k+1;kk++)for(int jj= j-1; jj<=j+1; jj++)for(int ii=i-1; ii<=i+1; ii++)
		{
			if(kk>=0&&kk<dimz && jj>=0&&jj<dimy && ii>=0&&ii<dimx)
			{
				uint read_idx = kk*dimx*dimy + jj*dimx + ii;
				for(p = start[read_idx]; p<end[read_idx]; p++)
				{
					if(p<num_vortexEle)
					{

						double3 omega = make_double3(boundary_vortx[p]-vort_x[idx], boundary_vorty[p]-vort_y[idx], boundary_vortz[p]-vort_z[idx]);
						float4 pposb = boundary_pos[p];
						pposb.x -= origin.x;
						pposb.y -= origin.y;
						pposb.z -= origin.z;
						double omg = shedding_r;
						double3 dir = make_double3(pposb.x-pposa.x, pposb.y-pposa.y, pposb.z-pposa.z);
						dir.x/=shedding_r; dir.y/=shedding_r; dir.z/=shedding_r;

						double dir_sqr = dir.x*dir.x + dir.y*dir.y + dir.z*dir.z;
						double weight = 0;
						if(dir_sqr<=4.0) weight = 0.269/(1+dir_sqr); 
						
						
						//total_weight += weight;
						dwdt.x += weight * omega.x;
						dwdt.y += weight * omega.y;
						dwdt.z += weight * omega.z;
					}
				}
			}
		}//end for


		////if(total_weight<1e-12) total_weight = 1e+24;
		//double invr2=1.0/shedding_r/shedding_r;
		vort_x[idx] += dt_c*dwdt.x;
		vort_y[idx] += dt_c*dwdt.y;
		vort_z[idx] += dt_c*dwdt.z;	
	}
}

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
					uint num_vortexEle)
{
	uint threads = 256;
	uint blocks = num_boundaryEle/threads + (!(num_boundaryEle%threads)?0:1);
	vortex_shedding_kernel<<<blocks,threads>>>(start,end,
		vort_pos,
		vort_x,vort_y,vort_z,
		boundary_pos,
		boundary_vortx,boundary_vorty,boundary_vortz,
		shedding_r,
		dt_c,
		dimx, dimy,dimz,
		inv_h,
		origin,
		num_boundaryEle,
		num_vortexEle);
}


#endif