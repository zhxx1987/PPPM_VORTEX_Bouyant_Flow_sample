#include "MultiGrid.h"

namespace gf{

	void MultiGridSolver3DPeriod::m_Vcycle(GpuArrayf* x, GpuArrayf* b, float tol, float &residual, int level)
	{
		xk[level] = x;
		bk[level] = b;

		for(int i=level+1; i<m_max_level; i++)
		{
			mg_zerofy(xk[i]->getDevicePtr(), xk[i]->getSize()*xk[i]->typeSize());
			mg_zerofy(bk[i]->getDevicePtr(), bk[i]->getSize()*bk[i]->typeSize());
			mg_zerofy(rk[i]->getDevicePtr(), rk[i]->getSize()*rk[i]->typeSize());
		}
		mg_zerofy(rk[level]->getDevicePtr(), rk[level]->getSize()*rk[level]->typeSize());
		int Loop  = 0 ;


		for(int i=level; i<m_max_level-1; ++i)
		{
			mg_RBGS_periodic3D(xk[i]->getDevicePtr(), 
				bk[i]->getDevicePtr(), 
				xk[i]->getDevicePtr(), 
				4, 
				systemk[i].gridx,
				systemk[i].gridy,
				systemk[i].gridz);

			mg_ComputeResidualPeriodic3D(xk[i]->getDevicePtr(),
				bk[i]->getDevicePtr(),
				rk[i]->getDevicePtr(),
				systemk[i].gridx,
				systemk[i].gridy,
				systemk[i].gridz);

			mg_RestrictionPeriodic3D(bk[i+1]->getDevicePtr(), 
				rk[i]->getDevicePtr(), 
				systemk[i+1].gridx,
				systemk[i+1].gridy,
				systemk[i+1].gridz);
		}

		mg_exact_periodic3D(xk[m_max_level-1]->getDevicePtr(),
			bk[m_max_level-1]->getDevicePtr(),
			xk[m_max_level-1]->getDevicePtr(),
			systemk[m_max_level-1].gridx,
			systemk[m_max_level-1].gridy,
			systemk[m_max_level-1].gridz);

		for(int i=m_max_level-2; i>=level; --i)
		{
			mg_ProlongationPeriodic3D(xk[i]->getDevicePtr(),
				xk[i+1]->getDevicePtr(),
				systemk[i].gridx,
				systemk[i].gridy,
				systemk[i].gridz);

			mg_RBGS_periodic3D(xk[i]->getDevicePtr(), 
				bk[i]->getDevicePtr(), 
				xk[i]->getDevicePtr(),  
				4, 
				systemk[i].gridx,
				systemk[i].gridy,
				systemk[i].gridz);
		}
		Loop++;
		for(int i=level+1; i<m_max_level; i++)
		{
			mg_zerofy(xk[i]->getDevicePtr(), xk[i]->getSize()*xk[i]->typeSize());
			mg_zerofy(bk[i]->getDevicePtr(), bk[i]->getSize()*bk[i]->typeSize());
			mg_zerofy(rk[i]->getDevicePtr(), rk[i]->getSize()*rk[i]->typeSize());

		}


	}
	void MultiGridSolver3DPeriod::m_FullMultiGrid(GpuArrayf* x, GpuArrayf* b, float tol, float &residual)
	{
		xk[0] = x ;
		bk[0] = b ;
		float res;
		for(int i=1; i<m_max_level; i++)
		{
			mg_zerofy(xk[i]->getDevicePtr(), xk[i]->getSize()*xk[i]->typeSize());
			mg_zerofy(bk[i]->getDevicePtr(), bk[i]->getSize()*bk[i]->typeSize());
			mg_zerofy(rk[i]->getDevicePtr(), rk[i]->getSize()*rk[i]->typeSize());
		}
		mg_zerofy(rk[0]->getDevicePtr(), rk[0]->getSize()*rk[0]->typeSize());
		for(int i=0; i<m_max_level-1; ++i)
		{
			mg_ComputeResidualPeriodic3D(xk[i]->getDevicePtr(),
				bk[i]->getDevicePtr(),
				rk[i]->getDevicePtr(),
				systemk[i].gridx,
				systemk[i].gridy,
				systemk[i].gridz);

			mg_RestrictionPeriodic3D(bk[i+1]->getDevicePtr(), 
				rk[i]->getDevicePtr(), 
				systemk[i+1].gridx,
				systemk[i+1].gridy,
				systemk[i+1].gridz);
		}
		mg_exact_periodic3D(xk[m_max_level-1]->getDevicePtr(),
			bk[m_max_level-1]->getDevicePtr(),
			xk[m_max_level-1]->getDevicePtr(),
			systemk[m_max_level-1].gridx,
			systemk[m_max_level-1].gridy,
			systemk[m_max_level-1].gridz);
		for(int i=m_max_level-2; i>=0; --i)
		{
			mg_ProlongationPeriodic3D(xk[i]->getDevicePtr(),
				xk[i+1]->getDevicePtr(),
				systemk[i].gridx,
				systemk[i].gridy,
				systemk[i].gridz);

			m_Vcycle(xk[i], bk[i], 1e-10, res, i);
		}
		for(int i=0; i<1; i++)
		{
			m_Vcycle(xk[0],b,1e-10, res,0);
		}

	}

	void MultiGridSolver3DPeriod_double::m_Vcycle(GpuArrayd* x, GpuArrayd* b, double tol, double &residual, int level)
	{
		xk[level] = x;
		bk[level] = b;

		for(int i=level+1; i<m_max_level; i++)
		{
			mg_zerofy(xk[i]->getDevicePtr(), xk[i]->getSize()*xk[i]->typeSize());
			mg_zerofy(bk[i]->getDevicePtr(), bk[i]->getSize()*bk[i]->typeSize());
			mg_zerofy(rk[i]->getDevicePtr(), rk[i]->getSize()*rk[i]->typeSize());
		}
		mg_zerofy(rk[level]->getDevicePtr(), rk[level]->getSize()*rk[level]->typeSize());
		int Loop  = 0 ;


		for(int i=level; i<m_max_level-1; ++i)
		{

			mg_RBGS_periodic3D2(xk[i]->getDevicePtr(), 
				bk[i]->getDevicePtr(), 
				xk[i]->getDevicePtr(),
				4, 
				systemk[i].gridx,
				systemk[i].gridy,
				systemk[i].gridz);

			mg_ComputeResidualPeriodic3D2(xk[i]->getDevicePtr(),
				bk[i]->getDevicePtr(),
				rk[i]->getDevicePtr(),
				systemk[i].gridx,
				systemk[i].gridy,
				systemk[i].gridz);

			mg_RestrictionPeriodic3D2(bk[i+1]->getDevicePtr(), 
				rk[i]->getDevicePtr(), 
				systemk[i+1].gridx,
				systemk[i+1].gridy,
				systemk[i+1].gridz);
		}

		mg_exact_periodic3D2(xk[m_max_level-1]->getDevicePtr(),
			bk[m_max_level-1]->getDevicePtr(),
			xk[m_max_level-1]->getDevicePtr(),
			systemk[m_max_level-1].gridx,
			systemk[m_max_level-1].gridy,
			systemk[m_max_level-1].gridz);

		for(int i=m_max_level-2; i>=level; --i)
		{
			mg_ProlongationPeriodic3D2(xk[i]->getDevicePtr(),
				xk[i+1]->getDevicePtr(),
				systemk[i].gridx,
				systemk[i].gridy,
				systemk[i].gridz);

			mg_RBGS_periodic3D2(xk[i]->getDevicePtr(), 
				bk[i]->getDevicePtr(), 
				xk[i]->getDevicePtr(),
				4, 
				systemk[i].gridx,
				systemk[i].gridy,
				systemk[i].gridz);
		}
		Loop++;
		for(int i=level+1; i<m_max_level; i++)
		{
			mg_zerofy(xk[i]->getDevicePtr(), xk[i]->getSize()*xk[i]->typeSize());
			mg_zerofy(bk[i]->getDevicePtr(), bk[i]->getSize()*bk[i]->typeSize());
			mg_zerofy(rk[i]->getDevicePtr(), rk[i]->getSize()*rk[i]->typeSize());

		}


	}
	void MultiGridSolver3DPeriod_double::m_FullMultiGrid(GpuArrayd* x, GpuArrayd* b, double tol, double &residual)
	{
		xk[0] = x ;
		bk[0] = b ;
		double res;
		for(int i=1; i<m_max_level; i++)
		{
			mg_zerofy(xk[i]->getDevicePtr(), xk[i]->getSize()*xk[i]->typeSize());
			mg_zerofy(bk[i]->getDevicePtr(), bk[i]->getSize()*bk[i]->typeSize());
			mg_zerofy(rk[i]->getDevicePtr(), rk[i]->getSize()*rk[i]->typeSize());
		}
		mg_zerofy(rk[0]->getDevicePtr(), rk[0]->getSize()*rk[0]->typeSize());
		for(int i=0; i<m_max_level-1; ++i)
		{
			mg_ComputeResidualPeriodic3D2(xk[i]->getDevicePtr(),
				bk[i]->getDevicePtr(),
				rk[i]->getDevicePtr(),
				systemk[i].gridx,
				systemk[i].gridy,
				systemk[i].gridz);

			mg_RestrictionPeriodic3D2(bk[i+1]->getDevicePtr(), 
				rk[i]->getDevicePtr(), 
				systemk[i+1].gridx,
				systemk[i+1].gridy,
				systemk[i+1].gridz);
		}
		mg_exact_periodic3D2(xk[m_max_level-1]->getDevicePtr(),
			bk[m_max_level-1]->getDevicePtr(),
			xk[m_max_level-1]->getDevicePtr(),
			systemk[m_max_level-1].gridx,
			systemk[m_max_level-1].gridy,
			systemk[m_max_level-1].gridz);
		for(int i=m_max_level-2; i>=0; --i)
		{
			mg_ProlongationPeriodic3D2(xk[i]->getDevicePtr(),
				xk[i+1]->getDevicePtr(),
				systemk[i].gridx,
				systemk[i].gridy,
				systemk[i].gridz);

			m_Vcycle(xk[i], bk[i], 1e-10, res, i);
		}
		for(int i=0; i<1; i++)
		{
			m_Vcycle(xk[0],b,1e-10, res,0);
		}

	}
}