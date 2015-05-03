#include "MultiGrid2D.h"

namespace gf{

	void MultiGridSolver2DPeriod_double::m_Vcycle(GpuArrayd* x, GpuArrayd* b, double tol, double &residual, int level)
	{
		xk[level] = x;
		bk[level] = b;

		for(int i=level+1; i<m_max_level; i++)
		{
			mg_zerofy2D(xk[i]->getDevicePtr(), xk[i]->getSize()*xk[i]->typeSize());
			mg_zerofy2D(bk[i]->getDevicePtr(), bk[i]->getSize()*bk[i]->typeSize());
			mg_zerofy2D(rk[i]->getDevicePtr(), rk[i]->getSize()*rk[i]->typeSize());
		}
		mg_zerofy2D(rk[level]->getDevicePtr(), rk[level]->getSize()*rk[level]->typeSize());
		int Loop  = 0 ;


		for(int i=level; i<m_max_level-1; ++i)
		{
			mg_RBGS_periodic2D2(xk[i]->getDevicePtr(), 
				bk[i]->getDevicePtr(), 
				xk[i]->getDevicePtr(),
				4, 
				systemk[i].gridx,
				systemk[i].gridy );

			mg_ComputeResidualPeriodic2D2(xk[i]->getDevicePtr(),
				bk[i]->getDevicePtr(),
				rk[i]->getDevicePtr(),
				systemk[i].gridx,
				systemk[i].gridy );

			mg_RestrictionPeriodic2D2(bk[i+1]->getDevicePtr(), 
				rk[i]->getDevicePtr(), 
				systemk[i+1].gridx,
				systemk[i+1].gridy );
		}

		mg_exact_periodic2D2(xk[m_max_level-1]->getDevicePtr(),
			bk[m_max_level-1]->getDevicePtr(),
			xk[m_max_level-1]->getDevicePtr(),
			systemk[m_max_level-1].gridx,
			systemk[m_max_level-1].gridy );

		for(int i=m_max_level-2; i>=level; --i)
		{
			mg_ProlongationPeriodic2D2(xk[i]->getDevicePtr(),
				xk[i+1]->getDevicePtr(),
				systemk[i].gridx,
				systemk[i].gridy );

			mg_RBGS_periodic2D2(xk[i]->getDevicePtr(), 
				bk[i]->getDevicePtr(), 
				xk[i]->getDevicePtr(),
				4, 
				systemk[i].gridx,
				systemk[i].gridy );
		}
		Loop++;
		for(int i=level+1; i<m_max_level; i++)
		{
			mg_zerofy2D(xk[i]->getDevicePtr(), xk[i]->getSize()*xk[i]->typeSize());
			mg_zerofy2D(bk[i]->getDevicePtr(), bk[i]->getSize()*bk[i]->typeSize());
			mg_zerofy2D(rk[i]->getDevicePtr(), rk[i]->getSize()*rk[i]->typeSize());

		}


	}
	void MultiGridSolver2DPeriod_double::m_FullMultiGrid(GpuArrayd* x, GpuArrayd* b, double tol, double &residual)
	{
		xk[0] = x ;
		bk[0] = b ;
		double res;
		for(int i=1; i<m_max_level; i++)
		{
			mg_zerofy2D(xk[i]->getDevicePtr(), xk[i]->getSize()*xk[i]->typeSize());
			mg_zerofy2D(bk[i]->getDevicePtr(), bk[i]->getSize()*bk[i]->typeSize());
			mg_zerofy2D(rk[i]->getDevicePtr(), rk[i]->getSize()*rk[i]->typeSize());
		}
		mg_zerofy2D(rk[0]->getDevicePtr(), rk[0]->getSize()*rk[0]->typeSize());
		for(int i=0; i<m_max_level-1; ++i)
		{
			mg_ComputeResidualPeriodic2D2(xk[i]->getDevicePtr(),
				bk[i]->getDevicePtr(),
				rk[i]->getDevicePtr(),
				systemk[i].gridx,
				systemk[i].gridy );

			mg_RestrictionPeriodic2D2(bk[i+1]->getDevicePtr(), 
				rk[i]->getDevicePtr(), 
				systemk[i+1].gridx,
				systemk[i+1].gridy );
		}
		mg_exact_periodic2D2(xk[m_max_level-1]->getDevicePtr(),
			bk[m_max_level-1]->getDevicePtr(),
			xk[m_max_level-1]->getDevicePtr(),
			systemk[m_max_level-1].gridx,
			systemk[m_max_level-1].gridy );

		for(int i=m_max_level-2; i>=0; --i)
		{
			mg_ProlongationPeriodic2D2(xk[i]->getDevicePtr(),
				xk[i+1]->getDevicePtr(),
				systemk[i].gridx,
				systemk[i].gridy );

			m_Vcycle(xk[i], bk[i], 1e-10, res, i);
		}
		for(int i=0; i<1; i++)
		{
			m_Vcycle(xk[0],b,1e-10, res,0);
		}

	}

	void MultiGridSolver2DPeriod::m_Vcycle(GpuArrayf* x, GpuArrayf* b, float tol, float &residual, int level)
	{
		xk[level] = x;
		bk[level] = b;

		for(int i=level+1; i<m_max_level; i++)
		{
			mg_zerofy2D(xk[i]->getDevicePtr(), xk[i]->getSize()*xk[i]->typeSize());
			mg_zerofy2D(bk[i]->getDevicePtr(), bk[i]->getSize()*bk[i]->typeSize());
			mg_zerofy2D(rk[i]->getDevicePtr(), rk[i]->getSize()*rk[i]->typeSize());
		}
		mg_zerofy2D(rk[level]->getDevicePtr(), rk[level]->getSize()*rk[level]->typeSize());
		//printf("zerofy done\n");
		int Loop  = 0 ;


		for(int i=level; i<m_max_level-1; ++i)
		{
			mg_RBGS_periodic2D(xk[i]->getDevicePtr(), 
				bk[i]->getDevicePtr(), 
				xk[i]->getDevicePtr(), 
				2, 
				systemk[i].gridx,
				systemk[i].gridy);
			//printf("mg_RBGS_periodic2D done\n");
			mg_ComputeResidualPeriodic2D(xk[i]->getDevicePtr(),
				bk[i]->getDevicePtr(),
				rk[i]->getDevicePtr(),
				systemk[i].gridx,
				systemk[i].gridy );
			//printf("mg_ComputeResidualPeriodic2D done\n");
			mg_RestrictionPeriodic2D(bk[i+1]->getDevicePtr(), 
				rk[i]->getDevicePtr(), 
				systemk[i+1].gridx,
				systemk[i+1].gridy );
			//printf("mg_RestrictionPeriodic2D done\n");
		}

		mg_exact_periodic2D(xk[m_max_level-1]->getDevicePtr(),
			bk[m_max_level-1]->getDevicePtr(),
			xk[m_max_level-1]->getDevicePtr(),
			systemk[m_max_level-1].gridx,
			systemk[m_max_level-1].gridy );
		//printf("mg_exact_periodic2D done\n");

		for(int i=m_max_level-2; i>=level; --i)
		{
			mg_ProlongationPeriodic2D(xk[i]->getDevicePtr(),
				xk[i+1]->getDevicePtr(),
				systemk[i].gridx,
				systemk[i].gridy );
			//printf("mg_ProlongationPeriodic2D done\n");

			mg_RBGS_periodic2D(xk[i]->getDevicePtr(), 
				bk[i]->getDevicePtr(), 
				xk[i]->getDevicePtr(),  
				2, 
				systemk[i].gridx,
				systemk[i].gridy );
			//printf("mg_RBGS_periodic2D done\n");
		}
		Loop++;
		for(int i=level+1; i<m_max_level; i++)
		{
			mg_zerofy2D(xk[i]->getDevicePtr(), xk[i]->getSize()*xk[i]->typeSize());
			mg_zerofy2D(bk[i]->getDevicePtr(), bk[i]->getSize()*bk[i]->typeSize());
			mg_zerofy2D(rk[i]->getDevicePtr(), rk[i]->getSize()*rk[i]->typeSize());

		}
		//printf("mg_zerofy2D done\n");

	}
	void MultiGridSolver2DPeriod::m_FullMultiGrid(GpuArrayf* x, GpuArrayf* b, float tol, float &residual)
	{
		xk[0] = x ;
		bk[0] = b ;
		float res;
		for(int i=1; i<m_max_level; i++)
		{
			mg_zerofy2D(xk[i]->getDevicePtr(), xk[i]->getSize()*xk[i]->typeSize());
			mg_zerofy2D(bk[i]->getDevicePtr(), bk[i]->getSize()*bk[i]->typeSize());
			mg_zerofy2D(rk[i]->getDevicePtr(), rk[i]->getSize()*rk[i]->typeSize());
		}

		mg_zerofy2D(rk[0]->getDevicePtr(), rk[0]->getSize()*rk[0]->typeSize());
		for(int i=0; i<m_max_level-1; ++i)
		{
			mg_ComputeResidualPeriodic2D(xk[i]->getDevicePtr(),
				bk[i]->getDevicePtr(),
				rk[i]->getDevicePtr(),
				systemk[i].gridx,
				systemk[i].gridy);
			mg_RestrictionPeriodic2D(bk[i+1]->getDevicePtr(), 
				rk[i]->getDevicePtr(), 
				systemk[i+1].gridx,
				systemk[i+1].gridy);
		}
		mg_exact_periodic2D(xk[m_max_level-1]->getDevicePtr(),
			bk[m_max_level-1]->getDevicePtr(),
			xk[m_max_level-1]->getDevicePtr(),
			systemk[m_max_level-1].gridx,
			systemk[m_max_level-1].gridy);

		for(int i=m_max_level-2; i>=0; --i)
		{
			mg_ProlongationPeriodic2D(xk[i]->getDevicePtr(),
				xk[i+1]->getDevicePtr(),
				systemk[i].gridx,
				systemk[i].gridy);

			m_Vcycle(xk[i], bk[i], 1e-10, res, i);
			//printf("m_Vcycle done\n");
		}
		for(int i=0; i<1; i++)
		{
			m_Vcycle(xk[0],b,1e-10, res,0);
		}

	}

}