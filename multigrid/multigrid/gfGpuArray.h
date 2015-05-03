#ifndef _gfgpu_array_h_
#define _gfgpu_array_h_
/*
here a higher level c++ style array structure is built
using the lower-level memory operations, the objective
is to get more convenience use of the data on the device
and make the code more clear
*/


#include <GL/glew.h>
#include <cuda_runtime.h>
#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
#include <cuda_gl_interop.h>
//#include <cutil_inline.h>
#include "MemoryOperation.cuh"
#include <stdio.h>
#include <helper_cuda.h> 
//using namespace std;
namespace gf{
	template <class T> 
	class gf_GpuArray {
	public:
		gf_GpuArray();
		~gf_GpuArray();

		enum Direction
		{
			HOST_TO_DEVICE,
			DEVICE_TO_HOST,
			DEVICE_TO_DEVICE
		};

		// allocate and free
		void allocPitch(size_t size, bool vbo=false, bool doubleBuffer=false, bool useElementArray=false);
		void alloc(size_t size, bool vbo=false, bool doubleBuffer=false, bool useElementArray=false);
		void free();

		// swap buffers for double buffering
		void swap();

		// when using vbo, must map before getting device ptr
		void map();
		void unmap();

		void copyPitch(Direction dir, size_t start=0, size_t count=0);
		void copy(Direction dir, size_t start=0, size_t count=0);
		void memset(T value, size_t start=0, size_t count=0);
		void read(Direction dir, T *data, size_t spitch, size_t start=0, size_t count=0);

		T *getDevicePtr() { return m_dptr[m_currentRead]; }
		GLuint getVbo() { return m_vbo[m_currentRead]; }

		T *getDeviceWritePtr() { return m_dptr[m_currentWrite]; }
		GLuint getWriteVbo() { return m_vbo[m_currentWrite]; }

		T *getHostPtr() { return m_hptr; }

		size_t getSize() const { return m_size; }
		size_t typeSize() const { return sizeof(T); }

	private:
		GLuint createVbo(size_t size, bool useElementArray);

		void allocDevicePitch();
		void allocDevice();
		void allocVbo(bool useElementArray);
		void allocHost();

		void freeDevice();
		void freeVbo();
		void freeHost();

		size_t m_size;
		T *m_dptr[2];
		GLuint m_vbo[2];
		struct cudaGraphicsResource *m_cuda_vbo_resource[2]; // handles OpenGL-CUDA exchange

		T *m_hptr;

		bool m_useVBO;
		bool m_doubleBuffer;
		uint m_currentRead, m_currentWrite;
		size_t pitch;
	};

	template <class T> 
	gf_GpuArray<T>::gf_GpuArray() :
	m_size(0),
		m_hptr(0),
		m_currentRead(0),
		m_currentWrite(0)
	{
		m_dptr[0] = 0;
		m_dptr[1] = 0;

		m_vbo[0] = 0;
		m_vbo[1] = 0;

		m_cuda_vbo_resource[0] = NULL;
		m_cuda_vbo_resource[1] = NULL;
	}

	template <class T> 
	gf_GpuArray<T>::~gf_GpuArray()
	{
		free();
	}

	template <class T> 
	void
		gf_GpuArray<T>::allocPitch(size_t size, bool vbo, bool doubleBuffer, bool useElementArray)
	{
		m_size = size;

		m_useVBO = vbo;
		m_doubleBuffer = doubleBuffer;
		if (m_doubleBuffer) {
			m_currentWrite = 1;
		}

		allocHost();
		if (vbo) {
			allocVbo(useElementArray);
		} else {
			allocDevicePitch();
		}
	}
	template <class T> 
	void
		gf_GpuArray<T>::alloc(size_t size, bool vbo, bool doubleBuffer, bool useElementArray)
	{
		m_size = size;

		m_useVBO = vbo;
		m_doubleBuffer = doubleBuffer;
		if (m_doubleBuffer) {
			m_currentWrite = 1;
		}

		allocHost();
		if (vbo) {
			allocVbo(useElementArray);
		} else {
			allocDevice();
		}
	}

	template <class T> 
	void
		gf_GpuArray<T>::free()
	{
		freeHost();
		if (m_useVBO) {
			freeVbo();
		} else {
			freeDevice();
		}
	}

	template <class T> 
	void
		gf_GpuArray<T>::allocHost()
	{
		m_hptr = (T *) new T [m_size];
	}

	template <class T> 
	void
		gf_GpuArray<T>::freeHost()
	{
		if (m_hptr) {
			delete [] m_hptr;
			m_hptr = 0;
		}
	}

	template <class T> 
	void
		gf_GpuArray<T>::allocDevicePitch()
	{
		AllocateMemoryPitch(m_size*sizeof(T), &pitch, (void**)&m_dptr[0]);
		if (m_doubleBuffer) {
			AllocateMemoryPitch(m_size*sizeof(T), &pitch, (void **)&m_dptr[1]);
		}
	}
	template <class T> 
	void
		gf_GpuArray<T>::allocDevice()
	{
		AllocateMemory(m_size*sizeof(T), (void**)&m_dptr[0]);
		if (m_doubleBuffer) {
			AllocateMemory(m_size*sizeof(T), (void **)&m_dptr[1]);
		}
	}
	template <class T> 
	void
		gf_GpuArray<T>::freeDevice()
	{
		if (m_dptr[0]) {
			FreeMemory(m_dptr[0]);
			m_dptr[0] = 0;
		}

		if (m_dptr[1]) {
			FreeMemory(m_dptr[1]);
			m_dptr[1] = 0;
		}
	}

	template <class T> 
	GLuint
		gf_GpuArray<T>::createVbo(size_t size, bool useElementArray)
	{
		GLuint vbo;
		glGenBuffers(1, &vbo);

		if (useElementArray) {
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		} else {
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}
		return vbo;
	}

	template <class T> 
	void
		gf_GpuArray<T>::allocVbo(bool useElementArray)
	{
		m_vbo[0] = createVbo(m_size*sizeof(T), useElementArray);
		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_cuda_vbo_resource[0], m_vbo[0], 
			cudaGraphicsMapFlagsWriteDiscard));
		if (m_doubleBuffer) {
			m_vbo[1] = createVbo(m_size*sizeof(T), useElementArray);
			checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_cuda_vbo_resource[1], m_vbo[1], 
				cudaGraphicsMapFlagsWriteDiscard));	
		}
	}

	template <class T> 
	void
		gf_GpuArray<T>::freeVbo()
	{
		if (m_vbo[0]) {
			checkCudaErrors(cudaGraphicsUnregisterResource(m_cuda_vbo_resource[0]));
			glDeleteBuffers(1, &m_vbo[0]);
			m_vbo[0] = 0;
		}

		if (m_vbo[1]) {
			checkCudaErrors(cudaGraphicsUnregisterResource(m_cuda_vbo_resource[1]));
			glDeleteBuffers(1, &m_vbo[1]);
			m_vbo[1] = 0;
		}
	}

	template <class T> 
	void
		gf_GpuArray<T>::swap()
	{
		//std::swap(m_currentRead, m_currentWrite);
		uint temp = m_currentRead;
		m_currentRead = m_currentWrite;
		m_currentWrite = temp;
	}

	template <class T> 
	void
		gf_GpuArray<T>::map()
	{
		if (m_vbo[0]) {
			checkCudaErrors(cudaGraphicsMapResources(1, &m_cuda_vbo_resource[0], 0));
			size_t num_bytes; 
			checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&m_dptr[0], &num_bytes,  
				m_cuda_vbo_resource[0]));
		} 
		if (m_doubleBuffer && m_vbo[1]) {
			checkCudaErrors(cudaGraphicsMapResources(1, &m_cuda_vbo_resource[1], 0));
			size_t num_bytes; 
			checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&m_dptr[1], &num_bytes,  
				m_cuda_vbo_resource[1]));
		}
	}

	template <class T> 
	void
		gf_GpuArray<T>::unmap()
	{
		if (m_vbo[0]) {
			checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cuda_vbo_resource[0], 0));
			m_dptr[0] = 0;
		}
		if (m_doubleBuffer && m_vbo[1]) {
			checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cuda_vbo_resource[1], 0));
			m_dptr[1] = 0;
		}
	}

	template <class T> 
	void
		gf_GpuArray<T>::copyPitch(Direction dir, size_t start, size_t count)
	{
		if (count==0) {
			count = (size_t) m_size;
		}

		map();
		switch(dir) {
	case HOST_TO_DEVICE:
		MemcpyHstToDevPitch((void*)(m_dptr[m_currentRead]+start), pitch, (void *)(m_hptr+start),sizeof(T)*count, sizeof(T)*count);
		break;

	case DEVICE_TO_HOST:
		MemcpyDevToHstPitch((void*)(m_dptr[m_currentRead]+start), pitch, (void *)(m_hptr+start),sizeof(T)*count, sizeof(T)*count);
		break;
		}
		
		unmap();
	}
	template <class T> 
	void
		gf_GpuArray<T>::copy(Direction dir, size_t start, size_t count)
	{
		if (count==0) {
			count = (size_t) m_size;
		}

		map();
		switch(dir) {
	case HOST_TO_DEVICE:
		MemcpyHstToDev((void*)(m_dptr[m_currentRead]+start), (void *)(m_hptr+start),sizeof(T)*count);
		break;

	case DEVICE_TO_HOST:
		MemcpyDevToHst((void*)(m_dptr[m_currentRead]+start), (void *)(m_hptr+start),sizeof(T)*count);
		break;
		}
		unmap();
	}
	template <class T> 
	void
		gf_GpuArray<T>::read(Direction dir, T *data, size_t spitch, size_t start=0, size_t count=0)
	{
		if (count==0) {
			count = (size_t) m_size;
		}


		switch(dir) {
	case HOST_TO_DEVICE:
		MemcpyHstToDevPitch((void*)(m_dptr[m_currentRead]+start), pitch, (void *)(data+start),sizeof(T)*count,sizeof(T)*count);
		break;

	case DEVICE_TO_HOST:
		MemcpyDevToHstPitch((void*)(data+start), pitch, (void *)(m_hptr+start),sizeof(T)*count,sizeof(T)*counts);
		break;
	case DEVICE_TO_DEVICE:
		MemcpyDevToDevPitch((void*)(m_dptr[m_currentRead]+start), pitch, (void *)(data+start), spitch, sizeof(T)*count);
		break;
		}
	}

	template <class T> 
	void
		gf_GpuArray<T>::memset(T value, size_t start, size_t count)
	{
		if (count==0) {
			count = (size_t) m_size;
		}
		cudaMemset(m_dptr[m_currentRead], 0, sizeof(T)*count);
		//memset(m_hptr, 0, sizeof(T)*count);
	}

	typedef gf_GpuArray<float> GpuArrayf;
	typedef gf_GpuArray<int>	GpuArrayi;
	typedef gf_GpuArray<char>	GpuArrayc;
	typedef gf_GpuArray<double> GpuArrayd;
	typedef gf_GpuArray<float2> GpuArrayf2;
	typedef gf_GpuArray<double2> GpuArrayd2;
	typedef gf_GpuArray<double3> GpuArrayd3;
	typedef gf_GpuArray<float3>  GpuArrayf3;
	typedef gf_GpuArray<double4> GpuArrayd4;
	typedef gf_GpuArray<float4>  GpuArrayf4;
}
#endif