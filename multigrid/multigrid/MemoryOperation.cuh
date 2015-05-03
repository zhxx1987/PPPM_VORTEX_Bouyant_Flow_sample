
/*In this file basic memory operations are defined
  which simplified the work to build up more complicated
  applications

*/
#include "vector_types.h"
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
//#include <cutil_inline.h>
typedef unsigned int uint;
extern "C"
{
	void InitCuda(int argc, char *argv[]);
	void InitCudaGL(int argc, char **argv);
	void AllocateMemoryPitch(size_t size, size_t * pitch, void **devPtr);
	void AllocateMemory(size_t size, void **devPtr);
	void FreeMemory(void *devPtr);
	void ThreadSync();
	void MemcpyHstToDevPitch(void* devPtr, size_t dPitch, const void* hstPtr, size_t hPitch, size_t size);
	void MemcpyDevToHstPitch(const void* devPtr, size_t dPitch, void* hstPtr, size_t hPitch, size_t size);
	void MemcpyDevToDevPitch(void *dstPtr, size_t dPitch, const void *srcPtr, size_t sPitch, size_t size);
	void MemcpyHstToDev(void* devPtr, const void* hstPtr, size_t size);
	void MemcpyDevToHst(const void* devPtr, void* hstPtr, size_t size);
	void MemcpyDevToDev(void* dstPtr, const void* srcPtr, size_t size);
	

}