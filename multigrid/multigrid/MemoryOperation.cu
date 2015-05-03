//#include <cutil_inline.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

extern"C"
{
void InitCuda(int argc, char *argv[])
{
	int devID;

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    devID = findCudaDevice(argc, (const char **)argv);

    if (devID < 0)
    {
        printf("No CUDA Capable devices found, exiting...\n");
        exit(EXIT_SUCCESS);
    }
}

void InitCudaGL(int argc, char** argv)
{
	findCudaGLDevice(argc, (const char **)argv);
}

void AllocateMemoryPitch(size_t size, size_t * pitch, void **devPtr)
{
	checkCudaErrors(cudaMallocPitch(devPtr, pitch, size, 1));
	getLastCudaError("allocate mem pitch failed!\n");
}
void AllocateMemory(size_t size, void **devPtr)
{
	checkCudaErrors(cudaMalloc(devPtr, size));
	getLastCudaError("allocate mem failed!\n");
}

void FreeMemory(void *devPtr)
{
	checkCudaErrors(cudaFree(devPtr));
}

void ThreadSync()
{
	checkCudaErrors(cudaThreadSynchronize());
}

void MemcpyHstToDevPitch(void* devPtr, size_t dPitch, const void* hstPtr, size_t hPitch, size_t size)
{
	checkCudaErrors(cudaMemcpy2D(devPtr, dPitch, hstPtr, hPitch, size, 1, cudaMemcpyHostToDevice));
}
void MemcpyDevToHstPitch(const void* devPtr, size_t dPitch, void* hstPtr, size_t hPitch, size_t size)
{
	checkCudaErrors(cudaMemcpy2D(hstPtr, hPitch, devPtr, dPitch, size, 1, cudaMemcpyDeviceToHost));
}
void MemcpyDevToDevPitch(void *dstPtr, size_t dPitch, const void *srcPtr, size_t sPitch, size_t size)
{
	checkCudaErrors(cudaMemcpy2D(dstPtr, dPitch, srcPtr, sPitch, size, 1, cudaMemcpyDeviceToDevice));
}
void MemcpyHstToDev(void* devPtr, const void* hstPtr, size_t size)
{
	checkCudaErrors(cudaMemcpy(devPtr, hstPtr,size, cudaMemcpyHostToDevice));
}
void MemcpyDevToHst(const void* devPtr, void* hstPtr, size_t size)
{
	checkCudaErrors(cudaMemcpy(hstPtr,devPtr,size, cudaMemcpyDeviceToHost));
}
void MemcpyDevToDev(void* dstPtr, const void* srcPtr, size_t size)
{
	checkCudaErrors(cudaMemcpy(dstPtr,srcPtr,size, cudaMemcpyDeviceToDevice));
}

}
