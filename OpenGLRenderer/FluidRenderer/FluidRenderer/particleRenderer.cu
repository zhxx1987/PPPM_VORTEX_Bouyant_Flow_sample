////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/*
    This example demonstrates how to use the Cuda OpenGL bindings to
    dynamically modify a vertex buffer using a Cuda kernel.

    The steps are:
    1. Create an empty vertex buffer object (VBO)
    2. Register the VBO with Cuda
    3. Map the VBO for writing from Cuda
    4. Run Cuda kernel to modify the vertex positions
    5. Unmap the VBO
    6. Render the results using OpenGL

    Host code
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>
using namespace std;

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <vector_types.h>
#include <windows.h>
#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     10 //ms

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width  = 1440;
const unsigned int window_height = 900;

int N_particle = 16384;
int N_frame=1;
float *buffer_val=0;
float4 *posh=0;
float4 *posd;

// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;
float g_size = 3.0;
float g_alpha = 0.2;

bool g_animate = true;
bool g_reverse = false;
int g_frame_add=1;
// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = 60.0;
vector<float4> points;
vector<float > Temperature;
vector<vector<float4>> point_sequence;
vector<vector<float >> temperature_sequence;
StopWatchInterface *timer = NULL;
float g_bright = 10;

// Auto-Verification Code
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;

int *pArgc = NULL;
char **pArgv = NULL;
float g_Red = 1.0;
float g_Green = 1.0;
float g_Blue = 1.0;
#define MAX(a,b) ((a > b) ? a : b)

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(int argc, char **argv, char *ref_file);
void cleanup();

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

// Cuda functionality
void runCuda(struct cudaGraphicsResource **vbo_resource);
void runAutoTest(int devID, char **argv, char *ref_file);
void checkResultCuda(int argc, char **argv, const GLuint &vbo);

const char *sSDKsample = "simpleGL (VBO)";
float translateX = 0.0f, translateY = 0.0f, translateZ = -10.0f;

__global__ void simple_vbo_kernel(float4 *pos,float4 * posd, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<N)
	{
		float factor = 1;
		pos[idx] = make_float4(posd[idx].x*factor,
			posd[idx].y*factor,
			posd[idx].z*factor,
			1.0);
	}
   

}
void doKernel(float4 * pos, float4 * posd, int N)
{
	int threads = 512;
	int blocks = N/threads + (!(N%threads)?0:1);
	simple_vbo_kernel<<<blocks,threads>>>(pos,posd,N);
}


bool checkHW(char *name, const char *gpuType, int dev)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    strcpy(name, deviceProp.name);

    if (!STRNCASECMP(deviceProp.name, gpuType, strlen(gpuType)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

int findGraphicsGPU(char *name)
{
    int nGraphicsGPU = 0;
    int deviceCount = 0;
    bool bFoundGraphics = false;
    char firstGraphicsName[256], temp[256];

    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("> FAILED %s sample finished, exiting...\n", sSDKsample);
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        printf("> There are no device(s) supporting CUDA\n");
        return false;
    }
    else
    {
        printf("> Found %d CUDA Capable Device(s)\n", deviceCount);
    }

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        bool bGraphics = !checkHW(temp, (const char *)"Tesla", dev);
        printf("> %s\t\tGPU %d: %s\n", (bGraphics ? "Graphics" : "Compute"), dev, temp);

        if (bGraphics)
        {
            if (!bFoundGraphics)
            {
                strcpy(firstGraphicsName, temp);
            }

            nGraphicsGPU++;
        }
    }

    if (nGraphicsGPU)
    {
        strcpy(name, firstGraphicsName);
    }
    else
    {
        strcpy(name, "this hardware");
    }

    return nGraphicsGPU;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int g_Dx;
int g_Dy;
int g_Dz;
char filepath[256];
int main(int argc, char **argv)
{
    char *ref_file = NULL;

    pArgc = &argc;
    pArgv = argv;

    printf("%s starting...\n", sSDKsample);

    if (argc > 1)
    {
        if (checkCmdLineFlag(argc, (const char **)argv, "file"))
        {
            // In this mode, we are running non-OpenGL and doing a compare of the VBO was generated correctly
            getCmdLineArgumentString(argc, (const char **)argv, "file", (char **)&ref_file);
        }
    }

	if(argc < 7)
	{
		printf("enter file_path, total_frame, dimx, dimy, dimz, frame_step\n");
		return 0;
	}
	else
	{
		sscanf(argv[1],"%s", filepath);
		sscanf(argv[2],"%d",&N_frame);
		sscanf(argv[3],"%d",&g_Dx);
		sscanf(argv[4],"%d",&g_Dy);
		sscanf(argv[5],"%d",&g_Dz);
		sscanf(argv[6],"%d",&g_frame_add);
		printf("%d\n",N_frame);
	

	buffer_val = (float*)malloc(sizeof(float)*g_Dx*g_Dy*g_Dz);


	point_sequence.resize(N_frame);
	temperature_sequence.resize(N_frame);
	for(int i=0;i<N_frame;i++)
	{ point_sequence[i].resize(0);
	temperature_sequence[i].resize(0);}

	//posh = (float4*)malloc(sizeof(float4)*N_particle);
	//cudaMalloc((void**)&posd,sizeof(float4)*N_particle);
	

    printf("\n");

    runTest(argc, argv, ref_file);

    cudaDeviceReset();
    printf("%s completed, returned %s\n", sSDKsample, (g_TotalErrors == 0) ? "OK" : "ERROR!");
    exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
	}
}

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.f);

        sdkResetTimer(&timer);
    }

    char fps[256];
    sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps (Max 100Hz)", avgFPS);
    glutSetWindowTitle(fps);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Cuda GL Interop (VBO)");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);

    // initialize necessary OpenGL extensions
    glewInit();

    if (! glewIsSupported("GL_VERSION_2_0 "))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    

    // projection
    

    SDK_CHECK_ERROR_GL();

    return true;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
bool runTest(int argc, char **argv, char *ref_file)
{
    // Create the CUTIL timer
   // sdkCreateTimer(&timer);

    // command line mode only
    if (ref_file != NULL)
    {
    }
    else
    {
        // First initialize OpenGL context, so we can properly set the GL for CUDA.
        // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
        if (false == initGL(&argc, argv))
        {
            return false;
        }

        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        if (checkCmdLineFlag(argc, (const char **)argv, "device"))
        {
			if( gpuGLDeviceInit(argc, (const char **)argv) == -1 ) {
				return false;
			}
        }
        else
        {
            cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
        }

        // register callbacks
        glutDisplayFunc(display);
        glutKeyboardFunc(keyboard);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);

        // create VBO
        //createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

        // run the cuda part
        runCuda(&cuda_vbo_resource);

        // start rendering mainloop
        glutMainLoop();
        atexit(cleanup);
    }

    return true;
}

int g_frames=0;
double frand(double a, double b)
{
	return (((double)rand())/((double)RAND_MAX)*(b-a))+a;
}
////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource **vbo_resource)
{
    // map OpenGL buffer object for writing from CUDA
    //float4 *dptr;
    //checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    //size_t num_bytes;
    //checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
    //                                                     *vbo_resource));



	char filename[256];
	int n=sprintf(filename,"%s/Particle_data%04d.bin",filepath,g_frames);
	char filename3[256];
	n=sprintf(filename3,"%s/temperature.%04d.bin", filepath, g_frames);

	if(point_sequence[g_frames].size()==0) //needs read
	{

		//memset(posh,0,sizeof(float4)*N_particle);
		FILE *data = fopen(filename,"rb");
		if(data==NULL)
			printf("file error\n");
		//char cc;
		//cc=fgetc(data);
		////size_t result = fread(buffer_val,sizeof(float),g_Dx*g_Dy*g_Dz,data);
		//while(cc!='[')
		//{
		//	cc=fgetc(data);
		//}
		//cc=fgetc(data);
		//while(cc!='[')
		//{
		//	cc=fgetc(data);
		//}
		//cc=fgetc(data);
		//while(cc!='[')
		//{
		//	cc=fgetc(data);
		//}
		//for(int idx=0;idx<g_Dx*g_Dy*g_Dz;idx++){

		//	fscanf(data,"%f ",&(buffer_val[idx]));
		//	//printf("%f ", buffer_val[idx]);
		//}

		//fclose(data);

		//FILE *data2 = fopen(filename3, "rb");
		//float *buffer_val_t = 0;
		//if(data2!=NULL)
		//{
		//	buffer_val_t = (float*)malloc(sizeof(float)*g_Dx*g_Dy*g_Dz);
		//	size_t result = fread(buffer_val_t, 1, sizeof(float)*g_Dx*g_Dy*g_Dz, data2);
		//	fclose(data2);

		//}


		point_sequence[g_frames].clear();
		temperature_sequence[g_frames].clear();
		
		if(fseek(data, 0, SEEK_END)){
			fclose(data);
			
		}

		long int filesize=ftell(data);
		if(filesize<0){
			fclose(data);
			
		}

		if(fseek(data, 0, SEEK_SET)){
			//fclose(data);
			
		}

		point_sequence[g_frames].resize(filesize/(4*sizeof(float)));
		printf("n:%d\n",point_sequence[g_frames].size());
		
		for(int i=0; i<point_sequence[g_frames].size(); ++i){
			// read x y z for particle i
			fread(&(point_sequence[g_frames][i]), sizeof(float), 4, data);
			
		}

		fclose(data);
		//for(int k=0;k<g_Dz;k++)for(int j=0;j<g_Dy;j++)for(int i=0;i<g_Dx;i++)
		//{
		//	if(buffer_val[(k*g_Dy+j)*g_Dx + i]>0.001)
		//	{
		//		point_sequence[g_frames].push_back(make_float4((float)i/(float)g_Dx-0.5,(float)j/(float)g_Dx,(float)k/(float)g_Dx-0.5,buffer_val[(k*g_Dy+j)*g_Dx + i]));
		//		if(buffer_val_t!=0)
		//			temperature_sequence[g_frames].push_back(buffer_val_t[(k*g_Dy+j)*g_Dx + i]);
		//	}
		//}

		//free(buffer_val_t);

	//	float4 *pos;
	//	char filename2[128];
	//int n=sprintf(filename2,"C:\\Users\\xinxin\\Desktop\\particle_data\\Particle_data%04d.bin",g_frames);
	//pos = (float4*)(&(point_sequence[g_frames][0]));
	////for (int i=0;i<point_sequence[g_frames].size();i++)
	////{
	////	pos[i].w = 1.0;
	////}
	//FILE *data_file = fopen(filename2,"wb");
	//fwrite(pos,sizeof(float4),point_sequence[g_frames].size(),data_file);
	//fclose(data_file);
	//

	//if(temperature_sequence[g_frames].size()>0)
	//{
	//	float *temp;
	//	char filename4[128];
	//	int n=sprintf(filename4,"C:\\Users\\xinxin\\Desktop\\particle_data\\temp_data%04d.bin",g_frames);
	//	temp = (float*)(&(temperature_sequence[g_frames][0]));
	//	FILE *data_file4 = fopen(filename4,"wb");
	//	fwrite(temp,sizeof(float),temperature_sequence[g_frames].size(),data_file4);
	//	fclose(data_file4);
	//}

	//printf("timestep %d done\n",g_frames);
	}

	//if(posh)
	//{free(posh); posh=0;}

	//posh = (float4*)malloc(sizeof(float4)*points.size());
	//for(int p=0;p<points.size();p++)
	//{
	//	posh[p] = points[p];
	//}

	//cudaMemcpy(dptr, posh, sizeof(float4)*points.size(), cudaMemcpyHostToDevice);

	//doKernel(dptr,posd,N_particle);
    // unmap buffer object
    //checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
	if(g_animate&&!g_reverse)
		g_frames+=g_frame_add;
	if(g_animate&&g_reverse)
		g_frames-=g_frame_add;
	printf("%d,%d\n",g_frames,N_frame);
	g_frames = (g_frames+N_frame)%N_frame;
}

#ifdef _WIN32
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) fopen_s(&fHandle, filename, mode)
#endif
#else
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) (fHandle = fopen(filename, mode))
#endif
#endif

void sdkDumpBin2(void *data, unsigned int bytes, const char *filename)
{
    printf("sdkDumpBin: <%s>\n", filename);
    FILE *fp;
    FOPEN(fp, filename, "wb");
    fwrite(data, bytes, 1, fp);
    fflush(fp);
    fclose(fp);
}



////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags)
{
    assert(vbo);

    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    unsigned int size = N_particle * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

    SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{

    // unregister this buffer object with CUDA
    cudaGraphicsUnregisterResource(vbo_res);

    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    sdkStartTimer(&timer);

    // run CUDA kernel to generate vertex positions
    runCuda(&cuda_vbo_resource);
    glClearColor(1.0,1.0,1.0,0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	glViewport(0, 0, window_width, window_height);
	glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.01, 100.0);
    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(translateX, translateY, translateZ);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);
	
	int display_frame = (g_frames-1+N_frame)%N_frame;
	if(g_reverse==true) display_frame = (g_frames+1+N_frame)%N_frame;
	
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_POINT_SPRITE);
	glEnable(GL_POINT_SMOOTH);
	glPointSize(g_size);
    glBegin(GL_POINTS);
	for(int i=0;i<point_sequence[display_frame].size();i++)
	{
		if(temperature_sequence[display_frame].size()>0){

			float heat_index = (temperature_sequence[display_frame][i])/1000.0f;
			heat_index = max(min(heat_index, 1.0f),0.0f);

			float r = heat_index;
			float g = heat_index*0.4;
			float b = 0;
			glColor4f(r*g_bright*point_sequence[display_frame][i].w,g*g_bright*point_sequence[display_frame][i].w,b*g_bright*point_sequence[display_frame][i].w,g_alpha);
		}
		else{

			glColor4f(0,0,0,0.01);

		}
		glVertex4f(point_sequence[display_frame][i].x,point_sequence[display_frame][i].y,point_sequence[display_frame][i].z,1);
	}
	glEnd();
	
    // render from the vbo
    //glBindBuffer(GL_ARRAY_BUFFER, vbo);
    //glVertexPointer(4, GL_FLOAT, 0, 0);

    //glEnableClientState(GL_VERTEX_ARRAY);
    //glColor4f(0.0, 0.0, 0.0,g_alpha);
	//glPointSize(g_size);
    //glDrawArrays(GL_POINTS, 0, N_particle);
    //glDisableClientState(GL_VERTEX_ARRAY);

	Sleep(30);
    glutSwapBuffers();

	glutPostRedisplay();

	//printf("%f\n",translate_z);

    //sdkStopTimer(&timer);
    //computeFPS();
}

void timerEvent(int value)
{
    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    //if (vbo)
    //{
    //    deleteVBO(&vbo, cuda_vbo_resource);
    //}
}


////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case (27) :
            exit(EXIT_SUCCESS);
			break;
		case ' ':
            g_animate = !g_animate;
			break;
		case 'r':
			g_reverse = !g_reverse;
			break;
		case '=':
			g_alpha+=0.001;
			break;
		case '-':
			g_alpha-=0.001;
			break;

		case '0':
			g_size += 0.1;
			break;
		case'9':
			g_size -= 0.1;
			break;
		case '7':
			g_bright -= 0.1;
			break;
		case '8':
			g_bright += 0.1;
			break;

		case 'E':
			g_Red +=0.01;
			break;
		case 'e':
			g_Red -=0.01;
			break;

		case 'G':
			g_Green +=0.01;
			break;
		case 'g':
			g_Green -=0.01;
			break;

		case 'B':
			g_Blue +=0.01;
			break;
		case 'b':
			g_Blue -=0.01;
			break;
            
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1)
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }
	else if (mouse_buttons == 2)
    {
        translateX += dx * 0.01f;
        translateY -= dy * 0.01f;
    }
    else if (mouse_buttons & 4)
    {
        translateZ += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}


