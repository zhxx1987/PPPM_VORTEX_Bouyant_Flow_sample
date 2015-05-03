
To Advanturers : 

This is a more matured version of the original PPPM vortex method of the 
"fast PPPM summation for fluids and beyond" paper.

many things has been re-considered for stability purpose, in addition,
a clean and basic(but redundant and inefficient) interface is provided.

there could be bugs and issues and a huge space for performance improvements.

TO COMPILE:
I had successful builds of the hybridFastSummation on MicroSoft Visual Studio 2008,
with CUDA 5.0(one can have cuda installation from Nvidia's website)

if you are using similar environment(but with different cuda versions or so),
you would have to change the cuda_build_rule to make things work.




if you are on linux:
I am not familiar with makefile, so I just let you know:
hybridFastSummation will make use of
multigrid and particle_hash project, which shall be compiled as static libs.
(or whatever way you would like to link them)


Once you build and run the excutable, 
you'll get Particle_data%04d.bin  files, these files can be visualized using the OpenGLRenderer
again, need to be compiled with cuda and glew.

FluidRenderer.exe  [path(no space in the file name)] end_frame 1 1 1 1 (those 1's were parameters not used anymore).

one shall get visualizations as showed in ogl.gif

Particle_renderer is a CPU renderer implemented by Robert Bridson, this one does fancier rendering.
once compiled, one renders the data by
Particle_renderer.exe  [path(the folder where Particle data stores, no space in name)] particle_radius density_scale opacity light_x light_y light_z frame_start frame_end

which produces animations as shown in render.gif



