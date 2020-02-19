# NMPC_Papers


OS:"Ubuntu 18.04.3 LTS"
Compiler:/usr/bin/x86_64-linux-gnu-gcc-7


Implementation of NMPC papers. 


Commands:

git submodule update --init --recursive

Currently for python development to work the env vars need to be set to:
LD_LIBRARY_PATH=<path_to_git_folder>/NMPC_Papers/3rdParty/acados/lib


Why the build shared libraries isn't working needs to be investigated,potential paths:    
LD_LIBRARY_PATH=<path_to_git_folder>/NMPC_Papers/build/3rdParty/acados/lib isn't even a thing
 
LD_LIBRARY_PATH=<path_to_git_folder>/NMPC_Papers/build/3rdParty/acados/acados/ doesn't contain all the neseccary *.so files.

Necassary files are:
libacados.so  
libblasfeo.so <- not there  
libhpipm.so <- not there  
