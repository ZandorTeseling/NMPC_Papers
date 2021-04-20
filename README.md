# NMPC_Papers


OS:"Ubuntu 18.04.3 LTS"
Compiler:/usr/bin/x86_64-linux-gnu-gcc-7


Implementation of NMPC papers. 


Commands:  

git submodule update --init --recursive  

Currently for python development to work the env vars need to be set to:  
LD_LIBRARY_PATH=<path_to_git_folder>/NMPC_Papers/3rdParty/acados/lib  

For latex figure issues/exception in ubuntu 18.04  
sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super  


Configuration of PyCharm:  
In CLI:  
<path_to_git_folder>/NMPC_Papers/3rdParty/acados$ pip install  ./interfaces/acados_template/  
Py Interp:  
Name-Python3.8  
Path-<path_to_anaconda3>/envs/NMPC_Papers/bin/python  

ENV Vars:  
ACADOS_SOURCE_DIR=<path_to_git_folder>/NMPC_Papers/3rdParty/acados/  
LD_LIBRARY_PATH=<path_to_git_folder>/NMPC_Papers/3rdParty/acados/lib  


Managing Conda Env and set up to use pycharm

#CLI
1. conda create --name NMPC_Papers  
2. Note path for env  <path_to_anaconda3>/envs/NMPC_Papers  
3. conda activate NMPC_Papers  
4. Follow acados build/install instructions  
5. <path_to_git_folder>/NMPC_Papers/3rdParty/acados$ pip install  ./interfaces/acados_template/   
#PyCharm  
6. File->Settings->PythonInterreters->Add Python Interpreter (With following settings)  
7. Select: New Environment  
8. Location: <path_to_anaconda3>/envs/NMPC_Papers>   
9. Python Version: 3.9  
10. Conda Executable: <path_to_anaconda3>/bin/conda  
