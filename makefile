PETSC_DIR=/home/chengtianpei/petsc/
export OMPI_CC=clang
export OMPI_CXX=clang++
defaults: fi
# 编译器设置
CC = gcc
CXX = g++
FC = gfortran

CXXFLAGS += -I/usr/local/include -I./autodiff
CFLAGS = -Wall -O -Wuninitialized
CFLAGS = 


BIN_DIR=.


LDFLAGS = 
CXXFLAGS += -I/home/chengtianpei/code/unisolver/include \
            -I/home/chengtianpei/code/autodiff \
            -I/share/software/anaconda3/envs/unisolver/include

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# obj files
OBJS_FI = \
${BIN_DIR}/tools.o \
${BIN_DIR}/main_fi.o \
${BIN_DIR}/formfunction_fi.o   reaction.o
${BIN_DIR}/%.o : %.c def.h reaction.h PhysicalFieldUtils.h
	-${PETSC_COMPILE_SINGLE} $< -o $@
fi: ${OBJS_FI} 
	-${CLINKER} -o ${BIN_DIR}/fi ${OBJS_FI} ${PETSC_LIB}
clear:
	-${RM} ${OBJS} ${BIN_DIR}/main
runss:
	-@${MPIEXEC} mpirun -np 2 ./fi -preload 0 -n1 100 -n2 10 -p 0 \
	 -tsize 0.01 -tfinal 0.02 -tsmax 10000 -tsstart 0 -hj_max_nit 8 \
	 -global_nonlinear_atol 8.e-1 -local_stop_atol 1.e-8  \
	 -snes_converged_reason -snes_max_it 100  \
	 -snes_type newtonls -snes_linesearch_type bt -snes_linesearch_minlambda 1.e-12 -snes_linesearch_maxlambda 1 -snes_linesearch_max_it 100 \
	 -snes_linesearch_alpha 1.e-4 -snes_linesearch_order 1 -snes_linesearch_damping 1.0  \
	 -snes_atol 8.e-1 -snes_rtol 1.e-8 -snes_stol 1.e-1000  -ksp_type gmres -ksp_atol 1.e-10 -ksp_rtol 1.e-5  \
	 -ksp_gmres_restart 30 -ksp_pc_side right -pc_type asm  -pc_asm_type restrict -pc_asm_overlap 1  \
	 -sub_ksp_type preonly -sub_pc_type lu -snes_monitor -ksp_monitor  \



runold:
	-@${MPIEXEC} -np 4 ./fi -preload 0 -n1 100 -n2 100 \
-tsize 0.1 -tfinal 8 -tsmax 10000 -tsstart 0 \
-snes_converged_reason \
-snes_type vinewtonrsls -snes_linesearch_type bt -snes_linesearch_minlambda 1.e-4  -snes_linesearch_max_it  100 -snes_linesearch_alpha 1.e-4   -snes_linesearch_order 3  -snes_linesearch_damping 1.0 \
-snes_atol 1.e-10 -snes_rtol 1.e-6  -snes_stol 1.e-1000 	 \
-ksp_type gmres     -ksp_atol 1.e-10 -ksp_rtol 1.e-6 \
-ksp_gmres_restart 30 -ksp_pc_side right -pc_type asm \
-pc_asm_type restrict -pc_asm_overlap 1 \
-sub_ksp_type preonly -sub_pc_type lu \
-snes_monitor \
