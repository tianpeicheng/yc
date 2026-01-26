PETSC_DIR=/share/userfile/chengtianpei/petsc
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
-include ../../../petscdir.mk


BIN_DIR=.


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
	-@${MPIEXEC} mpirun -np 16 ./fi -preload 0 -n1 100 -n2 10 -p 0 user_aspin 0 \
	 -tsize 0.01 -tfinal 1 -tsmax 10000 -tsstart 0  \
	 -snes_converged_reason -snes_max_it 100  \
	 -snes_type newtonls -snes_linesearch_type bt -snes_linesearch_minlambda 1.e-12 -snes_linesearch_maxlambda 1 -snes_linesearch_max_it 100 \
	 -snes_linesearch_alpha 1.e-4 -snes_linesearch_order 1 -snes_linesearch_damping 1.0  \
	 -snes_atol 1.e-10 -snes_atol 1.e-4 -snes_stol 1.e-1000  -ksp_type gmres -ksp_atol 1.e-10 -ksp_rtol 1.e-5  \
	 -ksp_gmres_restart 30 -ksp_pc_side right -pc_type asm  -pc_asm_type restrict -pc_asm_overlap 1  \
	 -sub_ksp_type preonly -sub_pc_type lu -snes_monitor -ksp_monitor  \



runold:
	-@${MPIEXEC} -np 4 ./fi -preload 0 -n1 100 -n2 10 \
-tsize 0.1 -tfinal 10 -tsmax 10000 -tsstart 0 \
-snes_converged_reason \
-snes_type vinewtonrsls -snes_linesearch_type bt -snes_linesearch_minlambda 1.e-4  -snes_linesearch_max_it  100 -snes_linesearch_alpha 1.e-4   -snes_linesearch_order 3  -snes_linesearch_damping 1.0 \
-snes_atol 1.e-10 -snes_rtol 1.e-6  -snes_stol 1.e-1000 	 \
-ksp_type gmres     -ksp_atol 1.e-10 -ksp_rtol 1.e-6 \
-ksp_gmres_restart 30 -ksp_pc_side right -pc_type asm \
-pc_asm_type restrict -pc_asm_overlap 1 \
-sub_ksp_type preonly -sub_pc_type lu \
-snes_monitor \


runss_1:
	-@${MPIEXEC} mpirun -np 4 ./fi -preload 0 -n1 100 -n2 10 -p 0 -da_overlop 1 \
	 -tsize 0.01 -tfinal 1 -tsmax 10000 -tsstart 0  \
	 -snes_converged_reason -snes_atol 1.e-10 -snes_atol 1.e-4 -snes_stol 1.e-100 \
	 -snes_type aspin -npc_snes_type nasm -npc_sub_snes_type newtonls -npc_sub_snes_atol 1.e-10 -npc_sub_snes_rtol 1.e-6 -npc_sub_snes_stol 1.e-100  \
	 -npc_sub_snes_stol 1.e-10 -snes_linesearch_type basic -npc_sub_snes_linesearch_type bt  -snes_monitor_short  \
	 -npc_sub_ksp_type  gmres -npc_sub_pc_type lu  \



runss_2:
	-@${MPIEXEC} mpirun -np 16  ./fi -preload 0 -n1 2000 -n2 10 -p 0  -da_overlop 2 \
	 -tsize 0.1 -tfinal 0.9 -tsmax 10000 -tsstart 0  \
	 -snes_converged_reason -snes_atol 1.e-10 -snes_rtol 1.e-4 -snes_stol 1.e-100 \
	 -snes_type aspin -npc_snes_type nasm -npc_sub_snes_type newtonls -npc_sub_snes_atol 1.e-10 -npc_sub_snes_rtol 1.e-6 -npc_sub_snes_stol 1.e-100  \
	 -npc_sub_snes_stol 1.e-10 -snes_linesearch_type bt -npc_sub_snes_linesearch_type bt -snes_monitor_short  \
	 -npc_sub_ksp_type gmres -npc_sub_pc_type ilu  -npc_sub_pc_factor_levels 2 \




runss_3:
	-@${MPIEXEC} mpirun -np 16  ./fi -preload 0 -n1 220 -n2 60 -p 0 -flow_da_overlap 1 -perm_da_overlap 0  \
	 -tsize 0.01 -tfinal 100 -tsmax 10000 -tsstart 0 -reaction_da_overlap 0 -secondary_da_overlap 0  \
	 -snes_converged_reason -snes_atol 1.e-10 -snes_rtol 1.e-6 -snes_stol 1.e-100 \
	  -snes_type aspin   -npc_snes_type nasm -npc_sub_snes_type newtonls -npc_sub_snes_atol 1.e-10 -npc_sub_snes_rtol 1.e-6 -npc_sub_snes_stol 1.e-100  \
	 -npc_sub_snes_stol 1.e-10 -snes_linesearch_type bt -npc_sub_snes_linesearch_type bt -snes_monitor_short  \
	 -npc_sub_ksp_type gmres -npc_sub_pc_type ilu  -npc_sub_pc_factor_levels 2 \

