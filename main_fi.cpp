static char help[] = "3D single-phase flow, by LiRui .\n\\n";

#include <petscsnes.h>
#include <petscdmda.h>
#include <petscdm.h>
#include "def.h"
#include "stdlib.h"
#include "petscsys.h" 
#include "petsctime.h"

MPI_Comm    comm;
PetscMPIInt rank, size;
//PetscViewer viewer;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
	PetscErrorCode ierr;
	SNES	       snes,sub_snes;
	UserCtx        *user;
	PetscInt       n1, n2,n3;
	//DM             da;
	ParaCtx        param;
	TstepCtx       tsctx;
//      EventCtx       event; 

	PetscInitialize(&argc,&argv,(char *)0,help);

		PetscPreLoadBegin(PETSC_TRUE,"SetUp");

		param.PetscPreLoading = PetscPreLoading;

	comm = PETSC_COMM_WORLD;
	ierr = MPI_Comm_rank( comm, &rank ); CHKERRQ(ierr);
	ierr = MPI_Comm_size( comm, &size ); CHKERRQ(ierr);

	param.use_adaptive_dt = PETSC_FALSE;
#if 1
	param.use_adaptive_dt = PETSC_FALSE;
	tsctx.p    = 0.5;
	tsctx.smax = Smax;
	PetscOptionsHasName(PETSC_NULLPTR,PETSC_NULLPTR,"-p",&param.use_adaptive_dt);
	if (param.use_adaptive_dt) {
		ierr = PetscOptionsGetScalar(PETSC_NULLPTR,PETSC_NULLPTR,"-p",&tsctx.p,PETSC_NULLPTR); CHKERRQ(ierr);
		ierr = PetscOptionsGetScalar(PETSC_NULLPTR,PETSC_NULLPTR,"-smax",&tsctx.smax,PETSC_NULLPTR); CHKERRQ(ierr);
	}
#endif
    param.global_nonlinear_atol = 1e-10 ;
	param.local_stop_atol = 1e-5 ;
	param.hj_fnorm     = 0.0;
	param.hj_max_nit   = 10;


	ierr = PetscOptionsGetReal(PETSC_NULLPTR,PETSC_NULLPTR,"-global_nonlinear_atol",&param.global_nonlinear_atol,PETSC_NULLPTR);CHKERRQ(ierr); 
	ierr = PetscOptionsGetReal(PETSC_NULLPTR,PETSC_NULLPTR,"-local_stop_atol",&param.local_stop_atol,PETSC_NULLPTR);CHKERRQ(ierr);
	ierr = PetscOptionsGetInt(PETSC_NULLPTR,PETSC_NULLPTR,"-hj_max_nit",&param.hj_max_nit,PETSC_NULLPTR); CHKERRQ(ierr);
		tsctx.torder    = TORDER;
		tsctx.tstart    = TSTART;
		tsctx.tfinal    = TFINAL;
		tsctx.tsize     = TSIZE;
		tsctx.tsmax     = TSMAX;
		tsctx.tsstart   = 0;
		tsctx.fnorm     = 1;
	ierr = PetscOptionsGetInt(PETSC_NULLPTR,PETSC_NULLPTR,"-torder",&tsctx.torder,PETSC_NULLPTR); CHKERRQ(ierr);
	ierr = PetscOptionsGetScalar(PETSC_NULLPTR,PETSC_NULLPTR,"-tstart",&tsctx.tstart,PETSC_NULLPTR); CHKERRQ(ierr);
	ierr = PetscOptionsGetScalar(PETSC_NULLPTR,PETSC_NULLPTR,"-tfinal",&tsctx.tfinal,PETSC_NULLPTR); CHKERRQ(ierr);
	ierr = PetscOptionsGetScalar(PETSC_NULLPTR,PETSC_NULLPTR,"-tsize",&tsctx.tsize,PETSC_NULLPTR); CHKERRQ(ierr);
	tsctx.tcurr = tsctx.tstart;
	ierr = PetscOptionsGetInt(PETSC_NULLPTR,PETSC_NULLPTR,"-tsmax",&tsctx.tsmax,PETSC_NULLPTR); CHKERRQ(ierr);
	ierr = PetscOptionsGetInt(PETSC_NULLPTR,PETSC_NULLPTR,"-tsstart",&tsctx.tsstart,PETSC_NULLPTR); CHKERRQ(ierr);
	if ( tsctx.tsstart < 0 ) tsctx.tsstart = 0;
		tsctx.tscurr = tsctx.tsstart;
		tsctx.tsback = -1;
	ierr = PetscOptionsGetInt(PETSC_NULLPTR,PETSC_NULLPTR,"-tsback",&tsctx.tsback,PETSC_NULLPTR); CHKERRQ(ierr);

	if ( tsctx.tsback > tsctx.tsmax ) tsctx.tsback = -1;
		n1 = N1;
		n2 = N2;
    //n3 = N3;
	ierr = PetscOptionsGetInt(PETSC_NULLPTR,PETSC_NULLPTR,"-n1",&n1,PETSC_NULLPTR); CHKERRQ(ierr);
	ierr = PetscOptionsGetInt(PETSC_NULLPTR,PETSC_NULLPTR,"-n2",&n2,PETSC_NULLPTR); CHKERRQ(ierr);
	ierr = PetscOptionsGetInt(PETSC_NULLPTR,PETSC_NULLPTR,"-n3",&n3,PETSC_NULLPTR); CHKERRQ(ierr);

	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	   Create user context, set problem data, create vector data structures.	Also, compute the initial guess.
	   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
	ierr = PetscMalloc(sizeof(UserCtx),  &user); CHKERRQ(ierr);

	ierr = SNESCreate(comm,&snes);CHKERRQ(ierr);
	ierr = SNESCreate(comm,&sub_snes);CHKERRQ(ierr);
	ierr = DMDACreate2d(comm, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DMDA_STENCIL_STAR, n1, n2, PETSC_DECIDE, PETSC_DECIDE, DOF, WIDTH, 0, 0, &(user->da)); CHKERRQ(ierr);
	ierr = DMDACreate2d(comm, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DMDA_STENCIL_STAR, n1, n2, PETSC_DECIDE, PETSC_DECIDE, DOF_reaction, WIDTH, 0, 0, &(user->da_reaction)); CHKERRQ(ierr);
	ierr = DMDACreate2d(comm, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DMDA_STENCIL_STAR, n1, n2, PETSC_DECIDE, PETSC_DECIDE, DOF_perm, WIDTH, 0, 0, &(user->da_perm)); CHKERRQ(ierr);
	ierr = DMSetFromOptions(user->da);CHKERRQ(ierr);
	ierr = DMSetUp(user->da);CHKERRQ(ierr);
	ierr = DMSetUp(user->da_reaction);CHKERRQ(ierr);
	ierr = DMSetUp(user->da_perm);CHKERRQ(ierr);

	ierr = DMDASetFieldName(user->da, 0, "pressure");CHKERRQ(ierr);
	ierr = DMDASetFieldName(user->da, 1, "h+");CHKERRQ(ierr);
	ierr = DMDASetFieldName(user->da, 2, "hco3-");CHKERRQ(ierr);
	ierr = DMDASetFieldName(user->da, 3, "ca2+");CHKERRQ(ierr);
	ierr = DMDASetFieldName(user->da, 4, "mg2+");CHKERRQ(ierr);
	ierr = DMDASetFieldName(user->da, 5, "fe2+");CHKERRQ(ierr);
	ierr = SNESSetDM(snes,user->da);CHKERRQ(ierr);
	ierr = SNESSetDM(sub_snes,user->da);CHKERRQ(ierr);
	ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
	ierr = SNESSetFromOptions(sub_snes);CHKERRQ(ierr); 
		user->Q0      = PETSC_NULLPTR;
		user->myF     = PETSC_NULLPTR;
        user->perm    = PETSC_NULLPTR;
		user->tsctx   = &tsctx;
		user->param   = &param;
    ierr = DMDAGetInfo(user->da,0,&(user->n1),&(user->n2),0,0,0,0,0,0,0,0,0,0); CHKERRQ(ierr);
		user->dx       = L1 / (PetscScalar)(user->n1);
		user->dy       = L2 / (PetscScalar)(user->n2);


    ierr = DMCreateGlobalVector(user->da,&user->sol);


	ierr = DMGetLocalVector(user->da_reaction, &(user->initial_ref)); CHKERRQ(ierr);
	ierr = DMGetLocalVector(user->da_reaction, &(user->mineral_conc_old)); CHKERRQ(ierr);
	ierr = DMGetLocalVector(user->da_reaction, &(user->_sec_conc_old));CHKERRQ(ierr);
	ierr = DMGetLocalVector(user->da_reaction, &(user->_sec_conc));CHKERRQ(ierr);	
	ierr = DMGetLocalVector(user->da_reaction, &(user->_reaction_rate));CHKERRQ(ierr);	
	ierr = DMGetLocalVector(user->da_reaction, &(user->_mineral_sat));CHKERRQ(ierr);
	ierr = DMGetLocalVector(user->da_reaction,  &(user->kinetic_k)); CHKERRQ(ierr);
	ierr = DMGetLocalVector(user->da_reaction, &(user->eqm_k)); CHKERRQ(ierr);
	ierr = DMGetLocalVector(user->da_reaction, &(user->_mass_frac)); CHKERRQ(ierr);
	ierr = DMGetLocalVector(user->da_reaction, &(user->_mass_frac_old)); CHKERRQ(ierr);

	ierr = VecDuplicate(user->sol, &(user->Q0)); CHKERRQ(ierr);
	ierr = VecDuplicate(user->sol, &(user->myF)); CHKERRQ(ierr);
	ierr = VecDuplicate(user->sol, &(user->sub_sol)); CHKERRQ(ierr);
	ierr = DMGetLocalVector(user->da_perm, &(user->phi)); CHKERRQ(ierr);
	ierr = DMGetLocalVector(user->da_perm, &(user->phi_old)); CHKERRQ(ierr);
    ierr = DMGetLocalVector(user->da_perm, &(user->perm)); CHKERRQ(ierr);
    ierr = DMDAGetArray(user->da,PETSC_TRUE,(void**)&(user->M));CHKERRQ(ierr);
	ierr = DMDAGetArray(user->da,PETSC_TRUE,(void**)&(user->flag));CHKERRQ(ierr);
	ierr = DMDAGetArray(user->da,PETSC_TRUE,(void**)&(user->global_sol));CHKERRQ(ierr);

	ierr = SNESSetFunction(snes,NULL,FormFunction,(void*)user);

	ierr = FormInitialValue(user); CHKERRQ(ierr);
	ierr = FormInitialValue_Perm(user); CHKERRQ(ierr);
	ierr = FormInitialValue_Reaction(user); CHKERRQ(ierr);

		user->snes     =  snes;
		user->sub_snes     =  sub_snes;
	if (!param.PetscPreLoading) {
		ierr = PetscPrintf(comm, "\n+++++++++++++++++++++++ Problem parameters +++++++++++++++++++++\n"); CHKERRQ(ierr);
		ierr = PetscPrintf(comm," Single-phase flow, example: %d\n",EXAMPLE); CHKERRQ(ierr);
		ierr = PetscPrintf(comm, " Problem size %d, %d, Ncpu = %d \n", n1, n2, size ); CHKERRQ(ierr);

	if (param.use_adaptive_dt) {
		ierr = PetscPrintf(comm, " Torder = %d, TimeSteps = %g (adpt with %g) x %d, final time %f \n", tsctx.torder, tsctx.tsize, tsctx.p, tsctx.tsmax,tsctx.tfinal ); CHKERRQ(ierr);
	}else {
		ierr = PetscPrintf(comm, " Torder = %d, TimeSteps = %g x %d, final time %g \n", tsctx.torder, tsctx.tsize, tsctx.tsmax,tsctx.tfinal ); CHKERRQ(ierr);
	}
	ierr = PetscPrintf(comm, "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"); CHKERRQ(ierr);
	}


	PetscPreLoadStage("Solve");
	ierr = PetscObjectIncrementTabLevel((PetscObject)user->sub_snes,(PetscObject)user->snes,1); CHKERRQ(ierr);
	ierr = SNESSetOptionsPrefix(user->sub_snes,"local_");CHKERRQ(ierr);
	ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
	ierr = SNESSetFromOptions(sub_snes);CHKERRQ(ierr); 
	ierr = Update(user); CHKERRQ(ierr);

	ierr = VecDestroy(&user->sol); CHKERRQ(ierr);
    ierr = VecDestroy(&user->Q0); CHKERRQ(ierr);
	ierr = VecDestroy(&user->myF); CHKERRQ(ierr);	
	ierr = VecDestroy(&(user->sub_sol)); CHKERRQ(ierr);
	ierr = DMDARestoreArray(user->da,PETSC_TRUE,(void**)&(user->M));CHKERRQ(ierr);
	ierr = DMDARestoreArray(user->da,PETSC_TRUE,(void**)&(user->flag));CHKERRQ(ierr);
	ierr = DMDARestoreArray(user->da,PETSC_TRUE,(void**)&(user->global_sol));CHKERRQ(ierr);
	ierr = DMRestoreLocalVector(user->da_reaction, &user->kinetic_k); CHKERRQ(ierr);
	ierr = DMRestoreLocalVector(user->da_reaction, &(user->initial_ref)); CHKERRQ(ierr);
	ierr = DMRestoreLocalVector(user->da_reaction, &(user->mineral_conc_old)); CHKERRQ(ierr);
	ierr = DMRestoreLocalVector(user->da_reaction, &(user->_sec_conc_old));CHKERRQ(ierr);
	ierr = DMRestoreLocalVector(user->da_reaction, &(user->_sec_conc));CHKERRQ(ierr);	
	ierr = DMRestoreLocalVector(user->da_reaction, &(user->_reaction_rate));CHKERRQ(ierr);	
	ierr = DMRestoreLocalVector(user->da_reaction, &(user->_mineral_sat));CHKERRQ(ierr);	
	ierr = DMRestoreLocalVector(user->da_reaction, &(user->_mass_frac)); CHKERRQ(ierr);
	ierr = DMRestoreLocalVector(user->da_reaction, &(user->_mass_frac_old)); CHKERRQ(ierr);
	ierr = DMRestoreLocalVector(user->da_reaction,&user->eqm_k); CHKERRQ(ierr);


	ierr = DMDestroy(&user->da);CHKERRQ(ierr);
	ierr = DMDestroy(&user->da_reaction);CHKERRQ(ierr);
	ierr = DMDestroy(&user->da_perm);CHKERRQ(ierr);
	ierr = SNESDestroy(&snes);CHKERRQ(ierr);
	ierr = PetscFree(user); CHKERRQ(ierr);


	if (PetscPreLoading) {
	ierr = PetscPrintf(comm," PetscPreLoading over!\n"); CHKERRQ(ierr);
	}
	PetscPreLoadEnd();
	ierr = PetscFinalize(); CHKERRQ(ierr);
	return 0;
}

// #undef __FUNCT__  
// #define __FUNCT__ "DetermineNewPartition"
// PetscErrorCode DetermineNewPartition(void *ptr)
// {
//   PetscErrorCode ierr;
// 	UserCtx        *user = (UserCtx*) ptr;
//     ParaCtx *param = user->param;
// 	DM             da = user->da;
//   PetscInt       i, j, k,xl, yl, zl, nxl, nyl, nzl,nc;
// 	PhysicalField p;
// #ifdef UseNonlinearElimination
// 	PhysicalField rho;
// 	PetscScalar err[DOF];
// 	PetscScalar error[DOF];
// 	PetscReal hj_norm[DOF];

// 	rho = param->rho;
// #endif
// 	PetscReal unknowns; // the total number of unknowns=dof*mx*my
// 	unknowns = (PetscReal)(user->n1 * user->n2);
//   PetscFunctionBegin;
//   ierr = DMDAGetCorners( da, &xl, &yl, &zl, &nxl, &nyl, &nzl ); CHKERRQ(ierr);

// #ifdef UseNonlinearElimination
// 	for (i = 0; i < 2; i++)
// 	{
// 		err[i] = 0.0;
// 	}

// 	for (j = yl; j < yl + nyl; j++)
// 	{
// 		for (i = xl; i < xl + nxl; i++)
// 		{
// 			err[0] = max(err[0], user->M[j][i].pw);
			
// 				 	for (nc = 0; nc < DOF_reaction; ++nc){
// 			err[nc] = max(err[1], user->M[j][i].cw[nc]);
// 					}

// 		}
// 	}

// 	/*use the max-norm to define the bad region*/
// 	ierr = MPI_Allreduce(&err[0], &error[0], 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);
// 	CHKERRQ(ierr);
// 				 	for (nc = 0; nc < DOF_reaction; ++nc){
// 	ierr = MPI_Allreduce(&err[nc], &error[nc], 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);
// 	CHKERRQ(ierr);
// 					}



// 	/*use the max-norm to define the bad region*/
// 	hj_norm[0] =   error[0]/ unknowns ;//
// for (nc = 0; nc < DOF_reaction; ++nc){
// 	hj_norm[nc] =  error[nc]/ unknowns;;
// 						}
// #endif
// 	 for ( j=yl; j < yl+nyl; j++) {
// 		for ( i=xl; i < xl+nxl; i++) {

// 			p = user->M[j][i];
// 			for (nc = 0; nc < DOF_reaction; ++nc){
// 			if(hj_norm[0]>hj_norm[nc])
// 				{
// 			user->flag[j][i].pw = 0;
// 			user->flag[j][i].cw[nc] = 1;
// 				}
// 			}
// 			#if 0	  
// 			if (user->sub_type % 2 == 0)
// 			{
// 				user->flag[j][i].pw = 0;
// 				for (nc = 0; nc < DOF_reaction; ++nc){
// 				user->flag[j][i].cw[nc] = 0;
// 				}
        
// 				if (p.pw >= hj_norm[0])
// 				{
// 					user->flag[j][i].pw = 1;

// 				}else{
			
// 						for (nc = 0; nc < DOF_reaction; ++nc){
// 				if (p.cw[nc] >= hj_norm[nc])
// 				{
// //printf("i=%d, j=%d, p.pw=%g, hj_norm[0]=%g\n",i, j, p.cw[nc], hj_norm[nc]); 
// 					//user->flag[j][i].cw[nc] = 1;
// 						}


// 				}
// 				}

// 			}
// 				#endif

// 		}
// 	}


//   PetscFunctionReturn(0);
// }

#undef __FUNCT__  
#define __FUNCT__ "DetermineNewPartition"
PetscErrorCode DetermineNewPartition(void *ptr)
{
  PetscErrorCode ierr;
  UserCtx       *user = (UserCtx*)ptr;
  DM             da = user->da;
  PetscInt       i, j, xl, yl, zl, nxl, nyl, nzl, nc;

  PetscFunctionBegin;
  ierr = DMDAGetCorners( da, &xl, &yl, &zl, &nxl, &nyl, &nzl ); CHKERRQ(ierr);

	 for ( j=yl; j < yl+nyl; j++) {
		for ( i=xl; i < xl+nxl; i++) {
			if(user->number==0){
          user->flag[j][i].pw    = 1 ;
		  for (nc = 0; nc < DOF_reaction; ++nc){
           user->flag[j][i].cw[nc]    = 0 ;  
		}
		}else{
		   user->flag[j][i].pw    = 0 ;	
		    for (nc = 0; nc < DOF_reaction; ++nc){
		   user->flag[j][i].cw[nc]    = 1 ;  
			  }
		}
		}

	}
	user->number = user->number+1;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Update"
PetscErrorCode Update(void *ptr)
{
	PetscErrorCode ierr;
	UserCtx        *user  = (UserCtx*) ptr;
	ParaCtx        *param = user->param;
	TstepCtx       *tsctx = user->tsctx;
	SNES	       snes   = user->snes;
	SNESConvergedReason reason;
	PetscInt       max_steps, max_it, i;    
	int    its=0,lits=0.0,fits=0;
	PetscScalar    res=0.0, res0=0.0, scaling=0.0;
	double time0=0.0,time1=0.0,totaltime=0.0;
    PetscScalar    fnorm;
	FILE           *fp;
	char           history[36], filename[PETSC_MAX_PATH_LEN-1];
	PetscScalar    hj_fnorm=0.0;
	PetscInt       globalksp = 0 , localksp = 0 , globalsnes = 0 , localsnes = 0 ;
	user->number=0;
	PetscFunctionBegin;

	if (param->PetscPreLoading) {
			max_steps = 1;
	} else {
			max_steps = tsctx->tsmax;
		}

	if (!param->PetscPreLoading ) {
		sprintf( history, "history_c%d_n%dx%d.data", size, user->n1, user->n2 );
		ierr = PetscFOpen(comm, history, "a", &fp); CHKERRQ(ierr);
		ierr = PetscFPrintf(comm,fp,"%% example, step, t, reason, fnorm, its_snes, its_ksp, its_fail, time\n" ); CHKERRQ(ierr);	
	}

	for (tsctx->tscurr = tsctx->tsstart + 1; (tsctx->tscurr <= tsctx->tsstart + max_steps); tsctx->tscurr++) {
		ierr = SNESComputeFunction(snes, user->Q0, user->myF);CHKERRQ(ierr);
		ierr = VecNorm(user->myF, NORM_2, &res); CHKERRQ(ierr);

	if (user->param->use_adaptive_dt) {
		if (tsctx->tscurr==tsctx->tsstart + 1) {
			res0 = res;
		} else {
			scaling = pow(res0/res, tsctx->p);
			if (scaling > tsctx->smax) scaling = tsctx->smax;
			if (scaling < 1./tsctx->smax) scaling = 1./tsctx->smax;
			tsctx->tsize = tsctx->tsize * scaling;
			if (( tsctx->tcurr < tsctx->tfinal - EPS )&&( tsctx->tcurr + tsctx->tsize >= tsctx->tfinal + EPS )) {
			tsctx->tsize = tsctx->tfinal - tsctx->tcurr;
						}
	//	if(tsctx->tsize > tsctx->tmax) tsctx->tsize = tsctx->tmax ;
			}
	ierr = PetscPrintf(comm, " current initial residual = %g with adpt method, current time size = %g\n", res, tsctx->tsize);
	}else{
	ierr = PetscPrintf(comm, " current initial residual = %g, current time size = %g\n", res, tsctx->tsize);	}

		tsctx->tcurr += tsctx->tsize;
	if (!param->PetscPreLoading ) {
	ierr = PetscPrintf(comm, "\n====================== Step: %d, time: %f ====================\n",
	                  tsctx->tscurr, tsctx->tcurr ); CHKERRQ(ierr);}
	ierr = PetscTime(&time0); CHKERRQ(ierr);
	ierr = SNESSolve(snes,0,user->sol);

	 while(0){
	  /*  Global update begin */
	 ierr = PetscPrintf(PETSC_COMM_WORLD,"Global update begin\n");CHKERRQ(ierr);	
	 max_it = param->hj_max_nit; 
        if( (hj_fnorm <= param->local_stop_atol) && (localsnes>=10) ) {
	       max_it = 300; 
	    }
        ierr = SNESSetTolerances(user->snes, 
              PETSC_DEFAULT,   
              PETSC_DEFAULT,    
              PETSC_DEFAULT, //stol  - convergence tolerance in terms of the norm of the change in the solution between steps, default: ,1E-08
              max_it, //maxit  - maximum number of iterations, default: 50
              PETSC_DEFAULT);//maxf  - maximum number of function evaluations, default: 10000
         CHKERRQ(ierr);

	ierr = SNESSetFunction(user->snes,NULL,FormFunction,user);
	//ierr = PetscLogEventBegin(global_snes_solve,snes,0,0,0);CHKERRQ(ierr);
        ierr = SNESSolve(user->snes,0,user->sol);

	//ierr = PetscLogEventEnd(global_snes_solve,snes,0,0,0);CHKERRQ(ierr);
	ierr = SNESGetIterationNumber(user->snes,&its);CHKERRQ(ierr);
        globalsnes = globalsnes + its;

        ierr = SNESGetLinearSolveIterations(user->snes,&lits);CHKERRQ(ierr);
        globalksp = globalksp + lits ;
	    ierr = PetscPrintf(PETSC_COMM_WORLD,"Global update end, Netwon iteration=%d,GMRES iteration=%d\n",globalsnes,globalksp);CHKERRQ(ierr);	
	    SNESLineSearch linesearch;
        SNESGetLineSearch(user->snes,&linesearch);
        SNESLineSearchGetNorms(linesearch,NULL,&hj_fnorm,NULL);
	    param->hj_fnorm = hj_fnorm;
        if(hj_fnorm < param->global_nonlinear_atol ) break ;
        user->sub_type = 0;  
      //set i<2 means using two different NE corrections,i<1 means using a NE correction.
        for (i=0;i<NE_TYPE;i++) {		    		
	ierr = PetscPrintf(PETSC_COMM_WORLD, "    Subspace correction begin: type=%d\n",user->sub_type);CHKERRQ(ierr);
	if(user->sub_type == 0){
	ierr = VecCopy(user->sol,user->sub_sol); CHKERRQ(ierr);
	}
	ierr = DetermineNewPartition(user);CHKERRQ(ierr);    /* Determine the new partition */ 
	ierr = SNESSetFunction(user->sub_snes,NULL,FormFunction_subspace,user);
	ierr = DMDAVecGetArray(user->da,  user->sub_sol, &user->global_sol);CHKERRQ(ierr);
	        ierr = SNESSetTolerances(user->sub_snes, 
              1.e-1,   
              1.e-6,    
              PETSC_DEFAULT, //stol  - convergence tolerance in terms of the norm of the change in the solution between steps, default: ,1E-08
              max_it, //maxit  - maximum number of iterations, default: 50
              PETSC_DEFAULT);//maxf  - maximum number of function evaluations, default: 10000
         CHKERRQ(ierr);
	ierr = SNESSolve(user->sub_snes,0,user->sub_sol);

	ierr = DMDAVecRestoreArray(user->da,user->sub_sol, &user->global_sol);CHKERRQ(ierr);//add by haijian
	ierr = SNESGetIterationNumber(user->sub_snes,&its);CHKERRQ(ierr);        
        localsnes = localsnes + its ;
        ierr = SNESGetLinearSolveIterations(user->sub_snes,&lits);CHKERRQ(ierr); 
	localksp = localksp + lits;
        ierr = PetscPrintf(PETSC_COMM_WORLD,"    Subspace correction: type=%d, Netwon iteration=%d,GMRES iteration=%d\n",user->sub_type,localsnes,localksp);CHKERRQ(ierr);
        user->sub_type = user->sub_type + 1;
	}
	ierr = VecCopy(user->sub_sol,user->sol); CHKERRQ(ierr);

	}	
	ierr = PetscTime(&time1); CHKERRQ(ierr);
    ierr = VecCopy(user->sol,user->Q0);
    ierr = VecCopy(user->_sec_conc,user->_sec_conc_old);
	ierr = VecCopy(user->_mass_frac,user->_mass_frac_old);
   

         PetscReal      litspit=0.0;
         PetscReal      its1=0.0;
         PetscReal      lits1=0.0;
         Vec F;
	ierr = SNESGetConvergedReason( snes, &reason ); CHKERRQ(ierr);
	ierr = SNESGetFunction(snes,&F,0,0);CHKERRQ(ierr);
	ierr = VecNorm(F,NORM_2,&fnorm);CHKERRQ(ierr);
	ierr = SNESGetIterationNumber(snes,&its); CHKERRQ(ierr);
        its1=its1+its;
	ierr = SNESGetLinearSolveIterations(snes,&lits); CHKERRQ(ierr);
        lits1=lits1+lits;
        litspit = ((PetscReal)lits1)/((PetscReal)its1);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Average Linear its / Newton = %g\n",litspit);CHKERRQ(ierr);  
	ierr = SNESGetNonlinearStepFailures( snes, &fits ); CHKERRQ(ierr);
	ierr = PetscPrintf(comm, " Snes converged reason = %d\n",  reason ); CHKERRQ(ierr);
	ierr = PetscPrintf(comm, " 2-norm of F = %g\n", fnorm ); CHKERRQ(ierr);
	ierr = PetscPrintf(comm, " number of Newton iterations = %f\n",its1); CHKERRQ(ierr);
	ierr = PetscPrintf(comm, " number of Linear iterations = %f\n",lits1); CHKERRQ(ierr);
	ierr = PetscPrintf(comm, " number of unsuccessful steps = %d\n", fits ); CHKERRQ(ierr);

						
	if (!param->PetscPreLoading ) {
	ierr = PetscFPrintf(comm,fp, "%d\t, %d\t, %g\t, %d\t, %g\t, %d\t, %d\t, %d\t, %g\n",EXAMPLE,tsctx->tscurr, tsctx->tsize, reason, fnorm, its, lits, fits, time1-time0 ); CHKERRQ(ierr);
		}

		res0 = res;



	
		ierr = MPI_Barrier(comm); CHKERRQ(ierr);
	if ( tsctx->tcurr > tsctx->tfinal - EPS ) {
			tsctx->tscurr++;
			break;
		}

}
	tsctx->tscurr--;


	ierr = PetscPrintf(comm, "\n+++++++++++++++++++++++ Summary +++++++++++++++++++++++++\n" ); CHKERRQ(ierr);
	ierr = PetscPrintf(comm, " Final time = %g, Cost time = %g\n", tsctx->tcurr, totaltime ); CHKERRQ(ierr);
# if 1	// For computing  Darcy velocity 
	Vec  global_perm;
        ierr = DMCreateGlobalVector(user->da_reaction,&global_perm);CHKERRQ(ierr);
	ierr = DMLocalToGlobalBegin(user->da_reaction,user->_sec_conc, INSERT_VALUES,global_perm);CHKERRQ(ierr);
	ierr = DMLocalToGlobalEnd(user->da_reaction,user->_sec_conc, INSERT_VALUES,global_perm);CHKERRQ(ierr);
 #if 1
sprintf( filename, "example=%dpermeability_xxascii.data",EXAMPLE);
ierr = DataSaveASCII(global_perm,filename);
 
  #endif 

#endif

	if (!param->PetscPreLoading) {
		ierr = PetscFClose(comm, fp); CHKERRQ(ierr);
	}

	PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FormInitialValue_Perm"
PetscErrorCode FormInitialValue_Perm(void* ptr)
{
	PetscErrorCode ierr;
	UserCtx        *user = (UserCtx*) ptr;
	DM             da = user->da,da_perm =user->da_perm;
	PermField      **perm,**phi;
	PetscInt       i, j, xg, yg, zg, nxg, nyg, nzg;
	PetscFunctionBeginUser;
    ierr = DMDAGetGhostCorners( da, &xg, &yg, &zg, &nxg, &nyg, &nzg ); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da_perm,user->perm,&perm); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da_perm,user->phi,&phi); CHKERRQ(ierr);

	for ( j=yg; j < yg+nyg; j++) {
		for ( i=xg; i < xg+nxg; i++) {
		for (int nc = 0; nc < DOF_perm; nc++){
			perm[j][i].xx[nc] = 1e-10;
			phi[j][i].xx[nc] = 0.2;
		}
			
		}
	
}

ierr = DMDAVecRestoreArray(da_perm,user->perm,&perm); CHKERRQ(ierr);
ierr = DMDAVecRestoreArray(da_perm,user->phi,&phi); CHKERRQ(ierr);
ierr = VecCopy(user->phi,user->phi_old);
	PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "FormInitialValue"
PetscErrorCode FormInitialValue(void* ptr)
{
	PetscErrorCode ierr;
	UserCtx        *user = (UserCtx*) ptr;
	DM             da = user->da;
    PhysicalField  **sol;
	PetscInt       i, j, y_loc,xl, yl, zl, nxl, nyl, nzl;
	PetscFunctionBeginUser;
	ierr = DMDAGetCorners( da, &xl, &yl, &zl, &nxl, &nyl, &nzl ); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da,user->sol,&sol); CHKERRQ(ierr);
	for ( j=yl; j < yl+nyl; j++) {
			y_loc = ((PetscScalar)j+0.5)*user->dy;
		for ( i=xl; i < xl+nxl; i++) {
			sol[j][i].pw = P_init;
		   for (int nc = 0; nc < DOF_reaction; ++nc){
		   if(y_loc<=2.5&&(i==0)){
		   sol[j][i].cw[nc] = c_BC_L;
		   }else{
		   sol[j][i].cw[nc] = c_BC_R;
		   }
				}
		
		}
	}
    ierr = DMDAVecRestoreArray(da,user->sol,&sol); CHKERRQ(ierr);
 	ierr = VecCopy(user->sol,user->Q0); CHKERRQ(ierr);
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormInitialValue_Reaction"
PetscErrorCode FormInitialValue_Reaction(void* ptr)
{
	PetscErrorCode ierr;
	UserCtx        *user = (UserCtx*) ptr;
	DM             da=user->da,da_reaction = user->da_reaction;
    ReactionField  **_sec_conc,**eqm_k, **_mass_frac;
	PetscInt       i, j, xg, yg, zg, nxg, nyg, nzg;
	PetscFunctionBeginUser;

    ierr = DMDAGetGhostCorners( da, &xg, &yg, &zg, &nxg, &nyg, &nzg ); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da_reaction,user->_sec_conc,&_sec_conc); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da_reaction,user->_mass_frac_old,&_mass_frac); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da_reaction,user->eqm_k,&eqm_k); CHKERRQ(ierr);



	for ( j=yg; j < yg+nyg; j++) {
		for ( i=xg; i < xg+nxg; i++) {
		for (int nc = 0; nc < DOF_reaction; ++nc){
		   	_sec_conc[j][i].reaction[nc]=1.e-7;
			_mass_frac[j][i].reaction[nc]=1.e-6;	

				}
			eqm_k[j][i].reaction[0]=2.19e6;
			eqm_k[j][i].reaction[1]=4.73e-11;
			eqm_k[j][i].reaction[2]=0.222;
			eqm_k[j][i].reaction[3]=1e-2;
			eqm_k[j][i].reaction[4]=1e-3;
			
		}
	
}
ierr = DMDAVecRestoreArray(da_reaction,user->eqm_k,&eqm_k); CHKERRQ(ierr);
ierr = DMDAVecRestoreArray(da_reaction,user->_mass_frac_old,&_mass_frac); CHKERRQ(ierr);
ierr = DMDAVecRestoreArray(da_reaction,user->_sec_conc,&_sec_conc); CHKERRQ(ierr);
ierr = VecCopy(user->_sec_conc,user->_sec_conc_old);

	PetscFunctionReturn(0);
}


