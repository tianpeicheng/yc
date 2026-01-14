static char help[] = "3D single-phase flow, by LiRui .\n\\n";
//aspin
#include <petscdm.h>
#include <petscdmda.h>
#include <petscsnes.h>
#include "def.h"
#include "petscsys.h"
#include "petsctime.h"
#include "reaction.h"
#include "stdlib.h"

MPI_Comm comm;
PetscMPIInt rank, size;
// PetscViewer viewer;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char** argv) {
    PetscErrorCode ierr;
    SNES snes;
    UserCtx* user;
    PetscInt n1, n2, n3;
    ParaCtx param;
    TstepCtx tsctx;
    Mat J;

    PetscInitialize(&argc, &argv, (char*)0, help);

    PetscPreLoadBegin(PETSC_TRUE, "SetUp");

    param.PetscPreLoading = PetscPreLoading;

    comm = PETSC_COMM_WORLD;
    ierr = MPI_Comm_rank(comm, &rank);
    CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm, &size);
    CHKERRQ(ierr);

    param.use_adaptive_dt = PETSC_FALSE;
#if 1
    param.use_adaptive_dt = PETSC_FALSE;
    tsctx.p = 0.5;
    tsctx.smax = Smax;
    PetscOptionsHasName(PETSC_NULLPTR, PETSC_NULLPTR, "-p",
                        &param.use_adaptive_dt);
    if (param.use_adaptive_dt) {
        ierr = PetscOptionsGetScalar(PETSC_NULLPTR, PETSC_NULLPTR, "-p",
                                     &tsctx.p, PETSC_NULLPTR);
        CHKERRQ(ierr);
        ierr = PetscOptionsGetScalar(PETSC_NULLPTR, PETSC_NULLPTR, "-smax",
                                     &tsctx.smax, PETSC_NULLPTR);
        CHKERRQ(ierr);
    }
#endif
    param.global_nonlinear_atol = 1e-10;
    param.local_stop_atol = 1e-5;
    param.hj_fnorm = 0.0;
    param.hj_max_nit = 10;
    ierr = PetscOptionsGetReal(PETSC_NULLPTR, PETSC_NULLPTR,
                               "-global_nonlinear_atol",
                               &param.global_nonlinear_atol, PETSC_NULLPTR);
    CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(PETSC_NULLPTR, PETSC_NULLPTR, "-local_stop_atol",
                               &param.local_stop_atol, PETSC_NULLPTR);
    CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(PETSC_NULLPTR, PETSC_NULLPTR, "-hj_max_nit",
                              &param.hj_max_nit, PETSC_NULLPTR);
    CHKERRQ(ierr);
    tsctx.torder = TORDER;
    tsctx.tstart = TSTART;
    tsctx.tfinal = TFINAL;
    tsctx.tsize = TSIZE;
    tsctx.tsmax = TSMAX;
    tsctx.tsstart = 0;
    tsctx.fnorm = 1;
    ierr = PetscOptionsGetInt(PETSC_NULLPTR, PETSC_NULLPTR, "-torder",
                              &tsctx.torder, PETSC_NULLPTR);
    CHKERRQ(ierr);
    ierr = PetscOptionsGetScalar(PETSC_NULLPTR, PETSC_NULLPTR, "-tstart",
                                 &tsctx.tstart, PETSC_NULLPTR);
    CHKERRQ(ierr);
    ierr = PetscOptionsGetScalar(PETSC_NULLPTR, PETSC_NULLPTR, "-tfinal",
                                 &tsctx.tfinal, PETSC_NULLPTR);
    CHKERRQ(ierr);
    ierr = PetscOptionsGetScalar(PETSC_NULLPTR, PETSC_NULLPTR, "-tsize",
                                 &tsctx.tsize, PETSC_NULLPTR);
    CHKERRQ(ierr);
    tsctx.tcurr = tsctx.tstart;
    ierr = PetscOptionsGetInt(PETSC_NULLPTR, PETSC_NULLPTR, "-tsmax",
                              &tsctx.tsmax, PETSC_NULLPTR);
    CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(PETSC_NULLPTR, PETSC_NULLPTR, "-tsstart",
                              &tsctx.tsstart, PETSC_NULLPTR);
    CHKERRQ(ierr);
    if (tsctx.tsstart < 0) tsctx.tsstart = 0;
    tsctx.tscurr = tsctx.tsstart;
    tsctx.tsback = -1;
    ierr = PetscOptionsGetInt(PETSC_NULLPTR, PETSC_NULLPTR, "-tsback",
                              &tsctx.tsback, PETSC_NULLPTR);
    CHKERRQ(ierr);

    if (tsctx.tsback > tsctx.tsmax) tsctx.tsback = -1;
    n1 = N1;
    n2 = N2;
    ierr = PetscOptionsGetInt(PETSC_NULLPTR, PETSC_NULLPTR, "-n1", &n1,
                              PETSC_NULLPTR);
    CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(PETSC_NULLPTR, PETSC_NULLPTR, "-n2", &n2,
                              PETSC_NULLPTR);
    CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(PETSC_NULLPTR, PETSC_NULLPTR, "-n3", &n3,
                              PETSC_NULLPTR);
    CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create user context, set problem data, create vector data structures.
       Also, compute the initial guess.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = PetscMalloc(sizeof(UserCtx), &user);
    CHKERRQ(ierr);

    ierr = SNESCreate(comm, &snes);
    CHKERRQ(ierr);

    ierr = DMDACreate2d(comm, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
                        DMDA_STENCIL_STAR, n1, n2, PETSC_DECIDE, PETSC_DECIDE,
                        DOF, WIDTH, 0, 0, &(user->da));
    CHKERRQ(ierr);
    ierr = DMDACreate2d(comm, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
                        DMDA_STENCIL_STAR, n1, n2, PETSC_DECIDE, PETSC_DECIDE,
                        DOF_reaction, WIDTH, 0, 0, &(user->da_reaction));
    CHKERRQ(ierr);
    ierr = DMDACreate2d(comm, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
                        DMDA_STENCIL_STAR, n1, n2, PETSC_DECIDE, PETSC_DECIDE,
                        DOF_perm, WIDTH, 0, 0, &(user->da_perm));
    CHKERRQ(ierr);
    ierr = DMSetFromOptions(user->da);
    CHKERRQ(ierr);
    ierr = DMSetUp(user->da);
    CHKERRQ(ierr);
    ierr = DMSetUp(user->da_reaction);
    CHKERRQ(ierr);
    ierr = DMSetUp(user->da_perm);
    CHKERRQ(ierr);

    ierr = DMDASetFieldName(user->da, 0, "pressure");
    CHKERRQ(ierr);
    ierr = DMDASetFieldName(user->da, 1, "h+");
    CHKERRQ(ierr);
    ierr = DMDASetFieldName(user->da, 2, "hco3-");
    CHKERRQ(ierr);
    ierr = DMDASetFieldName(user->da, 3, "ca2+");
    CHKERRQ(ierr);
    ierr = DMDASetFieldName(user->da, 4, "mg2+");
    CHKERRQ(ierr);
    ierr = DMDASetFieldName(user->da, 5, "fe2+");
    CHKERRQ(ierr);
    ierr = SNESSetDM(snes, user->da);
    CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);
    CHKERRQ(ierr);
    user->tsctx = &tsctx;
    user->param = &param;
    ierr = DMDAGetInfo(user->da, 0, &(user->n1), &(user->n2), 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0);
    CHKERRQ(ierr);
    user->dx = L1 / (PetscScalar)(user->n1);
    user->dy = L2 / (PetscScalar)(user->n2);

    ierr = DMCreateGlobalVector(user->da, &user->sol);
    ierr = VecDuplicate(user->sol, &(user->myF));
    CHKERRQ(ierr);
    ierr = DMDAGetArray(user->da_perm, PETSC_TRUE, (void**)&(user->phi_field));
    CHKERRQ(ierr);
    ierr =
        DMDAGetArray(user->da_perm, PETSC_TRUE, (void**)&(user->phi_old_field));
    CHKERRQ(ierr);
    ierr = DMDAGetArray(user->da_perm, PETSC_TRUE, (void**)&(user->perm_field));
    CHKERRQ(ierr);
    ierr = DMDAGetArray(user->da_reaction, PETSC_TRUE,
                        (void**)&(user->_sec_conc_old_field));
    CHKERRQ(ierr);
    ierr = DMDAGetArray(user->da_reaction, PETSC_TRUE,
                        (void**)&(user->eqm_k_field));
    CHKERRQ(ierr);
    ierr = DMDAGetArray(user->da_reaction, PETSC_TRUE,
                        (void**)&(user->_mass_frac_old_field));
    CHKERRQ(ierr);
    ierr = DMDAGetArray(user->da_reaction, PETSC_TRUE,
                        (void**)&(user->initial_ref_field));
    CHKERRQ(ierr);
    ierr = DMDAGetArray(user->da, PETSC_FALSE, (void**)&(user->xold));
    CHKERRQ(ierr);

    ierr = DMDASNESSetFunctionLocal(
        user->da, INSERT_VALUES,
        (PetscErrorCode (*)(DMDALocalInfo*, void*, void*,
                            void*))FormFunctionLocal,
        user);
    ierr = FormInitialValue_local(user);
    CHKERRQ(ierr);
    ierr = FormInitialValue_Perm_local(user);
    CHKERRQ(ierr);
    ierr = FormInitialValue_Reaction_local(user);
    CHKERRQ(ierr);
    user->snes = snes;
    if (!param.PetscPreLoading) {
        ierr = PetscPrintf(comm,
                           "\n+++++++++++++++++++++++ Problem parameters "
                           "+++++++++++++++++++++\n");
        CHKERRQ(ierr);
        ierr = PetscPrintf(comm, " Single-phase flow, example: %d\n", EXAMPLE);
        CHKERRQ(ierr);
        ierr = PetscPrintf(comm, " Problem size %d, %d, Ncpu = %d \n", n1, n2,
                           size);
        CHKERRQ(ierr);

        if (param.use_adaptive_dt) {
            ierr = PetscPrintf(comm,
                               " Torder = %d, TimeSteps = %g (adpt with %g) x "
                               "%d, final time %f \n",
                               tsctx.torder, tsctx.tsize, tsctx.p, tsctx.tsmax,
                               tsctx.tfinal);
            CHKERRQ(ierr);
        } else {
            ierr = PetscPrintf(
                comm, " Torder = %d, TimeSteps = %g x %d, final time %g \n",
                tsctx.torder, tsctx.tsize, tsctx.tsmax, tsctx.tfinal);
            CHKERRQ(ierr);
        }
        ierr = PetscPrintf(comm,
                           "+++++++++++++++++++++++++++++++++++++++++++++++++++"
                           "+++++++++++++++\n");
        CHKERRQ(ierr);
    }

    PetscPreLoadStage("Solve");
    ierr = SNESSetFromOptions(snes);
    CHKERRQ(ierr);
    ierr = Update(user);
    CHKERRQ(ierr);

    ierr = VecDestroy(&user->sol);
    CHKERRQ(ierr);
    ierr = VecDestroy(&user->myF);
    CHKERRQ(ierr);

    ierr =
        DMDARestoreArray(user->da_perm, PETSC_TRUE, (void**)&(user->phi_field));
    CHKERRQ(ierr);
    ierr = DMDARestoreArray(user->da_perm, PETSC_TRUE,
                            (void**)&(user->phi_old_field));
    CHKERRQ(ierr);
    ierr = DMDARestoreArray(user->da_perm, PETSC_TRUE,
                            (void**)&(user->perm_field));
    CHKERRQ(ierr);
    ierr = DMDARestoreArray(user->da_reaction, PETSC_TRUE,
                            (void**)&(user->_sec_conc_old_field));
    CHKERRQ(ierr);
    ierr = DMDARestoreArray(user->da_reaction, PETSC_TRUE,
                            (void**)&(user->eqm_k_field));
    CHKERRQ(ierr);
    ierr = DMDARestoreArray(user->da_reaction, PETSC_TRUE,
                            (void**)&(user->initial_ref_field));
    ierr = DMDARestoreArray(user->da, PETSC_FALSE, (void**)&(user->xold));
    CHKERRQ(ierr);
    ierr = DMDestroy(&user->da);
    CHKERRQ(ierr);
    ierr = DMDestroy(&user->da_reaction);
    CHKERRQ(ierr);
    ierr = DMDestroy(&user->da_perm);
    CHKERRQ(ierr);
    ierr = SNESDestroy(&snes);
    CHKERRQ(ierr);
    ierr = PetscFree(user);
    CHKERRQ(ierr);

    if (PetscPreLoading) {
        ierr = PetscPrintf(comm, " PetscPreLoading over!\n");
        CHKERRQ(ierr);
    }

    PetscPreLoadEnd();
    ierr = PetscFinalize();
    CHKERRQ(ierr);
    return 0;
}

#undef __FUNCT__
#define __FUNCT__ "Update"
PetscErrorCode Update(void* ptr) {
    PetscErrorCode ierr;
    UserCtx* user = (UserCtx*)ptr;
    ParaCtx* param = user->param;
    TstepCtx* tsctx = user->tsctx;
    SNES snes = user->snes;
    SNESConvergedReason reason;
    PetscInt max_steps, max_it, i;
    int its = 0, lits = 0.0, fits = 0;
    PetscScalar res = 0.0, res0 = 0.0, scaling = 0.0;
    double time0 = 0.0, time1 = 0.0, totaltime = 0.0;
    PetscScalar fnorm;
    FILE* fp;
    char history[36], filename[PETSC_MAX_PATH_LEN - 1];
    PetscScalar hj_fnorm = 0.0;
    PetscInt globalksp = 0, localksp = 0, globalsnes = 0, localsnes = 0;
    PetscFunctionBegin;

    if (param->PetscPreLoading) {
        max_steps = 1;
    } else {
        max_steps = tsctx->tsmax;
    }

    if (!param->PetscPreLoading) {
        sprintf(history, "history_c%d_n%dx%d.data", size, user->n1, user->n2);
        ierr = PetscFOpen(comm, history, "a", &fp);
        CHKERRQ(ierr);
        ierr = PetscFPrintf(comm, fp,
                            "%% example, step, t, reason, fnorm, its_snes, "
                            "its_ksp, its_fail, time\n");
        CHKERRQ(ierr);
    }

    for (tsctx->tscurr = tsctx->tsstart + 1;
         (tsctx->tscurr <= tsctx->tsstart + max_steps); tsctx->tscurr++) {
        ierr = SNESComputeFunction(snes, user->sol, user->myF);
        CHKERRQ(ierr);
        ierr = VecNorm(user->myF, NORM_2, &res);
        CHKERRQ(ierr);

        if (user->param->use_adaptive_dt) {
            if (tsctx->tscurr == tsctx->tsstart + 1) {
                res0 = res;
            } else {
                scaling = pow(res0 / res, tsctx->p);
                if (scaling > tsctx->smax) scaling = tsctx->smax;
                if (scaling < 1. / tsctx->smax) scaling = 1. / tsctx->smax;
                tsctx->tsize = tsctx->tsize * scaling;
                if ((tsctx->tcurr < tsctx->tfinal - EPS) &&
                    (tsctx->tcurr + tsctx->tsize >= tsctx->tfinal + EPS)) {
                    tsctx->tsize = tsctx->tfinal - tsctx->tcurr;
                }
            }
            ierr = PetscPrintf(comm,
                               " current initial residual = %g with adpt "
                               "method, current time size = %g\n",
                               res, tsctx->tsize);
        } else {
            ierr = PetscPrintf(
                comm,
                " current initial residual = %g, current time size = %g\n", res,
                tsctx->tsize);
        }

        tsctx->tcurr += tsctx->tsize;
        if (!param->PetscPreLoading) {
            ierr = PetscPrintf(comm,
                               "\n====================== Step: %d, time: %f "
                               "====================\n",
                               tsctx->tscurr, tsctx->tcurr);
            CHKERRQ(ierr);
        }
         ierr = PetscTime(&time0);
        CHKERRQ(ierr);
        ierr = SNESSolve(snes, 0, user->sol);
        ierr = PetscTime(&time1);
        CHKERRQ(ierr);
         ierr = Updata_Reaction(user);


        ierr = CopyOldVector(user->sol, user->xold, user);
        sprintf(filename, "example=%dpermeability_xxascii_%d.vts", EXAMPLE,
                tsctx->tscurr);
        ierr = DataSaveVTK(user->sol, filename);
        sprintf(filename, "example=%dpermeability_xxascii_%d.data", EXAMPLE,
                tsctx->tscurr);
        ierr = DataSaveASCII(user->sol, filename);
        PetscReal litspit = 0.0;
        PetscReal its1 = 0.0;
        PetscReal lits1 = 0.0;
        Vec F;
        ierr = SNESGetConvergedReason(snes, &reason);
        CHKERRQ(ierr);
        ierr = SNESGetFunction(snes, &F, 0, 0);
        CHKERRQ(ierr);
        ierr = VecNorm(F, NORM_2, &fnorm);
        CHKERRQ(ierr);
        ierr = SNESGetIterationNumber(snes, &its);
        CHKERRQ(ierr);
        its1 = its1 + its;
        ierr = SNESGetLinearSolveIterations(snes, &lits);
        CHKERRQ(ierr);
        lits1 = lits1 + lits;
        litspit = ((PetscReal)lits1) / ((PetscReal)its1);
        ierr = PetscPrintf(PETSC_COMM_WORLD,
                           "Average Linear its / Newton = %g\n", litspit);
        CHKERRQ(ierr);
        ierr = SNESGetNonlinearStepFailures(snes, &fits);
        CHKERRQ(ierr);
        ierr = PetscPrintf(comm, " Snes converged reason = %d\n", reason);
        CHKERRQ(ierr);
        ierr = PetscPrintf(comm, " 2-norm of F = %g\n", fnorm);
        CHKERRQ(ierr);
        ierr = PetscPrintf(comm, " number of Newton iterations = %f\n", its1);
        CHKERRQ(ierr);
        ierr = PetscPrintf(comm, " number of Linear iterations = %f\n", lits1);
        CHKERRQ(ierr);
        ierr = PetscPrintf(comm, " number of unsuccessful steps = %d\n", fits);
        CHKERRQ(ierr);

        if (!param->PetscPreLoading) {
            ierr = PetscFPrintf(
                comm, fp,
                "%d\t, %d\t, %g\t, %d\t, %g\t, %d\t, %d\t, %d\t, %g\n", EXAMPLE,
                tsctx->tscurr, tsctx->tsize, reason, fnorm, its, lits, fits,
                time1 - time0);
            CHKERRQ(ierr);
        }

        res0 = res;

        ierr = MPI_Barrier(comm);
        CHKERRQ(ierr);
        if (tsctx->tcurr > tsctx->tfinal - EPS) {
            tsctx->tscurr++;
            break;
        }
    }
    tsctx->tscurr--;

    ierr = PetscPrintf(
        comm, "\n+++++++++++++++++++++++ Summary +++++++++++++++++++++++++\n");
    CHKERRQ(ierr);
    ierr = PetscPrintf(comm, " Final time = %g, Cost time = %g\n", tsctx->tcurr,
                       totaltime);
    CHKERRQ(ierr);

    if (!param->PetscPreLoading) {
        ierr = PetscFClose(comm, fp);
        CHKERRQ(ierr);
    }

    PetscFunctionReturn(0);
}

PetscErrorCode FormInitialValue_Perm_local(void* ptr) {
    PetscErrorCode ierr;
    UserCtx* user = (UserCtx*)ptr;
    DM da = user->da;
    PetscInt i, j, xg, yg, zg, nxg, nyg, nzg;
    PetscFunctionBeginUser;
    ierr = DMDAGetGhostCorners(da, &xg, &yg, &zg, &nxg, &nyg, &nzg);
    CHKERRQ(ierr);
    for (j = yg; j < yg + nyg; j++) {
        for (i = xg; i < xg + nxg; i++) {
            for (int nc = 0; nc < DOF_perm; nc++) {
                user->perm_field[j][i].xx[nc] = 1e-10;
                user->phi_field[j][i].xx[nc] = 0.2;
                user->phi_old_field[j][i].xx[nc] = 0.2;
            }
        }
    }

    PetscFunctionReturn(0);
}

PetscErrorCode FormInitialValue_local(void* ptr) {
    PetscErrorCode ierr;
    UserCtx* user = (UserCtx*)ptr;
    DM da = user->da;
    PhysicalField** sol;
    PetscInt i, j, y_loc, xl, yl, zl, nxl, nyl, nzl;
    PetscFunctionBeginUser;
    ierr = DMDAGetCorners(da, &xl, &yl, &zl, &nxl, &nyl, &nzl);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, user->sol, &sol);
    CHKERRQ(ierr);
    for (j = yl; j < yl + nyl; j++) {
        y_loc = ((PetscScalar)j + 0.5) * user->dy;
        for (i = xl; i < xl + nxl; i++) {
            sol[j][i].pw = P_init;
            user->xold[j][i].pw = P_init;
            for (int nc = 0; nc < DOF_reaction - 1; ++nc) {
                if (y_loc <= 0.25 && (i == 0)) {
                    sol[j][i].cw[nc] = c_BC_L;
                    user->xold[j][i].cw[nc] = c_BC_L;
                } else {
                    sol[j][i].cw[nc] = c_BC_R;
                    user->xold[j][i].cw[nc] = c_BC_R;
                }
                double sum = 0.0, sum_1 = 0.0;
                for (int nc = 0; nc < DOF_reaction - 1; ++nc) {
                    sum += sol[j][i].cw[nc];
                    sum_1 += user->xold[j][i].cw[nc];
                }
                sol[j][i].cw[DOF_reaction - 1] = 1.0 - sum;
                user->xold[j][i].cw[DOF_reaction - 1] = 1.0 - sum_1;
            }
        }
    }
    ierr = DMDAVecRestoreArray(da, user->sol, &sol);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode FormInitialValue_Reaction_local(void* ptr) {
    PetscErrorCode ierr;
    UserCtx* user = (UserCtx*)ptr;
    DM da = user->da;
    double eqm_data[DOF_reaction] = {2.19e6, 4.73e-11, 0.222, 1e-2, 1e-3};
    PetscInt i, j, xg, yg, zg, nxg, nyg, nzg;
    PetscFunctionBeginUser;
    ierr = DMDAGetGhostCorners(da, &xg, &yg, &zg, &nxg, &nyg, &nzg);
    CHKERRQ(ierr);
    for (j = yg; j < yg + nyg; j++) {
        for (i = xg; i < xg + nxg; i++) {
            for (int nc = 0; nc < DOF_reaction; ++nc) {
                user->_mass_frac_old_field[j][i].reaction[nc] = 1.e-6;
                user->_sec_conc_old_field[j][i].reaction[nc] = 1.e-7;
                user->eqm_k_field[j][i].reaction[nc] = eqm_data[nc];
            }
        }
    }

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Updata_Reaction"
PetscErrorCode Updata_Reaction(void* ptr) {
    PetscErrorCode ierr;
    UserCtx* user = (UserCtx*)ptr;
    DM da = user->da, da_reaction = user->da_reaction, da_perm = user->da_perm;
    PhysicalField** sol;
    PetscInt i, j, nc, mx, my, xl, yl, zl, nxl, nyl, nzl, xg, yg, zg, nxg, nyg,
        nzg;
    Vec loc_X;
    PetscFunctionBeginUser;
    mx = user->n1;
    my = user->n2;
    ierr = DMGetLocalVector(da, &loc_X);
    CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, user->sol, INSERT_VALUES, loc_X);
    CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, user->sol, INSERT_VALUES, loc_X);
    CHKERRQ(ierr);
    ierr = DMDAGetGhostCorners(da, &xg, &yg, &zg, &nxg, &nyg, &nzg);
    CHKERRQ(ierr);
    ierr = DMDAGetCorners(da, &xl, &yl, &zl, &nxl, &nyl, &nzl);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, loc_X, &sol);
    CHKERRQ(ierr);
    if (xl == 0) {
        for (j = yg; j < yg + nyg; j++) {
            for (i = -WIDTH; i < 0; i++) {
                sol[j][i].pw = 2 * P_init - sol[j][-i - 1].pw;
                for (nc = 0; nc < DOF_reaction; ++nc) {
                    sol[j][i].cw[nc] = 2 * c_BC_L - sol[j][-i - 1].cw[nc];
                }
            }
        }
    }
    // right boundary (out flow): dirichlet boundary = P2
    if (xl + nxl == mx) {
        for (j = yg; j < yg + nyg; j++) {
            for (i = mx; i < mx + WIDTH; i++) {
                sol[j][i].pw = -sol[j][2 * mx - i - 1].pw;  //
                for (nc = 0; nc < DOF_reaction; ++nc) {
                    sol[j][mx].cw[nc] =
                        2 * c_BC_R - sol[j][2 * mx - i - 1].cw[nc];  //
                }
            }
        }
    }
    // bottom boundary (no flow): v_y=0 Neumann boundary
    if (yl == 0) {
        for (i = xg; i < xg + nxg; i++) {
            for (j = -WIDTH; j < 0; j++) {
                sol[j][i].pw = sol[-j - 1][i].pw;  //
                for (nc = 0; nc < DOF_reaction; ++nc) {
                    sol[j][i].cw[nc] = sol[-j - 1][i].cw[nc];
                }
            }
        }
    }
    // top boundary (no flow): v_y=0 Neumann boundary
    if (yl + nyl == my) {
        for (i = xg; i < xg + nxg; i++) {
            for (j = my; j < my + WIDTH; j++) {
                sol[j][i].pw = sol[2 * my - j - 1][i].pw;  //
                for (nc = 0; nc < DOF_reaction; ++nc) {
                    sol[j][i].cw[nc] = sol[2 * my - j - 1][i].cw[nc];  //
                }
            }
        }
    }

    for (j = yg; j < yg + nyg; j++) {
        for (i = xg; i < xg + nxg; i++) {
            ReactionField _sec_conc, _mass_frac, _reaction_rate,_mineral_sat;
            PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(
                &_sec_conc, &user->eqm_k_field[j][i],
                &_mass_frac, &sol[j][i], _equilibrium_constants_as_log10,
                user);
            PorousFlowAqueousPreDisChemistry_computeQpReactionRates(
                reference_temperature_pre, reference_saturation,
                user->phi_old_field[j][i].xx[0], &user->phi_field[j][i].xx[0], &_mineral_sat,
                &_reaction_rate, &user->_sec_conc_old_field[j][i], &_sec_conc,
                _equilibrium_constants_as_log10, user, &user->initial_ref_field[j][i]);
            PorousFlowAqueousPreDisMineral_computeQpProperties(
                reference_saturation, &user->_sec_conc_old_field[j][i], &_sec_conc,
                &_reaction_rate, user->phi_old_field[j][i].xx[0], user);
                user->_sec_conc_old_field[j][i]=_reaction_rate;
                user->_mass_frac_old_field[j][i]=_mass_frac;
        }
    }

    ierr = DMDAVecRestoreArray(da, loc_X, &sol);
    CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(da, &loc_X);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


PetscErrorCode  CopyOldVector(Vec sol, PhysicalField** xold, void *ptr){
    PetscErrorCode ierr;
    UserCtx* user = (UserCtx*)ptr;
    DM da = user->da;
    PhysicalField **sol_local;
    PetscInt i, j, nc, mx, my, xl, yl, zl, nxl, nyl, nzl;
        PetscFunctionBeginUser;
    ierr = DMDAGetCorners(da, &xl, &yl, &zl, &nxl, &nyl, &nzl);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, sol, &sol_local);
    CHKERRQ(ierr);
        for (j = yl; j < yl + nyl; j++) {
        for (i = xl; i < xl + nxl; i++) {
        user->xold[j][i].pw=sol_local[j][i].pw;
           for (int nc = 0; nc < DOF_reaction; ++nc) {
                    user->xold[j][i].cw[nc]=sol_local[j][i].cw[nc];
           }
        }
    }
    ierr = DMDAVecRestoreArray(da, sol, &sol_local);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);

}