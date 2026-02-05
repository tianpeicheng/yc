static char help[] = "3D single-phase flow, by TianpeiCheng .\n\\n";
#include <petscdm.h>
#include <petscdmda.h>
#include <petscsnes.h>
#include <iostream>
#include "def.h"
#include "petscsys.h"
#include "petsctime.h"
#include "reaction.h"
#include "stdlib.h"
MPI_Comm comm;
PetscMPIInt rank, size;

static PetscErrorCode ScatterNamedToSubDMLocal(DM dm, DM subdm,
                                               const char* name) {
    Vec g, l;
    VecScatter *iscat = NULL, *oscat = NULL, *gscat = NULL;
    PetscFunctionBeginUser;
    PetscCall(DMGetNamedGlobalVector(dm, name, &g));
    PetscCall(DMGetNamedLocalVector(subdm, name, &l));
    PetscCall(DMCreateDomainDecompositionScatters(dm, 1, &subdm, &iscat, &oscat,
                                                  &gscat));
    PetscCall(VecScatterBegin(*gscat, g, l, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(*gscat, g, l, INSERT_VALUES, SCATTER_FORWARD));
    // PetscCall(VecScatterDestroy(iscat));
    // PetscCall(VecScatterDestroy(oscat));
    // PetscCall(VecScatterDestroy(gscat));
    PetscCall(DMRestoreNamedGlobalVector(dm, name, &g));
    PetscCall(DMRestoreNamedLocalVector(subdm, name, &l));
    PetscFunctionReturn(0);
}

static PetscErrorCode ScatterNamedToSubDMGlobal(DM dm, DM subdm,
                                                const char* name) {
    Vec g, l;
    VecScatter *iscat = NULL, *oscat = NULL, *gscat = NULL;
    PetscFunctionBeginUser;
    PetscCall(DMGetNamedGlobalVector(dm, name, &g));
    PetscCall(DMGetNamedGlobalVector(subdm, name, &l));
    PetscCall(DMCreateDomainDecompositionScatters(dm, 1, &subdm, &iscat, &oscat,
                                                  &gscat));
    PetscCall(VecScatterBegin(*oscat, g, l, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(*oscat, g, l, INSERT_VALUES, SCATTER_FORWARD));
    // PetscCall(VecScatterDestroy(iscat));
    // PetscCall(VecScatterDestroy(oscat));
    // PetscCall(VecScatterDestroy(gscat));
    PetscCall(DMRestoreNamedGlobalVector(dm, name, &g));
    PetscCall(DMRestoreNamedGlobalVector(subdm, name, &l));
    PetscFunctionReturn(0);
}

static PetscErrorCode CoefficientSubDomainHook(DM dm, DM subdm, void* ctx) {
    DM perm_dm = NULL, secondary_dm = NULL, reaction_dm = NULL;
#if EXAMPLE == 2
    DM mineral_dm = NULL;
#endif
    PetscFunctionBeginUser;
    PetscCall(
        PetscObjectQuery((PetscObject)dm, "perm_dm", (PetscObject*)&perm_dm));
    if (perm_dm) {
        PetscInt dof = 0;
        DM perm_subdm = NULL;
        PetscCall(PetscObjectQuery((PetscObject)subdm, "perm_dm",
                                   (PetscObject*)&perm_subdm));
        if (!perm_subdm) {
            PetscCall(DMDAGetInfo(perm_dm, PETSC_NULLPTR, PETSC_NULLPTR,
                                  PETSC_NULLPTR, PETSC_NULLPTR, PETSC_NULLPTR,
                                  PETSC_NULLPTR, PETSC_NULLPTR, &dof,
                                  PETSC_NULLPTR, PETSC_NULLPTR, PETSC_NULLPTR,
                                  PETSC_NULLPTR, PETSC_NULLPTR));
            PetscCall(DMDACreateCompatibleDMDA(subdm, dof, &perm_subdm));
            PetscCall(PetscObjectCompose((PetscObject)subdm, "perm_dm",
                                         (PetscObject)perm_subdm));
        }
        PetscCall(ScatterNamedToSubDMGlobal(perm_dm, perm_subdm, "perm"));
        PetscCall(ScatterNamedToSubDMGlobal(perm_dm, perm_subdm, "phi"));
        PetscCall(ScatterNamedToSubDMGlobal(perm_dm, perm_subdm, "phi_old"));
        PetscCall(ScatterNamedToSubDMLocal(perm_dm, perm_subdm, "perm"));
        PetscCall(ScatterNamedToSubDMLocal(perm_dm, perm_subdm, "phi"));
        PetscCall(ScatterNamedToSubDMLocal(perm_dm, perm_subdm, "phi_old"));
        // PetscCall(DMDestroy(&perm_subdm));
    }
    PetscCall(PetscObjectQuery((PetscObject)dm, "secondary_dm",
                               (PetscObject*)&secondary_dm));
    if (secondary_dm) {
        PetscInt dof = 0;
        DM secondary_subdm = NULL;
        PetscCall(PetscObjectQuery((PetscObject)subdm, "secondary_dm",
                                   (PetscObject*)&secondary_subdm));
        if (!secondary_subdm) {
            PetscCall(DMDAGetInfo(secondary_dm, PETSC_NULLPTR, PETSC_NULLPTR,
                                  PETSC_NULLPTR, PETSC_NULLPTR, PETSC_NULLPTR,
                                  PETSC_NULLPTR, PETSC_NULLPTR, &dof,
                                  PETSC_NULLPTR, PETSC_NULLPTR, PETSC_NULLPTR,
                                  PETSC_NULLPTR, PETSC_NULLPTR));
            PetscCall(DMDACreateCompatibleDMDA(subdm, dof, &secondary_subdm));
            PetscCall(PetscObjectCompose((PetscObject)subdm, "secondary_dm",
                                         (PetscObject)secondary_subdm));
        }
        PetscCall(
            ScatterNamedToSubDMGlobal(secondary_dm, secondary_subdm, "eqm_k"));
        PetscCall(ScatterNamedToSubDMGlobal(secondary_dm, secondary_subdm,
                                            "sec_conc_old"));
        // PetscCall(DMDestroy(&secondary_subdm));
    }
    PetscCall(PetscObjectQuery((PetscObject)dm, "reaction_dm",
                               (PetscObject*)&reaction_dm));
    if (reaction_dm) {
        PetscInt dof = 0;
        DM reaction_subdm = NULL;
        PetscCall(PetscObjectQuery((PetscObject)subdm, "reaction_dm",
                                   (PetscObject*)&reaction_subdm));
        if (!reaction_subdm) {
            PetscCall(DMDAGetInfo(reaction_dm, PETSC_NULLPTR, PETSC_NULLPTR,
                                  PETSC_NULLPTR, PETSC_NULLPTR, PETSC_NULLPTR,
                                  PETSC_NULLPTR, PETSC_NULLPTR, &dof,
                                  PETSC_NULLPTR, PETSC_NULLPTR, PETSC_NULLPTR,
                                  PETSC_NULLPTR, PETSC_NULLPTR));
            PetscCall(DMDACreateCompatibleDMDA(subdm, dof, &reaction_subdm));
            PetscCall(PetscObjectCompose((PetscObject)subdm, "reaction_dm",
                                         (PetscObject)reaction_subdm));
        }
        PetscCall(ScatterNamedToSubDMGlobal(reaction_dm, reaction_subdm,
                                            "mass_frac_old"));
        PetscCall(ScatterNamedToSubDMGlobal(reaction_dm, reaction_subdm,
                                            "initial_ref"));
        // PetscCall(DMDestroy(&reaction_subdm));
    }
#if EXAMPLE == 2
    ierr = PetscObjectQuery((PetscObject)dm, "mineral_dm",
                            (PetscObject*)&mineral_dm);
    CHKERRQ(ierr);
    if (mineral_dm) {
        PetscInt dof = 0;
        DM mineral_subdm = NULL;
        ierr =
            DMDAGetInfo(mineral_dm, PETSC_NULLPTR, PETSC_NULLPTR, PETSC_NULLPTR,
                        PETSC_NULLPTR, PETSC_NULLPTR, PETSC_NULLPTR,
                        PETSC_NULLPTR, &dof, PETSC_NULLPTR, PETSC_NULLPTR,
                        PETSC_NULLPTR, PETSC_NULLPTR, PETSC_NULLPTR);
        CHKERRQ(ierr);
        ierr = DMDACreateCompatibleDMDA(subdm, dof, &mineral_subdm);
        CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject)subdm, "mineral_dm",
                                  (PetscObject)mineral_subdm);
        CHKERRQ(ierr);
        ierr = ScatterNamedToSubDMGlobal(mineral_dm, mineral_subdm,
                                         "eqm_k_mineral");
        CHKERRQ(ierr);
        ierr = DMDestroy(&mineral_subdm);
        CHKERRQ(ierr);
    }
#endif
    PetscCall(ScatterNamedToSubDMGlobal(dm, subdm, "xold"));
    PetscFunctionReturn(0);
}
PetscErrorCode CoefficientRestrictHook(DM global, VecScatter out, VecScatter in,
                                       DM block, void* ctx) {
    PetscFunctionBeginUser;
    PetscCall(CoefficientSubDomainHook(global, block, ctx));
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char** argv) {
    PetscErrorCode ierr;
    SNES snes;
    UserCtx* user;
    PetscInt n1, n2, n3;
    ParaCtx param;
    TstepCtx tsctx;

    PetscInitialize(&argc, &argv, (char*)0, help);

    PetscPreLoadBegin(PETSC_TRUE, "SetUp");

    param.PetscPreLoading = PetscPreLoading;

    comm = PETSC_COMM_WORLD;
    ierr = MPI_Comm_rank(comm, &rank);
    CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm, &size);
    CHKERRQ(ierr);

    param.use_adaptive_dt = PETSC_FALSE;
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
    PetscCall(PetscMalloc(sizeof(UserCtx), &user));

    PetscCall(SNESCreate(comm, &snes));

    PetscCall(DMDACreate2d(comm, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
                           DMDA_STENCIL_STAR, n1, n2, PETSC_DECIDE,
                           PETSC_DECIDE, DOF, WIDTH, 0, 0, &(user->da)));
    PetscCall(DMDACreate2d(comm, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
                           DMDA_STENCIL_STAR, n1, n2, PETSC_DECIDE,
                           PETSC_DECIDE, DOF_reaction, WIDTH, 0, 0,
                           &(user->da_reaction)));
    PetscCall(DMDACreate2d(comm, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
                           DMDA_STENCIL_STAR, n1, n2, PETSC_DECIDE,
                           PETSC_DECIDE, DOF_perm, WIDTH, 0, 0,
                           &(user->da_perm)));
    PetscCall(DMDACreate2d(comm, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
                           DMDA_STENCIL_STAR, n1, n2, PETSC_DECIDE,
                           PETSC_DECIDE, DOF_Secondary, WIDTH, 0, 0,
                           &(user->da_secondary)));
    if (DOF_mineral != 0) {
        PetscCall(DMDACreate2d(comm, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
                               DMDA_STENCIL_STAR, n1, n2, PETSC_DECIDE,
                               PETSC_DECIDE, DOF_mineral, WIDTH, 0, 0,
                               &(user->da_mineral)));
    }
    PetscCall(DMSetFromOptions(user->da));
    PetscCall(DMSetFromOptions(user->da_reaction));
    PetscCall(DMSetFromOptions(user->da_perm));
    PetscCall(DMSetFromOptions(user->da_secondary));
    PetscCall(DMSetUp(user->da));
    PetscCall(DMSetUp(user->da_reaction));
    PetscCall(DMSetUp(user->da_perm));
    PetscCall(DMSetUp(user->da_secondary));
#if EXAMPLE == 2
    PetscCall(DMSetUp(user->da_mineral));
#endif
    PetscCall(PetscObjectCompose((PetscObject)user->da, "perm_dm",
                                 (PetscObject)user->da_perm));
    PetscCall(PetscObjectCompose((PetscObject)user->da, "secondary_dm",
                                 (PetscObject)user->da_secondary));
    PetscCall(PetscObjectCompose((PetscObject)user->da, "reaction_dm",
                                 (PetscObject)user->da_reaction));
#if EXAMPLE == 2
    PetscCall(PetscObjectCompose((PetscObject)user->da, "mineral_dm",
                                 (PetscObject)user->da_mineral));
#endif
#if EXAMPLE == 1 || EXAMPLE == 3
    PetscCall(DMDASetFieldName(user->da, 0, "pressure"));
    PetscCall(DMDASetFieldName(user->da, 1, "h+"));
    PetscCall(DMDASetFieldName(user->da, 2, "hco3-"));
    PetscCall(DMDASetFieldName(user->da, 3, "ca2+"));
    PetscCall(DMDASetFieldName(user->da, 4, "mg2+"));
    PetscCall(DMDASetFieldName(user->da, 5, "fe2+"));
#elif EXAMPLE == 2
    PetscCall(DMDASetFieldName(user->da, 0, "pressure"));
    PetscCall(DMDASetFieldName(user->da, 1, "tracer"));
    PetscCall(DMDASetFieldName(user->da, 2, "hco3-"));
    PetscCall(DMDASetFieldName(user->da, 3, "h+"));
    PetscCall(DMDASetFieldName(user->da, 4, "ca2+"));
#endif
    PetscCall(SNESSetDM(snes, user->da));
    PetscCall(SNESSetFromOptions(snes));
    PetscCall(DMSubDomainHookAdd(user->da, CoefficientSubDomainHook,
                                 CoefficientRestrictHook, NULL));
    user->tsctx = &tsctx;
    user->param = &param;
    PetscCall(DMDAGetInfo(user->da, 0, &(user->n1), &(user->n2), 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0));
    user->dx = L1 / (PetscScalar)(user->n1);
    user->dy = L2 / (PetscScalar)(user->n2);

    PetscCall(DMCreateGlobalVector(user->da, &user->sol));
    PetscCall(DMGetNamedGlobalVector(user->da, "solution", &user->sol));
    PetscCall(DMGetNamedGlobalVector(user->da, "residual", &user->myF));
#if EXAMPLE == 2
    PetscCall(DMGetNamedGlobalVector(user->da_mineral, "sol_mineral",
                                     &user->sol_mineral));
    PetscCall(DMGetNamedGlobalVector(user->da_mineral, "sol_mineral_old",
                                     &user->sol_mineral_old));
#endif
#if EXAMPLE == 2
    ierr = DMDAGetArray(user->da_mineral, PETSC_TRUE,
                        (void**)&(user->eqm_k_mineral_field));
    CHKERRQ(ierr);
#endif
    PetscCall(DMDASNESSetFunctionLocal(
        user->da, INSERT_VALUES,
        (PetscErrorCode (*)(DMDALocalInfo*, void*, void*,
                            void*))FormFunctionLocal,
        user));
    PetscCall(FormInitialValue_local(user));
    PetscCall(FormInitialValue_Perm_local(user));
    PetscCall(FormInitialValue_Reaction_local(user));
    user->snes = snes;
    if (!param.PetscPreLoading) {
        PetscCall(PetscPrintf(comm,
                              "\n+++++++++++++++++++++++ Problem parameters "
                              "+++++++++++++++++++++\n"));
        PetscCall(
            PetscPrintf(comm, " Single-phase flow, example: %d\n", EXAMPLE));
        PetscCall(PetscPrintf(comm, " Problem size %d, %d, Ncpu = %d \n", n1,
                              n2, size));
        if (param.use_adaptive_dt) {
            PetscCall(PetscPrintf(
                comm,
                " Torder = %d, TimeSteps = %g (adpt with %g) x "
                "%d, final time %f \n",
                tsctx.torder, tsctx.tsize, tsctx.p, tsctx.tsmax, tsctx.tfinal));
        } else {
            PetscCall(PetscPrintf(
                comm, " Torder = %d, TimeSteps = %g x %d, final time %g \n",
                tsctx.torder, tsctx.tsize, tsctx.tsmax, tsctx.tfinal));
        }
        PetscCall(
            PetscPrintf(comm,
                        "+++++++++++++++++++++++++++++++++++++++++++++++++++"
                        "+++++++++++++++\n"));
    }

    PetscPreLoadStage("Solve");
    PetscCall(SNESSetFromOptions(snes));
    PetscCall(Update(user));

#if EXAMPLE == 2
    PetscCall(DMRestoreNamedGlobalVector(user->da_mineral, "sol_mineral",
                                         &user->sol_mineral));
    PetscCall(DMRestoreNamedGlobalVector(user->da_mineral, "sol_mineral_old",
                                         &user->sol_mineral_old));
#endif
    PetscCall(DMRestoreNamedGlobalVector(user->da, "solution", &user->sol));
    PetscCall(DMRestoreNamedGlobalVector(user->da, "residual", &user->myF));
    PetscCall(DMDestroy(&user->da));
    PetscCall(DMDestroy(&user->da_reaction));
    PetscCall(DMDestroy(&user->da_perm));
    PetscCall(DMDestroy(&user->da_secondary));
#if EXAMPLE == 2
    PetscCall(DMDestroy(&user->da_mineral));
#endif
    PetscCall(SNESDestroy(&snes));
    PetscCall(PetscFree(user));

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
    PetscInt max_steps;
    SNESConvergedReason reason;
    int its = 0, lits = 0.0, fits = 0;
    PetscScalar res = 0.0, res0 = 0.0, scaling = 0.0;
    double time0 = 0.0, time1 = 0.0, totaltime = 0.0;
    PetscScalar fnorm;
    FILE* fp;
    char history[36], filename[PETSC_MAX_PATH_LEN - 1];
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

        ierr = Updata_Reaction(user);

        ierr = CopyOldVector(user->sol, user->xold, user);
        if (tsctx->tscurr % 10 == 0) {
            sprintf(filename, "example=%dpermeability_xxascii_%d.vts", EXAMPLE,
                    tsctx->tscurr);
            ierr = DataSaveVTK(user->sol, filename);
            sprintf(filename, "example=%dpermeability_xxascii_%d.data", EXAMPLE,
                    tsctx->tscurr);
            ierr = DataSaveASCII(user->sol, filename);
            CHKERRQ(ierr);
        }
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
    DM da = user->da, da_perm = user->da_perm;
    PetscInt i, j, xg, yg, zg, nxg, nyg, nzg;
    PetscInt xl, yl, zl, nxl, nyl, nzl;
    PetscFunctionBeginUser;
    PetscCall(DMDAGetGhostCorners(da, &xg, &yg, &zg, &nxg, &nyg, &nzg));
    PetscCall(DMDAGetCorners(da, &xl, &yl, &zl, &nxl, &nyl, &nzl));
#if EXAMPLE == 3
    PermField** perm_field_local;
    PetscInt mx, my;
    mx = user->n1;
    my = user->n2;
    Vec perm_local, u_per;
    PetscViewer dataviewer;
    ierr = DMCreateGlobalVector(da_perm, &u_per);
    CHKERRQ(ierr);
    ierr = DMCreateLocalVector(da_perm, &perm_local);
    CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "SPE85.bin", FILE_MODE_READ,
                                 &dataviewer);
    CHKERRQ(ierr);
    ierr = VecLoad(u_per, dataviewer);
    CHKERRQ(ierr);
    ierr = VecScale(u_per, UNIT_MD);
    CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&dataviewer);
    CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da_perm, u_per, INSERT_VALUES, perm_local);
    CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da_perm, u_per, INSERT_VALUES, perm_local);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da_perm, perm_local, &perm_field_local);
    CHKERRQ(ierr);
#endif
    PermField **perm_field = NULL, **phi_field = NULL, **phi_old_field = NULL;
    Vec perm_vec = NULL, phi_vec = NULL, phi_old_vec = NULL;
    PetscCall(DMGetNamedGlobalVector(da_perm, "perm", &perm_vec));
    PetscCall(DMGetNamedGlobalVector(da_perm, "phi", &phi_vec));
    PetscCall(DMGetNamedGlobalVector(da_perm, "phi_old", &phi_old_vec));
    PetscCall(DMDAVecGetArray(da_perm, perm_vec, &perm_field));
    PetscCall(DMDAVecGetArray(da_perm, phi_vec, &phi_field));
    PetscCall(DMDAVecGetArray(da_perm, phi_old_vec, &phi_old_field));
    for (j = yl; j < yl + nyl; j++) {
        for (i = xl; i < xl + nxl; i++) {
#if EXAMPLE == 1
            for (int nc = 0; nc < DOF_perm; nc++) {
                perm_field[j][i].xx[nc] = 1e-10;
                phi_field[j][i].xx[nc] = 0.2;
                phi_old_field[j][i].xx[nc] = 0.2;
            }
#elif EXAMPLE == 2
            for (int nc = 0; nc < DOF_perm; nc++) {
                perm_field[j][i].xx[nc] = 1;
                phi_field[j][i].xx[nc] = 0.2;
                phi_old_field[j][i].xx[nc] = 0.2;
            }
#elif EXAMPLE == 3

            if (i < 0) {
                perm_field_local[j][i].xx[0] = perm_field_local[j][0].xx[0];
            }
            if (i > (mx - 1)) {
                perm_field_local[j][i].xx[0] =
                    perm_field_local[j][mx - 1].xx[0];
            }
            if (j < 0) {
                perm_field_local[j][i].xx[0] = perm_field_local[0][i].xx[0];
            }
            if (j > (my - 1)) {
                perm_field_local[j][i].xx[0] =
                    perm_field_local[my - 1][i].xx[0];
            }
            perm_field_local[j][i].xx[1] = perm_field_local[j][i].xx[0];
            for (int nc = 0; nc < DOF_perm; nc++) {
                perm_field[j][i].xx[nc] =
                    1.e-10;  // perm_field_local[j][i].xx[nc];
                phi_field[j][i].xx[nc] = 0.2;
                phi_old_field[j][i].xx[nc] = 0.2;
            }

#endif
        }
    }
#if EXAMPLE == 3
    PetscCall(DMDAVecRestoreArray(da_perm, perm_local, &perm_field_local));
    PetscCall(DMRestoreGlobalVector(da_perm, &u_per));
    PetscCall(DMRestoreLocalVector(da_perm, &perm_local));
#endif

    PetscCall(DMDAVecRestoreArray(da_perm, perm_vec, &perm_field));
    PetscCall(DMDAVecRestoreArray(da_perm, phi_vec, &phi_field));
    PetscCall(DMDAVecRestoreArray(da_perm, phi_old_vec, &phi_old_field));

    Vec perm_vec_local = NULL, phi_vec_local = NULL, phi_old_vec_local = NULL;
    PetscCall(DMGetNamedLocalVector(da_perm, "perm", &perm_vec_local));
    PetscCall(DMGetNamedLocalVector(da_perm, "phi", &phi_vec_local));
    PetscCall(DMGetNamedLocalVector(da_perm, "phi_old", &phi_old_vec_local));

    PetscCall(
        DMGlobalToLocalBegin(da_perm, perm_vec, INSERT_VALUES, perm_vec_local));
    PetscCall(
        DMGlobalToLocalEnd(da_perm, perm_vec, INSERT_VALUES, perm_vec_local));
    PetscCall(
        DMGlobalToLocalBegin(da_perm, phi_vec, INSERT_VALUES, phi_vec_local));
    PetscCall(
        DMGlobalToLocalEnd(da_perm, phi_vec, INSERT_VALUES, phi_vec_local));
    PetscCall(DMGlobalToLocalBegin(da_perm, phi_old_vec, INSERT_VALUES,
                                   phi_old_vec_local));
    PetscCall(DMGlobalToLocalEnd(da_perm, phi_old_vec, INSERT_VALUES,
                                 phi_old_vec_local));

    PetscCall(DMRestoreNamedLocalVector(da_perm, "perm", &perm_vec_local));
    PetscCall(DMRestoreNamedLocalVector(da_perm, "phi", &phi_vec_local));
    PetscCall(
        DMRestoreNamedLocalVector(da_perm, "phi_old", &phi_old_vec_local));

    PetscCall(DMRestoreNamedGlobalVector(da_perm, "perm", &perm_vec));
    PetscCall(DMRestoreNamedGlobalVector(da_perm, "phi", &phi_vec));
    PetscCall(DMRestoreNamedGlobalVector(da_perm, "phi_old", &phi_old_vec));
    PetscFunctionReturn(0);
}

PetscErrorCode FormInitialValue_local(void* ptr) {
    PetscErrorCode ierr;
    UserCtx* user = (UserCtx*)ptr;
    DM da = user->da;
    PhysicalField** sol;
    PhysicalField** xold_field;
    Vec xold = NULL;
    PetscInt i, j, y_loc, x_loc, xl, yl, zl, nxl, nyl, nzl;
    PetscFunctionBeginUser;
    ierr = DMDAGetCorners(da, &xl, &yl, &zl, &nxl, &nyl, &nzl);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, user->sol, &sol);
    CHKERRQ(ierr);
    ierr = DMGetNamedGlobalVector(da, "xold", &xold);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, xold, &xold_field);
    CHKERRQ(ierr);
    for (j = yl; j < yl + nyl; j++) {
        y_loc = ((PetscScalar)j + 0.5) * user->dy;
        for (i = xl; i < xl + nxl; i++) {
            x_loc = ((PetscScalar)i + 0.5) * user->dx;
#if EXAMPLE == 1 || EXAMPLE == 3
            sol[j][i].pw = P_init;
            xold_field[j][i].pw = P_init;
            for (int nc = 0; nc < DOF_reaction; ++nc) {
                if (y_loc <= L2 && (i == 0)) {
                    sol[j][i].cw[nc] = c_BC_L;
                    xold_field[j][i].cw[nc] = c_BC_L;
                } else {
                    sol[j][i].cw[nc] = c_BC_R;
                    xold_field[j][i].cw[nc] = c_BC_R;
                }
            }

#elif EXAMPLE == 2
            double c_init[DOF_reaction] = {0.0, 5.0e-2, 1.e-7, 1.0e-6};
            double c_init_1[DOF_reaction] = {1.0, 1.0e-6, 1.e-7, 1.0e-6};
            sol[j][i].pw = 60 - 50 * x_loc;
            xold_field[j][i].pw = 60 - 50 * x_loc;
            for (int nc = 0; nc < DOF_reaction; ++nc) {
                sol[j][i].cw[nc] = c_init[nc];
                xold_field[j][i].cw[nc] = c_init[nc];
            }

#endif
        }
    }
    ierr = DMDAVecRestoreArray(da, user->sol, &sol);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, xold, &xold_field);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedGlobalVector(da, "xold", &xold);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode FormInitialValue_Reaction_local(void* ptr) {
    PetscErrorCode ierr;
    UserCtx* user = (UserCtx*)ptr;
    DM da = user->da;

#if EXAMPLE == 1 || EXAMPLE == 3
    double eqm_data[DOF_Secondary] = {2.19e6, 4.73e-11, 0.222, 1e-2, 1e-3};
#elif EXAMPLE == 2
    double eqm_data[DOF_Secondary] = {6.341,  -10.325, -7.009,
                                      -0.653, -12.85,  -13.991};
#endif
    PetscInt i, j, xl, yl, zl, nxl, nyl, nzl;
    PetscFunctionBeginUser;
    Vec eqm_k = NULL, sec_conc_old = NULL;
    Vec mass_frac_old = NULL, initial_ref = NULL;
    Vec xold = NULL;
#if EXAMPLE == 2
    Vec eqm_k_mineral_local = NULL;
#endif
    ierr = DMDAGetCorners(da, &xl, &yl, &zl, &nxl, &nyl, &nzl);
    CHKERRQ(ierr);
    SecondaryReactionField **eqm_k_field = NULL, **sec_conc_old_field = NULL;
    ReactionField **mass_frac_old_field = NULL, **initial_ref_field = NULL;

    ierr = DMGetNamedGlobalVector(user->da_secondary, "eqm_k", &eqm_k);
    CHKERRQ(ierr);
    ierr = DMGetNamedGlobalVector(user->da_secondary, "sec_conc_old",
                                  &sec_conc_old);
    CHKERRQ(ierr);
    ierr = DMGetNamedGlobalVector(user->da_reaction, "mass_frac_old",
                                  &mass_frac_old);
    CHKERRQ(ierr);
    ierr =
        DMGetNamedGlobalVector(user->da_reaction, "initial_ref", &initial_ref);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da_secondary, eqm_k, &eqm_k_field);
    CHKERRQ(ierr);
    ierr =
        DMDAVecGetArray(user->da_secondary, sec_conc_old, &sec_conc_old_field);
    CHKERRQ(ierr);
    ierr =
        DMDAVecGetArray(user->da_reaction, mass_frac_old, &mass_frac_old_field);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da_reaction, initial_ref, &initial_ref_field);
    CHKERRQ(ierr);
#if EXAMPLE == 2
    ierr = DMGetNamedLocalVector(user->da_mineral, "eqm_k_mineral",
                                 &eqm_k_mineral_local);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da_mineral, eqm_k_mineral_local,
                           &eqm_k_mineral_field);
    CHKERRQ(ierr);
#endif

    for (j = yl; j < yl + nyl; j++) {
        for (i = xl; i < xl + nxl; i++) {
#if EXAMPLE == 1 || EXAMPLE == 3
            for (int nc = 0; nc < DOF_reaction; ++nc) {
                mass_frac_old_field[j][i].reaction[nc] = 1.e-6;
            }
            for (int nc = 0; nc < DOF_Secondary; ++nc) {
                sec_conc_old_field[j][i].reaction_secondary[nc] = 1.e-7;
                eqm_k_field[j][i].reaction_secondary[nc] = eqm_data[nc];
            }

#elif EXAMPLE == 2

            double eqm_mineral_data[DOF_mineral] = {1.8487};
            for (int nc = 0; nc < DOF_Secondary; ++nc)

            {
                eqm_k_field[j][i].reaction_secondary[nc] = eqm_data[nc];
                sec_conc_old_field[j][i].reaction_secondary[nc] = 0.0;
            }

            for (int nc = 0; nc < DOF_mineral; ++nc)

            {
                eqm_k_mineral_field[j][i].reaction_mineral[nc] =
                    eqm_mineral_data[nc];
            }

#endif
        }
    }
#if EXAMPLE == 2
    for (j = yl; j < yl + nyl; j++) {
        for (i = xl; i < xl + nxl; i++) {
            for (int nc = 0; nc < DOF_reaction; ++nc) {
                SecondaryReactionField _sec_conc;
                PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(
                    &_sec_conc, &eqm_k_field[j][i], &_mass_frac_old_field[j][i],
                    &xold_field[j][i], _equilibrium_constants_as_log10, user,
                    tsctx->tcurr);
                //  user->_mass_frac_old_field[j][i].reaction[nc] = 0.0;
            }
        }
    }
#endif
#if EXAMPLE == 2
    ierr = DMDAVecRestoreArray(da, loc_X, &sol);
    CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(da, &loc_X);
    CHKERRQ(ierr);
#endif
    ierr = DMDAVecRestoreArray(user->da_secondary, eqm_k, &eqm_k_field);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(user->da_secondary, sec_conc_old,
                               &sec_conc_old_field);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(user->da_reaction, mass_frac_old,
                               &mass_frac_old_field);
    CHKERRQ(ierr);
    ierr =
        DMDAVecRestoreArray(user->da_reaction, initial_ref, &initial_ref_field);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedGlobalVector(user->da_secondary, "eqm_k", &eqm_k);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedGlobalVector(user->da_secondary, "sec_conc_old",
                                      &sec_conc_old);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedGlobalVector(user->da_reaction, "mass_frac_old",
                                      &mass_frac_old);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedGlobalVector(user->da_reaction, "initial_ref",
                                      &initial_ref);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Updata_Reaction"
PetscErrorCode Updata_Reaction(void* ptr) {
    PetscErrorCode ierr;
    UserCtx* user = (UserCtx*)ptr;
    TstepCtx* tsctx = user->tsctx;
    DM da = user->da;
    PhysicalField** sol;
    PetscInt i, j, nc, xl, yl, zl, nxl, nyl, nzl;
    PetscFunctionBeginUser;
    ierr = DMDAGetCorners(da, &xl, &yl, &zl, &nxl, &nyl, &nzl);
    CHKERRQ(ierr);
    Vec phi = NULL, phi_old = NULL;
    Vec eqm_k = NULL, sec_conc_old = NULL;
    Vec mass_frac_old = NULL, initial_ref = NULL;
    PermField **phi_field = NULL, **phi_old_field = NULL;
    SecondaryReactionField **eqm_k_field = NULL, **sec_conc_old_field = NULL;
    ReactionField **mass_frac_old_field = NULL, **initial_ref_field = NULL;

    ierr = DMGetNamedGlobalVector(user->da_perm, "phi", &phi);
    CHKERRQ(ierr);
    ierr = DMGetNamedGlobalVector(user->da_perm, "phi_old", &phi_old);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da_perm, phi, &phi_field);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da_perm, phi_old, &phi_old_field);
    CHKERRQ(ierr);
    ierr = DMGetNamedGlobalVector(user->da_secondary, "eqm_k", &eqm_k);
    CHKERRQ(ierr);
    ierr = DMGetNamedGlobalVector(user->da_secondary, "sec_conc_old",
                                  &sec_conc_old);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da_secondary, eqm_k, &eqm_k_field);
    CHKERRQ(ierr);
    ierr =
        DMDAVecGetArray(user->da_secondary, sec_conc_old, &sec_conc_old_field);
    CHKERRQ(ierr);
    ierr = DMGetNamedGlobalVector(user->da_reaction, "mass_frac_old",
                                  &mass_frac_old);
    CHKERRQ(ierr);
    ierr =
        DMGetNamedGlobalVector(user->da_reaction, "initial_ref", &initial_ref);
    CHKERRQ(ierr);
    ierr =
        DMDAVecGetArray(user->da_reaction, mass_frac_old, &mass_frac_old_field);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da_reaction, initial_ref, &initial_ref_field);
    ierr = DMDAVecGetArray(da, user->sol, &sol);
    CHKERRQ(ierr);

    for (j = yl; j < yl + nyl; j++) {
        for (i = xl; i < xl + nxl; i++) {
#if EXAMPLE == 1 || EXAMPLE == 3
            SecondaryReactionField _sec_conc;
            ReactionField _mass_frac, _reaction_rate, _mineral_sat;
            PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(
                &_sec_conc, &eqm_k_field[j][i], &_mass_frac, &sol[j][i],
                _equilibrium_constants_as_log10, user, tsctx->tcurr);
            PorousFlowAqueousPreDisChemistry_computeQpReactionRates(
                reference_temperature_pre, reference_saturation,
                phi_old_field[j][i].xx[0], &phi_field[j][i].xx[0],
                &_mineral_sat, &_reaction_rate, &sec_conc_old_field[j][i],
                &_sec_conc, _equilibrium_constants_as_log10, user,
                &initial_ref_field[j][i]);
            PorousFlowAqueousPreDisMineral_computeQpProperties(
                reference_saturation, &sec_conc_old_field[j][i], &_sec_conc,
                &_reaction_rate, phi_old_field[j][i].xx[0], user);
            sec_conc_old_field[j][i] = _sec_conc;
            mass_frac_old_field[j][i] = _mass_frac;
#elif EXAMPLE == 2
            SecondaryReactionField _sec_conc;
            ReactionField _mass_frac;
            PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(
                &_sec_conc, &user->eqm_k_field[j][i], &_mass_frac, &sol[j][i],
                _equilibrium_constants_as_log10, user, tsctx->tcurr);
            user->_sec_conc_old_field[j][i] = _sec_conc;
            user->_mass_frac_old_field[j][i] = _mass_frac;
#endif
        }
    }

    ierr = DMDAVecRestoreArray(user->da_perm, phi, &phi_field);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(user->da_perm, phi_old, &phi_old_field);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(user->da_secondary, eqm_k, &eqm_k_field);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(user->da_secondary, sec_conc_old,
                               &sec_conc_old_field);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(user->da_reaction, mass_frac_old,
                               &mass_frac_old_field);
    CHKERRQ(ierr);
    ierr =
        DMDAVecRestoreArray(user->da_reaction, initial_ref, &initial_ref_field);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedGlobalVector(user->da_perm, "phi", &phi);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedGlobalVector(user->da_perm, "phi_old", &phi_old);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedGlobalVector(user->da_secondary, "eqm_k", &eqm_k);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedGlobalVector(user->da_secondary, "sec_conc_old",
                                      &sec_conc_old);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedGlobalVector(user->da_reaction, "mass_frac_old",
                                      &mass_frac_old);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedGlobalVector(user->da_reaction, "initial_ref",
                                      &initial_ref);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, user->sol, &sol);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode CopyOldVector(Vec sol, PhysicalField** xold, void* ptr) {
    PetscErrorCode ierr;
    UserCtx* user = (UserCtx*)ptr;
    DM da = user->da;
    Vec xold_vec = NULL;
    PhysicalField** sol_local;
    PhysicalField** xold_global;
    PetscInt i, j, xl, yl, zl, nxl, nyl, nzl;
    PetscFunctionBeginUser;
    ierr = DMDAGetCorners(da, &xl, &yl, &zl, &nxl, &nyl, &nzl);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, sol, &sol_local);
    CHKERRQ(ierr);
    ierr = DMGetNamedGlobalVector(da, "xold", &xold_vec);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, xold_vec, &xold_global);
    CHKERRQ(ierr);
    for (j = yl; j < yl + nyl; j++) {
        for (i = xl; i < xl + nxl; i++) {
            xold_global[j][i].pw = sol_local[j][i].pw;
            for (int nc = 0; nc < DOF_reaction; ++nc) {
                xold_global[j][i].cw[nc] = sol_local[j][i].cw[nc];
            }
        }
    }
    ierr = DMDAVecRestoreArray(da, sol, &sol_local);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, xold_vec, &xold_global);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedGlobalVector(da, "xold", &xold_vec);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
#if EXAMPLE == 2
PetscErrorCode KineticDisPreConcAux(Vec _vals, Vec _mineral_old, Vec _mineral,
                                    double _dt, void* ptr) {
    PetscErrorCode ierr;
    UserCtx* user = (UserCtx*)ptr;
    DM da = user->da, da_mineral = user->da_mineral;
    MineralField **mineral_species, **mineral_species_old;
    PhysicalField** sol;
    PetscInt i, j, xl, yl, zl, nxl, nyl, nzl;
    PetscFunctionBeginUser;
    ierr = DMDAGetCorners(da, &xl, &yl, &zl, &nxl, &nyl, &nzl);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da_mineral, _mineral, &mineral_species);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da_mineral, _mineral_old, &mineral_species_old);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, _vals, &sol);
    CHKERRQ(ierr);
    for (j = yl; j < yl + nyl; j++) {
        for (i = xl; i < xl + nxl; i++) {
            for (int q = 0; q < DOF_mineral; ++q) {
                MineralField kinetic_rate;
                KineticDisPreRateAux_All(&sol[j][i],
                                         &user->eqm_k_mineral_field[j][i],
                                         &kinetic_rate);
                mineral_species[j][i].reaction_mineral[q] =
                    mineral_species_old[j][i].reaction_mineral[q] +
                    kinetic_rate.reaction_mineral[q] * _dt;
                if (mineral_species[j][i].reaction_mineral[q] < 0.0)
                    mineral_species[j][i].reaction_mineral[q] = 0.0;
            };
        }
    }

    ierr = DMDAVecRestoreArray(da_mineral, _mineral, &mineral_species);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, _vals, &sol);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da_mineral, _mineral_old, &mineral_species_old);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
#endif
