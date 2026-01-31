static char help[] = "3D single-phase flow, by TianpeiCheng .\n\\n";
#include <petscdm.h>
#include <petscdmda.h>
#include <petscsnes.h>
#include "def.h"
#include "petscsys.h"
#include "petsctime.h"
#include "reaction.h"
#include "stdlib.h"
#include <iostream>
MPI_Comm comm;
PetscMPIInt rank, size;

static PetscErrorCode ScatterNamedToSubDMLocal(DM dm, DM subdm, const char *name)
{
    PetscErrorCode ierr;
    Vec g, l;
    VecScatter *iscat = NULL, *oscat = NULL, *gscat = NULL;
    PetscFunctionBeginUser;
    ierr = DMGetNamedGlobalVector(dm, name, &g);
    CHKERRQ(ierr);
    ierr = DMGetNamedLocalVector(subdm, name, &l);
    CHKERRQ(ierr);
    ierr = DMCreateDomainDecompositionScatters(dm, 1, &subdm, &iscat, &oscat,
                                               &gscat);
    CHKERRQ(ierr);
    ierr = VecScatterBegin(*gscat, g, l, INSERT_VALUES, SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr = VecScatterEnd(*gscat, g, l, INSERT_VALUES, SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr = VecScatterDestroy(iscat);
    CHKERRQ(ierr);
    ierr = VecScatterDestroy(oscat);
    CHKERRQ(ierr);
    ierr = VecScatterDestroy(gscat);
    CHKERRQ(ierr);
    ierr = PetscFree(iscat);
    CHKERRQ(ierr);
    ierr = PetscFree(oscat);
    CHKERRQ(ierr);
    ierr = PetscFree(gscat);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedGlobalVector(dm, name, &g);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedLocalVector(subdm, name, &l);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

static PetscErrorCode ScatterNamedToSubDMGlobal(DM dm, DM subdm, const char *name)
{
    PetscErrorCode ierr;
    Vec g, l;
    VecScatter *iscat = NULL, *oscat = NULL, *gscat = NULL;
    PetscFunctionBeginUser;
    ierr = DMGetNamedGlobalVector(dm, name, &g);
    CHKERRQ(ierr);
    ierr = DMGetNamedGlobalVector(subdm, name, &l);
    CHKERRQ(ierr);
    ierr = DMCreateDomainDecompositionScatters(dm, 1, &subdm, &iscat, &oscat,
                                               &gscat);
    CHKERRQ(ierr);
    ierr = VecScatterBegin(*oscat, g, l, INSERT_VALUES, SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr = VecScatterEnd(*oscat, g, l, INSERT_VALUES, SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr = VecScatterDestroy(iscat);
    CHKERRQ(ierr);
    ierr = VecScatterDestroy(oscat);
    CHKERRQ(ierr);
    ierr = VecScatterDestroy(gscat);
    CHKERRQ(ierr);
    ierr = PetscFree(iscat);
    CHKERRQ(ierr);
    ierr = PetscFree(oscat);
    CHKERRQ(ierr);
    ierr = PetscFree(gscat);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedGlobalVector(dm, name, &g);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedGlobalVector(subdm, name, &l);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

static PetscErrorCode CoefficientSubDomainHook(DM dm, DM subdm,
                                                       void *ctx)
{
    PetscErrorCode ierr;
    DM perm_dm = NULL, secondary_dm = NULL, reaction_dm = NULL;
#if EXAMPLE == 2
    DM mineral_dm = NULL;
#endif
    PetscFunctionBeginUser;
    ierr = PetscObjectQuery((PetscObject)dm, "perm_dm", (PetscObject *)&perm_dm);
    CHKERRQ(ierr);
    if (perm_dm)
    {
        PetscInt dof = 0;
        DM perm_subdm = NULL;
        ierr = DMDAGetInfo(perm_dm, PETSC_NULLPTR, PETSC_NULLPTR, PETSC_NULLPTR,
                           PETSC_NULLPTR, PETSC_NULLPTR, PETSC_NULLPTR,
                           PETSC_NULLPTR, &dof, PETSC_NULLPTR, PETSC_NULLPTR,
                           PETSC_NULLPTR, PETSC_NULLPTR, PETSC_NULLPTR);
        CHKERRQ(ierr);
        ierr = DMDACreateCompatibleDMDA(subdm, dof, &perm_subdm);
        CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject)subdm, "perm_dm",
                                  (PetscObject)perm_subdm);
        CHKERRQ(ierr);
        ierr = ScatterNamedToSubDMLocal(perm_dm, perm_subdm, "perm");
        CHKERRQ(ierr);
        ierr = ScatterNamedToSubDMLocal(perm_dm, perm_subdm, "phi");
        CHKERRQ(ierr);
        ierr = ScatterNamedToSubDMLocal(perm_dm, perm_subdm, "phi_old");
        CHKERRQ(ierr);
        ierr = DMDestroy(&perm_subdm);
        CHKERRQ(ierr);
    }
    ierr = PetscObjectQuery((PetscObject)dm, "secondary_dm",
                            (PetscObject *)&secondary_dm);
    CHKERRQ(ierr);
    if (secondary_dm)
    {
        PetscInt dof = 0;
        DM secondary_subdm = NULL;
        ierr = DMDAGetInfo(secondary_dm, PETSC_NULLPTR, PETSC_NULLPTR,
                           PETSC_NULLPTR, PETSC_NULLPTR, PETSC_NULLPTR,
                           PETSC_NULLPTR, PETSC_NULLPTR, &dof, PETSC_NULLPTR,
                           PETSC_NULLPTR, PETSC_NULLPTR, PETSC_NULLPTR,
                           PETSC_NULLPTR);
        CHKERRQ(ierr);
        ierr = DMDACreateCompatibleDMDA(subdm, dof, &secondary_subdm);
        CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject)subdm, "secondary_dm",
                                  (PetscObject)secondary_subdm);
        CHKERRQ(ierr);
        ierr = ScatterNamedToSubDMGlobal(secondary_dm, secondary_subdm, "eqm_k");
        CHKERRQ(ierr);
        ierr = ScatterNamedToSubDMGlobal(secondary_dm, secondary_subdm, "sec_conc_old");
        CHKERRQ(ierr);
        ierr = DMDestroy(&secondary_subdm);
        CHKERRQ(ierr);
    }
    ierr = PetscObjectQuery((PetscObject)dm, "reaction_dm",
                            (PetscObject *)&reaction_dm);
    CHKERRQ(ierr);
    if (reaction_dm)
    {
        PetscInt dof = 0;
        DM reaction_subdm = NULL;
        ierr = DMDAGetInfo(reaction_dm, PETSC_NULLPTR, PETSC_NULLPTR,
                           PETSC_NULLPTR, PETSC_NULLPTR, PETSC_NULLPTR,
                           PETSC_NULLPTR, PETSC_NULLPTR, &dof, PETSC_NULLPTR,
                           PETSC_NULLPTR, PETSC_NULLPTR, PETSC_NULLPTR,
                           PETSC_NULLPTR);
        CHKERRQ(ierr);
        ierr = DMDACreateCompatibleDMDA(subdm, dof, &reaction_subdm);
        CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject)subdm, "reaction_dm",
                                  (PetscObject)reaction_subdm);
        CHKERRQ(ierr);
        ierr = ScatterNamedToSubDMGlobal(reaction_dm, reaction_subdm, "mass_frac_old");
        CHKERRQ(ierr);
        ierr = ScatterNamedToSubDMGlobal(reaction_dm, reaction_subdm, "initial_ref");
        CHKERRQ(ierr);
        ierr = DMDestroy(&reaction_subdm);
        CHKERRQ(ierr);
    }
#if EXAMPLE == 2
    ierr = PetscObjectQuery((PetscObject)dm, "mineral_dm",
                            (PetscObject *)&mineral_dm);
    CHKERRQ(ierr);
    if (mineral_dm)
    {
        PetscInt dof = 0;
        DM mineral_subdm = NULL;
        ierr = DMDAGetInfo(mineral_dm, PETSC_NULLPTR, PETSC_NULLPTR,
                           PETSC_NULLPTR, PETSC_NULLPTR, PETSC_NULLPTR,
                           PETSC_NULLPTR, PETSC_NULLPTR, &dof, PETSC_NULLPTR,
                           PETSC_NULLPTR, PETSC_NULLPTR, PETSC_NULLPTR,
                           PETSC_NULLPTR);
        CHKERRQ(ierr);
        ierr = DMDACreateCompatibleDMDA(subdm, dof, &mineral_subdm);
        CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject)subdm, "mineral_dm",
                                  (PetscObject)mineral_subdm);
        CHKERRQ(ierr);
        ierr = ScatterNamedToSubDMGlobal(mineral_dm, mineral_subdm, "eqm_k_mineral");
        CHKERRQ(ierr);
        ierr = DMDestroy(&mineral_subdm);
        CHKERRQ(ierr);
    }
#endif
    ierr = ScatterNamedToSubDMGlobal(dm, subdm, "xold");
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
PetscErrorCode CoefficientRestrictHook(DM global, VecScatter out, VecScatter in, DM block, void *ctx) {
    PetscErrorCode ierr;
    PetscFunctionBeginUser;
    ierr = CoefficientSubDomainHook(global, block, ctx);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
    PetscErrorCode ierr;
    SNES snes;
    UserCtx *user;
    PetscInt n1, n2, n3;
    ParaCtx param;
    TstepCtx tsctx;

    PetscInitialize(&argc, &argv, (char *)0, help);

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
    if (param.use_adaptive_dt)
    {
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
    if (tsctx.tsstart < 0)
        tsctx.tsstart = 0;
    tsctx.tscurr = tsctx.tsstart;
    tsctx.tsback = -1;
    ierr = PetscOptionsGetInt(PETSC_NULLPTR, PETSC_NULLPTR, "-tsback",
                              &tsctx.tsback, PETSC_NULLPTR);
    CHKERRQ(ierr);

    if (tsctx.tsback > tsctx.tsmax)
        tsctx.tsback = -1;
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
    ierr = DMDACreate2d(comm, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
                        DMDA_STENCIL_STAR, n1, n2, PETSC_DECIDE, PETSC_DECIDE,
                        DOF_Secondary, WIDTH, 0, 0, &(user->da_secondary));
    CHKERRQ(ierr);
    if (DOF_mineral != 0)
    {
        ierr = DMDACreate2d(comm, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
                            DMDA_STENCIL_STAR, n1, n2, PETSC_DECIDE, PETSC_DECIDE,
                            DOF_mineral, WIDTH, 0, 0, &(user->da_mineral));
        CHKERRQ(ierr);
    }
    ierr = DMSetFromOptions(user->da);
    CHKERRQ(ierr);
    ierr = DMSetUp(user->da);
    CHKERRQ(ierr);
    ierr = DMSetUp(user->da_reaction);
    CHKERRQ(ierr);
    ierr = DMSetUp(user->da_perm);
    CHKERRQ(ierr);
    ierr = DMSetUp(user->da_secondary);
    CHKERRQ(ierr);
#if EXAMPLE == 2
    ierr = DMSetUp(user->da_mineral);
    CHKERRQ(ierr);
#endif
    ierr = PetscObjectCompose((PetscObject)user->da, "perm_dm",
                              (PetscObject)user->da_perm);
    CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)user->da, "secondary_dm",
                              (PetscObject)user->da_secondary);
    CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)user->da, "reaction_dm",
                              (PetscObject)user->da_reaction);
    CHKERRQ(ierr);
#if EXAMPLE == 2
    ierr = PetscObjectCompose((PetscObject)user->da, "mineral_dm",
                              (PetscObject)user->da_mineral);
    CHKERRQ(ierr);
#endif
#if EXAMPLE == 1 || EXAMPLE == 3
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
#elif EXAMPLE == 2
    ierr = DMDASetFieldName(user->da, 0, "pressure");
    CHKERRQ(ierr);
    ierr = DMDASetFieldName(user->da, 1, "tracer");
    CHKERRQ(ierr);
    ierr = DMDASetFieldName(user->da, 2, "hco3-");
    CHKERRQ(ierr);
    ierr = DMDASetFieldName(user->da, 3, "h+");
    CHKERRQ(ierr);
    ierr = DMDASetFieldName(user->da, 4, "ca2+");
    CHKERRQ(ierr);
#endif
    ierr = SNESSetDM(snes, user->da);
    CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);
    CHKERRQ(ierr);
    ierr = DMSubDomainHookAdd(user->da, CoefficientSubDomainHook, CoefficientRestrictHook,
                              NULL);
    CHKERRQ(ierr);
    user->tsctx = &tsctx;
    user->param = &param;
    ierr = DMDAGetInfo(user->da, 0, &(user->n1), &(user->n2), 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0);
    CHKERRQ(ierr);
    user->dx = L1 / (PetscScalar)(user->n1);
    user->dy = L2 / (PetscScalar)(user->n2);

    ierr = DMGetNamedGlobalVector(user->da, "solution", &user->sol);
    CHKERRQ(ierr);
    ierr = DMGetNamedGlobalVector(user->da, "residual", &user->myF);
    CHKERRQ(ierr);
#if EXAMPLE == 2
    ierr = DMGetNamedGlobalVector(user->da_mineral, "sol_mineral",
                                  &user->sol_mineral);
    CHKERRQ(ierr);
    ierr = DMGetNamedGlobalVector(user->da_mineral, "sol_mineral_old",
                                  &user->sol_mineral_old);
    CHKERRQ(ierr);
#endif
#if EXAMPLE == 2
    ierr = DMDAGetArray(user->da_mineral, PETSC_TRUE, (void **)&(user->eqm_k_mineral_field));
    CHKERRQ(ierr);
#endif

    ierr = DMDASNESSetFunctionLocal(
        user->da, INSERT_VALUES,
        (PetscErrorCode (*)(DMDALocalInfo *, void *, void *,
                            void *))FormFunctionLocal,
        user);
    ierr = FormInitialValue_local(user);
    CHKERRQ(ierr);
    ierr = FormInitialValue_Perm_local(user);
    CHKERRQ(ierr);
    ierr = FormInitialValue_Reaction_local(user);
    CHKERRQ(ierr);
    user->snes = snes;
    if (!param.PetscPreLoading)
    {
        ierr = PetscPrintf(comm,
                           "\n+++++++++++++++++++++++ Problem parameters "
                           "+++++++++++++++++++++\n");
        CHKERRQ(ierr);
        ierr = PetscPrintf(comm, " Single-phase flow, example: %d\n", EXAMPLE);
        CHKERRQ(ierr);
        ierr = PetscPrintf(comm, " Problem size %d, %d, Ncpu = %d \n", n1, n2,
                           size);
        CHKERRQ(ierr);

        if (param.use_adaptive_dt)
        {
            ierr = PetscPrintf(comm,
                               " Torder = %d, TimeSteps = %g (adpt with %g) x "
                               "%d, final time %f \n",
                               tsctx.torder, tsctx.tsize, tsctx.p, tsctx.tsmax,
                               tsctx.tfinal);
            CHKERRQ(ierr);
        }
        else
        {
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

#if EXAMPLE == 2
    ierr = DMRestoreNamedGlobalVector(user->da_mineral, "sol_mineral",
                                      &user->sol_mineral);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedGlobalVector(user->da_mineral, "sol_mineral_old",
                                      &user->sol_mineral_old);
    CHKERRQ(ierr);
#endif
    ierr = DMRestoreNamedGlobalVector(user->da, "solution", &user->sol);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedGlobalVector(user->da, "residual", &user->myF);
    CHKERRQ(ierr);
    ierr = DMDestroy(&user->da);
    CHKERRQ(ierr);
    ierr = DMDestroy(&user->da_reaction);
    CHKERRQ(ierr);
    ierr = DMDestroy(&user->da_perm);
    CHKERRQ(ierr);
    ierr = DMDestroy(&user->da_secondary);
    CHKERRQ(ierr);
#if EXAMPLE == 2
    ierr = DMDestroy(&user->da_mineral);
    CHKERRQ(ierr);
#endif
    ierr = SNESDestroy(&snes);
    CHKERRQ(ierr);
    ierr = PetscFree(user);
    CHKERRQ(ierr);

    if (PetscPreLoading)
    {
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
PetscErrorCode Update(void *ptr)
{
    PetscErrorCode ierr;
    UserCtx *user = (UserCtx *)ptr;
    ParaCtx *param = user->param;
    TstepCtx *tsctx = user->tsctx;
    SNES snes = user->snes;
    PetscInt max_steps;
    SNESConvergedReason reason;
    int its = 0, lits = 0.0, fits = 0;
    PetscScalar res = 0.0, res0 = 0.0, scaling = 0.0;
    double time0 = 0.0, time1 = 0.0, totaltime = 0.0;
    PetscScalar fnorm;
    FILE *fp;
    char history[36], filename[PETSC_MAX_PATH_LEN - 1];
    PetscFunctionBegin;

    if (param->PetscPreLoading)
    {
        max_steps = 1;
    }
    else
    {
        max_steps = tsctx->tsmax;
    }

    if (!param->PetscPreLoading)
    {
        sprintf(history, "history_c%d_n%dx%d.data", size, user->n1, user->n2);
        ierr = PetscFOpen(comm, history, "a", &fp);
        CHKERRQ(ierr);
        ierr = PetscFPrintf(comm, fp,
                            "%% example, step, t, reason, fnorm, its_snes, "
                            "its_ksp, its_fail, time\n");
        CHKERRQ(ierr);
    }

    for (tsctx->tscurr = tsctx->tsstart + 1;
         (tsctx->tscurr <= tsctx->tsstart + max_steps); tsctx->tscurr++)
    {
        ierr = SNESComputeFunction(snes, user->sol, user->myF);
        CHKERRQ(ierr);
        ierr = VecNorm(user->myF, NORM_2, &res);
        CHKERRQ(ierr);

        if (user->param->use_adaptive_dt)
        {
            if (tsctx->tscurr == tsctx->tsstart + 1)
            {
                res0 = res;
            }
            else
            {
                scaling = pow(res0 / res, tsctx->p);
                if (scaling > tsctx->smax)
                    scaling = tsctx->smax;
                if (scaling < 1. / tsctx->smax)
                    scaling = 1. / tsctx->smax;
                tsctx->tsize = tsctx->tsize * scaling;
                if ((tsctx->tcurr < tsctx->tfinal - EPS) &&
                    (tsctx->tcurr + tsctx->tsize >= tsctx->tfinal + EPS))
                {
                    tsctx->tsize = tsctx->tfinal - tsctx->tcurr;
                }
            }
            ierr = PetscPrintf(comm,
                               " current initial residual = %g with adpt "
                               "method, current time size = %g\n",
                               res, tsctx->tsize);
        }
        else
        {
            ierr = PetscPrintf(
                comm,
                " current initial residual = %g, current time size = %g\n", res,
                tsctx->tsize);
        }

        tsctx->tcurr += tsctx->tsize;
        if (!param->PetscPreLoading)
        {
            ierr = PetscPrintf(comm,
                               "\n====================== Step: %d, time: %f "
                               "====================\n",
                               tsctx->tscurr, tsctx->tcurr);
            CHKERRQ(ierr);
        }
        ierr = PetscTime(&time0);
        CHKERRQ(ierr);

        ierr = SNESSolve(snes, 0, user->sol);
        CHKERRQ(ierr);

        ierr = PetscTime(&time1);

        ierr = Updata_Reaction(user);


        ierr = CopyOldVector(user->sol, user);
        CHKERRQ(ierr);

        if (tsctx->tscurr % 1 == 0)
        {
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

        if (!param->PetscPreLoading)
        {
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
        if (tsctx->tcurr > tsctx->tfinal - EPS)
        {
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

    if (!param->PetscPreLoading)
    {
        ierr = PetscFClose(comm, fp);
        CHKERRQ(ierr);
    }

    PetscFunctionReturn(0);
}

PetscErrorCode FormInitialValue_Perm_local(void *ptr)
{
    PetscErrorCode ierr;
    UserCtx *user = (UserCtx *)ptr;
    DM da = user->da, da_perm = user->da_perm;
    PetscInt i, j, xl, yl, zl, nxl, nyl, nzl, mx, my;
    PetscFunctionBeginUser;
    ierr = DMDAGetCorners(da, &xl, &yl, &zl, &nxl, &nyl, &nzl);
    CHKERRQ(ierr);
#if EXAMPLE == 3
    mx = user->n1;
    my = user->n2;
    Vec u_per;
    PetscViewer dataviewer;
    ierr = DMCreateGlobalVector(da_perm, &u_per);
    CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "SPE85.bin", FILE_MODE_READ, &dataviewer);
    CHKERRQ(ierr);
    ierr = VecLoad(u_per, dataviewer);
    CHKERRQ(ierr);
    ierr = VecScale(u_per, UNIT_MD);
    CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&dataviewer);
    CHKERRQ(ierr);
#endif
    PermField **perm_field = NULL, **phi_field = NULL, **phi_old_field = NULL;
    Vec perm_vec = NULL, phi_vec = NULL, phi_old_vec = NULL;
    ierr = DMGetNamedGlobalVector(da_perm, "perm", &perm_vec);
    CHKERRQ(ierr);
    ierr = DMGetNamedGlobalVector(da_perm, "phi", &phi_vec);
    CHKERRQ(ierr);
    ierr = DMGetNamedGlobalVector(da_perm, "phi_old", &phi_old_vec);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da_perm, perm_vec, &perm_field);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da_perm, phi_vec, &phi_field);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da_perm, phi_old_vec, &phi_old_field);
    CHKERRQ(ierr);
    for (j = yl; j < yl + nyl; j++)
    {
        for (i = xl; i < xl + nxl; i++)
        {

#if EXAMPLE == 1
            for (int nc = 0; nc < DOF_perm; nc++)
            {
                user->perm_field[j][i].xx[nc] = 1e-10;
                user->phi_field[j][i].xx[nc] = 0.2;
                user->phi_old_field[j][i].xx[nc] = 0.2;
            }
#elif EXAMPLE == 2
            for (int nc = 0; nc < DOF_perm; nc++)
            {
                user->perm_field[j][i].xx[nc] = 1;
                user->phi_field[j][i].xx[nc] = 0.2;
                user->phi_old_field[j][i].xx[nc] = 0.2;
            }
#elif EXAMPLE == 3
            for (int nc = 0; nc < DOF_perm; nc++)
            {
                perm_field[j][i].xx[nc] = 1e-10;//perm_field_local[j][i].xx[nc];
                phi_field[j][i].xx[nc] = 0.2;
                phi_old_field[j][i].xx[nc] = 0.2;
            }

#endif
        }
    }
#if EXAMPLE == 3
    ierr = DMRestoreGlobalVector(da_perm, &u_per);
    CHKERRQ(ierr);
#endif
    ierr = DMDAVecRestoreArray(da_perm, perm_vec, &perm_field);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da_perm, phi_vec, &phi_field);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da_perm, phi_old_vec, &phi_old_field);
    CHKERRQ(ierr);


    Vec perm_vec_local = NULL, phi_vec_local = NULL, phi_old_vec_local = NULL;
    ierr = DMGetNamedLocalVector(da_perm, "perm", &perm_vec_local);
    CHKERRQ(ierr);
    ierr = DMGetNamedLocalVector(da_perm, "phi", &phi_vec_local);
    CHKERRQ(ierr);
    ierr = DMGetNamedLocalVector(da_perm, "phi_old", &phi_old_vec_local);
    CHKERRQ(ierr);

    ierr = DMGlobalToLocalBegin(da_perm, perm_vec, INSERT_VALUES, perm_vec_local);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da_perm, perm_vec, INSERT_VALUES, perm_vec_local);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da_perm, phi_vec, INSERT_VALUES, phi_vec_local);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da_perm, phi_vec, INSERT_VALUES, phi_vec_local);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da_perm, phi_old_vec, INSERT_VALUES, phi_old_vec_local);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da_perm, phi_old_vec, INSERT_VALUES, phi_old_vec_local);CHKERRQ(ierr);

    ierr = DMRestoreNamedLocalVector(da_perm, "perm", &perm_vec_local);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedLocalVector(da_perm, "phi", &phi_vec_local);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedLocalVector(da_perm, "phi_old", &phi_old_vec_local);
    CHKERRQ(ierr);

    ierr = DMRestoreNamedGlobalVector(da_perm, "perm", &perm_vec);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedGlobalVector(da_perm, "phi", &phi_vec);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedGlobalVector(da_perm, "phi_old", &phi_old_vec);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode FormInitialValue_local(void *ptr)
{
    PetscErrorCode ierr;
    UserCtx *user = (UserCtx *)ptr;
    DM da = user->da;
    PhysicalField **sol;
    PhysicalField **xold_field;
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
    for (j = yl; j < yl + nyl; j++)
    {
        y_loc = ((PetscScalar)j + 0.5) * user->dy;
        for (i = xl; i < xl + nxl; i++)
        {
            x_loc = ((PetscScalar)i + 0.5) * user->dx;
#if EXAMPLE == 1 || EXAMPLE == 3
            sol[j][i].pw = P_init;
            xold_field[j][i].pw = P_init;
            for (int nc = 0; nc < DOF_reaction; ++nc)
            {
                if (y_loc <= 0.25 && (i == 0))
                {
                    sol[j][i].cw[nc] = c_BC_L;
                    xold_field[j][i].cw[nc] = c_BC_L;
                }
                else
                {
                    sol[j][i].cw[nc] = c_BC_R;
                    xold_field[j][i].cw[nc] = c_BC_R;
                }
            }

#elif EXAMPLE == 2
            double c_init[DOF_reaction] = {0.0, 5.0e-2, 1.e-7, 1.0e-6};
            double c_init_1[DOF_reaction] = {1.0, 1.0e-6, 1.e-7, 1.0e-6};
            sol[j][i].pw = 60 - 50 * x_loc;
            xold_field[j][i].pw = 60 - 50 * x_loc;
            for (int nc = 0; nc < DOF_reaction; ++nc)
            {
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

PetscErrorCode FormInitialValue_Reaction_local(void *ptr)
{
    PetscErrorCode ierr;
    UserCtx *user = (UserCtx *)ptr;
    TstepCtx *tsctx = user->tsctx;
    DM da = user->da;

#if EXAMPLE == 1 || EXAMPLE == 3
    double eqm_data[DOF_Secondary] = {2.19e6, 4.73e-11, 0.222, 1e-2, 1e-3};
#elif EXAMPLE == 2
    double eqm_data[DOF_Secondary] = {6.341, -10.325, -7.009, -0.653, -12.85, -13.991};
#endif
    PetscInt i, j, xg, yg, zg, nxg, nyg, nzg;
    PetscInt xl, yl, zl, nxl, nyl, nzl;
    PetscFunctionBeginUser;
    Vec eqm_k = NULL, sec_conc_old = NULL;
    Vec mass_frac_old = NULL, initial_ref = NULL;
    Vec xold = NULL;
#if EXAMPLE == 2
    Vec eqm_k_mineral_local = NULL;
#endif
    SecondaryReactionField **eqm_k_field = NULL, **sec_conc_old_field = NULL;
    ReactionField **mass_frac_old_field = NULL, **initial_ref_field = NULL;
    PhysicalField **xold_field = NULL;

    ierr = DMGetNamedGlobalVector(user->da_secondary, "eqm_k", &eqm_k);
    CHKERRQ(ierr);
    ierr = DMGetNamedGlobalVector(user->da_secondary, "sec_conc_old",
                                 &sec_conc_old);
    CHKERRQ(ierr);
    ierr = DMGetNamedGlobalVector(user->da_reaction, "mass_frac_old",
                                 &mass_frac_old);
    CHKERRQ(ierr);
    ierr = DMGetNamedGlobalVector(user->da_reaction, "initial_ref",
                                 &initial_ref);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da_secondary, eqm_k,
                           &eqm_k_field);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da_secondary, sec_conc_old,
                           &sec_conc_old_field);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da_reaction, mass_frac_old,
                           &mass_frac_old_field);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da_reaction, initial_ref,
                           &initial_ref_field);
    CHKERRQ(ierr);
#if EXAMPLE == 2
    ierr = DMGetNamedLocalVector(user->da_mineral, "eqm_k_mineral",
                                 &eqm_k_mineral_local);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da_mineral, eqm_k_mineral_local,
                           &user->eqm_k_mineral_field);
    CHKERRQ(ierr);
#endif
    ierr = DMGetNamedGlobalVector(da, "xold", &xold);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, xold, &xold_field);
    CHKERRQ(ierr);
    ierr = DMDAGetGhostCorners(da, &xg, &yg, &zg, &nxg, &nyg, &nzg);
    CHKERRQ(ierr);
    ierr = DMDAGetCorners(da, &xl, &yl, &zl, &nxl, &nyl, &nzl);
    CHKERRQ(ierr);
#if EXAMPLE == 2
    Vec loc_X;
    PhysicalField **sol;
    ierr = DMGetLocalVector(da, &loc_X);
    CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, user->sol, INSERT_VALUES, loc_X);
    CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, user->sol, INSERT_VALUES, loc_X);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, loc_X, &sol);
    CHKERRQ(ierr);
#endif
    for (j = yl; j < yl + nyl; j++)
    {
        for (i = xl; i < xl + nxl; i++)
        {
#if EXAMPLE == 1 || EXAMPLE == 3
            for (int nc = 0; nc < DOF_reaction; ++nc)
            {
                mass_frac_old_field[j][i].reaction[nc] = 1.e-6;
            }
            for (int nc = 0; nc < DOF_Secondary; ++nc)
            {
                sec_conc_old_field[j][i].reaction_secondary[nc] = 1.e-7;
                eqm_k_field[j][i].reaction_secondary[nc] = eqm_data[nc];
            }

#elif EXAMPLE == 2

            double eqm_mineral_data[DOF_mineral] = {1.8487};
            for (int nc = 0; nc < DOF_Secondary; ++nc)

            {
                user->eqm_k_field[j][i].reaction_secondary[nc] = eqm_data[nc];
                user->_sec_conc_old_field[j][i].reaction_secondary[nc] = 0.0;
            }

            for (int nc = 0; nc < DOF_mineral; ++nc)

            {
                user->eqm_k_mineral_field[j][i].reaction_mineral[nc] = eqm_mineral_data[nc];
            }

#endif
        }
    }
#if EXAMPLE == 2
    for (j = yl; j < yl + nyl; j++)
    {
        for (i = xl; i < xl + nxl; i++)
        {
            for (int nc = 0; nc < DOF_reaction; ++nc)
            {
                SecondaryReactionField _sec_conc;
                PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(
                    &_sec_conc, &user->eqm_k_field[j][i],
                    &user->_mass_frac_old_field[j][i], &user->xold[j][i], _equilibrium_constants_as_log10,
                    user, tsctx->tcurr);
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
#if EXAMPLE == 2
    {
        Vec eqm_k_mineral_vec = NULL;
        ierr = DMGetNamedGlobalVector(user->da_mineral, "eqm_k_mineral",
                                      &eqm_k_mineral_vec);
        CHKERRQ(ierr);
        ierr = DMLocalToGlobalBegin(user->da_mineral, eqm_k_mineral_local,
                                    INSERT_VALUES, eqm_k_mineral_vec);
        CHKERRQ(ierr);
        ierr = DMLocalToGlobalEnd(user->da_mineral, eqm_k_mineral_local,
                                  INSERT_VALUES, eqm_k_mineral_vec);
        CHKERRQ(ierr);
        ierr = DMRestoreNamedGlobalVector(user->da_mineral, "eqm_k_mineral",
                                          &eqm_k_mineral_vec);
        CHKERRQ(ierr);
    }
#endif
    ierr = DMDAVecRestoreArray(da, xold, &xold_field);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedGlobalVector(da, "xold", &xold);
    CHKERRQ(ierr);
#if EXAMPLE == 2
    ierr = DMDAVecRestoreArray(user->da_mineral, eqm_k_mineral_local,
                               &user->eqm_k_mineral_field);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedLocalVector(user->da_mineral, "eqm_k_mineral",
                                     &eqm_k_mineral_local);
    CHKERRQ(ierr);
#endif
    ierr = DMDAVecRestoreArray(user->da_secondary, eqm_k,
                               &eqm_k_field);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(user->da_secondary, sec_conc_old,
                               &sec_conc_old_field);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(user->da_reaction, mass_frac_old,
                               &mass_frac_old_field);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(user->da_reaction, initial_ref,
                               &initial_ref_field);
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
PetscErrorCode Updata_Reaction(void *ptr)
{
    PetscErrorCode ierr;
    UserCtx *user = (UserCtx *)ptr;
    TstepCtx *tsctx = user->tsctx;
    DM da = user->da;
    PhysicalField **sol;
    PetscInt i, j, nc, mx, my, xl, yl, zl, nxl, nyl, nzl, xg, yg, zg, nxg, nyg,
        nzg;
    Vec loc_X;
    PetscFunctionBeginUser;
    Vec perm = NULL, phi = NULL, phi_old = NULL;
    Vec eqm_k = NULL, sec_conc_old = NULL;
    Vec mass_frac_old = NULL, initial_ref = NULL;
    PermField **perm_field = NULL, **phi_field = NULL, **phi_old_field = NULL;
    SecondaryReactionField **eqm_k_field = NULL, **sec_conc_old_field = NULL;
    ReactionField **mass_frac_old_field = NULL, **initial_ref_field = NULL;

    ierr = DMGetNamedGlobalVector(user->da_perm, "perm", &perm);
    CHKERRQ(ierr);
    ierr = DMGetNamedGlobalVector(user->da_perm, "phi", &phi);
    CHKERRQ(ierr);
    ierr = DMGetNamedGlobalVector(user->da_perm, "phi_old", &phi_old);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da_perm, perm, &perm_field);
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
    ierr = DMDAVecGetArray(user->da_secondary, eqm_k,
                           &eqm_k_field);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da_secondary, sec_conc_old,
                           &sec_conc_old_field);
    CHKERRQ(ierr);
    ierr = DMGetNamedGlobalVector(user->da_reaction, "mass_frac_old",
                                 &mass_frac_old);
    CHKERRQ(ierr);
    ierr = DMGetNamedGlobalVector(user->da_reaction, "initial_ref",
                                 &initial_ref);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da_reaction, mass_frac_old,
                           &mass_frac_old_field);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da_reaction, initial_ref,
                           &initial_ref_field);
    CHKERRQ(ierr);
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
#if EXAMPLE == 1 || EXAMPLE == 3
    if (xl == 0)
    {
        for (j = yg; j < yg + nyg; j++)
        {
            for (i = -WIDTH; i < 0; i++)
            {
                sol[j][i].pw = 2 * P_init - sol[j][-i - 1].pw;
                for (nc = 0; nc < DOF_reaction; ++nc)
                {
                    sol[j][i].cw[nc] = 2 * c_BC_L - sol[j][-i - 1].cw[nc];
                }
            }
        }
    }
    if (xl + nxl == mx)
    {
        for (j = yg; j < yg + nyg; j++)
        {
            for (i = mx; i < mx + WIDTH; i++)
            {
                sol[j][i].pw = -sol[j][2 * mx - i - 1].pw; //
                for (nc = 0; nc < DOF_reaction; ++nc)
                {
                    sol[j][mx].cw[nc] =
                        2 * c_BC_R - sol[j][2 * mx - i - 1].cw[nc]; //
                }
            }
        }
    }
    if (yl == 0)
    {
        for (i = xg; i < xg + nxg; i++)
        {
            for (j = -WIDTH; j < 0; j++)
            {
                sol[j][i].pw = sol[-j - 1][i].pw; //
                for (nc = 0; nc < DOF_reaction; ++nc)
                {
                    sol[j][i].cw[nc] = sol[-j - 1][i].cw[nc];
                }
            }
        }
    }
    if (yl + nyl == my)
    {
        for (i = xg; i < xg + nxg; i++)
        {
            for (j = my; j < my + WIDTH; j++)
            {
                sol[j][i].pw = sol[2 * my - j - 1][i].pw;
                for (nc = 0; nc < DOF_reaction; ++nc)
                {
                    sol[j][i].cw[nc] = sol[2 * my - j - 1][i].cw[nc];
                }
            }
        }
    }
#elif EXAMPLE == 2
    if (xl == 0)
    {
        for (j = yg; j < yg + nyg; j++)
        {
            for (i = -WIDTH; i < 0; i++)
            {

                sol[j][i].pw = 2 * 60 - sol[j][-i - 1].pw;
                sol[j][i].cw[0] = 2 * 1 - sol[j][-i - 1].cw[0];
                if (tsctx->tcurr < T_final)
                {

                    sol[j][i].cw[1] = 2 * (5.e-2 + (1.e-6 - 5.e-2) * (sin(0.5 * 3.14 * tsctx->tcurr / T_final))) - sol[j][-i - 1].cw[1];
                    sol[j][i].cw[3] = 2 * (1.e-6 + (5.e-2 - 1.e-6) * (sin(0.5 * 3.14 * tsctx->tcurr / T_final))) - sol[j][-i - 1].cw[3];
                }
                else
                {
                    sol[j][i].cw[1] = 2 * 1.e-6 - sol[j][-i - 1].cw[1];
                    sol[j][i].cw[3] = 2 * 5.e-2 - sol[j][-i - 1].cw[3];
                }
                sol[j][i].cw[2] = 2 * 1.e-7 - sol[j][-i - 1].cw[2];
            }
        }
    }

    if (xl + nxl == mx)
    {
        for (j = yg; j < yg + nyg; j++)
        {
            for (i = mx; i < mx + WIDTH; i++)
            {

                sol[j][i].pw = 2 * 10 - sol[j][2 * mx - i - 1].pw;
                sol[j][i].cw[0] = sol[j][2 * mx - i - 1].cw[0];
                sol[j][i].cw[1] = sol[j][2 * mx - i - 1].cw[1];
                sol[j][i].cw[2] = sol[j][2 * mx - i - 1].cw[2];
                sol[j][i].cw[3] = sol[j][2 * mx - i - 1].cw[3];
            }
        }
    }

    if (yl == 0)
    {
        for (i = xg; i < xg + nxg; i++)
        {
            for (j = -WIDTH; j < 0; j++)
            {

                sol[j][i].pw = sol[-j - 1][i].pw;
                sol[j][i].cw[0] = sol[-j - 1][i].cw[0];
                sol[j][i].cw[1] = sol[-j - 1][i].cw[1];
                sol[j][i].cw[2] = sol[-j - 1][i].cw[2];
                sol[j][i].cw[3] = sol[-j - 1][i].cw[3];
            }
        }
    }

    if (yl + nyl == my)
    {
        for (i = xg; i < xg + nxg; i++)
        {
            for (j = my; j < my + WIDTH; j++)
            {

                sol[j][i].pw = sol[2 * my - j - 1][i].pw;
                sol[j][i].cw[0] = sol[2 * my - j - 1][i].cw[0];
                sol[j][i].cw[1] = sol[2 * my - j - 1][i].cw[1];
                sol[j][i].cw[2] = sol[2 * my - j - 1][i].cw[2];
                sol[j][i].cw[3] = sol[2 * my - j - 1][i].cw[3];
            }
        }
    }
#endif
    for (j = yl; j < yl + nyl; j++)
    {
        for (i = xl; i < xl + nxl; i++)
        {
#if EXAMPLE == 1 || EXAMPLE == 3
            SecondaryReactionField _sec_conc;
            ReactionField _mass_frac, _reaction_rate, _mineral_sat;
            PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(
                &_sec_conc, &eqm_k_field[j][i],
                &_mass_frac, &sol[j][i], _equilibrium_constants_as_log10,
                user, tsctx->tcurr);
            PorousFlowAqueousPreDisChemistry_computeQpReactionRates(
                reference_temperature_pre, reference_saturation,
                phi_old_field[j][i].xx[0], &phi_field[j][i].xx[0], &_mineral_sat,
                &_reaction_rate, &sec_conc_old_field[j][i], &_sec_conc,
                _equilibrium_constants_as_log10, user, &initial_ref_field[j][i]);
            PorousFlowAqueousPreDisMineral_computeQpProperties(
                reference_saturation, &sec_conc_old_field[j][i], &_sec_conc,
                &_reaction_rate, phi_old_field[j][i].xx[0], user);
            sec_conc_old_field[j][i] = _sec_conc;
            mass_frac_old_field[j][i] = _mass_frac;
#elif EXAMPLE == 2
            SecondaryReactionField _sec_conc;
            ReactionField _mass_frac;
            PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(
                &_sec_conc, &user->eqm_k_field[j][i],
                &_mass_frac, &sol[j][i], _equilibrium_constants_as_log10,
                user, tsctx->tcurr);
            user->_sec_conc_old_field[j][i] = _sec_conc;
            user->_mass_frac_old_field[j][i] = _mass_frac;
#endif
        }
    }
    ierr = DMDAVecRestoreArray(user->da_perm, perm, &perm_field);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(user->da_perm, phi, &phi_field);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(user->da_perm, phi_old, &phi_old_field);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(user->da_secondary, eqm_k,
                               &eqm_k_field);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(user->da_secondary, sec_conc_old,
                               &sec_conc_old_field);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(user->da_reaction, mass_frac_old,
                               &mass_frac_old_field);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(user->da_reaction, initial_ref,
                               &initial_ref_field);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedGlobalVector(user->da_perm, "perm", &perm);
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
    ierr = DMDAVecRestoreArray(da, loc_X, &sol);
    CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(da, &loc_X);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode CopyOldVector(Vec sol,  void *ptr)
{
    PetscErrorCode ierr;
    UserCtx *user = (UserCtx *)ptr;
    DM da = user->da;
    PhysicalField **sol_local;
    PhysicalField **xold_global;
    Vec xold_vec = NULL;
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
    for (j = yl; j < yl + nyl; j++)
    {
        for (i = xl; i < xl + nxl; i++)
        {
            xold_global[j][i].pw = sol_local[j][i].pw;
            for (int nc = 0; nc < DOF_reaction; ++nc)
            {
                xold_global[j][i].cw[nc] = sol_local[j][i].cw[nc];
               // printf("xold_global[%d][%d].cw[%d]=%g\n", j, i, nc, xold_global[j][i].cw[nc]);
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
PetscErrorCode KineticDisPreConcAux(Vec _vals, Vec _mineral_old, Vec _mineral, double _dt, void *ptr)
{

    PetscErrorCode ierr;
    UserCtx *user = (UserCtx *)ptr;
    DM da = user->da, da_mineral = user->da_mineral;
    MineralField **mineral_species, **mineral_species_old;
    PhysicalField **sol;
    PetscInt i, j, xl, yl, zl, nxl, nyl, nzl;
    PetscFunctionBeginUser;
    Vec eqm_k_mineral_local = NULL;
    ierr = DMGetNamedLocalVector(da_mineral, "eqm_k_mineral",
                                 &eqm_k_mineral_local);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da_mineral, eqm_k_mineral_local,
                           &user->eqm_k_mineral_field);
    CHKERRQ(ierr);
    ierr = DMDAGetCorners(da, &xl, &yl, &zl, &nxl, &nyl, &nzl);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da_mineral, _mineral, &mineral_species);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da_mineral, _mineral_old, &mineral_species_old);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, _vals, &sol);
    CHKERRQ(ierr);
    for (j = yl; j < yl + nyl; j++)
    {
        for (i = xl; i < xl + nxl; i++)
        {

            for (int q = 0; q < DOF_mineral; ++q)
            {
                MineralField kinetic_rate;
                KineticDisPreRateAux_All(&sol[j][i], &user->eqm_k_mineral_field[j][i],
                                         &kinetic_rate);
                mineral_species[j][i].reaction_mineral[q] = mineral_species_old[j][i].reaction_mineral[q] + kinetic_rate.reaction_mineral[q] * _dt;
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
    ierr = DMDAVecRestoreArray(da_mineral, eqm_k_mineral_local,
                               &user->eqm_k_mineral_field);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedLocalVector(da_mineral, "eqm_k_mineral",
                                     &eqm_k_mineral_local);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
#endif
