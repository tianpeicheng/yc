#include <float.h> 
#include <limits.h>
#include <math.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscsnes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h> 
#include "def.h"
#include "petscsys.h"
#include "reaction.h"
#include <iostream>
#if EXAMPLE == 1||EXAMPLE==3
double qn[DOF_reaction] = {0, 0, 0, 0, 0};
#elif EXAMPLE == 2
double qn[DOF_reaction] = {0, 0, 0, 0};
#endif
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PhysicalField **x, PhysicalField **f, void *ptr)
{
    PetscErrorCode ierr;
    UserCtx *user = (UserCtx *)ptr;
    TstepCtx *tsctx = user->tsctx;
    DM dm = info->da;
    DM perm_dm = NULL, secondary_dm = NULL, reaction_dm = NULL;
    PetscInt xints, xinte, yints, yinte, i, j, nc, mx, my;
    PetscScalar alpha[DOF_reaction];
    PetscScalar diff, U_L, U_R, U_B, U_T;
    PetscScalar fluxL, fluxR, fluxB, fluxT;
    PetscScalar fluxL1, fluxR1, fluxB1, fluxT1;
    PetscScalar fluxL2, fluxR2, fluxB2, fluxT2;
    PetscScalar dx, dy;
    PetscInt xg, yg, zg, nxg, nyg, nzg;
    PetscFunctionBeginUser;
    ierr = DMDAGetGhostCorners(dm, &xg, &yg, &zg, &nxg, &nyg, &nzg);
    mx = (info->mx);
    my = (info->my);
    dx = user->dx;
    dy = user->dy;
    xints = info->xs;
    xinte = info->xs + info->xm;
    yints = info->ys;
    yinte = info->ys + info->ym;

    Vec perm_local = NULL, phi_local = NULL, phi_old_local = NULL;
    Vec eqm_k = NULL, sec_conc_old = NULL;
    Vec mass_frac_old = NULL, initial_ref = NULL;
    Vec xold = NULL;
    PermField **perm_field = NULL, **phi_field = NULL, **phi_old_field = NULL;
    SecondaryReactionField **eqm_k_field = NULL, **sec_conc_old_field = NULL;
    ReactionField **mass_frac_old_field = NULL, **initial_ref_field = NULL;
    PhysicalField **xold_field = NULL;

    ierr = PetscObjectQuery((PetscObject)dm, "perm_dm",
                            (PetscObject *)&perm_dm);
    CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)dm, "secondary_dm",
                            (PetscObject *)&secondary_dm);
    CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)dm, "reaction_dm",
                            (PetscObject *)&reaction_dm);
    CHKERRQ(ierr);
    if (!perm_dm || !secondary_dm || !reaction_dm)
    {
        SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE,
                "subdm coefficient DMs are not set up");
    }
    ierr = DMGetNamedLocalVector(perm_dm, "perm", &perm_local);
    CHKERRQ(ierr);
    ierr = DMGetNamedLocalVector(perm_dm, "phi", &phi_local);
    CHKERRQ(ierr);
    ierr = DMGetNamedLocalVector(perm_dm, "phi_old", &phi_old_local);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArrayRead(perm_dm, perm_local, &perm_field);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArrayRead(perm_dm, phi_local, &phi_field);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArrayRead(perm_dm, phi_old_local, &phi_old_field);
    CHKERRQ(ierr);

    for (j = yg; j < yg + nyg; j++)
    {
        for (i = xg; i < xg + nxg; i++)
        {
    // 对perm_field, phi_field, phi_old_field 的外区赋值.
            if (i < 0)
            {
                // perm_field_local[j][i].xx[0] = perm_field_local[j][0].xx[0];
                for (int nc = 0; nc < DOF_perm; nc++)
                {
                    perm_field[j][i].xx[nc] = 1e-10;//perm_field_local[j][i].xx[nc];
                    phi_field[j][i].xx[nc] = 0.2;
                    phi_old_field[j][i].xx[nc] = 0.2;
                }
            }
            if (i > (mx - 1))
            {
                // perm_field_local[j][i].xx[0] = perm_field_local[j][mx - 1].xx[0];
                for (int nc = 0; nc < DOF_perm; nc++)
                {
                    perm_field[j][i].xx[nc] = 1e-10;//perm_field_local[j][i].xx[nc];
                    phi_field[j][i].xx[nc] = 0.2;
                    phi_old_field[j][i].xx[nc] = 0.2;
                }
            }
            if (j < 0)
            {
                // perm_field_local[j][i].xx[0] = perm_field_local[0][i].xx[0];
                for (int nc = 0; nc < DOF_perm; nc++)
                {
                    perm_field[j][i].xx[nc] = 1e-10;//perm_field_local[j][i].xx[nc];
                    phi_field[j][i].xx[nc] = 0.2;
                    phi_old_field[j][i].xx[nc] = 0.2;
                }
            }
            if (j > (my - 1))
            {
                // perm_field_local[j][i].xx[0] = perm_field_local[my - 1][i].xx[0];
                for (int nc = 0; nc < DOF_perm; nc++)
                {
                    perm_field[j][i].xx[nc] = 1e-10;//perm_field_local[j][i].xx[nc];
                    phi_field[j][i].xx[nc] = 0.2;
                    phi_old_field[j][i].xx[nc] = 0.2;
                }
            }
        }
    }


    ierr = DMGetNamedGlobalVector(secondary_dm, "eqm_k", &eqm_k);
    CHKERRQ(ierr);
    ierr =
        DMGetNamedGlobalVector(secondary_dm, "sec_conc_old",
                                &sec_conc_old);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArrayRead(secondary_dm, eqm_k, &eqm_k_field);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArrayRead(secondary_dm, sec_conc_old,
                                &sec_conc_old_field);
    CHKERRQ(ierr);
    ierr = DMGetNamedGlobalVector(reaction_dm, "mass_frac_old",
                                    &mass_frac_old);
    CHKERRQ(ierr);
    ierr =
        DMGetNamedGlobalVector(reaction_dm, "initial_ref",
                                &initial_ref);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArrayRead(reaction_dm, mass_frac_old,
                                &mass_frac_old_field);
    CHKERRQ(ierr);
    PetscInt N, n;
PetscCall(VecGetSize(mass_frac_old, &N));      // 全局大小
PetscCall(VecGetLocalSize(mass_frac_old, &n)); // 本 rank 局部大小

//PetscPrintf(PETSC_COMM_WORLD,
  //         "mass_frac_oldpetsc: global size = %d, local size = %d\n",
     //     (int)N, (int)n);

    ierr = DMDAVecGetArrayRead(reaction_dm, initial_ref,
                                &initial_ref_field);
    CHKERRQ(ierr);
    ierr = DMGetNamedGlobalVector(dm, "xold", &xold);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArrayRead(dm, xold, &xold_field);
    CHKERRQ(ierr);

#if EXAMPLE == 1||EXAMPLE==3
#define conc_1(i, j) (1)
#define diffusivity(i, j) (0)
#elif EXAMPLE == 2
#define conc_1(i, j) (2.e-4)
#define diffusivity(i, j) (1.e-7)
#endif
#define K1_xx(i, j) ((perm_field[j][i].xx[0]) / (mu))
#define K1_yy(i, j) ((perm_field[j][i].xx[1]) / (mu))
#if EXAMPLE == 1||EXAMPLE == 3
#define RHO_OLD(i, j)                                                \
    (rho_init * PetscExpReal(xold_field[j][i].pw / _bulk_modulus -          \
                             _thermal_expansion * temp_ref))
#elif EXAMPLE == 2
#define RHO_OLD(i, j) (rho_init)
#endif

    for (j = yints; j < yinte; j++)
    {
        for (i = xints; i < xinte; i++)
        {
            double x_loc = (i + 0.5) * dx;
            PhysicalField x_center = x[j][i];
            PhysicalField x_left, x_right, x_bottom, x_top;
            if (i == 0)
            {
#if EXAMPLE == 1||EXAMPLE==3
                x_left.pw = 2 * P_init - x_center.pw;
                for (nc = 0; nc < DOF_reaction; ++nc)
                {
                    x_left.cw[nc] = 2 * c_BC_L - x_center.cw[nc];
                }
#elif EXAMPLE == 2
                x_left.pw = 2 * P_left - x_center.pw;
                x_left.cw[0] = 2 * 1 - x_center.cw[0];
                if (tsctx->tcurr < T_final)
                {
                    x_left.cw[1] = 2 * (5.e-2 + (1.e-6 - 5.e-2) * (sin(0.5 * 3.14159 * tsctx->tcurr / T_final))) - x_center.cw[1];
                    x_left.cw[3] = 2 * (1.e-6 + (5.e-2 - 1.e-6) * (sin(0.5 * 3.14159 * tsctx->tcurr / T_final))) - x_center.cw[3];
                }
                else
                {
                    x_left.cw[1] = 2 * 1.e-6 - x_center.cw[1];
                    x_left.cw[3] = 2 * 5.e-2 - x_center.cw[3];
                }
                x_left.cw[2] = 2 * 1.e-7 - x_center.cw[2];

#endif
            }
            else
            {
                x_left = x[j][i - 1];
            }
            if (i == mx - 1)
            {
#if EXAMPLE == 1||EXAMPLE==3
                x_right.pw = -x_center.pw;
                for (nc = 0; nc < DOF_reaction; ++nc)
                {
                    x_right.cw[nc] = 2 * c_BC_R - x_center.cw[nc];
                }
#elif EXAMPLE == 2
                x_right.pw = 2 * 10 - x_center.pw;
                for (nc = 0; nc < DOF_reaction; ++nc)
                {
                    x_right.cw[nc] = x_center.cw[nc];
                }
#endif
            }
            else
            {
                x_right = x[j][i + 1];
            }
            if (j == 0)
            {
                x_bottom.pw = x_center.pw;
                for (nc = 0; nc < DOF_reaction; ++nc)
                {
                    x_bottom.cw[nc] = x_center.cw[nc];
                }
            }
            else
            {
                x_bottom = x[j - 1][i];
            }
            if (j == my - 1)
            {
                x_top.pw = x_center.pw;
                for (nc = 0; nc < DOF_reaction; ++nc)
                {
                    x_top.cw[nc] = x_center.cw[nc];
                }
            }
            else
            {
                x_top = x[j + 1][i];
            }
             
            ReactionField _mass_frac_left, _mass_frac_right, _mass_frac_bottom,
                _mass_frac_top, _mass_frac;
            SecondaryReactionField _sec_conc_left, _sec_conc_right, _sec_conc_bottom,
                _sec_conc_top, _sec_conc;

            PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(
                &_sec_conc_left, &eqm_k_field[j][i],
                &_mass_frac_left, &x_left, _equilibrium_constants_as_log10,
                user,tsctx->tcurr);
    
            PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(
                &_sec_conc_right, &eqm_k_field[j][i],
                &_mass_frac_right, &x_right, _equilibrium_constants_as_log10,
                user,tsctx->tcurr);

            PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(
                &_sec_conc, &eqm_k_field[j][i], &_mass_frac,
                &x_center, _equilibrium_constants_as_log10, user,tsctx->tcurr);

            PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(
                &_sec_conc_bottom, &eqm_k_field[j][i],
                &_mass_frac_bottom, &x_bottom, _equilibrium_constants_as_log10,
                user,tsctx->tcurr);

            PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(
                &_sec_conc_top, &eqm_k_field[j][i], &_mass_frac_top,
                &x_top, _equilibrium_constants_as_log10, user,tsctx->tcurr);
            

            ReactionField _mineral_sat, _reaction_rate ;


#if EXAMPLE == 1||EXAMPLE==3
            PorousFlowAqueousPreDisChemistry_computeQpReactionRates(
                reference_temperature_pre, reference_saturation,
                phi_old_field[j][i].xx[0], &phi_field[j][i].xx[0], &_mineral_sat,
                &_reaction_rate, &sec_conc_old_field[j][i], &_sec_conc,
                _equilibrium_constants_as_log10, user, &initial_ref_field[j][i]);
              //  printf("FormFunctionLocal: i=%d, j=%d, reaction_rate[0]=%g\n", i, j,sec_conc_old_field[j][i].reaction_secondary[0]);
#endif

            diff = 0.5 * (dx / (K1_xx(i - 1, j)) + dx / (K1_xx(i, j)));
            U_L = -conc_1(i, j) * ((x_center.pw - x_left.pw)) / diff;

            diff = 0.5 * (dx / (K1_xx(i + 1, j)) + dx / (K1_xx(i, j)));
            U_R = -conc_1(i, j) * ((x_right.pw - x_center.pw)) / diff;

            diff = 0.5 * (dy / (K1_yy(i, j - 1)) + dy / (K1_yy(i, j)));
            U_B = -conc_1(i, j) * ((x_center.pw - x_bottom.pw)) / diff;

            diff = 0.5 * (dy / (K1_yy(i, j + 1)) + dy / (K1_yy(i, j)));
            U_T = -conc_1(i, j) * ((x_top.pw - x_center.pw)) / diff;

            fluxL = rho(i - 1, j) * max(U_L, 0.0) + rho(i, j) * min(U_L, 0.0);

            fluxR = rho(i + 1, j) * min(U_R, 0.0) + rho(i, j) * max(U_R, 0.0);

            fluxB = rho(i, j - 1) * max(U_B, 0.0) + rho(i, j) * min(U_B, 0.0);

            fluxT = rho(i, j + 1) * min(U_T, 0.0) + rho(i, j) * max(U_T, 0.0); //

            for (nc = 0; nc < DOF_reaction; ++nc)
            {
                fluxL1 = _mass_frac_left.reaction[nc] * rho(i - 1, j) *
                             max(U_L, 0.0) +
                         _mass_frac.reaction[nc] * rho(i, j) * min(U_L, 0.0);
                fluxL2 = diffusivity(i, j) * (_mass_frac.reaction[nc] - _mass_frac_left.reaction[nc]) / dx;
                fluxR1 = _mass_frac_right.reaction[nc] * rho(i + 1, j) *
                             min(U_R, 0.0) +
                         _mass_frac.reaction[nc] * rho(i, j) * max(U_R, 0.0);
                fluxR2 = diffusivity(i, j) * (_mass_frac_right.reaction[nc] - _mass_frac.reaction[nc]) / dx;
                fluxB1 = _mass_frac_bottom.reaction[nc] * rho(i, j - 1) *
                             max(U_B, 0.0) +
                         _mass_frac.reaction[nc] * rho(i, j) * min(U_B, 0.0);
                fluxB2 = diffusivity(i, j) * (_mass_frac.reaction[nc] - _mass_frac_bottom.reaction[nc]) / dy;
                fluxT1 = _mass_frac_top.reaction[nc] * rho(i, j + 1) *
                             min(U_T, 0.0) +
                         _mass_frac.reaction[nc] * rho(i, j) * max(U_T, 0.0);
                fluxT2 = diffusivity(i, j) * (_mass_frac_top.reaction[nc] - _mass_frac.reaction[nc]) / dy;
                alpha[nc] =
                    (rho(i, j) * phi_field[j][i].xx[0] * _mass_frac.reaction[nc] -
                     RHO_OLD(i, j) * phi_old_field[j][i].xx[0] *
                         mass_frac_old_field[j][i].reaction[nc]) /
                    tsctx->tsize;
    
#if EXAMPLE==1||EXAMPLE==3
                for (int q = 0; q < (DOF - DOF_reaction); ++q)
                {
                    qn[nc] = stoichiometry[nc] * _mineral_density[nc] *
                             _reaction_rate.reaction[q] * phi_field[j][i].xx[0];
                
                }
#endif  


double scaling_1=1;
#if EXAMPLE==2
if(nc==2){
    scaling_1=1.e+6;
}
#endif  
     f[j][i].cw[nc] = scaling_1*(alpha[nc] + (fluxR1 - fluxL1) / dx +
                                 (fluxT1 - fluxB1) / dy - ((fluxR2 - fluxL2) / dx + (fluxT2 - fluxB2) / dy) + qn[nc]);
     
            }
#if EXAMPLE == 1
            f[j][i].pw = (fluxR - fluxL) / dx + (fluxT - fluxB) / dy;
#elif EXAMPLE == 2
            f[j][i].pw = x[j][i].pw - (60 - 50 * x_loc);
#elif EXAMPLE==3
            double alpha1=0;
           alpha1 = (rho(i, j) * phi_field[j][i].xx[0] -
                    RHO_OLD(i, j) * phi_old_field[j][i].xx[0]) /
                    tsctx->tsize;
            f[j][i].pw = alpha1+(fluxR - fluxL) / dx + (fluxT - fluxB) / dy;
#endif
        }
    }
    PetscCall(PetscLogFlops(84.0 * info->ym * info->xm));
    ierr = DMDAVecRestoreArrayRead(perm_dm, perm_local, &perm_field);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArrayRead(perm_dm, phi_local, &phi_field);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArrayRead(perm_dm, phi_old_local, &phi_old_field);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArrayRead(secondary_dm, eqm_k, &eqm_k_field);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArrayRead(secondary_dm, sec_conc_old,
                                   &sec_conc_old_field);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArrayRead(reaction_dm, mass_frac_old,
                                   &mass_frac_old_field);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArrayRead(reaction_dm, initial_ref,
                                   &initial_ref_field);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArrayRead(dm, xold, &xold_field);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedLocalVector(perm_dm, "perm", &perm_local);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedLocalVector(perm_dm, "phi", &phi_local);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedLocalVector(perm_dm, "phi_old", &phi_old_local);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedGlobalVector(secondary_dm, "eqm_k", &eqm_k);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedGlobalVector(secondary_dm, "sec_conc_old",
                                     &sec_conc_old);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedGlobalVector(reaction_dm, "mass_frac_old",
                                     &mass_frac_old);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedGlobalVector(reaction_dm, "initial_ref",
                                     &initial_ref);
    CHKERRQ(ierr);
    ierr = DMRestoreNamedGlobalVector(dm, "xold", &xold);
    CHKERRQ(ierr);
    PetscFunctionReturn(PETSC_SUCCESS);
}
