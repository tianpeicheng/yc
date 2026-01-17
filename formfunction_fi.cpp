#include <float.h> // 用于DBL_MAX
#include <limits.h>
#include <math.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscsnes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h> // 包含 abs 函数的头文件
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
    UserCtx *user = (UserCtx *)ptr;
    TstepCtx *tsctx = user->tsctx;
    PetscInt xints, xinte, yints, yinte, i, j, nc, mx, my;
    PetscScalar alpha[DOF_reaction];
    PetscScalar diff, U_L, U_R, U_B, U_T;
    PetscScalar fluxL, fluxR, fluxB, fluxT;
    PetscScalar fluxL1, fluxR1, fluxB1, fluxT1;
    PetscScalar fluxL2, fluxR2, fluxB2, fluxT2;
    PetscScalar dx, dy;
    PetscFunctionBeginUser;
    mx = (info->mx);
    my = (info->my);
    dx = user->dx;
    dy = user->dy;
    xints = info->xs;
    xinte = info->xs + info->xm;
    yints = info->ys;
    yinte = info->ys + info->ym;
#if EXAMPLE == 1||EXAMPLE==3
#define conc_1(i, j) (1)
#define diffusivity(i, j) (0)
#elif EXAMPLE == 2
#define conc_1(i, j) (2.e-4)
#define diffusivity(i, j) (1.e-7)
#endif
#define K1_xx(i, j) ((user->perm_field[j][i].xx[0]) / (mu))
#define K1_yy(i, j) ((user->perm_field[j][i].xx[1]) / (mu))

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

                    x_left.cw[1] = 2 * (5.e-2 + (1.e-6 - 5.e-2) * (sin(0.5 * 3.14 * tsctx->tcurr / T_final))) - x_center.cw[1];
                    x_left.cw[3] = 2 * (1.e-6 + (5.e-2 - 1.e-6) * (sin(0.5 * 3.14 * tsctx->tcurr / T_final))) - x_center.cw[3];
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
                &_sec_conc_left, &user->eqm_k_field[j][i],
                &_mass_frac_left, &x_left, _equilibrium_constants_as_log10,
                user);

            PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(
                &_sec_conc_right, &user->eqm_k_field[j][i],
                &_mass_frac_right, &x_right, _equilibrium_constants_as_log10,
                user);

            PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(
                &_sec_conc, &user->eqm_k_field[j][i], &_mass_frac,
                &x_center, _equilibrium_constants_as_log10, user);

            PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(
                &_sec_conc_bottom, &user->eqm_k_field[j][i],
                &_mass_frac_bottom, &x_bottom, _equilibrium_constants_as_log10,
                user);

            PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(
                &_sec_conc_top, &user->eqm_k_field[j][i], &_mass_frac_top,
                &x_top, _equilibrium_constants_as_log10, user);
            

            ReactionField _mineral_sat, _reaction_rate ;


#if EXAMPLE == 1||EXAMPLE==3
            PorousFlowAqueousPreDisChemistry_computeQpReactionRates(
                reference_temperature_pre, reference_saturation,
                user->phi_old_field[j][i].xx[0], &user->phi_field[j][i].xx[0], &_mineral_sat,
                &_reaction_rate, &user->_sec_conc_old_field[j][i], &_sec_conc,
                _equilibrium_constants_as_log10, user, &user->initial_ref_field[j][i]);
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
                    (rho(i, j) * user->phi_field[j][i].xx[0] * _mass_frac.reaction[nc] -
                     rho_old(i, j) * user->phi_old_field[j][i].xx[0] *
                         user->_mass_frac_old_field[j][i].reaction[nc]) /
                    tsctx->tsize;
#if EXAMPLE==1||EXAMPLE==3
                for (int q = 0; q < (DOF - DOF_reaction); ++q)
                {
                    qn[nc] = stoichiometry[nc] * _mineral_density[nc] *
                             _reaction_rate.reaction[q] * user->phi_field[j][i].xx[0];
                
                }
#endif
                f[j][i].cw[nc] = alpha[nc] + (fluxR1 - fluxL1) / dx +
                                 (fluxT1 - fluxB1) / dy - ((fluxR2 - fluxL2) / dx + (fluxT2 - fluxB2) / dy) + qn[nc];
            }
#if EXAMPLE == 1||EXAMPLE==3
            f[j][i].pw = (fluxR - fluxL) / dx + (fluxT - fluxB) / dy;
#elif EXAMPLE == 2
            f[j][i].pw = (fluxR - fluxL) / dx + (fluxT - fluxB) / dy;// x[j][i].pw - (60 - 50 * x_loc);
#endif
        }
    }
    PetscCall(PetscLogFlops(84.0 * info->ym * info->xm));
    PetscFunctionReturn(PETSC_SUCCESS);
}
