#include <float.h>  // 用于DBL_MAX
#include <limits.h>
#include <math.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscsnes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>  // 包含 abs 函数的头文件
#include "def.h"
#include "petscsys.h"
#include "reaction.h"
#include <iostream>
#include </home/chengtianpei/petsc/include/petscdmdatypes.h>
//aspin
#include <autodiff/forward/dual.hpp>

double qn[DOF_reaction] = {0, 0, 0, 0, 0};
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PhysicalField **x, PhysicalField **f, void *ptr)
{
  UserCtx     *user = (UserCtx *)ptr;
  TstepCtx    *tsctx = user->tsctx;
  PetscInt    xints, xinte, yints, yinte, i, j, nc,mx, my;
  PetscScalar alpha[DOF_reaction];
  PetscScalar diff, U_L, U_R, U_B, U_T;
  PetscScalar fluxL, fluxR, fluxB, fluxT;
  PetscScalar fluxL1, fluxR1, fluxB1, fluxT1;
  PetscScalar dx, dy;
  PetscFunctionBeginUser;
  mx   = (info->mx);
  my   = (info->my);
  dx=user->dx;
  dy=user->dy;
  xints = info->xs;
  xinte = info->xs + info->xm;
  yints = info->ys;
  yinte = info->ys + info->ym;

     #define K1_xx(i, j) ((user->perm_field[j][i].xx[0]) / (mu))
     #define K1_yy(i, j) ((user->perm_field[j][i].xx[1]) / (mu))


  for (j = yints; j < yinte; j++) {
    for (i = xints; i < xinte; i++) {
             PhysicalField x_center = x[j][i];
            PhysicalField x_left, x_right, x_bottom, x_top;
            if (i == 0) {
                x_left.pw = 2 * P_init - x_center.pw;
                for (nc = 0; nc < DOF_reaction; ++nc) {
                    x_left.cw[nc] = 2 * c_BC_L - x_center.cw[nc];
                }
            } else {
                x_left = x[j][i - 1];
            }
            if (i == mx - 1) {
                x_right.pw = -x_center.pw;
                for (nc = 0; nc < DOF_reaction; ++nc) {
                    x_right.cw[nc] = 2 * c_BC_R - x_center.cw[nc];
                }
            } else {
                x_right = x[j][i + 1];
            }
            if (j == 0) {
                x_bottom.pw = x_center.pw;
                for (nc = 0; nc < DOF_reaction; ++nc) {
                    x_bottom.cw[nc] = x_center.cw[nc];
                }
            } else {
                x_bottom = x[j - 1][i];
            }
            if (j == my - 1) {
                x_top.pw = x_center.pw;
                for (nc = 0; nc < DOF_reaction; ++nc) {
                    x_top.cw[nc] = x_center.cw[nc];
                }
            } else {
                x_top = x[j + 1][i];
            }

            ReactionField _mass_frac_left, _mass_frac_right, _mass_frac_bottom,
                _mass_frac_top, _mass_frac;
            ReactionField _mineral_sat_left, _mineral_sat_right,
                _mineral_sat_bottom, _mineral_sat_top, _mineral_sat;
            ReactionField _reaction_rate_left, _reaction_rate_right,
                _reaction_rate_bottom, _reaction_rate_top, _reaction_rate;
            ReactionField _sec_conc_left, _sec_conc_right, _sec_conc_bottom,
                _sec_conc_top, _sec_conc;
            PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(
                &_sec_conc_left, &user->eqm_k_field[j][i],
                &_mass_frac_left, &x_left, _equilibrium_constants_as_log10,
                user);
            PorousFlowAqueousPreDisChemistry_computeQpReactionRates(
                reference_temperature_pre, reference_saturation,
                user->phi_old_field[j][i - 1].xx[0], &user->phi_field[j][i].xx[0], &_mineral_sat_left,
                &_reaction_rate_left, &user->_sec_conc_old_field[j][i - 1], &_sec_conc_left,
                _equilibrium_constants_as_log10, user, &user->initial_ref_field[j][i - 1]);
       
            PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(
                &_sec_conc_right, &user->eqm_k_field[j][i],
                &_mass_frac_right, &x_right, _equilibrium_constants_as_log10,
                user);
            PorousFlowAqueousPreDisChemistry_computeQpReactionRates(
                reference_temperature_pre, reference_saturation,
                user->phi_old_field[j][i + 1].xx[0], &user->phi_field[j][i].xx[0], &_mineral_sat_right,
                &_reaction_rate_right, &user->_sec_conc_old_field[j][i + 1],
                &_sec_conc_right, _equilibrium_constants_as_log10, user,
                &user->initial_ref_field[j][i + 1]);
                 
            PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(
                &_sec_conc, &user->eqm_k_field[j][i], &_mass_frac,
                &x_center, _equilibrium_constants_as_log10, user);
            PorousFlowAqueousPreDisChemistry_computeQpReactionRates(
                reference_temperature_pre, reference_saturation,
                user->phi_old_field[j][i].xx[0], &user->phi_field[j][i].xx[0], &_mineral_sat,
                &_reaction_rate, &user->_sec_conc_old_field[j][i], &_sec_conc,
                _equilibrium_constants_as_log10, user, &user->initial_ref_field[j][i]);
       
            PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(
                &_sec_conc_bottom, &user->eqm_k_field[j][i],
                &_mass_frac_bottom, &x_bottom, _equilibrium_constants_as_log10,
                user);
            PorousFlowAqueousPreDisChemistry_computeQpReactionRates(
                reference_temperature_pre, reference_saturation,
                user->phi_old_field[j - 1][i].xx[0], &user->phi_field[j - 1][i].xx[0],
                &_mineral_sat_bottom, &_reaction_rate_bottom,
                &user->_sec_conc_old_field[j - 1][i], &_sec_conc_bottom,
                _equilibrium_constants_as_log10, user, &user->initial_ref_field[j - 1][i]);
                      
            PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(
                &_sec_conc_top, &user->eqm_k_field[j][i], &_mass_frac_top,
                &x_top, _equilibrium_constants_as_log10, user);
            PorousFlowAqueousPreDisChemistry_computeQpReactionRates(
                reference_temperature_pre, reference_saturation,
                user->phi_old_field[j + 1][i].xx[0], &user->phi_field[j + 1][i].xx[0],
                &_mineral_sat_top, &_reaction_rate_top,
                &user->_sec_conc_old_field[j + 1][i], &_sec_conc_top,
                _equilibrium_constants_as_log10, user, &user->initial_ref_field[j + 1][i]);
 
            diff = 0.5 * (dx / (K1_xx(i - 1, j)) + dx / (K1_xx(i, j)));
            U_L = -((x_center.pw - x_left.pw)) / diff;

            diff = 0.5 * (dx / (K1_xx(i + 1, j)) + dx / (K1_xx(i, j)));
            U_R = -((x_right.pw - x_center.pw)) / diff;

            diff = 0.5 * (dy / (K1_yy(i, j - 1)) + dy / (K1_yy(i, j)));
            U_B = -((x_center.pw - x_bottom.pw)) / diff;  //

            diff = 0.5 * (dy / (K1_yy(i, j + 1)) + dy / (K1_yy(i, j)));
            U_T = -((x_top.pw - x_center.pw)) / diff;  //
          
            fluxL = rho(i - 1, j) * max(U_L, 0.0) + rho(i, j) * min(U_L, 0.0);
          
            fluxR = rho(i + 1, j) * min(U_R, 0.0) + rho(i, j) * max(U_R, 0.0);
    
            fluxB = rho(i, j - 1) * max(U_B, 0.0) + rho(i, j) * min(U_B, 0.0);
          
            fluxT = rho(i, j + 1) * min(U_T, 0.0) + rho(i, j) * max(U_T, 0.0); // 

            for (nc = 0; nc < DOF_reaction; ++nc) {
                fluxL1 = _mass_frac_left.reaction[nc] * rho(i - 1, j) *
                             max(U_L, 0.0) +
                         _mass_frac.reaction[nc] * rho(i, j) * min(U_L, 0.0);
                fluxR1 = _mass_frac_right.reaction[nc] * rho(i + 1, j) *
                             min(U_R, 0.0) +
                         _mass_frac.reaction[nc] * rho(i, j) * max(U_R, 0.0);
                fluxB1 = _mass_frac_bottom.reaction[nc] * rho(i, j - 1) *
                             max(U_B, 0.0) +
                         _mass_frac.reaction[nc] * rho(i, j) * min(U_B, 0.0);
                fluxT1 = _mass_frac_top.reaction[nc] * rho(i, j + 1) *
                             min(U_T, 0.0) +
                         _mass_frac.reaction[nc] * rho(i, j) * max(U_T, 0.0);
                alpha[nc] =
                    (rho(i, j) * user->phi_field[j][i].xx[0] * _mass_frac.reaction[nc] -
                     rho_old(i, j) * user->phi_old_field[j][i].xx[0] *
                         user->_mass_frac_old_field[j][i].reaction[nc]) /
                    tsctx->tsize;

                for (int q = 0; q < (DOF - DOF_reaction); ++q) {
                    qn[nc] = stoichiometry[nc] * _mineral_density[nc] *
                             _reaction_rate.reaction[q] * user->phi_field[j][i].xx[0];
                }
                f[j][i].cw[nc] = alpha[nc] + (fluxR1 - fluxL1) / dx +
                                 (fluxT1 - fluxB1) / dy + qn[nc];
    
        
   
            }
            f[j][i].pw = (fluxR - fluxL) / dx + (fluxT - fluxB) / dy;
 
        }
    }
    PetscCall(PetscLogFlops(84.0 * info->ym * info->xm));
  PetscFunctionReturn(PETSC_SUCCESS);
}
