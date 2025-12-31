#include <petscsnes.h>
#include <petscdmda.h>
#include "def.h"
#include "reaction.h"
#include <stdio.h>
#include "petscsys.h"
#include <stdlib.h> // 包含 abs 函数的头文件
#include <math.h>
#include <float.h> // 用于DBL_MAX
#include <stdbool.h>
#include <limits.h>
double qn[DOF_reaction] = {0, 0, 0, 0, 0};
#undef __FUNCT__
#define __FUNCT__ "FormFunction"
PetscErrorCode FormFunction(SNES snes, Vec X, Vec F, void *ptr)
{
	PetscErrorCode ierr;
	UserCtx *user = (UserCtx *)ptr;
	TstepCtx *tsctx = user->tsctx;
	DM da = user->da, da_reaction = user->da_reaction, da_perm = user->da_perm;
	PetscScalar dx = user->dx, dy = user->dy;
	PetscScalar alpha[DOF_reaction];
	PetscInt i, j, nc, mx, my, xl, yl, zl, nxl, nyl, nzl, xg, yg, zg, nxg, nyg, nzg;
	PetscScalar diff, U_L, U_R, U_B, U_T;
	PetscScalar fluxL, fluxR, fluxB, fluxT;
	PetscScalar fluxL1, fluxR1, fluxB1, fluxT1;
	Vec loc_X, loc_Xold;
	PhysicalField **x, **f, **xold;
	PermField **perm, **phi, **phi_old;
	ReactionField **initial_ref, **_equilibrium_constants;
	ReactionField **_sec_conc_old, **mineral_conc_old, **_mass_frac_old;

	PetscFunctionBeginUser;
	mx = user->n1;
	my = user->n2;
	ierr = DMDAGetGhostCorners(da, &xg, &yg, &zg, &nxg, &nyg, &nzg);
	CHKERRQ(ierr);
	ierr = DMDAGetCorners(da, &xl, &yl, &zl, &nxl, &nyl, &nzl);
	CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da_perm, user->phi, &phi);
	CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da_perm, user->phi_old, &phi_old);
	CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da_perm, user->perm, &perm);
	CHKERRQ(ierr);

	ierr = DMGetLocalVector(da, &loc_X);
	CHKERRQ(ierr);
	ierr = DMGetLocalVector(da, &loc_Xold);
	CHKERRQ(ierr);
	ierr = DMGlobalToLocalBegin(da, X, INSERT_VALUES, loc_X);
	CHKERRQ(ierr);
	ierr = DMGlobalToLocalEnd(da, X, INSERT_VALUES, loc_X);
	CHKERRQ(ierr);
	ierr = DMGlobalToLocalBegin(da, user->Q0, INSERT_VALUES, loc_Xold);
	CHKERRQ(ierr);
	ierr = DMGlobalToLocalEnd(da, user->Q0, INSERT_VALUES, loc_Xold);
	CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da, loc_X, &x);
	CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da, F, &f);
	CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da, loc_Xold, &xold);
	CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da_reaction, user->eqm_k, &_equilibrium_constants);
	CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da_reaction, user->initial_ref, &initial_ref);
	CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da_reaction, user->mineral_conc_old, &mineral_conc_old);
	CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da_reaction, user->_mass_frac_old, &_mass_frac_old);
	CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da_reaction, user->_sec_conc_old, &_sec_conc_old);
	CHKERRQ(ierr);


#define K_xx(i, j) ((perm[j][i].xx[0]) / (mu))
#define K_yy(i, j) ((perm[j][i].xx[1]) / (mu))

	for (j = yl; j < yl + nyl; j++)
	{
		for (i = xl; i < xl + nxl; i++)
		{
			PhysicalField x_center = x[j][i];
			PhysicalField x_left, x_right, x_bottom, x_top;
			if (i == 0) {
				x_left.pw = 2 * P_init - x_center.pw;
				for (nc = 0; nc < DOF_reaction; ++nc)
				{
					x_left.cw[nc] = 2 * c_BC_L - x_center.cw[nc];
				}
			} else {
				x_left = x[j][i - 1];
			}
			if (i == mx - 1) {
				x_right.pw = -x_center.pw;
				for (nc = 0; nc < DOF_reaction; ++nc)
				{
					x_right.cw[nc] = 2 * c_BC_R - x_center.cw[nc];
				}
			} else {
				x_right = x[j][i + 1];
			}
			if (j == 0) {
				x_bottom.pw = x_center.pw;
				for (nc = 0; nc < DOF_reaction; ++nc)
				{
					x_bottom.cw[nc] = x_center.cw[nc];
				}
			} else {
				x_bottom = x[j - 1][i];
			}
			if (j == my - 1) {
				x_top.pw = x_center.pw;
				for (nc = 0; nc < DOF_reaction; ++nc)
				{
					x_top.cw[nc] = x_center.cw[nc];
				}
			} else {
				x_top = x[j + 1][i];
			}

			ReactionField _mass_frac_left, _mass_frac_right, _mass_frac_bottom, _mass_frac_top, _mass_frac;
			ReactionField _mineral_sat_left, _mineral_sat_right, _mineral_sat_bottom, _mineral_sat_top, _mineral_sat;
			ReactionField _reaction_rate_left, _reaction_rate_right, _reaction_rate_bottom, _reaction_rate_top, _reaction_rate;
			ReactionField _sec_conc_left, _sec_conc_right, _sec_conc_bottom, _sec_conc_top, _sec_conc;
			PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(&_sec_conc_left, &_equilibrium_constants[j][i], &_mass_frac_left, &x_left, _equilibrium_constants_as_log10, user);
			PorousFlowAqueousPreDisChemistry_computeQpReactionRates(reference_temperature_pre, reference_saturation, phi_old[j][i-1].xx[0], &phi[j][i].xx[0], &_mineral_sat_left, &_reaction_rate_left, &_sec_conc_old[j][i - 1], &_sec_conc_left, _equilibrium_constants_as_log10, user, &mineral_conc_old[j][i - 1], &initial_ref[j][i - 1]);
			PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(&_sec_conc_right, &_equilibrium_constants[j][i], &_mass_frac_right, &x_right, _equilibrium_constants_as_log10, user);
			PorousFlowAqueousPreDisChemistry_computeQpReactionRates(reference_temperature_pre, reference_saturation, phi_old[j][i+1].xx[0], &phi[j][i].xx[0], &_mineral_sat_right, &_reaction_rate_right, &_sec_conc_old[j][i + 1], &_sec_conc_right, _equilibrium_constants_as_log10, user, &mineral_conc_old[j][i + 1], &initial_ref[j][i + 1]);
			PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(&_sec_conc, &_equilibrium_constants[j][i], &_mass_frac, &x_center, _equilibrium_constants_as_log10, user);
			PorousFlowAqueousPreDisChemistry_computeQpReactionRates(reference_temperature_pre, reference_saturation, phi_old[j][i].xx[0], &phi[j][i].xx[0], &_mineral_sat, &_reaction_rate, &_sec_conc_old[j][i], &_sec_conc, _equilibrium_constants_as_log10, user, &mineral_conc_old[j][i], &initial_ref[j][i]);
			PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(&_sec_conc_bottom, &_equilibrium_constants[j][i], &_mass_frac_bottom, &x_bottom, _equilibrium_constants_as_log10, user);
			PorousFlowAqueousPreDisChemistry_computeQpReactionRates(reference_temperature_pre, reference_saturation, phi_old[j - 1][i].xx[0], &phi[j - 1][i].xx[0], &_mineral_sat_bottom, &_reaction_rate_bottom, &_sec_conc_old[j - 1][i], &_sec_conc_bottom, _equilibrium_constants_as_log10, user, &mineral_conc_old[j - 1][i], &initial_ref[j - 1][i]);
			PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(&_sec_conc_top, &_equilibrium_constants[j][i], &_mass_frac_top, &x_top, _equilibrium_constants_as_log10, user);
			PorousFlowAqueousPreDisChemistry_computeQpReactionRates(reference_temperature_pre, reference_saturation, phi_old[j + 1][i].xx[0], &phi[j + 1][i].xx[0], &_mineral_sat_top, &_reaction_rate_top, &_sec_conc_old[j + 1][i], &_sec_conc_top, _equilibrium_constants_as_log10, user, &mineral_conc_old[j + 1][i], &initial_ref[j + 1][i]);
			// PorousFlowAqueousPreDisMineral_computeQpProperties(reference_saturation,&_sec_conc_old, &_sec_conc, &_reaction_rate,  phi_old[j][i].xx[0],user);

			diff = 0.5 * (dx / (K_xx(i - 1, j)) + dx / (K_xx(i, j)));
			U_L = -((x_center.pw - x_left.pw)) / diff;

			diff = 0.5 * (dx / (K_xx(i + 1, j)) + dx / (K_xx(i, j)));
			U_R = -((x_right.pw - x_center.pw)) / diff;

			diff = 0.5 * (dy / (K_yy(i, j - 1)) + dy / (K_yy(i, j)));
			U_B = -((x_center.pw - x_bottom.pw)) / diff; //

			diff = 0.5 * (dy / (K_yy(i, j + 1)) + dy / (K_yy(i, j)));
			U_T = -((x_top.pw - x_center.pw)) / diff; //
			fluxL = rho(i - 1, j) * max(U_L, 0.0) + rho(i, j) * min(U_L, 0.0);
			fluxR = rho(i + 1, j) * min(U_R, 0.0) + rho(i, j) * max(U_R, 0.0);
			fluxB = rho(i, j - 1) * max(U_B, 0.0) + rho(i, j) * min(U_B, 0.0);
			fluxT = rho(i, j + 1) * min(U_T, 0.0) + rho(i, j) * max(U_T, 0.0);

			for (nc = 0; nc < DOF_reaction; ++nc)
			{
				fluxL1 = _mass_frac_left.reaction[nc] * rho(i - 1, j) * max(U_L, 0.0) + _mass_frac.reaction[nc] * rho(i, j) * min(U_L, 0.0);
				fluxR1 = _mass_frac_right.reaction[nc] * rho(i + 1, j) * min(U_R, 0.0) + _mass_frac.reaction[nc] * rho(i, j) * max(U_R, 0.0);
				fluxB1 = _mass_frac_bottom.reaction[nc] * rho(i, j - 1) * max(U_B, 0.0) + _mass_frac.reaction[nc] * rho(i, j) * min(U_B, 0.0);
				fluxT1 = _mass_frac_top.reaction[nc] * rho(i, j + 1) * min(U_T, 0.0) + _mass_frac.reaction[nc] * rho(i, j) * max(U_T, 0.0);
				alpha[nc] = (rho(i, j) * phi[j][i].xx[0] * _mass_frac.reaction[nc] - rho_old(i, j) * phi[j][i].xx[0] * _mass_frac_old[j][i].reaction[nc]) / tsctx->tsize;

				for (int q = 0; q < (DOF - DOF_reaction); ++q)
				{
					qn[nc] = stoichiometry[nc] * _mineral_density[nc] * _reaction_rate.reaction[q] * phi[j][i].xx[0];
				}
				f[j][i].cw[nc] = alpha[nc] + (fluxR1 - fluxL1) / dx + (fluxT1 - fluxB1) / dy + qn[nc];
	
			}
			f[j][i].pw = (fluxR - fluxL) / dx + (fluxT - fluxB) / dy;
		}
	}


	ierr = DMDAVecRestoreArray(da_reaction, user->_sec_conc_old, &_sec_conc_old);
	CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(da_reaction, user->mineral_conc_old, &mineral_conc_old);
	CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(da_reaction, user->_mass_frac_old, &_mass_frac_old);
	CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(da_reaction, user->initial_ref, &initial_ref);
	CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(da_perm, user->perm, &perm);
	CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(da_perm, user->phi, &phi);
	CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(da_perm, user->phi_old, &phi_old);
	CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(da_reaction, user->eqm_k, &_equilibrium_constants);
	CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(da, loc_X, &x);
	CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(da, F, &f);
	CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(da, loc_Xold, &xold);
	CHKERRQ(ierr);
	ierr = DMRestoreLocalVector(da, &loc_X);
	CHKERRQ(ierr);
	ierr = DMRestoreLocalVector(da, &loc_Xold);
	CHKERRQ(ierr);

	PetscFunctionReturn(0);
}
