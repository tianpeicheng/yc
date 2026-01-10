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

#include <autodiff/forward/dual.hpp>

double qn[DOF_reaction] = {0, 0, 0, 0, 0};

template <typename T>
struct PhysicalFieldT {
	T pw;
	T cw[DOF_reaction];
};

template <typename T>
struct ReactionFieldT {
	T reaction[DOF_reaction];
};

template <typename T>
static inline void ComputeSecondaryConcentrations(ReactionFieldT<T> *sec_conc,
												  const ReactionField *equilibrium_constants,
												  const PhysicalFieldT<T> *x,
												  bool equilibrium_constants_as_log10)
{
	for (int r = 0; r < DOF_reaction; ++r)
	{
		sec_conc->reaction[r] = static_cast<T>(1.0);
		for (int i = 0; i < DOF_reaction; ++i)
		{
			T gamp = static_cast<T>(_primary_activity_coefficients[i]) * x->cw[r];
			if (gamp <= static_cast<T>(0.0))
			{
				if (stoichiometry_Secondary(r, i) < 0.0)
				{
					sec_conc->reaction[r] = static_cast<T>(LONG_MAX);
				}
				else if (stoichiometry_Secondary(r, i) == 0.0)
				{
					sec_conc->reaction[r] = sec_conc->reaction[r] * static_cast<T>(1.0);
				}
				else
				{
					sec_conc->reaction[r] = static_cast<T>(0.0);
					break;
				}
			}
			else
			{
				sec_conc->reaction[r] *= pow(gamp, static_cast<T>(stoichiometry_Secondary(r, i)));
			}
		}
		T eq = equilibrium_constants_as_log10
				   ? pow(static_cast<T>(10.0), static_cast<T>(equilibrium_constants->reaction[r]))
				   : static_cast<T>(equilibrium_constants->reaction[r]);
		sec_conc->reaction[r] *= eq;
		sec_conc->reaction[r] /= static_cast<T>(_secondary_activity_coefficients[r]);
	}
}

template <typename T>
static inline void ComputeMassFraction(ReactionFieldT<T> *mass_frac,
									   ReactionFieldT<T> *sec_conc,
									   const ReactionField *equilibrium_constants,
									   const PhysicalFieldT<T> *x,
									   bool equilibrium_constants_as_log10)
{
	ComputeSecondaryConcentrations(sec_conc, equilibrium_constants, x, equilibrium_constants_as_log10);
	for (int i = 0; i < DOF_reaction; ++i)
	{
		mass_frac->reaction[i] = x->cw[i];
		for (int r = 0; r < DOF_reaction; ++r)
		{
			mass_frac->reaction[i] += static_cast<T>(stoichiometry_Secondary(r, i)) * sec_conc->reaction[r];
		}
	}
}

template <typename T>
static inline T DensityFromPw(const T &pw)
{
	return static_cast<T>(rho_init) *
		   exp(pw / static_cast<T>(_bulk_modulus) - static_cast<T>(_thermal_expansion * temp_ref));
}

template <typename T>
static inline T UpwindMax(const T &a, const T &b)
{
	return (a >= b) ? a : b;
}

template <typename T>
static inline T UpwindMin(const T &a, const T &b)
{
	return (a <= b) ? a : b;
}

template <typename T>
static inline void ComputeResidualAtCell(const PhysicalFieldT<T> &x_center,
										 const PhysicalFieldT<T> &x_left,
										 const PhysicalFieldT<T> &x_right,
										 const PhysicalFieldT<T> &x_bottom,
										 const PhysicalFieldT<T> &x_top,
										 const ReactionField &equilibrium_constants,
										 const ReactionField &mass_frac_old_center,
										 const ReactionField &reaction_rate_center,
										 PetscScalar phi_center,
										 PetscScalar rho_old_center,
										 PetscScalar Kxx_left,
										 PetscScalar Kxx_center,
										 PetscScalar Kxx_right,
										 PetscScalar Kyy_bottom,
										 PetscScalar Kyy_center,
										 PetscScalar Kyy_top,
										 PetscScalar dx,
										 PetscScalar dy,
										 TstepCtx *tsctx,
										 ReactionFieldT<T> *out_cw,
										 T *out_pw)
{
	ReactionFieldT<T> mass_frac_left, mass_frac_right, mass_frac_bottom, mass_frac_top, mass_frac_center;
	ReactionFieldT<T> sec_conc_left, sec_conc_right, sec_conc_bottom, sec_conc_top, sec_conc_center;

	ComputeMassFraction(&mass_frac_left, &sec_conc_left, &equilibrium_constants, &x_left, _equilibrium_constants_as_log10);
	ComputeMassFraction(&mass_frac_right, &sec_conc_right, &equilibrium_constants, &x_right, _equilibrium_constants_as_log10);
	ComputeMassFraction(&mass_frac_bottom, &sec_conc_bottom, &equilibrium_constants, &x_bottom, _equilibrium_constants_as_log10);
	ComputeMassFraction(&mass_frac_top, &sec_conc_top, &equilibrium_constants, &x_top, _equilibrium_constants_as_log10);
	ComputeMassFraction(&mass_frac_center, &sec_conc_center, &equilibrium_constants, &x_center, _equilibrium_constants_as_log10);

	T diff = static_cast<T>(0.5) * (static_cast<T>(dx / Kxx_left) + static_cast<T>(dx / Kxx_center));
	T U_L = -(x_center.pw - x_left.pw) / diff;

	diff = static_cast<T>(0.5) * (static_cast<T>(dx / Kxx_right) + static_cast<T>(dx / Kxx_center));
	T U_R = -(x_right.pw - x_center.pw) / diff;

	diff = static_cast<T>(0.5) * (static_cast<T>(dy / Kyy_bottom) + static_cast<T>(dy / Kyy_center));
	T U_B = -(x_center.pw - x_bottom.pw) / diff;

	diff = static_cast<T>(0.5) * (static_cast<T>(dy / Kyy_top) + static_cast<T>(dy / Kyy_center));
	T U_T = -(x_top.pw - x_center.pw) / diff;

	T rho_left = DensityFromPw(x_left.pw);
	T rho_right = DensityFromPw(x_right.pw);
	T rho_bottom = DensityFromPw(x_bottom.pw);
	T rho_top = DensityFromPw(x_top.pw);
	T rho_center = DensityFromPw(x_center.pw);

	T fluxL = rho_left * UpwindMax(U_L, static_cast<T>(0.0)) + rho_center * UpwindMin(U_L, static_cast<T>(0.0));
	T fluxR = rho_right * UpwindMin(U_R, static_cast<T>(0.0)) + rho_center * UpwindMax(U_R, static_cast<T>(0.0));
	T fluxB = rho_bottom * UpwindMax(U_B, static_cast<T>(0.0)) + rho_center * UpwindMin(U_B, static_cast<T>(0.0));
	T fluxT = rho_top * UpwindMin(U_T, static_cast<T>(0.0)) + rho_center * UpwindMax(U_T, static_cast<T>(0.0));

	for (int nc = 0; nc < DOF_reaction; ++nc)
	{
		T fluxL1 = mass_frac_left.reaction[nc] * rho_left * UpwindMax(U_L, static_cast<T>(0.0)) +
				   mass_frac_center.reaction[nc] * rho_center * UpwindMin(U_L, static_cast<T>(0.0));
		T fluxR1 = mass_frac_right.reaction[nc] * rho_right * UpwindMin(U_R, static_cast<T>(0.0)) +
				   mass_frac_center.reaction[nc] * rho_center * UpwindMax(U_R, static_cast<T>(0.0));
		T fluxB1 = mass_frac_bottom.reaction[nc] * rho_bottom * UpwindMax(U_B, static_cast<T>(0.0)) +
				   mass_frac_center.reaction[nc] * rho_center * UpwindMin(U_B, static_cast<T>(0.0));
		T fluxT1 = mass_frac_top.reaction[nc] * rho_top * UpwindMin(U_T, static_cast<T>(0.0)) +
				   mass_frac_center.reaction[nc] * rho_center * UpwindMax(U_T, static_cast<T>(0.0));

		T alpha = (rho_center * static_cast<T>(phi_center) * mass_frac_center.reaction[nc] -
				   static_cast<T>(rho_old_center) * static_cast<T>(phi_center) * static_cast<T>(mass_frac_old_center.reaction[nc])) /
				  static_cast<T>(tsctx->tsize);

		T qn_local = static_cast<T>(stoichiometry[nc]) * static_cast<T>(_mineral_density[nc]) *
					 static_cast<T>(reaction_rate_center.reaction[0]) * static_cast<T>(phi_center);

		out_cw->reaction[nc] = alpha + (fluxR1 - fluxL1) / static_cast<T>(dx) +
							   (fluxT1 - fluxB1) / static_cast<T>(dy) + qn_local;
	}
	*out_pw = (fluxR - fluxL) / static_cast<T>(dx) + (fluxT - fluxB) / static_cast<T>(dy);
}

static inline void InitDualField(PhysicalFieldT<autodiff::dual> *dst,
								 const PhysicalField *src,
								 bool seed_var,
								 int seed_field)
{
	dst->pw = src->pw;
	autodiff::seed<1>(dst->pw, (seed_var && seed_field == 0) ? 1.0 : 0.0);
	for (int nc = 0; nc < DOF_reaction; ++nc)
	{
		dst->cw[nc] = src->cw[nc];
		autodiff::seed<1>(dst->cw[nc], (seed_var && seed_field == (nc + 1)) ? 1.0 : 0.0);
	}
}

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
	ReactionField **_sec_conc_old, **_mass_frac_old;

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
			PorousFlowAqueousPreDisChemistry_computeQpReactionRates(reference_temperature_pre, reference_saturation, phi_old[j][i-1].xx[0], &phi[j][i].xx[0], &_mineral_sat_left, &_reaction_rate_left, &_sec_conc_old[j][i - 1], &_sec_conc_left, _equilibrium_constants_as_log10, user,  &initial_ref[j][i - 1]);
			PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(&_sec_conc_right, &_equilibrium_constants[j][i], &_mass_frac_right, &x_right, _equilibrium_constants_as_log10, user);
			PorousFlowAqueousPreDisChemistry_computeQpReactionRates(reference_temperature_pre, reference_saturation, phi_old[j][i+1].xx[0], &phi[j][i].xx[0], &_mineral_sat_right, &_reaction_rate_right, &_sec_conc_old[j][i + 1], &_sec_conc_right, _equilibrium_constants_as_log10, user, &initial_ref[j][i + 1]);
			PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(&_sec_conc, &_equilibrium_constants[j][i], &_mass_frac, &x_center, _equilibrium_constants_as_log10, user);
			PorousFlowAqueousPreDisChemistry_computeQpReactionRates(reference_temperature_pre, reference_saturation, phi_old[j][i].xx[0], &phi[j][i].xx[0], &_mineral_sat, &_reaction_rate, &_sec_conc_old[j][i], &_sec_conc, _equilibrium_constants_as_log10, user,  &initial_ref[j][i]);
			PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(&_sec_conc_bottom, &_equilibrium_constants[j][i], &_mass_frac_bottom, &x_bottom, _equilibrium_constants_as_log10, user);
			PorousFlowAqueousPreDisChemistry_computeQpReactionRates(reference_temperature_pre, reference_saturation, phi_old[j - 1][i].xx[0], &phi[j - 1][i].xx[0], &_mineral_sat_bottom, &_reaction_rate_bottom, &_sec_conc_old[j - 1][i], &_sec_conc_bottom, _equilibrium_constants_as_log10, user,  &initial_ref[j - 1][i]);
			PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(&_sec_conc_top, &_equilibrium_constants[j][i], &_mass_frac_top, &x_top, _equilibrium_constants_as_log10, user);
			PorousFlowAqueousPreDisChemistry_computeQpReactionRates(reference_temperature_pre, reference_saturation, phi_old[j + 1][i].xx[0], &phi[j + 1][i].xx[0], &_mineral_sat_top, &_reaction_rate_top, &_sec_conc_old[j + 1][i], &_sec_conc_top, _equilibrium_constants_as_log10, user,  &initial_ref[j + 1][i]);


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

#undef __FUNCT__
#define __FUNCT__ "FormJacobian"
PetscErrorCode FormJacobian(SNES snes, Vec X, Mat J, Mat P, void *ptr)
{
	PetscErrorCode ierr;
	UserCtx *user = (UserCtx *)ptr;
	TstepCtx *tsctx = user->tsctx;
	DM da = user->da, da_reaction = user->da_reaction, da_perm = user->da_perm;
	PetscScalar dx = user->dx, dy = user->dy;
	PetscInt i, j, nc, mx, my, xl, yl, zl, nxl, nyl, nzl, xg, yg, zg, nxg, nyg, nzg;
	Vec loc_X, loc_Xold;
	PhysicalField **x, **xold;
	PermField **perm, **phi, **phi_old;
	ReactionField **initial_ref, **_equilibrium_constants;
	ReactionField **_sec_conc_old, **_mass_frac_old;

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
	ierr = DMDAVecGetArray(da, loc_Xold, &xold);
	CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da_reaction, user->eqm_k, &_equilibrium_constants);
	CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da_reaction, user->initial_ref, &initial_ref);
	CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da_reaction, user->_mass_frac_old, &_mass_frac_old);
	CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da_reaction, user->_sec_conc_old, &_sec_conc_old);
	CHKERRQ(ierr);

#ifdef K_xx
#undef K_xx
#endif
#ifdef K_yy
#undef K_yy
#endif
#define K_xx(i, j) ((perm[j][i].xx[0]) / (mu))
#define K_yy(i, j) ((perm[j][i].xx[1]) / (mu))

	ierr = MatZeroEntries(P);
	CHKERRQ(ierr);
	if (J != P)
	{
		ierr = MatZeroEntries(J);
		CHKERRQ(ierr);
	}

	enum VarPos
	{
		VAR_CENTER = 0,
		VAR_LEFT = 1,
		VAR_RIGHT = 2,
		VAR_BOTTOM = 3,
		VAR_TOP = 4
	};
	struct VarRef
	{
		PetscInt i;
		PetscInt j;
		int c;
		VarPos pos;
	};

	for (j = yl; j < yl + nyl; j++)
	{
		for (i = xl; i < xl + nxl; i++)
		{
			PhysicalField x_center = x[j][i];
			PhysicalField x_left, x_right, x_bottom, x_top;
			if (i == 0)
			{
				x_left.pw = 2 * P_init - x_center.pw;
				for (nc = 0; nc < DOF_reaction; ++nc)
				{
					x_left.cw[nc] = 2 * c_BC_L - x_center.cw[nc];
				}
			}
			else
			{
				x_left = x[j][i - 1];
			}
			if (i == mx - 1)
			{
				x_right.pw = -x_center.pw;
				for (nc = 0; nc < DOF_reaction; ++nc)
				{
					x_right.cw[nc] = 2 * c_BC_R - x_center.cw[nc];
				}
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

			ReactionField sec_conc_center, mass_frac_center;
			ReactionField mineral_sat_center, reaction_rate_center;
			PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(
				&sec_conc_center, &_equilibrium_constants[j][i], &mass_frac_center, &x_center,
				_equilibrium_constants_as_log10, user);
			PorousFlowAqueousPreDisChemistry_computeQpReactionRates(
				reference_temperature_pre, reference_saturation, phi_old[j][i].xx[0], &phi[j][i].xx[0],
				&mineral_sat_center, &reaction_rate_center, &_sec_conc_old[j][i], &sec_conc_center,
				_equilibrium_constants_as_log10, user,  &initial_ref[j][i]);

			const PetscScalar rho_old_center = rho_old(i, j);
			const PetscScalar Kxx_left = K_xx(i - 1, j);
			const PetscScalar Kxx_center = K_xx(i, j);
			const PetscScalar Kxx_right = K_xx(i + 1, j);
			const PetscScalar Kyy_bottom = K_yy(i, j - 1);
			const PetscScalar Kyy_center = K_yy(i, j);
			const PetscScalar Kyy_top = K_yy(i, j + 1);

			VarRef vars[DOF * 5];
			int nvars = 0;
			auto add_vars = [&](VarPos pos, PetscInt ii, PetscInt jj) {
				for (int c = 0; c < DOF; ++c)
				{
					vars[nvars++] = VarRef{ii, jj, c, pos};
				}
			};

			add_vars(VAR_CENTER, i, j);
			if (i > 0)
			{
				add_vars(VAR_LEFT, i - 1, j);
			}
			if (i < mx - 1)
			{
				add_vars(VAR_RIGHT, i + 1, j);
			}
			if (j > 0)
			{
				add_vars(VAR_BOTTOM, i, j - 1);
			}
			if (j < my - 1)
			{
				add_vars(VAR_TOP, i, j + 1);
			}

			for (int v = 0; v < nvars; ++v)
			{
				const VarRef &var = vars[v];
				PhysicalFieldT<autodiff::dual> x_center_d, x_left_d, x_right_d, x_bottom_d, x_top_d;

				InitDualField(&x_center_d, &x_center, var.pos == VAR_CENTER, var.c);
				if (i == 0)
				{
					x_left_d.pw = 2 * P_init - x_center_d.pw;
					for (nc = 0; nc < DOF_reaction; ++nc)
					{
						x_left_d.cw[nc] = 2 * c_BC_L - x_center_d.cw[nc];
					}
				}
				else
				{
					InitDualField(&x_left_d, &x_left, var.pos == VAR_LEFT, var.c);
				}

				if (i == mx - 1)
				{
					x_right_d.pw = -x_center_d.pw;
					for (nc = 0; nc < DOF_reaction; ++nc)
					{
						x_right_d.cw[nc] = 2 * c_BC_R - x_center_d.cw[nc];
					}
				}
				else
				{
					InitDualField(&x_right_d, &x_right, var.pos == VAR_RIGHT, var.c);
				}

				if (j == 0)
				{
					x_bottom_d.pw = x_center_d.pw;
					for (nc = 0; nc < DOF_reaction; ++nc)
					{
						x_bottom_d.cw[nc] = x_center_d.cw[nc];
					}
				}
				else
				{
					InitDualField(&x_bottom_d, &x_bottom, var.pos == VAR_BOTTOM, var.c);
				}

				if (j == my - 1)
				{
					x_top_d.pw = x_center_d.pw;
					for (nc = 0; nc < DOF_reaction; ++nc)
					{
						x_top_d.cw[nc] = x_center_d.cw[nc];
					}
				}
				else
				{
					InitDualField(&x_top_d, &x_top, var.pos == VAR_TOP, var.c);
				}

				ReactionFieldT<autodiff::dual> res_cw;
				autodiff::dual res_pw;
				ComputeResidualAtCell(x_center_d, x_left_d, x_right_d, x_bottom_d, x_top_d,
									  _equilibrium_constants[j][i], _mass_frac_old[j][i], reaction_rate_center,
									  phi[j][i].xx[0], rho_old_center, Kxx_left, Kxx_center, Kxx_right,
									  Kyy_bottom, Kyy_center, Kyy_top, dx, dy, tsctx, &res_cw, &res_pw);

				MatStencil col;
				col.i = var.i;
				col.j = var.j;
				col.k = 0;
				col.c = var.c;

				MatStencil row;
				row.i = i;
				row.j = j;
				row.k = 0;

				PetscScalar val;
				row.c = 0;
				val = static_cast<PetscScalar>(autodiff::derivative(res_pw));
				ierr = MatSetValuesStencil(P, 1, &row, 1, &col, &val, INSERT_VALUES);
				CHKERRQ(ierr);
				if (J != P)
				{
					ierr = MatSetValuesStencil(J, 1, &row, 1, &col, &val, INSERT_VALUES);
					CHKERRQ(ierr);
				}

				for (nc = 0; nc < DOF_reaction; ++nc)
				{
					row.c = nc + 1;
					val = static_cast<PetscScalar>(autodiff::derivative(res_cw.reaction[nc]));
					ierr = MatSetValuesStencil(P, 1, &row, 1, &col, &val, INSERT_VALUES);
					CHKERRQ(ierr);
					if (J != P)
					{
						ierr = MatSetValuesStencil(J, 1, &row, 1, &col, &val, INSERT_VALUES);
						CHKERRQ(ierr);
					}
				}
			}
		}
	}

	ierr = DMDAVecRestoreArray(da_reaction, user->_sec_conc_old, &_sec_conc_old);
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
	ierr = DMDAVecRestoreArray(da, loc_Xold, &xold);
	CHKERRQ(ierr);
	ierr = DMRestoreLocalVector(da, &loc_X);
	CHKERRQ(ierr);
	ierr = DMRestoreLocalVector(da, &loc_Xold);
	CHKERRQ(ierr);

	ierr = MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY);
	CHKERRQ(ierr);
	ierr = MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY);
	CHKERRQ(ierr);
	if (J != P)
	{
		ierr = MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
		CHKERRQ(ierr);
		ierr = MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);
		CHKERRQ(ierr);
	}

	PetscFunctionReturn(0);
}
