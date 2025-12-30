#include <petscsnes.h>
#include <petscdmda.h>
#include "def.h"
#include <stdio.h>
#include "petscsys.h"
#include <stdlib.h>  // 包含 abs 函数的头文件
#include <math.h>
#include <float.h>  // 用于DBL_MAX
#include <stdbool.h>
#include <limits.h>
PetscInt _num_reactions=5;
PetscInt num_reactions_1=1;
PetscScalar  _primary_activity_coefficients[] = {1,1,1,1,1};
PetscScalar  _secondary_activity_coefficients[] = {1,1,1,1,1};
PetscScalar  _reactions_Secondary[]  = {1, 1, 0, 0, 0, -1, 1, 0,0, 0,0, 1, 1, 0, 0,0, 1, 0, 1, 0,0, 1, 0, 0, 1};
PetscScalar  _reactions[] = {-2,2,1,0.8,0.2};
PetscScalar  _ref_kcons[] ={3e-4,3e-4,3e-4,3e-4,3e-4};
PetscScalar  _r_area[5]={1.2e-08,1.2e-08,1.2e-08,1.2e-08,1.2e-08};
PetscScalar _eta_exponent[]={1,1,1,1,1};
PetscScalar _molar_volume[]={64365,64365,64365,64365,64365};
PetscInt  _e_act[5]={15000,15000,15000,15000,15000};
#undef __FUNCT__
#define __FUNCT__ "FormFunction"
PetscErrorCode FormFunction(SNES snes,Vec X,Vec F,void *ptr)
{
	PetscErrorCode ierr;
	UserCtx        *user = (UserCtx*) ptr ;
	TstepCtx       *tsctx = user->tsctx;
 	DM             da = user->da,da_reaction=user->da_reaction,da_perm=user->da_perm;
	PetscScalar    dx = user->dx, dy = user->dy;
    PetscScalar    alpha[DOF_reaction];    
	PetscInt       i, j, nc,mx, my, xl, yl, zl, nxl, nyl, nzl, xg, yg, zg, nxg, nyg, nzg;
	PetscScalar    diff,U_L, U_R, U_B, U_T;
	PetscScalar    fluxL, fluxR, fluxB, fluxT;
    PetscScalar    fluxL1, fluxR1, fluxB1, fluxT1;
	PetscScalar	   qn[DOF_reaction]={0,0,0,0,0};
	PetscScalar    stoichiometry[DOF_reaction]={-2,2,1,0.8,0.2};
	bool      _equilibrium_constants_as_log10=false;
	PetscScalar    _mineral_density[DOF_reaction]={2875.0, 2875.0, 2875.0, 2875.0, 2875.0};
	Vec            loc_X, loc_Xold;
	PhysicalField  **x, **f,**xold;
	PermField     **perm,**phi,**phi_old;
    ReactionField **_mineral_sat,**_reaction_rate,**initial_ref,**_equilibrium_constants;
    ReactionField **_sec_conc_old,**_sec_conc,**mineral_conc_old,**_mass_frac,**_mass_frac_old;
	PhysicalField **global_sol,**M,**flag;//ne by haijian
	M            = user->M ;
	flag         = user->flag ;
	global_sol   = user->global_sol ;
	PetscFunctionBegin;
	mx = user->n1; my = user->n2;
	ierr = DMDAGetGhostCorners( da, &xg, &yg, &zg, &nxg, &nyg, &nzg ); CHKERRQ(ierr);
	ierr = DMDAGetCorners( da, &xl, &yl, &zl, &nxl, &nyl, &nzl ); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da_perm,user->phi,&phi ); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da_perm,user->phi_old,&phi_old); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da_perm, user->perm, &perm); CHKERRQ(ierr);


	ierr = DMGetLocalVector( da, &loc_X  ); CHKERRQ(ierr);
	ierr = DMGetLocalVector( da, &loc_Xold ); CHKERRQ(ierr);
	ierr = DMGlobalToLocalBegin(da, X, INSERT_VALUES, loc_X); CHKERRQ(ierr);
	ierr = DMGlobalToLocalEnd(  da, X, INSERT_VALUES, loc_X); CHKERRQ(ierr);
	ierr = DMGlobalToLocalBegin(da, user->Q0, INSERT_VALUES, loc_Xold); CHKERRQ(ierr);
	ierr = DMGlobalToLocalEnd(  da, user->Q0, INSERT_VALUES, loc_Xold); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da, loc_X, &x); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da, F,     &f); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da, loc_Xold, &xold); CHKERRQ(ierr);


	  /*reaction*/
	ierr = DMDAVecGetArray(da_reaction,user->_mineral_sat,&_mineral_sat);CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da_reaction,user->_reaction_rate,&_reaction_rate);CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da_reaction,user->_sec_conc_old,&_sec_conc_old); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da_reaction,user->_sec_conc,&_sec_conc); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da_reaction,user->initial_ref,&initial_ref); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da_reaction,user->mineral_conc_old,&mineral_conc_old); CHKERRQ(ierr);
  	ierr = DMDAVecGetArray(da_reaction,user->eqm_k,&_equilibrium_constants); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da_reaction,user->_mass_frac,&_mass_frac); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da_reaction,user->_mass_frac_old,&_mass_frac_old); CHKERRQ(ierr);

// left boundary (in flow): 0-30m dirichlet boundary = P1 ; else v_x=0 Neumann

	if (xl==0) {
		for ( j=yg; j < yg+nyg; j++ ) {
			for ( i=-WIDTH; i<0; i++ ) {

             x[j][i].pw =2*P_init-x[j][-i-1].pw; 
			 	for (nc = 0; nc < DOF_reaction; ++nc){
             x[j][i].cw[nc] =2*c_BC_L-x[j][-i-1].cw[nc];
				}	
			}
		}
	}
// right boundary (out flow): dirichlet boundary = P2
	if ( xl+nxl==mx ) {
		for (  j=yg; j < yg+nyg; j++ ) {
			for ( i=mx; i<mx+WIDTH; i++ ) {
               x[j][i].pw =-x[j][2*mx-i-1].pw;//
			     for (nc = 0; nc < DOF_reaction; ++nc){
               x[j][mx].cw[nc] =2*c_BC_R-x[j][2*mx-i-1].cw[nc];//
			   			}
			}
		}
	}
// bottom boundary (no flow): v_y=0 Neumann boundary
	if ( yl==0 ) {
		for ( i=xg; i < xg+nxg; i++ ) {
			for ( j=-WIDTH; j<0; j++ ) {
              x[j][i].pw =x[-j-1][i].pw;//
			  for (nc = 0; nc < DOF_reaction; ++nc){
              x[j][i].cw[nc] = x[-j-1][i].cw[nc];
			  }
			}
		}
	}
// top boundary (no flow): v_y=0 Neumann boundary
	if ( yl+nyl==my ) {
		for ( i=xg; i < xg+nxg; i++ ) {
			for ( j=my; j<my+WIDTH; j++ ) {
				x[j][i].pw =x[2*my-j-1][i].pw;//
				for (nc = 0; nc < DOF_reaction; ++nc){
               x[j][i].cw[nc] =x[2*my-j-1][i].cw[nc];//
				}
			}
		}
	}

		for (j = yg; j < yg + nyg; j++)
		{
			for (i = xg; i < xg + nxg; i++)
			{
PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(&_sec_conc[j][i], &_equilibrium_constants[j][i],&_mass_frac[j][i], &x[j][i],  _equilibrium_constants_as_log10,user);
PorousFlowAqueousPreDisChemistry_computeQpReactionRates(reference_temperature_pre,reference_saturation, phi_old[j][i].xx[0], &phi[j][i].xx[0], &_mineral_sat[j][i],  &_reaction_rate[j][i],  &_sec_conc_old[j][i], &_sec_conc[j][i],_equilibrium_constants_as_log10,user,&mineral_conc_old[j][i],&initial_ref[j][i]);
PorousFlowAqueousPreDisMineral_computeQpProperties(reference_saturation,&_sec_conc_old[j][i], &_sec_conc[j][i], &_reaction_rate[j][i],  phi_old[j][i].xx[0],user);

			
      }
    }

#define K_xx(i,j)   ((perm[j][i].xx[0])/(mu))
#define K_yy(i,j)   ((perm[j][i].xx[1])/(mu))

	for ( j=yl; j < yl+nyl; j++) {
		for ( i=xl; i < xl+nxl; i++) {

		diff = 0.5*( dx/(K_xx(i-1,j)) + dx/(K_xx(i,j)));
		U_L = -(( x[j][i].pw - x[j][i-1].pw ))/diff;

       diff = 0.5*( dx/(K_xx(i+1,j)) + dx/(K_xx(i,j)));
	   U_R = -(( x[j][i+1].pw - x[j][i].pw ))/diff;

		diff = 0.5*( dy/(K_yy(i,j-1)) + dy/(K_yy(i,j)));
		U_B = -(( x[j][i].pw - x[j-1][i].pw ))/diff;//

		diff = 0.5*( dy/(K_yy(i,j+1)) + dy/(K_yy(i,j)));
		U_T = -(( x[j+1][i].pw - x[j][i].pw ))/diff;//
        fluxL = rho(i-1,j)*max(U_L,0.0) + rho(i,j)*min(U_L,0.0);
        fluxR = rho(i+1,j)*min(U_R,0.0) + rho(i,j)*max(U_R,0.0);
     	fluxB = rho(i,j-1)*max(U_B,0.0) + rho(i,j)*min(U_B,0.0);
      	fluxT = rho(i,j+1)*min(U_T,0.0) + rho(i,j)*max(U_T,0.0);
	
		for (nc = 0; nc < DOF_reaction; ++nc){
	    fluxL1= _mass_frac[j][i-1].reaction[nc]*rho(i-1,j)*max(U_L,0.0) +_mass_frac[j][i].reaction[nc]*rho(i,j)*min(U_L,0.0);
        fluxR1= _mass_frac[j][i+1].reaction[nc]*rho(i+1,j)*min(U_R,0.0) +_mass_frac[j][i].reaction[nc]*rho(i,j)*max(U_R,0.0);
        fluxB1= _mass_frac[j-1][i].reaction[nc]*rho(i,j-1)*max(U_B,0.0) +_mass_frac[j][i].reaction[nc]*rho(i,j)*min(U_B,0.0);  
	    fluxT1= _mass_frac[j+1][i].reaction[nc]*rho(i,j+1)*min(U_T,0.0) +_mass_frac[j][i].reaction[nc]*rho(i,j)*max(U_T,0.0);
		alpha[nc]=(rho(i,j)*phi[j][i].xx[0]*_mass_frac[j][i].reaction[nc]-rho_old(i,j)*phi[j][i].xx[0]*_mass_frac_old[j][i].reaction[nc])/tsctx->tsize;
	 
		for (int q = 0; q < num_reactions_1; ++q){
        qn[nc]=stoichiometry[nc] * _mineral_density[nc]* _reaction_rate[j][i].reaction[q]*phi[j][i].xx[0];
		}   
        f[j][i].cw[nc] = alpha[nc]+(fluxR1-fluxL1)/dx + (fluxT1-fluxB1)/dy+qn[nc];
       }
	    f[j][i].pw = (fluxR-fluxL)/dx + (fluxT-fluxB)/dy;
		}
	}

		#if 1
	for ( j=yl; j < yl+nyl; j++) {
		for ( i=xl; i < xl+nxl; i++) {
			M[j][i].pw = fabs(f[j][i].pw);
				for (nc = 0; nc < DOF_reaction; ++nc){
			   M[j][i].cw[nc] = fabs(f[j][i].cw[nc]);
				}

		}
	}

#endif

   ierr = DMDAVecRestoreArray(da_reaction,user->_mineral_sat,&_mineral_sat);CHKERRQ(ierr);
   ierr = DMDAVecRestoreArray(da_reaction,user->_reaction_rate,&_reaction_rate);CHKERRQ(ierr);
   ierr = DMDAVecRestoreArray(da_reaction,user->_sec_conc_old,&_sec_conc_old); CHKERRQ(ierr);
   ierr = DMDAVecRestoreArray(da_reaction,user->_sec_conc,&_sec_conc); CHKERRQ(ierr);
   ierr = DMDAVecRestoreArray(da_reaction,user->initial_ref,&initial_ref); CHKERRQ(ierr);
   ierr = DMDAVecRestoreArray(da_reaction,user->mineral_conc_old,&mineral_conc_old); CHKERRQ(ierr);
   ierr = DMDAVecRestoreArray(da_reaction,user->eqm_k,&_equilibrium_constants); CHKERRQ(ierr);
   ierr = DMDAVecRestoreArray(da_reaction,user->_mass_frac,&_mass_frac); CHKERRQ(ierr);
   ierr = DMDAVecRestoreArray(da_reaction,user->_mass_frac_old,&_mass_frac_old); CHKERRQ(ierr);


	ierr = DMDAVecRestoreArray(da_perm, user->perm, &perm); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(da_perm,user->phi,&phi ); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(da_perm,user->phi_old,&phi_old ); CHKERRQ(ierr);

	ierr = DMDAVecRestoreArray(da, loc_X, &x); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(da, F,     &f); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(da, loc_Xold, &xold); CHKERRQ(ierr);
	ierr = DMRestoreLocalVector(da, &loc_X ); CHKERRQ(ierr);
	ierr = DMRestoreLocalVector(da, &loc_Xold ); CHKERRQ(ierr);

	
	PetscFunctionReturn(0);
}


PetscErrorCode  PorousFlowAqueousPreDisChemistry_computeQpReactionRates(double temp, double Saturation, PetscScalar phi_old,PetscScalar *phi, ReactionField *_mineral_sat, ReactionField *_reaction_rate, ReactionField *_sec_conc_old, ReactionField *_sec_conc, bool _equilibrium_constants_as_log10,void *ptr,ReactionField *mineral_conc_old,ReactionField *initial_ref){
	UserCtx        *user = (UserCtx*) ptr ;
	TstepCtx       *tsctx   = user->tsctx;
	bool          _bounded_rate[DOF_reaction];
	PetscScalar gamp ,fac,sgn ,unbounded_rr, por_times_rr_dt ;
	PetscScalar      _primary_activity_coefficient[DOF_reaction],_primary[DOF_reaction],_theta_exponent[DOF_reaction];
  PetscFunctionBeginUser;
  InitializeArray(_primary_activity_coefficient,  DOF_reaction);
  InitializeArray(_primary, DOF_reaction);
  InitializeArray(_theta_exponent,  DOF_reaction);
for (int r = 0; r <_num_reactions; ++r) {  
_mineral_sat->reaction[r]  = (_equilibrium_constants_as_log10 ? pow(10.0, -(358 - (temp)) / (358 - 294))
                                             : 1.0 / -(358 -  (temp)) / (358 - 294)); 
					
for (int j = 0; j < DOF_reaction; ++j){	
 gamp = _primary_activity_coefficient[j] * _primary[j];

	if (gamp <= 0.0){
		if (stoichiometry(r, j) < 0.0){
      _mineral_sat->reaction[r]  = DBL_MAX;
		}else if(stoichiometry(r, j) ==0.0){
		_mineral_sat->reaction[r]   = _mineral_sat->reaction[r]*1.0;	

		}else{
		_mineral_sat->reaction[r]   = 0.0;	
		break;
		}
	}else{
	    _mineral_sat->reaction[r]  *= pow(gamp, stoichiometry(r, j));	
	}
  }
	 fac = 1.0 - pow(_mineral_sat->reaction[r] , _theta_exponent[r]);
  
    // if fac > 0 then dissolution occurs; if fac < 0 then precipitation occurs.
    sgn = (fac < 0 ? -1.0 : 1.0);
    unbounded_rr = -sgn * rateConstantQp(r,reference_temperature) * _r_area[r] * _molar_volume[r] *
                            pow(fabs(fac), _eta_exponent[r]);	  
    por_times_rr_dt = phi_old * (Saturation)* unbounded_rr * tsctx->tsize;
	
    if (_sec_conc_old->reaction[r] + por_times_rr_dt > 1.0)
    {

      _bounded_rate[r] = true;
      _reaction_rate->reaction[r] =(1.0 - _sec_conc_old->reaction[r]) / (phi_old *(Saturation) *tsctx->tsize);

    }
    else if (_sec_conc_old->reaction[r] + por_times_rr_dt < 0.0)
    {
      _bounded_rate[r] = true;
      _reaction_rate->reaction[r] =-_sec_conc_old->reaction[r] / phi_old/ (Saturation)/tsctx->tsize;

    }
    else
    {
      _bounded_rate[r] = false;
      _reaction_rate->reaction[r]= unbounded_rr;

    }
    }
//PorousFlowPorosity_atNegInfinityQp(true,phi,_reaction_rate,mineral_conc_old,initial_ref, phi_old, Saturation,user);



	PetscFunctionReturn(0);
}



PetscScalar  stoichiometry(PetscInt reaction_num, PetscInt primary_num)
{
 const int index = reaction_num * DOF_reaction + primary_num;
  return _reactions[index];
}
PetscScalar  stoichiometry_Secondary(PetscInt reaction_num, PetscInt primary_num)
{
 const int index = reaction_num * DOF_reaction + primary_num;
  return _reactions_Secondary[index];
}


PetscScalar rateConstantQp(int reaction_num, PetscScalar temp) 
{

 return _ref_kcons[reaction_num] * exp(_e_act[reaction_num] / _gas_const *
                                              (_one_over_ref_temp - 1.0 / temp));
}
void PorousFlowAqueousPreDisMineral_computeQpProperties(double _saturation,ReactionField *_sec_conc_old, ReactionField *_sec_conc, ReactionField *_reaction_rate, double _porosity_old,void *ptr)
{

	UserCtx        *user = (UserCtx*) ptr ;
	TstepCtx       *tsctx   = user->tsctx;
    for (int r = 0; r < DOF_reaction; ++r){
    _sec_conc->reaction[r] = _sec_conc_old->reaction[r] + _porosity_old * _reaction_rate->reaction[r] *
                                                    _saturation * tsctx->tsize;
                                                  
    }
}


void InitializeArray(PetscScalar *array, PetscInt size) {
    for (PetscInt i = 0; i < size; i++) {
        array[i] = 1.0;
    }

}

void PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(ReactionField *_sec_conc, ReactionField *_equilibrium_constants,ReactionField *_mass_frac, PhysicalField *x, bool _equilibrium_constants_as_log10,void *ptr)
{
	UserCtx        *user = (UserCtx*) ptr ;
	TstepCtx       *tsctx   = user->tsctx;
	bool           isRestarting=false;
  // Compute the secondary concentrations
    if ((tsctx->tcurr== 0)&&(isRestarting)){
    PorousFlowMassFractionAqueousEquilibriumChemistry_initQpSecondaryConcentrations(_sec_conc);
	} else{
    PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpSecondaryConcentrations(_sec_conc, _equilibrium_constants,x,  _equilibrium_constants_as_log10);
    }
	 for (int i = 0; i < DOF_reaction; ++i)
  {

    _mass_frac->reaction[i] =x->cw[i];
 
    for (int r = 0; r < _num_reactions; ++r){

      _mass_frac->reaction[i] = _mass_frac->reaction[i]+  stoichiometry_Secondary(r, i) * _sec_conc->reaction[r];
      }

    // remove mass-fraction from the H20 component
   // _mass_frac[_qp][_aq_ph][_num_components - 1] -= _mass_frac[_qp][_aq_ph][i];
  }


}
void PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpSecondaryConcentrations(ReactionField *_sec_conc, ReactionField *_equilibrium_constants, PhysicalField *x,bool _equilibrium_constants_as_log10){
  for (int r = 0; r < _num_reactions; ++r)
  {
    _sec_conc->reaction[r] = 1.0;
    for (int i = 0; i < DOF_reaction; ++i)
    {
      double gamp = _primary_activity_coefficients[i] * x->cw[r];

      if (gamp <= 0.0)
      {
        if (stoichiometry_Secondary(r, i) < 0.0){
          _sec_conc->reaction[r] = LONG_MAX;
        }else if (stoichiometry_Secondary(r, i) == 0.0){
          _sec_conc->reaction[r] *= 1.0;
		  				
       } else
        {
          _sec_conc->reaction[r] = 0.0;
          break;
        }
      }
      else{
        _sec_conc->reaction[r] *= pow(gamp, stoichiometry_Secondary(r, i));
        }
    }
    _sec_conc->reaction[r] *=
        (_equilibrium_constants_as_log10 ? pow(10.0, (_equilibrium_constants->reaction[r]))
                                         : (_equilibrium_constants->reaction[r]));
    _sec_conc->reaction[r] /= _secondary_activity_coefficients[r];
  }
}
void PorousFlowMassFractionAqueousEquilibriumChemistry_initQpSecondaryConcentrations(ReactionField *_sec_conc)
{
for (int r = 0; r < _num_reactions; ++r)
  {
  _sec_conc->reaction[r]=0.0;
  }
}




