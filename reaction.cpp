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

PetscScalar  _primary_activity_coefficients[] = {1,1,1,1,1};
PetscScalar  _secondary_activity_coefficients[] = {1,1,1,1,1};
PetscScalar  _reactions_Secondary[]  = {1, 1, 0, 0, 0, -1, 1, 0,0, 0,0, 1, 1, 0, 0,0, 1, 0, 1, 0,0, 1, 0, 0, 1};
PetscScalar  _reactions[] = {-2,2,1,0.8,0.2};
PetscScalar  _ref_kcons[] ={3e-4,3e-4,3e-4,3e-4,3e-4};
PetscScalar  _r_area[5]={1.2e-08,1.2e-08,1.2e-08,1.2e-08,1.2e-08};
PetscScalar _eta_exponent[]={1,1,1,1,1};
PetscScalar _molar_volume[]={64365,64365,64365,64365,64365};
PetscInt  _e_act[5]={15000,15000,15000,15000,15000};
double    stoichiometry[DOF_reaction]={-2,2,1,0.8,0.2};
bool      _equilibrium_constants_as_log10=false;
double    _mineral_density[DOF_reaction]={2875.0, 2875.0, 2875.0, 2875.0, 2875.0};


bool           isRestarting=false;
PetscErrorCode  PorousFlowAqueousPreDisChemistry_computeQpReactionRates(double temp, double Saturation, PetscScalar phi_old,PetscScalar *phi, ReactionField *_mineral_sat, ReactionField *_reaction_rate, ReactionField *_sec_conc_old, ReactionField *_sec_conc, bool _equilibrium_constants_as_log10,void *ptr,ReactionField *initial_ref){
	UserCtx        *user = (UserCtx*) ptr ;
	TstepCtx       *tsctx   = user->tsctx;
	bool          _bounded_rate[DOF_reaction];
	PetscScalar gamp ,fac,sgn ,unbounded_rr, por_times_rr_dt ;
	PetscScalar      _primary_activity_coefficient[DOF_reaction],_primary[DOF_reaction],_theta_exponent[DOF_reaction];
  PetscFunctionBeginUser;
  InitializeArray(_primary_activity_coefficient,  DOF_reaction);
  InitializeArray(_primary, DOF_reaction);
  InitializeArray(_theta_exponent,  DOF_reaction);
for (int r = 0; r <DOF_reaction; ++r) {  
_mineral_sat->reaction[r]  = (_equilibrium_constants_as_log10 ? pow(10.0, -(358 - (temp)) / (358 - 294))
                                             : 1.0 / -(358 -  (temp)) / (358 - 294)); 
					
for (int j = 0; j < DOF_reaction; ++j){	
 gamp = _primary_activity_coefficient[j] * _primary[j];

	if (gamp <= 0.0){
		if (stoichiometry_primary(r, j) < 0.0){
      _mineral_sat->reaction[r]  = DBL_MAX;
		}else if(stoichiometry_primary(r, j) ==0.0){
		_mineral_sat->reaction[r]   = _mineral_sat->reaction[r]*1.0;	

		}else{
		_mineral_sat->reaction[r]   = 0.0;	
		break;
		}
	}else{
	    _mineral_sat->reaction[r]  *= pow(gamp, stoichiometry_primary(r, j));	
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



PetscScalar  stoichiometry_primary(PetscInt reaction_num, PetscInt primary_num)
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

 
    if ((tsctx->tcurr== 0)&&(isRestarting)){
    PorousFlowMassFractionAqueousEquilibriumChemistry_initQpSecondaryConcentrations(_sec_conc);
	} else{
    PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpSecondaryConcentrations(_sec_conc, _equilibrium_constants,x,  _equilibrium_constants_as_log10);
    }
	 for (int i = 0; i < DOF_reaction; ++i)
  {

    _mass_frac->reaction[i] =x->cw[i];
 
    for (int r = 0; r < DOF_reaction; ++r){

      _mass_frac->reaction[i] = _mass_frac->reaction[i]+  stoichiometry_Secondary(r, i) * _sec_conc->reaction[r];
      }

  }


}
void PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpSecondaryConcentrations(ReactionField *_sec_conc, ReactionField *_equilibrium_constants, PhysicalField *x,bool _equilibrium_constants_as_log10){
  for (int r = 0; r < DOF_reaction; ++r)
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
for (int r = 0; r < DOF_reaction; ++r)
  {
  _sec_conc->reaction[r]=0.0;
  }
}




