#ifndef My_headfile
// aspin
#define My_headfile
#include <map>
#include <vector>

#define Smax 1.5
#define EXAMPLE 3
#define TORDER 2
#define limiter 0 // slope limiter
#define MATTYPE 0 // 0: aij, 1: baij
#define MATFD 0
#define NE_TYPE 1
#define WIDTH 1
#define WIDTH_perm 2
#if EXAMPLE == 1||EXAMPLE == 3
#define DOF 6
#define DOF_reaction 5
#define DOF_perm 2
#define DOF_Secondary 5
#define DOF_mineral 0
#define DOF_Preparation 1
#elif EXAMPLE == 2
#define DOF 5
#define DOF_reaction 4
#define DOF_Preparation 0
#define DOF_perm 3
#define DOF_Secondary 6
#define DOF_mineral 1
#endif
#if EXAMPLE == 1||EXAMPLE==3
#define rho_old(i, j)                                              \
    (rho_init * PetscExpReal(user->xold[j][i].pw / _bulk_modulus - \
                             _thermal_expansion * temp_ref))
#define phi_old(i, j) (phi_init)
#define rho(i, j) \
    (rho_init *   \
     PetscExpReal(x[j][i].pw / _bulk_modulus - _thermal_expansion * temp_ref))
#define phi(i, j) (phi_init)
#elif EXAMPLE == 2
#define rho_old(i, j) (rho_init)
#define phi_old(i, j) (phi_init)
#define rho(i, j)  (rho_init)
#define phi(i, j) (phi_init)
#endif
#define UNIT_L (0.3048)
#define UNIT_T (1.0)
#define UNIT_VIS (1e-3) 
#define UNIT_MD   (9.869233e-16)

#if EXAMPLE == 1 
#define L1 (1)
#define L2 (0.25)
#define _bulk_modulus (2.0e9)
#define _thermal_expansion (2.14e-4)
#define reference_temperature (298.15)
#define reference_temperature_pre (326.2)
#define reference_saturation (1)
#define temp_ref (298.15)
#define biot (1.0)
#define _gas_const (8.314)
#define _one_over_ref_temp (1 / 298.15)
#define kinetic_rate_constant (3e-4)
#define reference_chemistry (0.1)
#define P_BC_L (2 * 100 * 1e6)
#define c_BC_L (5e-2)
#define c_BC_R (1.e-6)
#define P_init (1.e+6)    // psi
#define rho_init (1000.0) // 56 lb/cu-ft =897.033948kg/m^3
#define phi_init (0.1)
#define c_init (0.0)
#define mu (1e-3)
#define N1 100
#define N2 100
#define TSIZE 0.01
#elif EXAMPLE == 2
#define L1 (1)
#define L2 (0.25)
#define T_final 1
#define _weight (2.0)
#define _one_over_ref_temp (1 / 298.15)
#define P_bottom (0)
#define P_left (60)
#define reference_saturation (1)
#define phi_init (0.2)
#define P_init (0)
#define _bulk_modulus (2.0e9)
#define _thermal_expansion (2.14e-4)
#define mu (1)
#define temp_ref (298.15)
#define reference_temperature (298.15)
#define reference_temperature_pre (298.15)
#define _gas_const (8.314)
#define rho_init (1)
#define N1 100
#define N2 100
#define TSIZE 0.01
#elif EXAMPLE==3
#define L1 (220*UNIT_L)
#define L2 (120*UNIT_L)
#define _bulk_modulus (2.0e9)
#define _thermal_expansion (2.14e-4)
#define reference_temperature (298.15)
#define reference_temperature_pre (326.2)
#define reference_saturation (1)
#define temp_ref (298.15)
#define biot (1.0)
#define _gas_const (8.314)
#define _one_over_ref_temp (1 / 298.15)
#define kinetic_rate_constant (3e-4)
#define reference_chemistry (0.1)
#define c_BC_L (5e-2)
#define c_BC_R (1.e-6)
#define P_init (1.e+6)   
#define rho_init (1000.0) 
#define c_init (0.0)
#define mu (1e-3)
#define N1 100
#define N2 100
#define TSIZE 0.01
#endif

#define MATFIX 0

#define TSTART 0.0
#define TFINAL 0.5
#define TSMAX 1

/* fixed parameters */
#define EPS (1.0E-13)
#define EPS00 (0.0)
#define pi (PETSC_PI)
#define ee (2.718281828459)
// reaction.h

#define FLOP_FABS (1)
#define FLOP_SQRT (5)
#define FLOP_SIN (5)
#define FLOP_COS (5)
#define FLOP_TAN (5)
#define FLOP_ASIN (5)
#define FLOP_ATAN2 (5)

#define ATOL 1.e-5      // safeguard, TEST!!!!!!
#define SNES_atol 1.e-8 // TODO
#define SNES_rtol 1.e-6 // TODO
#define KSP_atol 1.e-9  // TODO
#define KSP_rtol 1.e-3  // TODO

#define sign(a) ((a) > 0 ? (1.0) : ((a) < 0 ? (-1.0) : (0.0)))
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define pow2(a) ((a) * (a))
#define pow3(a) ((a) * (a) * (a))
#define pow4(a) ((a) * (a) * (a) * (a))
#define pow5(a) ((a) * (a) * (a) * (a) * (a))
#define pow6(a) (pow3(a) * pow3(a))
#define pow7(a) (pow4(a) * pow3(a))
#define pow8(a) (pow5(a) * pow3(a))
#define pow9(a) (pow5(a) * pow4(a))
#define pow10(a) (pow5(a) * pow5(a))

typedef struct
{
    PetscInt tscurr;    // the count number of the current time step
    PetscInt tsstart;   // start time step (default: 0)
    PetscInt tsmax;     // the total number of time steps we wish to take
    PetscInt tsback;    // backups solution for later restarts
    PetscInt tscomp;    // compare with true solution every CompSteps step
    PetscScalar tsize;  // the size of (\Delta t)
    PetscScalar tstart; // start time
    PetscScalar tfinal; // final time
    PetscScalar tcurr;  // the current time accumulation (in Alfven units)
    PetscInt torder;    // order of temporal discretization
    PetscScalar fnorm;  // holder for current L2-norm of the nonlinear function
    PetscScalar p, smax;
} TstepCtx; // time stepping related information ( CTX - context )

typedef struct
{
    PetscLogEvent func, jac, func_apply, mpi;
} EventCtx;

typedef struct
{
    PetscScalar pw, cw[DOF_reaction];
} PhysicalField;
typedef struct
{
    PetscScalar xx[DOF_perm];
} PermField;
typedef struct
{
    PetscScalar reaction[DOF_reaction];
} ReactionField;

typedef struct
{
    PetscScalar reaction_secondary[DOF_Secondary];
} SecondaryReactionField;
#if DOF_mineral != 0
typedef struct
{
    PetscScalar reaction_mineral[DOF_mineral];
} MineralField;
#endif
typedef struct
{
    PetscInt matfd, mattype, matfree, matfix;
    PetscBool PetscPreLoading;
    PetscBool use_adaptive_dt;
    PetscReal global_nonlinear_atol, local_stop_atol;
    PetscReal hj_fnorm;
    PhysicalField rho;
    PetscInt hj_max_nit;
} ParaCtx; 

typedef struct
{
    TstepCtx *tsctx;
    ParaCtx *param;
    EventCtx *event;
    PetscInt n1, n2;
    PetscScalar dx, dy;
    Vec myF, sol, sol_mineral, sol_mineral_old;
    DM da, da_reaction, da_perm, da_secondary, da_mineral;
    SNES snes;
    PermField **phi_field, **perm_field, **phi_old_field;
    ReactionField **initial_ref_field, **_mass_frac_old_field;
    SecondaryReactionField **eqm_k_field, **_sec_conc_old_field;
#if EXAMPLE==2
    MineralField **eqm_k_mineral_field;
#endif
    PhysicalField **xold;
} UserCtx;

EXTERN_C_BEGIN

extern MPI_Comm comm;
extern PetscMPIInt rank, size;
extern PetscViewer viewer;

extern PetscErrorCode DataSaveASCII(Vec x, char *filename);
extern PetscErrorCode DataSaveBin(Vec x, char *filename);
extern PetscErrorCode DataLoadBin(Vec x, char *filename);
extern PetscErrorCode MY_ApplyFunction(PhysicalField **x, PhysicalField **f,
                                       PermField **perm, Vec loc_X, void *ptr);
extern PetscErrorCode FormInitialValue(void *);
// extern PetscErrorCode FormFunction(DMDALocalInfo* ,PetscScalar**
// ,PetscScalar** ,void*);
extern PetscErrorCode FormFunction(SNES, Vec, Vec, void *);
extern PetscErrorCode FormJacobian(SNES, Vec, Mat, Mat, void *);
extern PetscErrorCode Update(void *);
extern PetscErrorCode FormResidual(SNES, Vec, Vec, void *);
extern PetscErrorCode ComputeParameter(Vec, Vec, void *);
extern PetscErrorCode FormBounds(SNES, Vec, Vec);
extern PetscErrorCode PorousFlowAqueousPreDisChemistry(
    Vec X, bool _equilibrium_constants_as_log10, PetscInt _num_reactions,
    PetscInt _num_primary, PhysicalField **_reaction_rate, void *ptr);
PetscScalar stoichiometry_primary(PetscInt reaction_num, PetscInt primary_num);
extern PetscScalar rateConstantQp(int reaction_num, PetscScalar temp);
extern void computeQpProperties(PetscInt _num_reactions, PermField **phi_old,
                                PhysicalField **x,
                                PhysicalField **_sec_conc_old,
                                PhysicalField **_sec_conc,
                                PhysicalField **_reaction_rate, void *ptr);
extern PetscErrorCode updataprosity(void *ptr, Vec _sec_conc_old, Vec _sec_conc,
                                    Vec _mass_frac_old1, Vec _mass_frac_1);
extern PetscErrorCode computeQpSecondaryConcentrations(
    bool _equilibrium_constants_as_log10, PetscInt _num_reactions,
    PetscInt _num_primary, PhysicalField **x, Vec _sec_conc,
    PetscScalar *reactions, void *ptr);
extern PetscErrorCode computeQpProperties_Chemical(PetscInt _num_reactions,
                                                   PetscInt _num_primary,
                                                   PhysicalField **x,
                                                   void *ptr);
PetscErrorCode PorousFlowAqueousPreDisChemistry_computeQpReactionRates(
    double temp, double Saturation, PetscScalar phi_old, PetscScalar *phi,
    ReactionField *_mineral_sat, ReactionField *_reaction_rate,
    SecondaryReactionField *_sec_conc_old, SecondaryReactionField *_sec_conc,
    bool _equilibrium_constants_as_log10, void *ptr,
    ReactionField *initial_ref);
void PorousFlowAqueousPreDisMineral_computeQpProperties(
    double _saturation, SecondaryReactionField *_sec_conc_old, SecondaryReactionField *_sec_conc,
    ReactionField *_reaction_rate, PetscScalar _porosity_old, void *ptr);
PetscScalar rateConstantQp(int reaction_num, PetscScalar temp);
PetscScalar stoichiometry_Secondary(PetscInt reaction_num,
                                    PetscInt primary_num);
PetscErrorCode FormInitialValue_Reaction(void *ptr);
PetscErrorCode FormInitialValue_Perm(void *ptr);
PetscErrorCode FormFunction_subspace(SNES snes, Vec X, Vec F, void *ptr);
void InitializeArray(PetscScalar *array, PetscInt size);
void PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpSecondaryConcentrations(
    SecondaryReactionField *_sec_conc, SecondaryReactionField *_equilibrium_constants,
    PhysicalField *x, bool _equilibrium_constants_as_log10, double tsize);
void PorousFlowMassFractionAqueousEquilibriumChemistry_initQpSecondaryConcentrations(
    SecondaryReactionField *_sec_conc);
void PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties(
    SecondaryReactionField *_sec_conc, SecondaryReactionField *_equilibrium_constants,
    ReactionField *_mass_frac, PhysicalField *x,
    bool _equilibrium_constants_as_log10, void *ptr, double tsize);
extern PetscErrorCode DetermineNewPartition(void *ptr);
extern PetscErrorCode PorousFlowAqueousPreDisChemistry_subspace(
    Vec X, bool _equilibrium_constants_as_log10, PetscInt _num_reactions,
    PetscInt _num_primary, PhysicalField **_reaction_rate, void *ptr);
PetscScalar stoichiometry_subspace(PetscInt reaction_num, PetscInt primary_num);
extern PetscScalar rateConstantQp_subspace(int reaction_num, PetscScalar temp);
extern void computeQpProperties_subspace(PetscInt _num_reactions,
                                         PermField **phi_old, PhysicalField **x,
                                         PhysicalField **_sec_conc_old,
                                         PhysicalField **_sec_conc,
                                         PhysicalField **_reaction_rate,
                                         void *ptr);
extern PetscErrorCode updataprosit_subspace(void *ptr, Vec _sec_conc_old,
                                            Vec _sec_conc, Vec _mass_frac_old1,
                                            Vec _mass_frac_1);
extern PetscErrorCode computeQpSecondaryConcentrations_subspace(
    bool _equilibrium_constants_as_log10, PetscInt _num_reactions,
    PetscInt _num_primary, PhysicalField **x, Vec _sec_conc,
    PetscScalar *reactions, void *ptr);
extern PetscErrorCode computeQpProperties_Chemical_subspace(
    PetscInt _num_reactions, PetscInt _num_primary, PhysicalField **x,
    void *ptr);
void PorousFlowAqueousPreDisMineral_computeQpProperties_subspace(
    double _saturation, ReactionField *_sec_conc_old, ReactionField *_sec_conc,
    ReactionField *_reaction_rate, PetscScalar _porosity_old, void *ptr);
PetscScalar rateConstantQp_subspace(int reaction_num, PetscScalar temp);
PetscScalar stoichiometry_Secondary_subspace(PetscInt reaction_num,
                                             PetscInt primary_num);
PetscErrorCode FormInitialValue_Reaction_subspace(void *ptr);
PetscErrorCode FormInitialValue_Perm_subspace(void *ptr);
void InitializeArray_subspace(PetscScalar *array, PetscInt size);
void PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpSecondaryConcentrations_subspace(
    SecondaryReactionField *_sec_conc, ReactionField *_equilibrium_constants,
    PhysicalField *x, bool _equilibrium_constants_as_log10);
void PorousFlowMassFractionAqueousEquilibriumChemistry_initQpSecondaryConcentrations_subspace(
    SecondaryReactionField *_sec_conc);
void PorousFlowMassFractionAqueousEquilibriumChemistry_computeQpProperties_subspace(
    SecondaryReactionField *_sec_conc, SecondaryReactionField *_equilibrium_constants,
    ReactionField *_mass_frac, PhysicalField *x,
    bool _equilibrium_constants_as_log10, void *ptr);
PetscErrorCode DataSaveVTK(Vec x, char *filename);
PetscErrorCode Updata_Reaction(void *ptr);
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PhysicalField **x,
                                 PhysicalField **f, void *ptr);
PetscErrorCode FormInitialValue_Perm_local(void *ptr);
PetscErrorCode FormInitialValue_Reaction_local(void *ptr);
PetscErrorCode FormInitialValue_local(void *ptr);
PetscErrorCode CopyOldVector(Vec sol, PhysicalField **xold, void *ptr);
#if EXAMPLE == 2
void KineticDisPreRateAux_All(const PhysicalField * _vals,
                              const MineralField  * _equilibrium_constants_mineral,
                              MineralField * kinetic_rate_out);
#endif
EXTERN_C_END

#endif
