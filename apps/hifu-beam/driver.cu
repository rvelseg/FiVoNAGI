//=======================================================================
//
// Name : Finite Volume Nonlinear Acoustics GPU Implementation (FiVoNAGI)
//
// Authors : Roberto Velasco Segura and Pablo L. Rend\'on
//
// License : see licence.txt in the root directory of the repository.
//
//=======================================================================

// For comparison with CLAWPACK see: clawpack-4.6.1/apps/acoustics/2d/example1/driver.f
// and clawpack-4.6.1/apps/acoustics/2d/example1/Makefile

#define EXPAND1(x) x
#define EXPAND2(x) #x
#define CONCAT_STR(x, y) EXPAND1(x)y
#define QUOTE(x) EXPAND2(x)

#ifndef ROOT
# error ROOT variable not set.
#else
#define ROOT_PATH(FILE) QUOTE(CONCAT_STR(EXPAND1(ROOT), EXPAND1(FILE)))
#endif /* ROOT */

// use 1 for single precision and 2 for double precision
#ifndef PRECISION
#define PRECISION 1
#endif /* PRECISION */

#if PRECISION == 1 
#define DATATYPEV float
#define DATATYPET float
#elif PRECISION == 2
#define DATATYPEV double
#define DATATYPET int2
#else /* PRECISION value */
# error unresolved PRECISION value
#endif /* PRECISION value */

#include ROOT_PATH(/common/numbers.h)
#include "parameters.h"
#include ROOT_PATH(/common/data_definitions.h)
#include "init.h"
#include "boundary.h"
#include ROOT_PATH(/common/filter.h)
#include "draw_float_cut.h"
#include "data_export.h"
#include "data_collect.h"
#include "../na/source.h"
#include "../na/fv.h"

#include ROOT_PATH(/common/main.cu)
