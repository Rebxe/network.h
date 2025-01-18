#pragma once

#include <cstring>
#include <ctime>
#include <fstream>
#include <random>
#include <cfloat>
#include <tuple>
#include <assert.h>
#include <cmath>
#include <type_traits>

#ifdef ENABLE_OPENMP
	#include <omp.h>
#endif

/***************** Begin of definitions***************/

typedef std::tuple<int,int,int> SHAPE3D;

#define ext_assert(_Expression,_Fail_Code) assert((_Expression)||((_Fail_Code),false))

#define INIT_HE       1
#define INIT_XAVIER   2

/***************** End of definitions***************/

#include "fast_calc.h"
#include "file_io.h"
#include "auto_dao.h"

#ifdef ENABLE_AUTO_SL
	#include "auto_saveload.h"
#endif
