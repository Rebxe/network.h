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

#include "fast_calc.h"
#include "file_io.h"
#include "auto_dao.h"

#ifdef ENABLE_AUTO_SL
	#include "auto_saveload.h"
#endif

typedef std::tuple<int,int,int> SHAPE3D; 

#define MAX_POOLING    1
#define MEAN_POOLING   2

#define INIT_HE       1
#define INIT_XAVIER   2
