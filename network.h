// define ENABLE_GPU to enable gpu

// use -mavx2 -mfma to enable AVX256

// define ENABLE_OPENMP and use -fopenmp to enable openmp, 
//           and use THREAD_NUM to control the count of threads

// define ENABLE_AUTO_SL to enable auto save/load/delthis(see details in auto_saveload.h)

#pragma once

#include "defines.h"

#include "Layers/fc.h"
#include "Layers/bias.h"
#include "Layers/conv.h"
#include "Layers/deconv.h"
#include "Layers/pooling.h"
#include "Layers/ext.h"

#include "Layers/bn.h"
#include "Layers/gn.h"
#include "Layers/Softmax.h"

#include "Layers/ReLU.h"
#include "Layers/Leaky_ReLU.h"
#include "Layers/Sigmoid.h"
#include "Layers/Tanh.h"

#include "Optimizer/SGD.h"
#include "Optimizer/Adam.h"
