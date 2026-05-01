#pragma once

// SharedMath::ML — umbrella header
// Usage: #include <ML/ml.h>

// Autograd engine
#include "AutogradTensor.h"

// Neural-network layers
#include "Module.h"

// Optimizers
#include "Optimizer.h"

// Losses
#include "Loss.h"

// Classical ML models
#include "Models.h"

// Regularized linear models
#include "RegularizedModels.h"

// Decision trees & random forests
#include "TreeModels.h"

// Naive Bayes
#include "NaiveBayes.h"

// SVM
#include "SVM.h"

// Dataset / DataLoader
#include "DataLoader.h"

// Training loop
#include "Trainer.h"

// Preprocessing utilities
#include "Preprocessing.h"

// Dimensionality reduction (PCA alias)
#include "DimensionalityReduction.h"

// Model selection (KFold, StratifiedKFold, cross_val_score)
#include "ModelSelection.h"

// Metrics
#include "Metrics.h"
