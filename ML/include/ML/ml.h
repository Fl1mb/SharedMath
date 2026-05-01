#pragma once

/**
 * @file ml.h
 * @brief Umbrella header — include this single file to access all ML features.
 *
 * @defgroup ML Machine Learning Module
 *
 * The ML module provides a complete, sklearn-like machine-learning framework
 * built on top of `SharedMath::LinearAlgebra::Tensor`:
 *
 * - **Autograd** — dynamic computation graph with `AutoTensor` (@ref ML_Autograd)
 * - **Layers** — `Linear`, `Conv2d`, `Sequential`, activations, norm layers (@ref ML_Layers)
 * - **Optimizers** — `SGD`, `Adam`, `AdamW`, … (@ref ML_Optimizers)
 * - **Losses** — `MSELoss`, `CrossEntropyLoss`, `BCELoss`, … (@ref ML_Losses)
 * - **Classical models** — regression, classification, clustering, trees, forests
 * - **Preprocessing** — `StandardScaler`, `MinMaxScaler`, `train_test_split` (@ref ML_Preprocessing)
 * - **Model selection** — `KFold`, `StratifiedKFold`, `cross_val_score` (@ref ML_ModelSelection)
 * - **Metrics** — accuracy, F1, MSE, R², silhouette_score, … (@ref ML_Metrics)
 */

/// SharedMath::ML — umbrella header
/// Usage: #include <ML/ml.h>

/// Autograd engine
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
