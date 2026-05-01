#pragma once

#include <sharedmath_ml_export.h>

#include "DataLoader.h"
#include "Loss.h"
#include "Module.h"
#include "Optimizer.h"

#include <cstddef>
#include <vector>

namespace SharedMath::ML {

class SHAREDMATH_ML_EXPORT Trainer {
public:
    Trainer(Module& model, Optimizer& optimizer, Loss& loss);

    std::vector<double> fit(DataLoader& loader, size_t epochs);
    double train_epoch(DataLoader& loader);

private:
    Module& m_model;
    Optimizer& m_optimizer;
    Loss& m_loss;
};

} // namespace SharedMath::ML
