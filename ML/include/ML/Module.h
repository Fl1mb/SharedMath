#pragma once

/**
 * @file Module.h
 * @brief Neural-network layer API.
 *
 * @defgroup ML_Layers Neural Network Layers
 * @ingroup ML
 * @{
 *
 * All layers inherit from @ref SharedMath::ML::Module.  The two key virtual
 * methods are `forward()` and `parameters()`.  The base class provides
 * `train()`, `eval()`, `zero_grad()`, `save()`, and `load()`.
 *
 * **Available layers**
 * - Linear — fully-connected layer (y = x W + b)
 * - Sequential — ordered container of modules
 * - Dropout — inverted-scaling training dropout
 * - Flatten — reshape feature dimensions for Linear
 * - ReLU, LeakyReLU, ELU, SiLU, GELU, Sigmoid, Tanh, Softmax — activations
 * - LayerNorm — normalisation over the last dimension
 * - Embedding — lookup table for token IDs
 * - MultiHeadAttention — scaled dot-product self-attention
 * - Conv2d — NCHW 2-D convolution with backward pass
 * - BatchNorm1d / BatchNorm2d — batch normalisation
 * - MaxPool2d / AvgPool2d — spatial pooling
 *
 * @}
 */

#include <sharedmath_ml_export.h>

#include "AutogradTensor.h"
#include "LinearAlgebra/Tensor.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace SharedMath::ML {

using Tensor = SharedMath::LinearAlgebra::Tensor;

class SHAREDMATH_ML_EXPORT Module {
public:
    virtual ~Module() = default;

    virtual AutoTensor forward(const AutoTensor& x) = 0;
    AutoTensor operator()(const AutoTensor& x);

    virtual std::vector<AutoTensor*> parameters();

    virtual void train(bool mode = true);
    void eval();
    bool is_training() const noexcept;

    void zero_grad();
    virtual void to(SharedMath::LinearAlgebra::Device device,
                    int device_id = -1);
    void cuda(int device_id = -1);
    void cuda_auto();
    void cpu();
    void save(const std::string& path) const;
    void load(const std::string& path);

protected:
    bool m_training = true;
};

class SHAREDMATH_ML_EXPORT Linear : public Module {
public:
    Linear(size_t in_features, size_t out_features, bool use_bias = true);

    AutoTensor forward(const AutoTensor& x) override;
    std::vector<AutoTensor*> parameters() override;

    AutoTensor& weight();
    AutoTensor& bias();
    const AutoTensor& weight() const;
    const AutoTensor& bias() const;

    size_t in_features()  const noexcept;
    size_t out_features() const noexcept;

private:
    size_t     m_in;
    size_t     m_out;
    bool       m_use_bias;
    AutoTensor m_weight;
    AutoTensor m_bias;
};

class SHAREDMATH_ML_EXPORT Sequential : public Module {
public:
    Sequential();
    explicit Sequential(std::vector<std::shared_ptr<Module>> layers);

    void add(std::shared_ptr<Module> layer);

    AutoTensor forward(const AutoTensor& x) override;
    std::vector<AutoTensor*> parameters() override;
    void train(bool mode = true) override;

    size_t size() const noexcept;

private:
    std::vector<std::shared_ptr<Module>> m_layers;
};

class SHAREDMATH_ML_EXPORT Dropout : public Module {
public:
    explicit Dropout(double p = 0.5);

    AutoTensor forward(const AutoTensor& x) override;
    double drop_prob() const noexcept;

private:
    double        m_p;
    std::uint64_t m_seed = 12345;
};

class SHAREDMATH_ML_EXPORT Flatten : public Module {
public:
    explicit Flatten(size_t start_dim = 1);

    AutoTensor forward(const AutoTensor& x) override;

    size_t start_dim() const noexcept;

private:
    size_t m_start_dim;
};

class SHAREDMATH_ML_EXPORT ReLU : public Module {
public:
    AutoTensor forward(const AutoTensor& x) override;
};

class SHAREDMATH_ML_EXPORT Sigmoid : public Module {
public:
    AutoTensor forward(const AutoTensor& x) override;
};

class SHAREDMATH_ML_EXPORT Tanh : public Module {
public:
    AutoTensor forward(const AutoTensor& x) override;
};

class SHAREDMATH_ML_EXPORT GELU : public Module {
public:
    AutoTensor forward(const AutoTensor& x) override;
};

class SHAREDMATH_ML_EXPORT Softmax : public Module {
public:
    explicit Softmax(size_t axis = 1);

    AutoTensor forward(const AutoTensor& x) override;

    size_t axis() const noexcept;

private:
    size_t m_axis;
};

class SHAREDMATH_ML_EXPORT LayerNorm : public Module {
public:
    LayerNorm(size_t normalized_shape, double eps = 1e-5, bool affine = true);

    AutoTensor forward(const AutoTensor& x) override;
    std::vector<AutoTensor*> parameters() override;

    AutoTensor& gamma();
    AutoTensor& beta();
    const AutoTensor& gamma() const;
    const AutoTensor& beta() const;

private:
    size_t     m_normalized_shape;
    double     m_eps;
    bool       m_affine;
    AutoTensor m_gamma;
    AutoTensor m_beta;
};

class SHAREDMATH_ML_EXPORT Embedding : public Module {
public:
    Embedding(size_t num_embeddings, size_t embedding_dim);

    AutoTensor forward(const AutoTensor& indices) override;
    std::vector<AutoTensor*> parameters() override;

    AutoTensor& weight();
    const AutoTensor& weight() const;

private:
    size_t     m_num_embeddings;
    size_t     m_embedding_dim;
    AutoTensor m_weight;
};

class SHAREDMATH_ML_EXPORT MultiHeadAttention : public Module {
public:
    MultiHeadAttention(size_t embed_dim, size_t num_heads);

    AutoTensor forward(const AutoTensor& x) override;
    std::vector<AutoTensor*> parameters() override;

private:
    size_t     m_embed_dim;
    size_t     m_num_heads;
    size_t     m_head_dim;
    AutoTensor m_wq;
    AutoTensor m_wk;
    AutoTensor m_wv;
    AutoTensor m_wo;
};

class SHAREDMATH_ML_EXPORT Conv2d : public Module {
public:
    Conv2d(size_t in_channels,
           size_t out_channels,
           size_t kernel_size,
           size_t stride = 1,
           size_t padding = 0,
           bool use_bias = true);

    AutoTensor forward(const AutoTensor& x) override;
    std::vector<AutoTensor*> parameters() override;

    AutoTensor& weight();
    AutoTensor& bias();
    const AutoTensor& weight() const;
    const AutoTensor& bias() const;

    size_t in_channels() const noexcept;
    size_t out_channels() const noexcept;
    size_t kernel_size() const noexcept;
    size_t stride() const noexcept;
    size_t padding() const noexcept;

private:
    size_t     m_in_channels;
    size_t     m_out_channels;
    size_t     m_kernel_size;
    size_t     m_stride;
    size_t     m_padding;
    bool       m_use_bias;
    AutoTensor m_weight;
    AutoTensor m_bias;
};

class SHAREDMATH_ML_EXPORT BatchNorm1d : public Module {
public:
    BatchNorm1d(size_t num_features,
                double eps = 1e-5,
                double momentum = 0.1,
                bool affine = true);

    AutoTensor forward(const AutoTensor& x) override;
    std::vector<AutoTensor*> parameters() override;
    void to(SharedMath::LinearAlgebra::Device device,
            int device_id = -1) override;

    AutoTensor& gamma();
    AutoTensor& beta();
    const AutoTensor& gamma() const;
    const AutoTensor& beta() const;
    const Tensor& running_mean() const noexcept;
    const Tensor& running_var() const noexcept;

private:
    size_t     m_num_features;
    double     m_eps;
    double     m_momentum;
    bool       m_affine;
    AutoTensor m_gamma;
    AutoTensor m_beta;
    Tensor     m_running_mean;
    Tensor     m_running_var;
};

class SHAREDMATH_ML_EXPORT BatchNorm2d : public Module {
public:
    BatchNorm2d(size_t num_features,
                double eps = 1e-5,
                double momentum = 0.1,
                bool affine = true);

    AutoTensor forward(const AutoTensor& x) override;
    std::vector<AutoTensor*> parameters() override;
    void to(SharedMath::LinearAlgebra::Device device,
            int device_id = -1) override;

    AutoTensor& gamma();
    AutoTensor& beta();
    const AutoTensor& gamma() const;
    const AutoTensor& beta() const;
    const Tensor& running_mean() const noexcept;
    const Tensor& running_var() const noexcept;

private:
    size_t     m_num_features;
    double     m_eps;
    double     m_momentum;
    bool       m_affine;
    AutoTensor m_gamma;
    AutoTensor m_beta;
    Tensor     m_running_mean;
    Tensor     m_running_var;
};

class SHAREDMATH_ML_EXPORT MaxPool2d : public Module {
public:
    MaxPool2d(size_t kernel_size, size_t stride = 0, size_t padding = 0);

    AutoTensor forward(const AutoTensor& x) override;

    size_t kernel_size() const noexcept;
    size_t stride() const noexcept;
    size_t padding() const noexcept;

private:
    size_t m_kernel_size;
    size_t m_stride;
    size_t m_padding;
};

class SHAREDMATH_ML_EXPORT AvgPool2d : public Module {
public:
    AvgPool2d(size_t kernel_size, size_t stride = 0, size_t padding = 0);

    AutoTensor forward(const AutoTensor& x) override;

    size_t kernel_size() const noexcept;
    size_t stride() const noexcept;
    size_t padding() const noexcept;

private:
    size_t m_kernel_size;
    size_t m_stride;
    size_t m_padding;
};

class SHAREDMATH_ML_EXPORT LeakyReLU : public Module {
public:
    explicit LeakyReLU(double negative_slope = 0.01);

    AutoTensor forward(const AutoTensor& x) override;

    double negative_slope() const noexcept;

private:
    double m_negative_slope;
};

class SHAREDMATH_ML_EXPORT ELU : public Module {
public:
    explicit ELU(double alpha = 1.0);

    AutoTensor forward(const AutoTensor& x) override;

    double alpha() const noexcept;

private:
    double m_alpha;
};

class SHAREDMATH_ML_EXPORT SiLU : public Module {
public:
    AutoTensor forward(const AutoTensor& x) override;
};

} // namespace SharedMath::ML
