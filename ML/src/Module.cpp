#include "Module.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace SharedMath::ML {
namespace {

size_t checkedPoolOutDim(size_t input, size_t kernel, size_t stride,
                         size_t padding, const char* name)
{
    if (kernel == 0 || stride == 0)
        throw std::invalid_argument(std::string(name) + ": kernel and stride must be > 0");
    const size_t padded = input + 2 * padding;
    if (padded < kernel)
        throw std::invalid_argument(std::string(name) + ": kernel larger than padded input");
    return (padded - kernel) / stride + 1;
}

} // namespace

AutoTensor Module::operator()(const AutoTensor& x) {
    return forward(x);
}

std::vector<AutoTensor*> Module::parameters() {
    return {};
}

void Module::train(bool mode) {
    m_training = mode;
}

void Module::eval() {
    train(false);
}

bool Module::is_training() const noexcept {
    return m_training;
}

void Module::zero_grad() {
    for (auto* p : parameters()) p->zero_grad();
}

void Module::to(SharedMath::LinearAlgebra::Device device, int device_id) {
    for (auto* p : parameters()) {
        p->data() = p->data().to(device, device_id);
        if (p->has_grad())
            p->set_grad(p->grad().to(device, device_id));
    }
}

void Module::cuda(int device_id) {
    to(SharedMath::LinearAlgebra::Device::CUDA, device_id);
}

void Module::cuda_auto() {
    to(SharedMath::LinearAlgebra::Device::CUDA, -1);
}

void Module::cpu() {
    to(SharedMath::LinearAlgebra::Device::CPU);
}

void Module::save(const std::string& path) const {
    auto params = const_cast<Module*>(this)->parameters();
    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw std::runtime_error("Module::save: cannot open file for writing");

    const std::uint64_t count = static_cast<std::uint64_t>(params.size());
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const auto* p : params) {
        Tensor cpu = p->data().cpu();
        const std::uint64_t rank = static_cast<std::uint64_t>(cpu.ndim());
        out.write(reinterpret_cast<const char*>(&rank), sizeof(rank));
        for (size_t d : cpu.shape()) {
            const std::uint64_t dim = static_cast<std::uint64_t>(d);
            out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        }
        const std::uint64_t n = static_cast<std::uint64_t>(cpu.size());
        out.write(reinterpret_cast<const char*>(&n), sizeof(n));
        out.write(reinterpret_cast<const char*>(cpu.data().data()),
                  static_cast<std::streamsize>(n * sizeof(double)));
    }
}

void Module::load(const std::string& path) {
    auto params = parameters();
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("Module::load: cannot open file for reading");

    std::uint64_t count = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (count != params.size())
        throw std::runtime_error("Module::load: parameter count mismatch");

    for (auto* p : params) {
        std::uint64_t rank = 0;
        in.read(reinterpret_cast<char*>(&rank), sizeof(rank));
        Tensor::Shape shape(rank);
        for (size_t i = 0; i < rank; ++i) {
            std::uint64_t dim = 0;
            in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
            shape[i] = static_cast<size_t>(dim);
        }
        std::uint64_t n = 0;
        in.read(reinterpret_cast<char*>(&n), sizeof(n));
        std::vector<double> values(static_cast<size_t>(n));
        in.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(n * sizeof(double)));
        Tensor loaded(shape, std::move(values));
        if (loaded.shape() != p->data().shape())
            throw std::runtime_error("Module::load: parameter shape mismatch");
        p->data() = loaded.to(p->data().device(), p->data().device_id());
    }
}

Linear::Linear(size_t in_features, size_t out_features, bool use_bias)
    : m_in(in_features),
      m_out(out_features),
      m_use_bias(use_bias)
{
    if (in_features == 0 || out_features == 0)
        throw std::invalid_argument("Linear: feature dimensions must be > 0");

    double limit = 1.0 / std::sqrt(static_cast<double>(in_features));
    m_weight = AutoTensor::from(
        Tensor::uniform({in_features, out_features}, -limit, limit, 42),
        true);

    if (use_bias)
        m_bias = AutoTensor::from(Tensor::zeros({1, out_features}), true);
}

AutoTensor Linear::forward(const AutoTensor& x) {
    if (x.data().ndim() != 2)
        throw std::invalid_argument(
            "Linear::forward: input must be 2-D [batch, in_features]");
    if (x.data().dim(1) != m_in)
        throw std::invalid_argument("Linear::forward: input feature dim mismatch");

    auto out = x.matmul(m_weight);
    if (m_use_bias)
        out = out + m_bias;
    return out;
}

std::vector<AutoTensor*> Linear::parameters() {
    if (m_use_bias) return {&m_weight, &m_bias};
    return {&m_weight};
}

AutoTensor& Linear::weight() { return m_weight; }
AutoTensor& Linear::bias() { return m_bias; }
const AutoTensor& Linear::weight() const { return m_weight; }
const AutoTensor& Linear::bias() const { return m_bias; }

size_t Linear::in_features() const noexcept { return m_in; }
size_t Linear::out_features() const noexcept { return m_out; }

Sequential::Sequential() = default;

Sequential::Sequential(std::vector<std::shared_ptr<Module>> layers)
    : m_layers(std::move(layers))
{}

void Sequential::add(std::shared_ptr<Module> layer) {
    m_layers.push_back(std::move(layer));
}

AutoTensor Sequential::forward(const AutoTensor& x) {
    AutoTensor out = x;
    for (auto& layer : m_layers) {
        layer->train(m_training);
        out = layer->forward(out);
    }
    return out;
}

std::vector<AutoTensor*> Sequential::parameters() {
    std::vector<AutoTensor*> all;
    for (auto& layer : m_layers) {
        auto p = layer->parameters();
        all.insert(all.end(), p.begin(), p.end());
    }
    return all;
}

void Sequential::train(bool mode) {
    m_training = mode;
    for (auto& layer : m_layers) layer->train(mode);
}

size_t Sequential::size() const noexcept {
    return m_layers.size();
}

Dropout::Dropout(double p)
    : m_p(p)
{
    if (p < 0.0 || p >= 1.0)
        throw std::invalid_argument("Dropout: p must be in [0, 1)");
}

AutoTensor Dropout::forward(const AutoTensor& x) {
    if (!m_training || m_p == 0.0) return x;

    const double keep = 1.0 - m_p;
    const double scale = 1.0 / keep;

    Tensor mask = Tensor::bernoulli(x.data().shape(), keep, m_seed++);
    Tensor out_data = x.data() * mask * scale;

    if (!x.requires_grad())
        return AutoTensor::from(std::move(out_data));

    auto xi = x.impl();
    return AutoTensor::make_result(std::move(out_data), true,
        [xi, mask, scale](const Tensor& g) {
            xi->propagate(g * mask * scale);
        });
}

double Dropout::drop_prob() const noexcept {
    return m_p;
}

Flatten::Flatten(size_t start_dim)
    : m_start_dim(start_dim)
{}

AutoTensor Flatten::forward(const AutoTensor& x) {
    if (x.data().ndim() == 0)
        throw std::invalid_argument("Flatten::forward: scalar tensors are not supported");
    if (m_start_dim >= x.data().ndim())
        throw std::invalid_argument("Flatten::forward: start_dim out of range");

    Tensor::Shape out_shape;
    out_shape.reserve(m_start_dim + 1);
    for (size_t i = 0; i < m_start_dim; ++i)
        out_shape.push_back(x.data().dim(i));

    size_t flattened = 1;
    for (size_t i = m_start_dim; i < x.data().ndim(); ++i)
        flattened *= x.data().dim(i);
    out_shape.push_back(flattened);

    Tensor out_data = x.data().reshape(out_shape);
    if (!x.requires_grad()) return AutoTensor::from(std::move(out_data));

    auto xi = x.impl();
    const Tensor::Shape original_shape = x.data().shape();
    return AutoTensor::make_result(std::move(out_data), true,
        [xi, original_shape](const Tensor& g) {
            xi->propagate(g.reshape(original_shape));
        });
}

size_t Flatten::start_dim() const noexcept {
    return m_start_dim;
}

AutoTensor ReLU::forward(const AutoTensor& x) {
    return x.relu();
}

AutoTensor Sigmoid::forward(const AutoTensor& x) {
    return x.sigmoid();
}

AutoTensor Tanh::forward(const AutoTensor& x) {
    return x.tanh();
}

AutoTensor GELU::forward(const AutoTensor& x) {
    Tensor out_data = x.data().gelu();
    if (!x.requires_grad()) return AutoTensor::from(std::move(out_data));

    auto xi = x.impl();
    return AutoTensor::make_result(std::move(out_data), true,
        [xi](const Tensor& g) {
            Tensor grad = xi->data.apply([](double v) {
                constexpr double inv_sqrt_2pi = 0.39894228040143267794;
                const double cdf = 0.5 * (1.0 + std::erf(v / std::sqrt(2.0)));
                const double pdf = inv_sqrt_2pi * std::exp(-0.5 * v * v);
                return cdf + v * pdf;
            });
            xi->propagate(g * grad);
        });
}

Softmax::Softmax(size_t axis)
    : m_axis(axis)
{}

AutoTensor Softmax::forward(const AutoTensor& x) {
    if (x.data().ndim() == 0)
        throw std::invalid_argument("Softmax::forward: scalar tensors are not supported");
    if (m_axis >= x.data().ndim())
        throw std::invalid_argument("Softmax::forward: axis out of range");

    Tensor out_data = x.data().softmax(m_axis);
    if (!x.requires_grad()) return AutoTensor::from(std::move(out_data));

    auto xi = x.impl();
    const Tensor y = out_data;
    const Tensor::Shape shape = x.data().shape();
    const size_t axis = m_axis;
    return AutoTensor::make_result(std::move(out_data), true,
        [xi, y, shape, axis](const Tensor& g) {
            size_t outer = 1;
            for (size_t i = 0; i < axis; ++i) outer *= shape[i];
            const size_t axis_dim = shape[axis];
            size_t inner = 1;
            for (size_t i = axis + 1; i < shape.size(); ++i) inner *= shape[i];

            Tensor dx = Tensor::zeros(shape);
            for (size_t o = 0; o < outer; ++o) {
                for (size_t in = 0; in < inner; ++in) {
                    double dot = 0.0;
                    for (size_t a = 0; a < axis_dim; ++a) {
                        const size_t idx = o * axis_dim * inner + a * inner + in;
                        dot += g.flat(idx) * y.flat(idx);
                    }
                    for (size_t a = 0; a < axis_dim; ++a) {
                        const size_t idx = o * axis_dim * inner + a * inner + in;
                        dx.flat(idx) = y.flat(idx) * (g.flat(idx) - dot);
                    }
                }
            }
            xi->propagate(dx);
        });
}

size_t Softmax::axis() const noexcept {
    return m_axis;
}

LayerNorm::LayerNorm(size_t normalized_shape, double eps, bool affine)
    : m_normalized_shape(normalized_shape),
      m_eps(eps),
      m_affine(affine)
{
    if (normalized_shape == 0)
        throw std::invalid_argument("LayerNorm: normalized_shape must be > 0");
    if (eps <= 0.0)
        throw std::invalid_argument("LayerNorm: eps must be > 0");
    if (affine) {
        m_gamma = AutoTensor::from(Tensor::ones({normalized_shape}), true);
        m_beta = AutoTensor::from(Tensor::zeros({normalized_shape}), true);
    }
}

AutoTensor LayerNorm::forward(const AutoTensor& x) {
    if (x.data().ndim() < 1 || x.data().shape().back() != m_normalized_shape)
        throw std::invalid_argument("LayerNorm::forward: last dimension mismatch");

    Tensor x_cpu = x.data().cpu();
    const size_t F = m_normalized_shape;
    const size_t rows = x_cpu.size() / F;
    Tensor out_cpu = Tensor::zeros(x_cpu.shape());
    Tensor xhat_cpu = Tensor::zeros(x_cpu.shape());
    Tensor inv_std({rows});

    Tensor gamma_cpu = m_affine ? m_gamma.data().cpu() : Tensor{};
    Tensor beta_cpu = m_affine ? m_beta.data().cpu() : Tensor{};

    for (size_t r = 0; r < rows; ++r) {
        double mean = 0.0;
        for (size_t f = 0; f < F; ++f) mean += x_cpu.flat(r * F + f);
        mean /= static_cast<double>(F);

        double var = 0.0;
        for (size_t f = 0; f < F; ++f) {
            double d = x_cpu.flat(r * F + f) - mean;
            var += d * d;
        }
        var /= static_cast<double>(F);
        inv_std.flat(r) = 1.0 / std::sqrt(var + m_eps);

        for (size_t f = 0; f < F; ++f) {
            double xhat = (x_cpu.flat(r * F + f) - mean) * inv_std.flat(r);
            xhat_cpu.flat(r * F + f) = xhat;
            double gamma = m_affine ? gamma_cpu.flat(f) : 1.0;
            double beta = m_affine ? beta_cpu.flat(f) : 0.0;
            out_cpu.flat(r * F + f) = xhat * gamma + beta;
        }
    }

    Tensor out = out_cpu.to(x.data().device(), x.data().device_id());
    const bool needs_grad = x.requires_grad() ||
        (m_affine && (m_gamma.requires_grad() || m_beta.requires_grad()));
    if (!needs_grad) return AutoTensor::from(std::move(out));

    auto xi = x.impl();
    auto gi = m_affine ? m_gamma.impl() : nullptr;
    auto bi = m_affine ? m_beta.impl() : nullptr;
    const bool affine = m_affine;

    return AutoTensor::make_result(std::move(out), true,
        [xi, gi, bi, affine, xhat_cpu, inv_std, F, rows](const Tensor& g) {
            Tensor g_cpu = g.cpu();
            Tensor dx_cpu = Tensor::zeros(xhat_cpu.shape());
            Tensor dgamma_cpu = affine ? Tensor::zeros({F}) : Tensor{};
            Tensor dbeta_cpu = affine ? Tensor::zeros({F}) : Tensor{};
            Tensor gamma_cpu = affine ? gi->data.cpu() : Tensor{};

            for (size_t r = 0; r < rows; ++r) {
                double sum_dy = 0.0;
                double sum_dy_xhat = 0.0;
                for (size_t f = 0; f < F; ++f) {
                    double dy = g_cpu.flat(r * F + f);
                    if (affine) {
                        dgamma_cpu.flat(f) += dy * xhat_cpu.flat(r * F + f);
                        dbeta_cpu.flat(f) += dy;
                        dy *= gamma_cpu.flat(f);
                    }
                    sum_dy += dy;
                    sum_dy_xhat += dy * xhat_cpu.flat(r * F + f);
                }
                for (size_t f = 0; f < F; ++f) {
                    double dy = g_cpu.flat(r * F + f);
                    if (affine) dy *= gamma_cpu.flat(f);
                    dx_cpu.flat(r * F + f) =
                        inv_std.flat(r) / static_cast<double>(F) *
                        (static_cast<double>(F) * dy - sum_dy -
                         xhat_cpu.flat(r * F + f) * sum_dy_xhat);
                }
            }

            xi->propagate(dx_cpu.to(xi->data.device(), xi->data.device_id()));
            if (affine) {
                gi->propagate(dgamma_cpu.to(gi->data.device(), gi->data.device_id()));
                bi->propagate(dbeta_cpu.to(bi->data.device(), bi->data.device_id()));
            }
        });
}

std::vector<AutoTensor*> LayerNorm::parameters() {
    if (!m_affine) return {};
    return {&m_gamma, &m_beta};
}

AutoTensor& LayerNorm::gamma() { return m_gamma; }
AutoTensor& LayerNorm::beta() { return m_beta; }
const AutoTensor& LayerNorm::gamma() const { return m_gamma; }
const AutoTensor& LayerNorm::beta() const { return m_beta; }

Embedding::Embedding(size_t num_embeddings, size_t embedding_dim)
    : m_num_embeddings(num_embeddings),
      m_embedding_dim(embedding_dim)
{
    if (num_embeddings == 0 || embedding_dim == 0)
        throw std::invalid_argument("Embedding: dimensions must be > 0");
    const double scale = 1.0 / std::sqrt(static_cast<double>(embedding_dim));
    m_weight = AutoTensor::from(
        Tensor::uniform({num_embeddings, embedding_dim}, -scale, scale, 42),
        true);
}

AutoTensor Embedding::forward(const AutoTensor& indices) {
    Tensor idx_cpu = indices.data().cpu();
    Tensor w_cpu = m_weight.data().cpu();
    Tensor out_cpu({idx_cpu.size(), m_embedding_dim});
    for (size_t i = 0; i < idx_cpu.size(); ++i) {
        int token = static_cast<int>(idx_cpu.flat(i));
        if (token < 0 || token >= static_cast<int>(m_num_embeddings))
            throw std::out_of_range("Embedding::forward: token id out of range");
        for (size_t d = 0; d < m_embedding_dim; ++d)
            out_cpu(i, d) = w_cpu(static_cast<size_t>(token), d);
    }
    Tensor out = out_cpu.to(m_weight.data().device(), m_weight.data().device_id());
    if (!m_weight.requires_grad()) return AutoTensor::from(std::move(out));

    auto wi = m_weight.impl();
    return AutoTensor::make_result(std::move(out), true,
        [wi, idx_cpu, emb_dim = m_embedding_dim](const Tensor& g) {
            Tensor g_cpu = g.cpu();
            Tensor dw_cpu = Tensor::zeros(wi->data.cpu().shape());
            for (size_t i = 0; i < idx_cpu.size(); ++i) {
                size_t token = static_cast<size_t>(idx_cpu.flat(i));
                for (size_t d = 0; d < emb_dim; ++d)
                    dw_cpu(token, d) += g_cpu(i, d);
            }
            wi->propagate(dw_cpu.to(wi->data.device(), wi->data.device_id()));
        });
}

std::vector<AutoTensor*> Embedding::parameters() { return {&m_weight}; }
AutoTensor& Embedding::weight() { return m_weight; }
const AutoTensor& Embedding::weight() const { return m_weight; }

MultiHeadAttention::MultiHeadAttention(size_t embed_dim, size_t num_heads)
    : m_embed_dim(embed_dim),
      m_num_heads(num_heads),
      m_head_dim(num_heads == 0 ? 0 : embed_dim / num_heads)
{
    if (embed_dim == 0 || num_heads == 0 || embed_dim % num_heads != 0)
        throw std::invalid_argument("MultiHeadAttention: embed_dim must be divisible by num_heads");

    const double scale = 1.0 / std::sqrt(static_cast<double>(embed_dim));
    m_wq = AutoTensor::from(Tensor::uniform({embed_dim, embed_dim}, -scale, scale, 11), true);
    m_wk = AutoTensor::from(Tensor::uniform({embed_dim, embed_dim}, -scale, scale, 12), true);
    m_wv = AutoTensor::from(Tensor::uniform({embed_dim, embed_dim}, -scale, scale, 13), true);
    m_wo = AutoTensor::from(Tensor::uniform({embed_dim, embed_dim}, -scale, scale, 14), true);
}

AutoTensor MultiHeadAttention::forward(const AutoTensor& x) {
    if (x.data().ndim() != 3 || x.data().dim(2) != m_embed_dim)
        throw std::invalid_argument("MultiHeadAttention::forward: input must be [N, S, E]");

    const size_t N = x.data().dim(0);
    const size_t S = x.data().dim(1);
    AutoTensor xf = x.reshape({N * S, m_embed_dim});
    AutoTensor q = xf.matmul(m_wq);
    AutoTensor k = xf.matmul(m_wk);
    AutoTensor v = xf.matmul(m_wv);

    Tensor q_cpu = q.data().cpu();
    Tensor k_cpu = k.data().cpu();
    Tensor v_cpu = v.data().cpu();
    Tensor ctx_cpu({N, S, m_embed_dim}, 0.0);
    Tensor attn_cpu({N, m_num_heads, S, S}, 0.0);
    const double scale = 1.0 / std::sqrt(static_cast<double>(m_head_dim));

    for (size_t n = 0; n < N; ++n) {
        for (size_t h = 0; h < m_num_heads; ++h) {
            for (size_t i = 0; i < S; ++i) {
                std::vector<double> scores(S);
                double max_score = -std::numeric_limits<double>::infinity();
                for (size_t j = 0; j < S; ++j) {
                    double dot = 0.0;
                    for (size_t d = 0; d < m_head_dim; ++d) {
                        size_t e = h * m_head_dim + d;
                        dot += q_cpu(n * S + i, e) * k_cpu(n * S + j, e);
                    }
                    scores[j] = dot * scale;
                    max_score = std::max(max_score, scores[j]);
                }
                double denom = 0.0;
                for (double& s : scores) {
                    s = std::exp(s - max_score);
                    denom += s;
                }
                for (size_t j = 0; j < S; ++j) {
                    double a = scores[j] / denom;
                    attn_cpu(n, h, i, j) = a;
                    for (size_t d = 0; d < m_head_dim; ++d) {
                        size_t e = h * m_head_dim + d;
                        ctx_cpu(n, i, e) += a * v_cpu(n * S + j, e);
                    }
                }
            }
        }
    }

    const bool needs_grad = q.requires_grad() || k.requires_grad() || v.requires_grad();
    Tensor ctx_data = ctx_cpu.reshape({N * S, m_embed_dim})
        .to(x.data().device(), x.data().device_id());
    AutoTensor ctx = needs_grad
        ? AutoTensor::make_result(std::move(ctx_data), true,
              [qi = q.impl(), ki = k.impl(), vi = v.impl(),
               q_cpu, k_cpu, v_cpu, attn_cpu,
               N, S, E = m_embed_dim, Hh = m_num_heads,
               Dh = m_head_dim, scale](const Tensor& g) {
                  Tensor grad_cpu = g.cpu().reshape({N, S, E});
                  Tensor dq({N * S, E}, 0.0);
                  Tensor dk({N * S, E}, 0.0);
                  Tensor dv({N * S, E}, 0.0);

                  for (size_t n = 0; n < N; ++n) {
                      for (size_t h = 0; h < Hh; ++h) {
                          for (size_t i = 0; i < S; ++i) {
                              std::vector<double> da(S, 0.0);
                              double weighted_da = 0.0;
                              for (size_t j = 0; j < S; ++j) {
                                  for (size_t d = 0; d < Dh; ++d) {
                                      const size_t e = h * Dh + d;
                                      da[j] += grad_cpu(n, i, e) * v_cpu(n * S + j, e);
                                      dv(n * S + j, e) += attn_cpu(n, h, i, j) *
                                                          grad_cpu(n, i, e);
                                  }
                                  weighted_da += da[j] * attn_cpu(n, h, i, j);
                              }

                              for (size_t j = 0; j < S; ++j) {
                                  const double ds = attn_cpu(n, h, i, j) *
                                      (da[j] - weighted_da);
                                  for (size_t d = 0; d < Dh; ++d) {
                                      const size_t e = h * Dh + d;
                                      dq(n * S + i, e) += ds * k_cpu(n * S + j, e) * scale;
                                      dk(n * S + j, e) += ds * q_cpu(n * S + i, e) * scale;
                                  }
                              }
                          }
                      }
                  }

                  qi->propagate(dq.to(qi->data.device(), qi->data.device_id()));
                  ki->propagate(dk.to(ki->data.device(), ki->data.device_id()));
                  vi->propagate(dv.to(vi->data.device(), vi->data.device_id()));
              })
        : AutoTensor::from(std::move(ctx_data));
    return ctx.matmul(m_wo).reshape({N, S, m_embed_dim});
}

std::vector<AutoTensor*> MultiHeadAttention::parameters() {
    return {&m_wq, &m_wk, &m_wv, &m_wo};
}

Conv2d::Conv2d(size_t in_channels,
               size_t out_channels,
               size_t kernel_size,
               size_t stride,
               size_t padding,
               bool use_bias)
    : m_in_channels(in_channels),
      m_out_channels(out_channels),
      m_kernel_size(kernel_size),
      m_stride(stride),
      m_padding(padding),
      m_use_bias(use_bias)
{
    if (in_channels == 0 || out_channels == 0 || kernel_size == 0 || stride == 0)
        throw std::invalid_argument("Conv2d: channels, kernel_size and stride must be > 0");

    const double fan_in = static_cast<double>(in_channels * kernel_size * kernel_size);
    const double limit = 1.0 / std::sqrt(fan_in);
    m_weight = AutoTensor::from(
        Tensor::uniform({out_channels, in_channels, kernel_size, kernel_size},
                        -limit, limit, 42),
        true);
    if (use_bias)
        m_bias = AutoTensor::from(Tensor::zeros({out_channels}), true);
}

AutoTensor Conv2d::forward(const AutoTensor& x) {
    if (x.data().ndim() != 4)
        throw std::invalid_argument("Conv2d::forward: input must be NCHW [N, C, H, W]");
    if (x.data().dim(1) != m_in_channels)
        throw std::invalid_argument("Conv2d::forward: input channel mismatch");

    const size_t H = x.data().dim(2);
    const size_t W = x.data().dim(3);
    const size_t K = m_kernel_size;
    (void)checkedPoolOutDim(H, K, m_stride, m_padding, "Conv2d::forward");
    (void)checkedPoolOutDim(W, K, m_stride, m_padding, "Conv2d::forward");

    const Tensor* bias_ptr = m_use_bias ? &m_bias.data() : nullptr;
    Tensor out = x.data().conv2d(m_weight.data(), bias_ptr, m_stride, m_padding);

    const bool needs_grad = x.requires_grad() || m_weight.requires_grad() ||
                            (m_use_bias && m_bias.requires_grad());
    if (!needs_grad) return AutoTensor::from(std::move(out));

    auto xi = x.impl();
    auto wi = m_weight.impl();
    auto bi = m_use_bias ? m_bias.impl() : nullptr;
    const bool use_bias = m_use_bias;
    const size_t stride = m_stride;
    const size_t padding = m_padding;

    return AutoTensor::make_result(std::move(out), true,
        [xi, wi, bi, use_bias, stride, padding](const Tensor& g) {
            Tensor dx = Tensor::conv2d_backward_input(
                g, wi->data, xi->data.shape(), stride, padding);
            Tensor dw = g.conv2d_backward_weight(
                xi->data, wi->data.shape(), stride, padding);
            xi->propagate(dx);
            wi->propagate(dw);
            if (use_bias) bi->propagate(g.conv2d_backward_bias());
        });
}

std::vector<AutoTensor*> Conv2d::parameters() {
    if (m_use_bias) return {&m_weight, &m_bias};
    return {&m_weight};
}

AutoTensor& Conv2d::weight() { return m_weight; }
AutoTensor& Conv2d::bias() { return m_bias; }
const AutoTensor& Conv2d::weight() const { return m_weight; }
const AutoTensor& Conv2d::bias() const { return m_bias; }
size_t Conv2d::in_channels() const noexcept { return m_in_channels; }
size_t Conv2d::out_channels() const noexcept { return m_out_channels; }
size_t Conv2d::kernel_size() const noexcept { return m_kernel_size; }
size_t Conv2d::stride() const noexcept { return m_stride; }
size_t Conv2d::padding() const noexcept { return m_padding; }

BatchNorm1d::BatchNorm1d(size_t num_features, double eps, double momentum, bool affine)
    : m_num_features(num_features),
      m_eps(eps),
      m_momentum(momentum),
      m_affine(affine),
      m_running_mean(Tensor::zeros({num_features})),
      m_running_var(Tensor::ones({num_features}))
{
    if (num_features == 0)
        throw std::invalid_argument("BatchNorm1d: num_features must be > 0");
    if (eps <= 0.0)
        throw std::invalid_argument("BatchNorm1d: eps must be > 0");
    if (momentum < 0.0 || momentum > 1.0)
        throw std::invalid_argument("BatchNorm1d: momentum must be in [0, 1]");

    if (affine) {
        m_gamma = AutoTensor::from(Tensor::ones({num_features}), true);
        m_beta = AutoTensor::from(Tensor::zeros({num_features}), true);
    }
}

AutoTensor BatchNorm1d::forward(const AutoTensor& x) {
    if (x.data().ndim() != 2 || x.data().dim(1) != m_num_features)
        throw std::invalid_argument("BatchNorm1d::forward: input must be [N, C]");

    Tensor x_cpu = x.data().cpu();
    Tensor gamma_cpu = m_affine ? m_gamma.data().cpu() : Tensor{};
    Tensor beta_cpu = m_affine ? m_beta.data().cpu() : Tensor{};
    const size_t N = x.data().dim(0);
    const size_t C = x.data().dim(1);
    Tensor mean = Tensor::zeros({C});
    Tensor var = Tensor::zeros({C});

    if (m_training) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t n = 0; n < N; ++n) mean.flat(c) += x_cpu(n, c);
            mean.flat(c) /= static_cast<double>(N);
            for (size_t n = 0; n < N; ++n) {
                const double d = x_cpu(n, c) - mean.flat(c);
                var.flat(c) += d * d;
            }
            var.flat(c) /= static_cast<double>(N);
        }
        const auto stats_device = m_running_mean.device();
        const int stats_device_id = m_running_mean.device_id();
        m_running_mean = (m_running_mean.cpu() * (1.0 - m_momentum) +
                          mean * m_momentum).to(stats_device, stats_device_id);
        m_running_var = (m_running_var.cpu() * (1.0 - m_momentum) +
                         var * m_momentum).to(stats_device, stats_device_id);
    } else {
        mean = m_running_mean.cpu();
        var = m_running_var.cpu();
    }

    Tensor xhat = Tensor::zeros(x_cpu.shape());
    Tensor out = Tensor::zeros(x_cpu.shape());
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            const double inv = 1.0 / std::sqrt(var.flat(c) + m_eps);
            xhat(n, c) = (x_cpu(n, c) - mean.flat(c)) * inv;
            const double gamma = m_affine ? gamma_cpu.flat(c) : 1.0;
            const double beta = m_affine ? beta_cpu.flat(c) : 0.0;
            out(n, c) = xhat(n, c) * gamma + beta;
        }
    }
    out = out.to(x.data().device(), x.data().device_id());

    const bool needs_grad = x.requires_grad() ||
        (m_affine && (m_gamma.requires_grad() || m_beta.requires_grad()));
    if (!needs_grad) return AutoTensor::from(std::move(out));

    auto xi = x.impl();
    auto gi = m_affine ? m_gamma.impl() : nullptr;
    auto be = m_affine ? m_beta.impl() : nullptr;
    const bool affine = m_affine;
    const bool training = m_training;
    const double eps = m_eps;

    return AutoTensor::make_result(std::move(out), true,
        [xi, gi, be, affine, training, xhat, mean, var, gamma_cpu, eps, N, C](const Tensor& g) {
            Tensor g_cpu = g.cpu();
            Tensor dx = Tensor::zeros(xi->data.cpu().shape());
            Tensor dgamma = affine ? Tensor::zeros(gi->data.cpu().shape()) : Tensor{};
            Tensor dbeta = affine ? Tensor::zeros(be->data.cpu().shape()) : Tensor{};

            for (size_t c = 0; c < C; ++c) {
                double sum_dy = 0.0;
                double sum_dy_xhat = 0.0;
                for (size_t n = 0; n < N; ++n) {
                    const double dy = g_cpu(n, c);
                    sum_dy += dy;
                    sum_dy_xhat += dy * xhat(n, c);
                    if (affine) {
                        dgamma.flat(c) += dy * xhat(n, c);
                        dbeta.flat(c) += dy;
                    }
                }

                const double gamma = affine ? gamma_cpu.flat(c) : 1.0;
                const double inv = 1.0 / std::sqrt(var.flat(c) + eps);
                for (size_t n = 0; n < N; ++n) {
                    if (training) {
                        dx(n, c) = gamma * inv / static_cast<double>(N) *
                            (static_cast<double>(N) * g_cpu(n, c) - sum_dy -
                             xhat(n, c) * sum_dy_xhat);
                    } else {
                        dx(n, c) = g_cpu(n, c) * gamma * inv;
                    }
                }
            }

            xi->propagate(dx.to(xi->data.device(), xi->data.device_id()));
            if (affine) {
                gi->propagate(dgamma.to(gi->data.device(), gi->data.device_id()));
                be->propagate(dbeta.to(be->data.device(), be->data.device_id()));
            }
        });
}

std::vector<AutoTensor*> BatchNorm1d::parameters() {
    if (!m_affine) return {};
    return {&m_gamma, &m_beta};
}

void BatchNorm1d::to(SharedMath::LinearAlgebra::Device device, int device_id) {
    Module::to(device, device_id);
    m_running_mean = m_running_mean.to(device, device_id);
    m_running_var = m_running_var.to(device, device_id);
}

AutoTensor& BatchNorm1d::gamma() { return m_gamma; }
AutoTensor& BatchNorm1d::beta() { return m_beta; }
const AutoTensor& BatchNorm1d::gamma() const { return m_gamma; }
const AutoTensor& BatchNorm1d::beta() const { return m_beta; }
const Tensor& BatchNorm1d::running_mean() const noexcept { return m_running_mean; }
const Tensor& BatchNorm1d::running_var() const noexcept { return m_running_var; }

BatchNorm2d::BatchNorm2d(size_t num_features, double eps, double momentum, bool affine)
    : m_num_features(num_features),
      m_eps(eps),
      m_momentum(momentum),
      m_affine(affine),
      m_running_mean(Tensor::zeros({num_features})),
      m_running_var(Tensor::ones({num_features}))
{
    if (num_features == 0)
        throw std::invalid_argument("BatchNorm2d: num_features must be > 0");
    if (eps <= 0.0)
        throw std::invalid_argument("BatchNorm2d: eps must be > 0");
    if (momentum < 0.0 || momentum > 1.0)
        throw std::invalid_argument("BatchNorm2d: momentum must be in [0, 1]");

    if (affine) {
        m_gamma = AutoTensor::from(Tensor::ones({num_features}), true);
        m_beta = AutoTensor::from(Tensor::zeros({num_features}), true);
    }
}

AutoTensor BatchNorm2d::forward(const AutoTensor& x) {
    if (x.data().ndim() != 4 || x.data().dim(1) != m_num_features)
        throw std::invalid_argument("BatchNorm2d::forward: input must be [N, C, H, W]");

    Tensor x_cpu = x.data().cpu();
    Tensor gamma_cpu = m_affine ? m_gamma.data().cpu() : Tensor{};
    Tensor beta_cpu = m_affine ? m_beta.data().cpu() : Tensor{};
    const size_t N = x.data().dim(0);
    const size_t C = x.data().dim(1);
    const size_t H = x.data().dim(2);
    const size_t W = x.data().dim(3);
    const double M = static_cast<double>(N * H * W);
    Tensor mean = Tensor::zeros({C});
    Tensor var = Tensor::zeros({C});

    if (m_training) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t n = 0; n < N; ++n)
                for (size_t h = 0; h < H; ++h)
                    for (size_t w = 0; w < W; ++w)
                        mean.flat(c) += x_cpu(n, c, h, w);
            mean.flat(c) /= M;
            for (size_t n = 0; n < N; ++n)
                for (size_t h = 0; h < H; ++h)
                    for (size_t w = 0; w < W; ++w) {
                        const double d = x_cpu(n, c, h, w) - mean.flat(c);
                        var.flat(c) += d * d;
                    }
            var.flat(c) /= M;
        }
        const auto stats_device = m_running_mean.device();
        const int stats_device_id = m_running_mean.device_id();
        m_running_mean = (m_running_mean.cpu() * (1.0 - m_momentum) +
                          mean * m_momentum).to(stats_device, stats_device_id);
        m_running_var = (m_running_var.cpu() * (1.0 - m_momentum) +
                         var * m_momentum).to(stats_device, stats_device_id);
    } else {
        mean = m_running_mean.cpu();
        var = m_running_var.cpu();
    }

    Tensor xhat = Tensor::zeros(x_cpu.shape());
    Tensor out = Tensor::zeros(x_cpu.shape());
    for (size_t n = 0; n < N; ++n)
        for (size_t c = 0; c < C; ++c)
            for (size_t h = 0; h < H; ++h)
                for (size_t w = 0; w < W; ++w) {
                    const double inv = 1.0 / std::sqrt(var.flat(c) + m_eps);
                    xhat(n, c, h, w) = (x_cpu(n, c, h, w) - mean.flat(c)) * inv;
                    const double gamma = m_affine ? gamma_cpu.flat(c) : 1.0;
                    const double beta = m_affine ? beta_cpu.flat(c) : 0.0;
                    out(n, c, h, w) = xhat(n, c, h, w) * gamma + beta;
                }
    out = out.to(x.data().device(), x.data().device_id());

    const bool needs_grad = x.requires_grad() ||
        (m_affine && (m_gamma.requires_grad() || m_beta.requires_grad()));
    if (!needs_grad) return AutoTensor::from(std::move(out));

    auto xi = x.impl();
    auto gi = m_affine ? m_gamma.impl() : nullptr;
    auto be = m_affine ? m_beta.impl() : nullptr;
    const bool affine = m_affine;
    const bool training = m_training;
    const double eps = m_eps;

    return AutoTensor::make_result(std::move(out), true,
        [xi, gi, be, affine, training, xhat, mean, var, gamma_cpu, eps, N, C, H, W, M](const Tensor& g) {
            Tensor g_cpu = g.cpu();
            Tensor dx = Tensor::zeros(xi->data.cpu().shape());
            Tensor dgamma = affine ? Tensor::zeros(gi->data.cpu().shape()) : Tensor{};
            Tensor dbeta = affine ? Tensor::zeros(be->data.cpu().shape()) : Tensor{};

            for (size_t c = 0; c < C; ++c) {
                double sum_dy = 0.0;
                double sum_dy_xhat = 0.0;
                for (size_t n = 0; n < N; ++n)
                    for (size_t h = 0; h < H; ++h)
                        for (size_t w = 0; w < W; ++w) {
                            const double dy = g_cpu(n, c, h, w);
                            sum_dy += dy;
                            sum_dy_xhat += dy * xhat(n, c, h, w);
                            if (affine) {
                                dgamma.flat(c) += dy * xhat(n, c, h, w);
                                dbeta.flat(c) += dy;
                            }
                        }

                const double gamma = affine ? gamma_cpu.flat(c) : 1.0;
                const double inv = 1.0 / std::sqrt(var.flat(c) + eps);
                for (size_t n = 0; n < N; ++n)
                    for (size_t h = 0; h < H; ++h)
                        for (size_t w = 0; w < W; ++w) {
                            if (training) {
                                dx(n, c, h, w) = gamma * inv / M *
                                    (M * g_cpu(n, c, h, w) - sum_dy -
                                     xhat(n, c, h, w) * sum_dy_xhat);
                            } else {
                                dx(n, c, h, w) = g_cpu(n, c, h, w) * gamma * inv;
                            }
                        }
            }

            xi->propagate(dx.to(xi->data.device(), xi->data.device_id()));
            if (affine) {
                gi->propagate(dgamma.to(gi->data.device(), gi->data.device_id()));
                be->propagate(dbeta.to(be->data.device(), be->data.device_id()));
            }
        });
}

std::vector<AutoTensor*> BatchNorm2d::parameters() {
    if (!m_affine) return {};
    return {&m_gamma, &m_beta};
}

void BatchNorm2d::to(SharedMath::LinearAlgebra::Device device, int device_id) {
    Module::to(device, device_id);
    m_running_mean = m_running_mean.to(device, device_id);
    m_running_var = m_running_var.to(device, device_id);
}

AutoTensor& BatchNorm2d::gamma() { return m_gamma; }
AutoTensor& BatchNorm2d::beta() { return m_beta; }
const AutoTensor& BatchNorm2d::gamma() const { return m_gamma; }
const AutoTensor& BatchNorm2d::beta() const { return m_beta; }
const Tensor& BatchNorm2d::running_mean() const noexcept { return m_running_mean; }
const Tensor& BatchNorm2d::running_var() const noexcept { return m_running_var; }

MaxPool2d::MaxPool2d(size_t kernel_size, size_t stride, size_t padding)
    : m_kernel_size(kernel_size),
      m_stride(stride == 0 ? kernel_size : stride),
      m_padding(padding)
{
    if (kernel_size == 0)
        throw std::invalid_argument("MaxPool2d: kernel_size must be > 0");
}

AutoTensor MaxPool2d::forward(const AutoTensor& x) {
    if (x.data().ndim() != 4)
        throw std::invalid_argument("MaxPool2d::forward: input must be NCHW [N, C, H, W]");

    Tensor out = x.data().max_pool2d(m_kernel_size, m_stride, m_padding);

    if (!x.requires_grad()) return AutoTensor::from(std::move(out));

    auto xi = x.impl();
    const size_t kernel_size = m_kernel_size;
    const size_t stride = m_stride;
    const size_t padding = m_padding;
    return AutoTensor::make_result(std::move(out), true,
        [xi, kernel_size, stride, padding](const Tensor& g) {
            xi->propagate(g.max_pool2d_backward(xi->data, kernel_size, stride, padding));
        });
}

size_t MaxPool2d::kernel_size() const noexcept { return m_kernel_size; }
size_t MaxPool2d::stride() const noexcept { return m_stride; }
size_t MaxPool2d::padding() const noexcept { return m_padding; }

AvgPool2d::AvgPool2d(size_t kernel_size, size_t stride, size_t padding)
    : m_kernel_size(kernel_size),
      m_stride(stride == 0 ? kernel_size : stride),
      m_padding(padding)
{
    if (kernel_size == 0)
        throw std::invalid_argument("AvgPool2d: kernel_size must be > 0");
}

AutoTensor AvgPool2d::forward(const AutoTensor& x) {
    if (x.data().ndim() != 4)
        throw std::invalid_argument("AvgPool2d::forward: input must be NCHW [N, C, H, W]");

    Tensor out = x.data().avg_pool2d(m_kernel_size, m_stride, m_padding);

    if (!x.requires_grad()) return AutoTensor::from(std::move(out));

    auto xi = x.impl();
    const size_t kernel_size = m_kernel_size;
    const size_t stride = m_stride;
    const size_t padding = m_padding;
    return AutoTensor::make_result(std::move(out), true,
        [xi, kernel_size, stride, padding](const Tensor& g) {
            xi->propagate(g.avg_pool2d_backward(xi->data.shape(), kernel_size, stride, padding));
        });
}

size_t AvgPool2d::kernel_size() const noexcept { return m_kernel_size; }
size_t AvgPool2d::stride() const noexcept { return m_stride; }
size_t AvgPool2d::padding() const noexcept { return m_padding; }

} // namespace SharedMath::ML
