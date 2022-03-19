
// layer_test.cpp

#include <random>

#include "batch_norm_2d.hpp"
#include "conv_2d.hpp"
#include "data_types.hpp"
#include "depthwise_conv_2d.hpp"
#include "linear.hpp"
#include "max_pool_2d.hpp"

#include "tb/test_util.hpp"

constexpr float kTolerance = 1.0e-6;

template <int C, int H, int W, int B>
void TestBatchNorm2dReLU()
{
  std::random_device random_dev;
  std::default_random_engine engine { random_dev() };
  std::uniform_real_distribution<float> dist { -0.1f, 0.1f };
  auto rnd = [&dist, &engine] { return dist(engine); };

  fixed_t x[C][H][W];
  fixed_t scale[C];
  fixed_t bias[C];
  fixed_t mean[C];
  fixed_t y0[C][H][W];
  fixed_t y1[C][H][W];

  GenerateRandomTensor3d<C, H, W>(x, rnd);
  GenerateRandomTensor1d<C>(scale, rnd);
  GenerateRandomTensor1d<C>(bias, rnd);
  GenerateRandomTensor1d<C>(mean, rnd);

  // Test the naive implementation
  BatchNorm2dReLU<C, H, W>(x, y0, scale, bias, mean);
  // Test the parallel implementation
  BatchNorm2dReLU2<C, H, W, B>(x, y1, scale, bias, mean);
  // Compare the results
  CompareTensor3d<C, H, W>(y0, y1, kTolerance, "BatchNorm2dReLU2");
}

template <int InCh, int OutCh, int H, int W, int OH, int OW,
          int K, int P, int S, int B>
void TestConv2d()
{
  std::random_device random_dev;
  std::default_random_engine engine { random_dev() };
  std::uniform_real_distribution<float> dist { -0.1f, 0.1f };
  auto rnd = [&dist, &engine] { return dist(engine); };

  fixed_t x[InCh][H][W];
  fixed_t weight[OutCh][InCh][K][K];
  fixed_t y0[OutCh][OH][OW];
  fixed_t y1[OutCh][OH][OW];
  fixed_t y2[OutCh][OH][OW];

  GenerateRandomTensor3d<InCh, H, W>(x, rnd);
  GenerateRandomTensor4d<OutCh, InCh, K, K>(weight, rnd);

  // Test the naive implementation
  Conv2d<InCh, OutCh, H, W, OH, OW, K, P, S>(x, y0, weight);
  // Test the parallel implementation
  Conv2d2<InCh, OutCh, H, W, OH, OW, K, P, S, B>(x, y1, weight);
  // Test the parallel implementation
  Conv2d3<InCh, OutCh, H, W, OH, OW, K, P, S, B>(x, y2, weight);

  // Compare the results
  CompareTensor3d<OutCh, OH, OW>(y0, y1, kTolerance, "Conv2d2");
  CompareTensor3d<OutCh, OH, OW>(y0, y2, kTolerance, "Conv2d3");
}

template <int C, int H, int W, int OH, int OW,
          int K, int P, int S, int B>
void TestDepthwiseConv2d()
{
  std::random_device random_dev;
  std::default_random_engine engine { random_dev() };
  std::uniform_real_distribution<float> dist { -0.1f, 0.1f };
  auto rnd = [&dist, &engine] { return dist(engine); };

  fixed_t x[C][H][W];
  fixed_t weight[C][K][K];
  fixed_t y0[C][OH][OW];
  fixed_t y1[C][OH][OW];

  GenerateRandomTensor3d<C, H, W>(x, rnd);
  GenerateRandomTensor3d<C, K, K>(weight, rnd);

  // Test the naive implementation
  DepthwiseConv2d<C, H, W, OH, OW, K, P, S>(x, y0, weight);
  // Test the parallel implementation
  DepthwiseConv2d2<C, H, W, OH, OW, K, P, S, B>(x, y1, weight);

  // Compare the results
  CompareTensor3d<C, OH, OW>(y0, y1, kTolerance, "DepthwiseConv2d2");
}

template <int C, int H, int W, int K, int B>
void TestMaxPool2d()
{
  std::random_device random_dev;
  std::default_random_engine engine { random_dev() };
  std::uniform_real_distribution<float> dist { -0.1f, 0.1f };
  auto rnd = [&dist, &engine] { return dist(engine); };

  constexpr int OH = H / K;
  constexpr int OW = W / K;

  fixed_t x[C][H][W];
  fixed_t y0[C][OH][OW];
  fixed_t y1[C][OH][OW];

  GenerateRandomTensor3d<C, H, W>(x, rnd);

  // Test the naive implementation
  MaxPool2d<C, H, W, K>(x, y0);
  // Test the parallel implementation
  MaxPool2d2<C, H, W, K, B>(x, y1);

  // Compare the results
  CompareTensor3d<C, OH, OW>(y0, y1, kTolerance, "MaxPool2d2");
}

template <int InDims, int OutDims, int B, bool ApplyReLU>
void TestLinear()
{
  std::random_device random_dev;
  std::default_random_engine engine { random_dev() };
  std::uniform_real_distribution<float> dist { -0.1f, 0.1f };
  auto rnd = [&dist, &engine] { return dist(engine); };

  fixed_t x[InDims];
  fixed_t weight[OutDims][InDims];
  fixed_t bias[OutDims];
  fixed_t y0[OutDims];
  fixed_t y1[OutDims];

  GenerateRandomTensor1d<InDims>(x, rnd);
  GenerateRandomTensor2d<OutDims, InDims>(weight, rnd);
  GenerateRandomTensor1d<OutDims>(bias, rnd);

  // Test the naive implementation
  Linear<InDims, OutDims, ApplyReLU>(x, weight, bias, y0);
  // Test the parallel implementation
  Linear2<InDims, OutDims, ApplyReLU, B>(x, weight, bias, y1);

  // Compare the results
  CompareTensor1d<OutDims>(y0, y1, kTolerance, "Linear2");
}

int main(int argc, char** argv)
{
  TestBatchNorm2dReLU<64, 8, 8, 8>();
  TestConv2d<32, 64, 10, 10, 10, 10, 3, 1, 1, 8>();
  TestConv2d<32, 64, 10, 10, 5, 5, 3, 1, 2, 8>();
  TestConv2d<6, 16, 14, 14, 10, 10, 5, 0, 1, 2>();
  TestDepthwiseConv2d<64, 10, 10, 10, 10, 3, 1, 1, 8>();
  TestDepthwiseConv2d<64, 10, 10, 5, 5, 3, 1, 2, 8>();
  TestDepthwiseConv2d<16, 14, 14, 10, 10, 5, 0, 1, 2>();
  TestMaxPool2d<64, 12, 12, 2, 8>();
  TestLinear<64, 128, 8, false>();
  TestLinear<64, 128, 8, true>();

  return EXIT_SUCCESS;
}
