
// test_util.hpp

#ifndef TOYNET_TB_TEST_UTIL_HPP
#define TOYNET_TB_TEST_UTIL_HPP

#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>

template <int D0>
void GenerateRandomTensor1d(fixed_t x[D0],
                            std::function<float()> rnd)
{
  for (int i = 0; i < D0; ++i)
    x[i] = static_cast<fixed_t>(rnd());
}

template <int D0, int D1>
void GenerateRandomTensor2d(fixed_t x[D0][D1],
                            std::function<float()> rnd)
{
  for (int i = 0; i < D0; ++i)
    for (int j = 0; j < D1; ++j)
      x[i][j] = static_cast<fixed_t>(rnd());
}

template <int D0, int D1, int D2>
void GenerateRandomTensor3d(fixed_t x[D0][D1][D2],
                            std::function<float()> rnd)
{
  for (int i = 0; i < D0; ++i)
    for (int j = 0; j < D1; ++j)
      for (int k = 0; k < D2; ++k)
        x[i][j][k] = static_cast<fixed_t>(rnd());
}

template <int D0, int D1, int D2, int D3>
void GenerateRandomTensor4d(fixed_t x[D0][D1][D2][D3],
                            std::function<float()> rnd)
{
  for (int i = 0; i < D0; ++i)
    for (int j = 0; j < D1; ++j)
      for (int k = 0; k < D2; ++k)
        for (int l = 0; l < D3; ++l)
          x[i][j][k][l] = static_cast<fixed_t>(rnd());
}

template <int D0>
void CompareTensor1d(const fixed_t x0[D0],
                     const fixed_t x1[D0],
                     const float tolerance,
                     const char* name)
{
  for (int i = 0; i < D0; ++i) {
    const float x0f = static_cast<float>(x0[i]);
    const float x1f = static_cast<float>(x1[i]);
    const float diff = std::fabs(x0f - x1f);
    if (diff > tolerance) {
      std::cerr << "Test for " << name << " failed: "
                << "Expected[" << i << "]: " << x0[i] << ", "
                << "Output[" << i << "]: " << x1[i] << '\n';
      std::exit(EXIT_FAILURE);
    }
  }

  std::cerr << "Test for " << name << " succeeded!\n";
}

template <int D0, int D1, int D2>
void CompareTensor3d(const fixed_t x0[D0][D1][D2],
                     const fixed_t x1[D0][D1][D2],
                     const float tolerance,
                     const char* name)
{
  for (int i = 0; i < D0; ++i) {
    for (int j = 0; j < D1; ++j) {
      for (int k = 0; k < D2; ++k) {
        const float x0f = static_cast<float>(x0[i][j][k]);
        const float x1f = static_cast<float>(x1[i][j][k]);
        const float diff = std::fabs(x0f - x1f);
        if (diff > tolerance) {
          std::cerr << "Test for " << name << " failed: "
                    << "Expected[" << i << ", " << j << ", " << k << "]: "
                    << x0[i][j][k] << ", "
                    << "Output[" << i << ", " << j << ", " << k << "]: "
                    << x1[i][j][k] << '\n';
          std::exit(EXIT_FAILURE);
        }
      }
    }
  }

  std::cerr << "Test for " << name << " succeeded!\n";
}

#endif // TOYNET_TB_TEST_UTIL_HPP
