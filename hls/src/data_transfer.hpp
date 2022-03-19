
// data_transfer.hpp

#ifndef TOYNET_DATA_TRANSFER_HPP
#define TOYNET_DATA_TRANSFER_HPP

#include "data_conversion.hpp"
#include "data_types.hpp"

// Read the 1D array from the AXI4-Stream interface
template <int D0>
void ReadArray1d(fixed_t x[D0],
                 hls::stream<axi_stream_data_t>& in_stream)
{
#pragma HLS INLINE off
  for (int i = 0; i < D0; ++i) {
#pragma HLS PIPELINE off
    axi_stream_data_t in_data = in_stream.read();
    float val = U32ToFloat(in_data.data.to_uint());
    x[i] = static_cast<fixed_t>(val);
  }
}

// Read the 2D array from the AXI4-Stream interface
template <int D0, int D1>
void ReadArray2d(fixed_t x[D0][D1],
                 hls::stream<axi_stream_data_t>& in_stream)
{
#pragma HLS INLINE off
  for (int i = 0; i < D0; ++i) {
#pragma HLS PIPELINE off
    for (int j = 0; j < D1; ++j) {
#pragma HLS PIPELINE off
      axi_stream_data_t in_data = in_stream.read();
      float val = U32ToFloat(in_data.data.to_uint());
      x[i][j] = static_cast<fixed_t>(val);
    }
  }
}

// Read the 3D array from the AXI4-Stream interface
template <int D0, int D1, int D2>
void ReadArray3d(fixed_t x[D0][D1][D2],
                 hls::stream<axi_stream_data_t>& in_stream)
{
#pragma HLS INLINE off
  for (int i = 0; i < D0; ++i) {
#pragma HLS PIPELINE off
    for (int j = 0; j < D1; ++j) {
#pragma HLS PIPELINE off
      for (int k = 0; k < D2; ++k) {
#pragma HLS PIPELINE off
        axi_stream_data_t in_data = in_stream.read();
        float val = U32ToFloat(in_data.data.to_uint());
        x[i][j][k] = static_cast<fixed_t>(val);
      }
    }
  }
}

// Read the 3D array from the AXI4-Stream interface
template <int D0, int D1, int D2>
void ReadArray3d2(fixed_t x[D0][D1][D2],
                  hls::stream<axi_stream_data_t>& in_stream)
{
#pragma HLS INLINE off
  for (int i = 0; i < D0; ++i) {
    for (int j = 0; j < D1; ++j) {
      for (int k = 0; k < D2; ++k) {
        axi_stream_data_t in_data = in_stream.read();
        float val = U32ToFloat(in_data.data.to_uint());
        x[i][j][k] = static_cast<fixed_t>(val);
      }
    }
  }
}

// Read the 4D array from the AXI4-Stream interface
template <int D0, int D1, int D2, int D3>
void ReadArray4d(fixed_t x[D0][D1][D2][D3],
                 hls::stream<axi_stream_data_t>& in_stream)
{
#pragma HLS INLINE off
  for (int i = 0; i < D0; ++i) {
#pragma HLS PIPELINE off
    for (int j = 0; j < D1; ++j) {
#pragma HLS PIPELINE off
      for (int k = 0; k < D2; ++k) {
#pragma HLS PIPELINE off
        for (int l = 0; l < D3; ++l) {
#pragma HLS PIPELINE off
          axi_stream_data_t in_data = in_stream.read();
          float val = U32ToFloat(in_data.data.to_uint());
          x[i][j][k][l] = static_cast<fixed_t>(val);
        }
      }
    }
  }
}

// Read the parameters for the 2D convolutional layer
template <int InCh, int OutCh, int K>
void ReadConv2dParams(fixed_t weight[OutCh][InCh][K][K],
                      hls::stream<axi_stream_data_t>& in_stream)
{
#pragma HLS INLINE
  ReadArray4d<OutCh, InCh, K, K>(weight, in_stream);
}

// Read the parameters for the 2D batch normalization layer
template <int C>
void ReadBatchNorm2dParams(fixed_t scale[C],
                           fixed_t bias[C],
                           fixed_t mean[C],
                           hls::stream<axi_stream_data_t>& in_stream)
{
#pragma HLS INLINE
  ReadArray1d<C>(scale, in_stream);
  ReadArray1d<C>(bias, in_stream);
  ReadArray1d<C>(mean, in_stream);
}

// Read the parameters for the fully-connected layer
template <int InDims, int OutDims>
void ReadLinearParams(fixed_t weight[OutDims][InDims],
                      fixed_t bias[OutDims],
                      hls::stream<axi_stream_data_t>& in_stream)
{
#pragma HLS INLINE
  ReadArray2d<OutDims, InDims>(weight, in_stream);
  ReadArray1d<OutDims>(bias, in_stream);
}

// Write the 1D array to the AXI4-Stream interface
template <int D0>
void WriteArray1d(const fixed_t x[D0],
                  hls::stream<axi_stream_data_t>& out_stream)
{
#pragma HLS INLINE off
  axi_stream_data_t out_data;
  out_data.keep = 0xF;
  out_data.strb = 0xF;

  for (int i = 0; i < D0; ++i) {
#pragma HLS PIPELINE off
    // Set all bits in `keep` and `strb` fields to 1
    float val = static_cast<float>(x[i]);
    out_data.data = FloatToU32(val);
    out_data.last = (i == D0 - 1);
    out_stream.write(out_data);
  }
}

// Write the 1D array to the AXI4-Stream interface
template <int D0>
void WriteArray1d2(const fixed_t x[D0],
                   hls::stream<axi_stream_data_t>& out_stream)
{
#pragma HLS INLINE off
  axi_stream_data_t out_data;
  out_data.keep = 0xF;
  out_data.strb = 0xF;

  for (int i = 0; i < D0; ++i) {
    // Set all bits in `keep` and `strb` fields to 1
    float val = static_cast<float>(x[i]);
    out_data.data = FloatToU32(val);
    out_data.last = (i == D0 - 1);
    out_stream.write(out_data);
  }
}

// Write the 2D array to the AXI4-Stream interface
template <int D0, int D1>
void WriteArray2d(const fixed_t x[D0][D1],
                  hls::stream<axi_stream_data_t>& out_stream)
{
#pragma HLS INLINE off
  axi_stream_data_t out_data;
  out_data.keep = 0xF;
  out_data.strb = 0xF;

  for (int i = 0; i < D0; ++i) {
#pragma HLS PIPELINE off
    for (int j = 0; j < D1; ++j) {
#pragma HLS PIPELINE off
      // Set all bits in `keep` and `strb` fields to 1
      float val = static_cast<float>(x[i][j]);
      out_data.data = FloatToU32(val);
      out_data.last = (i == D0 - 1 && j == D1 - 1);
      out_stream.write(out_data);
    }
  }
}

// Write the 3D array to the AXI4-Stream interface
template <int D0, int D1, int D2>
void WriteArray3d(const fixed_t x[D0][D1][D2],
                  hls::stream<axi_stream_data_t>& out_stream)
{
#pragma HLS INLINE off
  axi_stream_data_t out_data;
  out_data.keep = 0xF;
  out_data.strb = 0xF;

  for (int i = 0; i < D0; ++i) {
#pragma HLS PIPELINE off
    for (int j = 0; j < D1; ++j) {
#pragma HLS PIPELINE off
      for (int k = 0; k < D2; ++k) {
#pragma HLS PIPELINE off
        // Set all bits in `keep` and `strb` fields to 1
        float val = static_cast<float>(x[i][j][k]);
        out_data.data = FloatToU32(val);
        out_data.last = (i == D0 - 1 && j == D1 - 1 && k == D2 - 1);
        out_stream.write(out_data);
      }
    }
  }
}

// Write the acknowledgment message to the AXI4-Stream interface
void WriteAck(hls::stream<axi_stream_data_t>& out_stream)
{
#pragma HLS INLINE off
  // Set all bits in `keep` and `strb` fields to 1
  axi_stream_data_t ack_data;
  ack_data.data = 1;
  ack_data.keep = 0xF;
  ack_data.strb = 0xF;
  ack_data.last = 1;
  out_stream.write(ack_data);
}

#endif // TOYNET_DATA_TRANSFER_HPP
