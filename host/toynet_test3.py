# coding: utf-8
# toynet_test3.py

# Example:
# sudo XILINX_XRT=/usr python3 toynet_test3.py zcu104_toynet_naive.bit

import numpy as np
import os
import pynq
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets
import torchvision.transforms

from pynq import allocate, Overlay

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir)))

from net import ToyNet

def _copy_conv2d_weights(buf: pynq.buffer.PynqBuffer,
                         layer: nn.Conv2d,
                         offset: int) -> int:
    param_size = layer.out_channels * layer.in_channels * \
                 layer.kernel_size[0] * layer.kernel_size[1]
    buf[offset:offset+param_size] = layer.weight.data.view(-1)
    offset += param_size
    return offset

def _copy_batchnorm2d_weights(buf: pynq.buffer.PynqBuffer,
                              layer: nn.BatchNorm2d,
                              offset: int) -> int:
    param_size = layer.num_features
    stddev_inv = torch.sqrt(layer.running_var.data + layer.eps)
    stddev_inv = torch.reciprocal(stddev_inv)
    scale = stddev_inv * layer.weight.data
    buf[offset:offset+param_size] = scale.view(-1)
    offset += param_size
    buf[offset:offset+param_size] = layer.bias.data.view(-1)
    offset += param_size
    buf[offset:offset+param_size] = layer.running_mean.data.view(-1)
    offset += param_size
    return offset

def _copy_linear_weights(buf: pynq.buffer.PynqBuffer,
                         layer: nn.Linear,
                         offset: int) -> int:
    weight_size = layer.out_features * layer.in_features
    bias_size = layer.out_features
    buf[offset:offset+weight_size] = layer.weight.data.view(-1)
    offset += weight_size
    buf[offset:offset+bias_size] = layer.bias.data.view(-1)
    offset += bias_size
    return offset

def transfer_weights(dma: pynq.lib.DMA, model: ToyNet):
    # Compute the number of parameters in the model
    buf_len = 0
    buf_len += 6 * 1 * 5 * 5
    buf_len += 6 * 3
    buf_len += 16 * 6 * 5 * 5
    buf_len += 16 * 3
    buf_len += 120 * 400
    buf_len += 120
    buf_len += 84 * 120
    buf_len += 84
    buf_len += 10 * 84
    buf_len += 10

    # Allocate the buffers for transfer
    buf_in0 = allocate(shape=(1,), dtype=np.uint32, cacheable=False)
    buf_in1 = allocate(shape=(buf_len,), dtype=np.float32, cacheable=False)
    buf_out = allocate(shape=(1,), dtype=np.uint32, cacheable=False)

    # Fill the buffer
    buf_in0[0] = 1

    offset = 0
    offset = _copy_conv2d_weights(buf_in1, model.conv0, offset)
    offset = _copy_batchnorm2d_weights(buf_in1, model.bn0, offset)
    offset = _copy_conv2d_weights(buf_in1, model.conv1, offset)
    offset = _copy_batchnorm2d_weights(buf_in1, model.bn1, offset)
    offset = _copy_linear_weights(buf_in1, model.linear0, offset)
    offset = _copy_linear_weights(buf_in1, model.linear1, offset)
    offset = _copy_linear_weights(buf_in1, model.linear2, offset)
    assert offset == buf_len

    # Transfer the weights
    dma.sendchannel.transfer(buf_in0)
    dma.sendchannel.wait()
    dma.sendchannel.transfer(buf_in1)
    dma.sendchannel.wait()
    dma.recvchannel.transfer(buf_out)
    dma.recvchannel.wait()
    print(f"Ack: {buf_out[0]}")

def test(dma: pynq.lib.DMA,
         test_loader: torch.utils.data.DataLoader):
    correct = 0
    in_len = 1 * 28 * 28
    out_len = 10

    # Allocate the buffer for transfer
    buf_in0 = allocate(shape=(2,), dtype=np.uint32, cacheable=False)
    # buf_in0 = allocate(shape=(1,), dtype=np.uint32, cacheable=False)
    buf_in1 = allocate(shape=(in_len,), dtype=np.float32, cacheable=False)
    buf_out = allocate(shape=(out_len,), dtype=np.float32, cacheable=False)

    for idx, (data, target) in enumerate(test_loader):
        # Transfer the data sample and receive the result
        buf_in0[0] = 2
        buf_in0[1] = 1
        buf_in1[:] = data[0].view(-1)
        dma.sendchannel.transfer(buf_in0)
        dma.sendchannel.wait()
        dma.sendchannel.transfer(buf_in1)
        dma.sendchannel.wait()
        dma.recvchannel.transfer(buf_out)
        dma.recvchannel.wait()

        out = torch.from_numpy(buf_out).clone()
        out = out.view(1, out_len)
        out = F.log_softmax(out, dim=1)
        pred = out.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        if idx % 100 == 0:
            print("Index: {}, correct: {}".format(idx, correct))

    print("Test accuracy: {} / {} ({:.0f}%)".format(
          correct, len(test_loader.dataset),
          100.0 * correct / len(test_loader.dataset)))

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <Checkpoint> <Bitstream>")
        sys.exit(1)

    # Load the model
    model = ToyNet()
    model.load_state_dict(torch.load(sys.argv[1], map_location="cpu"))

    # Load the overlay
    overlay = Overlay(sys.argv[2])

    if not overlay.is_loaded():
        print(f"Failed to load the bitstream: {sys.argv[2]}")
        sys.exit(1)

    dma = overlay.axi_dma
    toynet_ip = overlay.toynet
    toynet_ip.register_map.CTRL.AP_START = 1
    toynet_ip.register_map.CTRL.AUTO_RESTART = 1

    # Transfer the weights
    transfer_weights(dma, model)
    print("Weight initialization successful")

    # Load the dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    test_set = torchvision.datasets.MNIST(
        "./data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=1)
    print("Test dataset is successfully loaded")

    # Test the model
    test(dma, test_loader)

if __name__ == "__main__":
    main()
