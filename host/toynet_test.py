# coding: utf-8
# toynet_test.py

# Example:
# sudo XILINX_XRT=/usr python3 toynet_test.py zcu104_toynet_naive.bit

import numpy as np
import sys
import time

from pynq import allocate, Overlay

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <Bitstream>")
        sys.exit(1)

    overlay = Overlay(sys.argv[1])

    if not overlay.is_loaded():
        print(f"Failed to load the bitstream: {sys.argv[1]}")
        sys.exit(1)

    dma = overlay.axi_dma
    toynet_ip = overlay.toynet
    toynet_ip.register_map.CTRL.AP_START = 1
    toynet_ip.register_map.CTRL.AUTO_RESTART = 1

    buf_in0 = allocate(shape=(1,), dtype=np.uint32, cacheable=False)
    buf_in1 = allocate(shape=(784,), dtype=np.float32, cacheable=False)
    buf_out = allocate(shape=(10,), dtype=np.float32, cacheable=False)

    buf_in0[0] = 2
    buf_in1[:] = np.random.rand(784)

    t0 = time.time()

    for _ in range(100):
        dma.sendchannel.transfer(buf_in0)
        dma.sendchannel.wait()
        dma.sendchannel.transfer(buf_in1)
        dma.sendchannel.wait()
        dma.recvchannel.transfer(buf_out)
        dma.recvchannel.wait()

    t1 = time.time()
    elapsed = (t1 - t0) * 1e3
    print("Elapsed time: {:.3f} ms".format(elapsed))
    print("Average inference time: {:.3f} ms".format(elapsed / 100))

if __name__ == "__main__":
    main()

