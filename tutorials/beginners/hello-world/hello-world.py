import os
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sionna.phy
import matplotlib.pyplot as plt
import numpy as np

batch_size = 1000
Bits = 4
binary_source = sionna.phy.mapping.BinarySource()
b = binary_source([batch_size, Bits])
print(b)
constellation = sionna.phy.mapping.Constellation("qam",Bits)
fig = constellation.show()
fig.savefig('qam16-constellation-map.png')

mapper = sionna.phy.mapping.Mapper(constellation=constellation)
x = mapper(b)

awgn = sionna.phy.channel.AWGN()
ebno_db = 15
no = sionna.phy.utils.ebnodb2no(ebno_db, Bits, coderate=1)
y= awgn(x,no)

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
plt.scatter(np.real(y), np.imag(y))
ax.set_aspect("equal", adjustable="box")
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.grid(True, which="both", axis="both")
plt.title("Received Symbols")
plt.savefig("sionna-15dB-constellation-map.png")