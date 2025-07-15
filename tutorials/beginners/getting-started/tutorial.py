## IMPORTS AND PRELIMS
import os # Configure which GPU
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
try:
    import sionna.phy
except ImportError as e:
    import sys
    if 'google.colab' in sys.modules:
       # Install Sionna in Google Colab
       print("Installing Sionna and restarting the runtime. Please run the cell again.")
       os.system("pip install sionna")
       os.kill(os.getpid(), 5)
    else:
       raise e

# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

import numpy as np

import matplotlib.pyplot as plt

# for performance measurements
import time

# RANDOM NUMBER GENERATION
sionna.phy.config.seed = 40

# Python RNG - use instead of
# import random
# random.randint(0, 10)
print(sionna.phy.config.py_rng.randint(0,10))

# NumPy RNG - use instead of
# import numpy as np
# np.random.randint(0, 10)
print(sionna.phy.config.np_rng.integers(0,10))

# TensorFlow RNG - use instead of
# import tensorflow as tf
# tf.random.uniform(shape=[1], minval=0, maxval=10, dtype=tf.int32)
print(sionna.phy.config.tf_rng.uniform(shape=[1], minval=0, maxval=10, dtype=tf.int32))

NUM_BITS_PER_SYMBOL = 2 # QPSK
constellation = sionna.phy.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)

# fig = constellation.show()
# fig.savefig('QPSKConstellationMapDemo.png')

#mappers and demappers
mapper = sionna.phy.mapping.Mapper(constellation=constellation)
demapper = sionna.phy.mapping.Demapper("maxlog", constellation=constellation)

#binary source/ data_gen in 16QAM toolbox
binary_source = sionna.phy.mapping.BinarySource()
awgn_channel = sionna.phy.channel.AWGN()
no = sionna.phy.utils.ebnodb2no(ebno_db=20.0, num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                                   coderate=1.0)

# BATCH_SIZE = 64
# bits = binary_source([BATCH_SIZE, 1024])
# print(bits.shape)
# x = mapper(bits)
# print(x.shape)
# y = awgn_channel(x,no)
# print(y.shape)
# llr = demapper(y,no)
# print(llr.shape)


# num_samples = 8 # how many samples shall be printed
# num_symbols = int(num_samples/NUM_BITS_PER_SYMBOL)

# print(f"First {num_samples} transmitted bits: {bits[0,:num_samples]}")
# print(f"First {num_symbols} transmitted symbols: {np.round(x[0,:num_symbols], 2)}")
# print(f"First {num_symbols} received symbols: {np.round(y[0,:num_symbols], 2)}")
# print(f"First {num_samples} demapped llrs: {np.round(llr[0,:num_samples], 2)}")

# plt.figure(figsize=(8,8))
# plt.axes().set_aspect(1)
# plt.grid(True)
# plt.title('Channel output')
# plt.xlabel('Real Part')
# plt.ylabel('Imaginary Part')
# plt.scatter(tf.math.real(y), tf.math.imag(y))
# plt.tight_layout()
# plt.show()


class UncodedSystemAWGN(sionna.phy.Block):
    def __init__(self, num_bits_per_symbol, block_length):
        """
        A Sionna Block for uncoded transmission over the AWGN channel

        Parameters
        ----------
        num_bits_per_symbol: int
            The number of bits per constellation symbol, e.g., 4 for QAM16.

        block_length: int
            The number of bits per transmitted message block (will be the codeword length later).

        Input
        -----
        batch_size: int
            The batch_size of the Monte-Carlo simulation.

        ebno_db: float
            The `Eb/No` value (=rate-adjusted SNR) in dB.

        Output
        ------
        (bits, llr):
            Tuple:

        bits: tf.float32
            A tensor of shape `[batch_size, block_length] of 0s and 1s
            containing the transmitted information bits.

        llr: tf.float32
            A tensor of shape `[batch_size, block_length] containing the
            received log-likelihood-ratio (LLR) values.
        """

        super().__init__() # Must call the block initializer

        self.num_bits_per_symbol = num_bits_per_symbol
        self.block_length = block_length
        self.constellation = sionna.phy.mapping.Constellation("qam", self.num_bits_per_symbol)
        self.mapper = sionna.phy.mapping.Mapper(constellation=self.constellation)
        self.demapper = sionna.phy.mapping.Demapper("app", constellation=self.constellation)
        self.binary_source = sionna.phy.mapping.BinarySource()
        self.awgn_channel = sionna.phy.channel.AWGN()

    # @tf.function # Enable graph execution to speed things up
    def call(self, batch_size, ebno_db):

        # no channel coding used; we set coderate=1.0
        no = sionna.phy.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=1.0)

        bits = self.binary_source([batch_size, self.block_length]) # Blocklength set to 1024 bits
        x = self.mapper(bits)
        y = self.awgn_channel(x, no)
        llr = self.demapper(y,no)
        return bits, llr
    
model_uncoded_awgn = UncodedSystemAWGN(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=1024)
EBN0_DB_MIN = -3.0 # Minimum value of Eb/N0 [dB] for simulations
EBN0_DB_MAX = 5.0 # Maximum value of Eb/N0 [dB] for simulations
BATCH_SIZE = 2000 # How many examples are processed by Sionna in parallel

ber_plots = sionna.phy.utils.PlotBER("AWGN")
# ber_plots.simulate(model_uncoded_awgn,
#                    ebno_dbs=np.linspace(EBN0_DB_MIN,EBN0_DB_MAX,20),
#                    batch_size=BATCH_SIZE,
#                    num_target_bit_errors=100,
#                    legend="Uncoded",
#                    soft_estimates=True,
#                    max_mc_iter= 100,
#                    show_fig=True)

k = 12
n = 20

encoder = sionna.phy.fec.ldpc.LDPC5GEncoder(k,n)
decoder = sionna.phy.fec.ldpc.LDPC5GDecoder(encoder, hard_out=True)

BATCH_SIZE = 1
u = binary_source([BATCH_SIZE, k])
print("Input bits are: \n", u.numpy())
print("Shape of u: ", u.shape)

c = encoder(u)
print("Encoded bits are: \n", c.numpy())
print("Shape of c: ", c.shape)

BATCH_SIZE = 10
num_basestations = 4
num_users = 5 #users per basestation
n = 1000 #codeword length peer transmitted codeword
coderate = 0.5 

k = int(coderate*n)
#instantiate new encoder for codewords of length n
encoder = sionna.phy.fec.ldpc.LDPC5GEncoder(k,n)
#decoder mus tbe linked to encoder
decoder = sionna.phy.fec.ldpc.LDPC5GDecoder(encoder,
                                            hard_out = True,
                                            num_iter = 20,
                                            cn_update = "boxplus-phi")

# draw random bits to encode
u = binary_source([BATCH_SIZE, num_basestations, num_users, k])
print("Shape of u: ", u.shape)

# We can immediately encode u for all users, basetation and samples
# This all happens with a single line of code
c = encoder(u)
print("Shape of c: ", c.shape)

print("Total number of processed bits: ", np.prod(c.shape))

k = 64
n = 128

encoder = sionna.phy.fec.polar.Polar5GEncoder(k, n)
decoder = sionna.phy.fec.polar.Polar5GDecoder(encoder,
                                      dec_type="SCL") # you can also use "SCL"

class CodedSystemAWGN(sionna.phy.Block):
    def __init__(self, num_bits_per_symbol, n, coderate):
        super().__init__() # Must call the Sionna block initializer

        self.num_bits_per_symbol = num_bits_per_symbol
        self.n = n
        self.k = int(n*coderate)
        self.coderate = coderate
        self.constellation = sionna.phy.mapping.Constellation("qam", self.num_bits_per_symbol)

        self.mapper = sionna.phy.mapping.Mapper(constellation=self.constellation)
        self.demapper = sionna.phy.mapping.Demapper("app", constellation=self.constellation)

        self.binary_source = sionna.phy.mapping.BinarySource()
        self.awgn_channel = sionna.phy.channel.AWGN()

        self.encoder = sionna.phy.fec.ldpc.LDPC5GEncoder(self.k, self.n)
        self.decoder = sionna.phy.fec.ldpc.LDPC5GDecoder(self.encoder, hard_out=True)

    #@tf.function # activate graph execution to speed things up
    def call(self, batch_size, ebno_db):
        no = sionna.phy.utils.ebnodb2no(ebno_db, num_bits_per_symbol=self.num_bits_per_symbol, coderate=self.coderate)

        bits = self.binary_source([batch_size, self.k])
        codewords = self.encoder(bits)
        x = self.mapper(codewords)
        y = self.awgn_channel(x, no)
        llr = self.demapper(y,no)
        bits_hat = self.decoder(llr)
        return bits, bits_hat

CODERATE = 0.5
BATCH_SIZE = 2000

model_coded_awgn = CodedSystemAWGN(num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                                   n=2048,
                                   coderate=CODERATE)
ber_plots.simulate(model_coded_awgn,
                   ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 15),
                   batch_size=BATCH_SIZE,
                   num_target_block_errors=500,
                   legend="Coded",
                   soft_estimates=False,
                   max_mc_iter=15,
                   show_fig=True,
                   forward_keyboard_interrupt=False)
