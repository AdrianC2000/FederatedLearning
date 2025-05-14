from common.enum.aggregation_method import AggregationMethod
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Default values for training

CENTRALISED_EPOCHS = 10
LOCAL_EPOCHS = 1
NUM_CLIENTS = 5
NUM_ROUNDS = 10
LEARNING_RATE = 0.01
MOMENTUM = 0.9
BATCH_SIZE = 32
TEST_BATCH_SIZE = 1000
AGGREGATION_METHOD = AggregationMethod.FED_AVG
FED_PROX_MU = 0.1

# Differential privacy
EPSILON = 0.1
DELTA = 1e-5
MAX_GRAD_NORM = 1.0

# Homomorphic encryption
POLY_MODULUS_DEGREE = 32768
SCALE = 2 ** 30

# Secure aggregation
MASK_NOISE_SCALE = 0.01

# Secure Multi Party Computation
SHARE_NOISE_SCALE = 0.01