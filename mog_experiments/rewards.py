import numpy as np
def reward_func_0(x):
    return 1*(x[:, 0] < -7).astype(float)