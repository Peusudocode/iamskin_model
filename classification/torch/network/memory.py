

import tensorflow


def memory():

    gpu = tensorflow.config.experimental.list_physical_devices('GPU')
    for g in gpu: tensorflow.config.experimental.set_memory_growth(g, True)
    return