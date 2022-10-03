

import tensorflow as tf
from tensorflow import compat
from tensorflow.python.keras import backend as K


class session:

    def __init__(self, rate=0.4):

        gpu_options = compat.v1.GPUOptions(per_process_gpu_memory_fraction=rate)
        sess = compat.v1.Session(config=compat.v1.ConfigProto(gpu_options=gpu_options))
        K.set_session(sess)


        # physical_devices = tensorflow.config.list_physical_devices('GPU') 
        # tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

        # # opts = tensorflow.GPUOptions(per_process_gpu_memory_fraction=rate)
        # sess = tensorflow.Session(config=tensorflow.ConfigProto(gpu_options=opts))
        # config = tensorflow.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = rate
        # sess = tensorflow.Session(config=config)
        # backend.set_session(sess)
        pass

#     pass



# # rate = 0.4


# import tensorflow as tf

# # 只使用 30% 的 GPU 記憶體
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# # 設定 Keras 使用的 TensorFlow Session
# tf.keras.backend.set_session(sess)
