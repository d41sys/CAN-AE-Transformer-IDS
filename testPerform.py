from trainer import Model
import os
import numpy as np
from utils import *
import logging
logging.getLogger('tensorflow').disabled = True

batch_size=1
model = Model(model='CAAE', data_dir='./Data/Train_0.3_Labeled_0.1/', batch_size=batch_size)
tf_data = data_from_tfrecord(['./Data/Train_0.3_Labeled_0.1/DoS/test'], batch_size, 1)
sess_cpu = tf.Session(config=tf.ConfigProto(device_count={'GPU':0}))
x, y = data_stream(tf_data, sess_cpu)

model_path = './model/Transformer'
time = model.timing(x, model_path, use_gpu=False)
print('Average test time for {} sample: {}ms'.format(batch_size, np.mean(time)*1000))
