#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing

import numpy as np
import matplotlib.pyplot as plt
import string
import pathlib
from win10toast import ToastNotifier
import random


# In[152]:


def notify(f, t):
    def xfunc():
        f()
        toaster.show_toast(t)
    return xfunc


# In[2]:


charset = ''.join([
    string.ascii_lowercase,
#     string.ascii_uppercase,
#     string.digits,
#     string.punctuation,
    ' '
])


# In[3]:


data_url = 'https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz'
dataset_dir = utils.get_file(
    'so_text_dataset',
    origin=data_url,
    untar=True,
    cache_dir='stack_overflow',
    cache_subdir='')


# In[4]:


dataset_dir = pathlib.Path(dataset_dir).parent


# In[5]:


dataset_dir


# In[6]:


raw_train_ds = preprocessing.text_dataset_from_directory(
    dataset_dir/'train',
    batch_size=1,
    validation_split=0.2,
    subset='training',
    seed=42)


# In[30]:


def one_hot(text, onehot=True):
    s = '[SEP]'
#     encoded = tf.keras.preprocessing.text.one_hot(
#         s.join(text),
#         len(charset),
#         filters='\t\n',
#         lower=False, split=s
#     )
    text = text.lower()
    encoded = []
    for c in text:
        if c in charset:
            encoded.append(charset.index(c))
        else:
            encoded.append(charset.index(' '))
    if onehot:
        encoded = tf.one_hot(encoded, len(charset))
    else:
        encoded = np.expand_dims(encoded, 1)
    return encoded

one_hot('test data', onehot=False)


# In[31]:


def prep_data(x):
    x = x.numpy().decode('UTF-8')
    x = x[:50]
    c = one_hot(x + ' ' * (50 - len(x)), onehot=False)
    return c


# In[40]:


def preprocess(n=250):
    text_data = []
    for a, b in raw_train_ds.take(n):
    #     print(len(a), len(b))
    #     for i in range(5):
    #         print(a[i])
    #         print(b[i])
    #     print(a[0].numpy().decode('ascii'))
        d = prep_data(a[0])
        if not isinstance(d, np.ndarray):
            d = d.numpy()
        text_data.append(d)
    text_data = np.stack(text_data)
    return text_data


# In[41]:


# text_data.shape


# In[183]:


# num_chars = len(charset)
num_chars = 1
model = tf.keras.models.Sequential()

encoder_shape = []

activation = 'softplus'
encoder_layers = [
    #timedistributed layer?
    layers.Flatten(input_shape=(50, num_chars,)),
    layers.Dense(20, activation=activation),
#     layers.LSTM(10, activation='tanh', return_sequences=True),
#     layers.LSTM(10, input_shape=(50, num_chars,), activation='tanh', return_sequences=False),
    layers.Dense(5, activation=activation),
#     layers.Dense(2, activation=activation)
]
decoder_layers = [
    layers.Dense(5, activation=activation),
#     layers.RepeatVector(50),
    layers.Dense(10, activation=activation),
    
    layers.Dense(50, activation=activation),
    layers.Reshape((50, 1))
#     layers.LSTM(num_chars, return_sequences=True, activation='elu'),
]
for l in encoder_layers + decoder_layers:
    model.add(l)

model.summary()


# In[185]:


def decode(t, onehot=False):
    if onehot:
        return ''.join(charset[int(np.clip(0, len(charset)-1, i))] for i in tf.argmax(t[0], axis=1))
    else:
        return ''.join(charset[int(np.clip(0, len(charset)-1, i))] for i in t[0])

def run_encoder(pred):
    for l in encoder_layers:
        pred = l(pred)
    return pred

def sample():
    noise = np.random.uniform(0, 1, [1, 5])
    pred = noise
    for l in decoder_layers:
        pred = l(pred)
    y = decode(pred)
    return y

f = sample()
print(f, len(f))
# reconstruct()


# In[186]:


text_data.shape


# In[187]:


model(text_data)


# In[188]:


encoder_layers[1](encoder_layers[0](text_data))


# In[189]:


model.compile(
    tf.keras.optimizers.Adam(learning_rate=0.001),
    losses.mean_squared_error
#     loss=tf.edit_distance
)


# In[190]:


text_data = preprocess(1000)


# In[191]:


history = model.fit(text_data, text_data, epochs=1000)


# In[164]:


import tensorflow.python.ops.numpy_ops.np_config as npcfg
npcfg.enable_numpy_behavior()


# In[197]:


# import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')
# text_data = preprocess(1000)
plt.scatter(*run_encoder(text_data).T[:2], alpha=1, s=6)
plt.show()


# In[198]:


plt.plot(history.history['loss'])


# In[199]:


def reconstruct():
    return decode(model(text_data))
    
print(decode(text_data))
reconstruct()


# In[200]:


for x in range(10):
    print(sample())


# In[ ]:




