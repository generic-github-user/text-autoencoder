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


# In[174]:


text_data = []
for a, b in raw_train_ds.take(10):
    print(len(a), len(b))
#     for i in range(5):
#         print(a[i])
#         print(b[i])
    print(a[0].numpy().decode('ascii'))
    text_data.append(prep_data(a[0]))
text_data = np.array(text_data)


# In[41]:


# text_data.shape


# In[183]:


num_chars = len(charset)
model = tf.keras.models.Sequential([
    layers.LSTM(5, input_shape=(50, num_chars,)),
])
decoder_layers = [
    layers.RepeatVector(50),
    layers.LSTM(num_chars, return_sequences=True),
]
for l in decoder_layers:
    model.add(l)

model.summary()


# In[185]:


def decode(t, onehot=False):
    if onehot:
        return ''.join(charset[int(np.clip(0, len(charset)-1, i))] for i in tf.argmax(t[0], axis=1))
    else:
        return ''.join(charset[int(np.clip(0, len(charset)-1, i))] for i in t[0])
def sample():
    noise = np.random.uniform(0, 1, [1, 5])
    pred = noise
    for l in decoder_layers:
        pred = l(pred)
    y = decode(pred)
    return y

sample()


# In[186]:


text_data.shape


# In[187]:


model(text_data)


# In[175]:


model.compile(
    tf.keras.optimizers.Adam(learning_rate=0.001),
    losses.mean_squared_error
)


# In[188]:


history = model.fit(text_data, text_data, epochs=1000)


# In[189]:


plt.plot(history.history['loss'])


# In[199]:


def reconstruct():
    return decode(model(text_data))
    
print(decode(text_data))
reconstruct()


# In[200]:


sample()


# In[ ]:




