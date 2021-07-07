#!/usr/bin/env python
# coding: utf-8

# In[156]:


import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing

import numpy as np
import matplotlib.pyplot as plt
import string
import pathlib


# In[47]:


charset = ''.join([string.ascii_lowercase, string.ascii_uppercase, string.digits, string.punctuation, ' '])


# In[9]:


data_url = 'https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz'
dataset_dir = utils.get_file(
    'so_text_dataset',
    origin=data_url,
    untar=True,
    cache_dir='stack_overflow',
    cache_subdir='')


# In[11]:


dataset_dir = pathlib.Path(dataset_dir).parent


# In[12]:


dataset_dir


# In[52]:


raw_train_ds = preprocessing.text_dataset_from_directory(
    dataset_dir/'train',
    batch_size=1,
    validation_split=0.2,
    subset='training',
    seed=42)


# In[90]:


def one_hot(text):
    s = '[SEP]'
#     encoded = tf.keras.preprocessing.text.one_hot(
#         s.join(text),
#         len(charset),
#         filters='\t\n',
#         lower=False, split=s
#     )
    encoded = []
    for c in text:
        if c in charset:
            encoded.append(charset.index(c))
        else:
            encoded.append(charset.index(' '))
    encoded = tf.one_hot(encoded, len(charset))
    return encoded

one_hot('test data')
