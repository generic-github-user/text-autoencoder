#!/usr/bin/env python
# coding: utf-8

# In[98]:


from scipy import spatial
import string
import numpy as np
from fuzzywuzzy import fuzz, process
import seaborn
import matplotlib.pyplot as plt
import random


# In[96]:


import nltk
nltk.download('words')


# In[99]:


word_list = words.words()


# In[83]:


seaborn.set_theme()


# In[76]:


ord('a')


# In[56]:


charset = ''.join([
    string.ascii_lowercase,
#     string.ascii_uppercase,
#     string.digits,
#     string.punctuation,
    ' '
])


# In[57]:


def pad(x, n=10):
#     s = ' ' * n
    return x + ' ' * (n - len(x))


# In[58]:


def one_hot(text, onehot=True):
    text = text.lower()
    encoded = []
    for c in text:
        if c in charset:
            encoded.append(charset.index(c))
        else:
            encoded.append(charset.index(' '))
#         encoded = np.expand_dims(encoded, 1)
    return np.array(encoded)
