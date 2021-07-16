#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'widget')

from scipy import spatial
import string
import numpy as np
from fuzzywuzzy import fuzz, process
import seaborn
import matplotlib.pyplot as plt
import random

import zlib
import base64
import itertools
import nltk


# In[27]:


import matplotlib


# In[2]:


bytes


# In[3]:


import nltk
word_data = nltk.download('words')


# In[4]:


word_list = nltk.corpus.words.words()


# In[5]:


seaborn.set_theme()


# In[6]:


ord('a')


# In[7]:


charset = ''.join([
    string.ascii_lowercase,
#     string.ascii_uppercase,
#     string.digits,
#     string.punctuation,
    ' '
])


# In[8]:


def pad(x, n=10):
#     s = ' ' * n
    return x + ' ' * (n - len(x))


# In[9]:


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


# In[10]:


# terms

# list(zip([words, terms]))

# [(w, t) for w, t in zip([words, terms])]


# In[61]:


# words = ['What a piece of work is a man! How noble in reason, how infinite in faculty!']
words = []
a = 'you may live to see man-made horrors beyond your comprehension'

words += a.replace('-', ' ').split()
words += random.choices(word_list, k=10000)

ref = 'complex analysis is the field that studies'

max_len = max(map(len, words+[ref]))
print(max_len)
ref_ = one_hot(pad(ref, n=max_len))
terms = [one_hot(pad(t, n=max_len)) for t in words]



positions, labels = zip(*[([np.linalg.norm(t-ref_), fuzz.token_set_ratio(w, ref)], w) for w, t in zip(words, terms)])
lengths = list(map(len, words))

x, y = np.array(positions).T
plot = seaborn.scatterplot(x=x, y=y, size=1, alpha=0.2, c=lengths)
seaborn.set(rc={'figure.figsize':(11, 11)})
offset = np.array([0, 0.3])
for i, l in enumerate(labels):
    if i % (len(labels) // 50) == 0:
        plot.text(*positions[i]+offset, 
             l, horizontalalignment='center')
# plt.show()


# In[12]:


positions


# In[13]:


list(zip(words, terms))[0]


# In[305]:


def comp_str(x):
    data = zlib.compress(x.encode('UTF-8'))
    return base64.b64encode(data)
#     return int.from_bytes(data, 'big')
#     return int(data, base=2)
#     return str(data, encoding='ASCII')
#     return ''.join(format(y, 'b') for y in bytearray(data))
# In[50]:


hamlet = nltk.corpus.gutenberg.sents('shakespeare-hamlet.txt')
def words_to_sent(w):
    s = w[0]
    s += ''.join([t if t in string.punctuation else ' '+t for t in w[1:]])
    return s
# hamlet = [words_to_sent(S) for S in hamlet]
hamlet = list(map(words_to_sent, hamlet))
hamlet = list(filter(lambda l: len(l) >= 10, hamlet))
