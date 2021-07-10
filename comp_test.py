#!/usr/bin/env python
# coding: utf-8

# In[45]:


import zlib
import math
import nltk
import itertools
import string
import random
import base64
import unicodedata


# In[70]:


a = zlib.compress(b'apples and oranges')
# a = zlib.compress(b'a')
# a[4] = 's'
byteorder = 'big'
# a = int.from_bytes(a, 'little') << 2
num_bytes = len(a)
a = int.from_bytes(a, byteorder)# + (2 ** 20)
print(a)
a = a.to_bytes(num_bytes, byteorder)
# a = str(a).encode()
print(zlib.decompress(a))

b = zlib.compress(b'plums and pears')


# In[142]:


z = zlib.compressobj(level=9)


# In[150]:


def int_comp(x):
    return int.from_bytes(
        zlib.compress(bytes(x, 'UTF-8')),
        byteorder
    )
ic = int_comp


# In[151]:


text = 'I have neither a fear, nor a presentiment, nor a hope of death. Why should I? With my hard constitution and temperate mode of living, and unperilous occupations, I ought to, and probably shall, remain above ground till there is scarcely a black hair on my head. And yet I cannot continue in this condition!'


# In[159]:


m = ic(text)
n = ic(text+' t')
print(m, n, m-(n*(10**round(len(str(m/n))))))


# In[2]:


nltk.download('gutenberg')


# In[3]:


hamlet = nltk.corpus.gutenberg.raw('shakespeare-hamlet.txt')
# hamlet = ' '.join(words)
print(hamlet[:100])
