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
