#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import numpy as np
import nevergrad as ng
import matplotlib.pyplot as plt


# In[2]:


# !dir


# In[3]:


# Via https://stackoverflow.com/a/7769424/10940584
def load_image( infilename, scale=0.05) :
#     img = Image.open( open(infilename, 'rb') )
    img = Image.open(infilename)
    img.load()
    data = np.asarray( img.resize(tuple((np.array(img.size) * scale).round().astype(int))), dtype="int32" )
    return data
