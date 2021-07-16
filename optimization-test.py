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


# In[4]:


image_data = (load_image('test-image.jpg')/255).astype(float)#.mean(axis=2)
plt.imshow(image_data, cmap='inferno')


# In[5]:


# ng.optimizers.NGOpt13


# In[6]:


image_data.shape


# In[20]:


plt.close('all')
ih, iw = image_data.shape[:2]
w, h = (5,)*2
limit = np.array([ih-float(w), iw-float(h)])
print(limit.dtype)
swatches = 300
