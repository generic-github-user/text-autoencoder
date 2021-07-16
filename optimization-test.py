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
positions = np.random.randint(0, limit, [swatches, 2])
color_init = np.random.uniform(-0.1, 0.1, [swatches, 3])

patch = np.random.uniform(-1., 1., [w, h, 3])


# In[29]:


def generate(zx, zy, z2):
    zx = np.array(zx)
    zy = np.array(zy)
    c = np.zeros_like(image_data, dtype=float)
#     Explicitly indexing the position array seems to alleviate the random kernel crashes
    for p in range(zx.shape[0]):
#         x, y = tuple(p.astype(int))
#         x, y = tuple(z1[p])
        x = zx[p]
        y = zy[p]
        x = round(x)
        y = round(y)
        g = z2[p]
#         print(x, y, x+w, y+h, c[x:x+w, y:y+h].shape, g.shape, g.dtype)
        c[x:x+w, y:y+h] += g#+= patch
#     return np.clip(c, 0., 1.)
    return c
    
def loss(qx:np.ndarray, qy:np.ndarray, q2:np.ndarray, noisy=True) -> float:
    canvas = generate(qx, qy, q2)
#     print(canvas.shape)
    d = np.mean((canvas - image_data) ** 2)
#     d = np.mean(np.abs(canvas - image_data))
#     print(d)
    return d


# In[34]:


lim = [int(u) for u in limit.astype(int)]
print(lim)

optimizer = ng.optimizers.NGOpt(
# optimizer = ng.optimizers.PortfolioDiscreteOnePlusOne(
# optimizer = ng.optimizers.TwoPointsDE(
#     parametrization=ng.p.Array(init=positions, lower=0., upper=limit),
#     parametrization=ng.p.TransitionChoice(),
    parametrization=ng.p.Instrumentation(
#         ng.p.Array(
#             ng.p.Scalar(),
#             init=positions, lower=0., upper=limit
#         ),
        ng.p.TransitionChoice(lim[0], repetitions=swatches),
        ng.p.TransitionChoice(lim[1], repetitions=swatches),
        ng.p.Array(init=color_init, lower=-1., upper=1., mutable_sigma=False),
#         .set_integer_casting()
    ),
    budget=100
)

# TODO: image composition with cyclic & trigonometric functions


# In[35]:


result = optimizer.minimize(generate)
# print(result.value)
# print(loss())
plt.imshow(generate(*result.value[0]), cmap='inferno')
