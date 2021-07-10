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
