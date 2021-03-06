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


# In[80]:


chars = string.punctuation + string.digits# + string.ascii_letters
chars = list(chars) + [chr(n) for n in range(161, 256)]
chars = set(chars) - set(['"', "'", '\\', '&'])
replacements = []
for l in range(1, 3):
    spans = [''.join(c) for c in itertools.combinations_with_replacement(chars, r=l)]
    random.shuffle(spans)
    replacements.extend(spans)
print(len(replacements))
print(replacements[:20])
# random.shuffle(replacements)


# In[6]:


# replacements[:100]


# In[56]:


def compress(input_text, log=False, window_size=(2, 10, 1), separator='&'):
    sections = {}
    w = 0
    reps = []
    checked = []
    
    original = input_text
    compressed = input_text
    
    for k in range(5):
        print(f'Compression pass {k+1}')
        for i in range(*window_size):
            if log:
                print(f'Scanning sequences of length {i}')
            for j in range(0, len(compressed)-i):
                window = compressed[j:j+i]
                if window and window not in checked and compressed.count(window) > 1 and window not in sections.values():
                    for rep in replacements:
                        if rep not in compressed and len(window) > len(rep)+len(separator):
                            sections[w] = window
                            reps.append(rep)
                            w += 1
                #             print(window)
        #                     r = f'[{w}]'
                            compressed = compressed.replace(window, rep+separator)
                            break
                checked.append(window)
            if log:
                print(len(compressed))
    return compressed, reps, sections


# In[57]:


def decompress(input_text, reps, sections, separator='&'):
    for i, r in enumerate(reps[::-1]):
        input_text = input_text.replace(r+separator, sections[len(reps)-i-1])
    return input_text


# In[103]:


argdict = {
    'separator': ''
}
a, r, s = compress('grapes, bananas, apples, peaches, and bananas', window_size=(2, 10, 1), **argdict)
print(a)
print(decompress(a, r, s, **argdict))


# In[106]:


decompress('gr??+??6??pl+peach+and ??', r, s, **argdict)


# In[108]:


s = 500
h = hamlet[:1000]
wa = (2, 20, 1)
wb = (10, 2, -1)
compressed, reps, sections = compress(h, window_size=wa, log=True, **argdict)
original = h
print(original[:s], '\n'*4, compressed[:], '\n'*4)
print(decompress(compressed, reps, sections, **argdict))


# In[344]:


enc = 'UTF-8'
c = zlib.compress(original.encode(enc))#.decode(enc)
base64.b64encode(c)


# In[67]:


reconstructed = compressed
# u = random.randint(0, len(reconstructed))
# reconstructed = reconstructed[:u] + '?' + reconstructed[u:]


# In[124]:


def detach(text, x, f=''):
    return text[:x] + f + text[x+1:]

variations = []
for i in range(10):
    newstring = compressed
    for j in range(random.randint(1, 400)):
        c = random.choice(list(chars)+['']*2)
        newstring = detach(newstring, random.randint(0, len(newstring)), c)
    variations.append(newstring)

reconstructed = variations[0]

print(decompress(reconstructed[:s], reps, sections, **argdict))
# todo: sort by frequency


# In[208]:


len(hamlet)

