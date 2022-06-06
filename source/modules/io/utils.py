import torch.nn as nn
import torch
import math
import numpy as np

# def random_light_sampling(L, minLightNum, maxLightNum, minLightRadius = 20, maxLightRadius = 90):
#     elevation = 180*np.arccos(L[:,2])/np.pi
#     valid_light_list = ((elevation[:] < np.random.randint(minLightRadius,maxLightRadius)).nonzero())[0]
#     if len(valid_light_list) > minLightNum:
#         indeces = np.random.permutation(valid_light_list)
#         if len(valid_light_list) < maxLightNum:
#             n = np.random.randint(minLightNum, len(valid_light_list))
#         else:
#             n = np.random.randint(minLightNum, maxLightNum)
#         idx = indeces[:n]
#     else:
#         idx = valid_light_list
#     return idx

def random_light_sampling(L, min_nimg, max_nimg):
    indeces = np.random.permutation(L.shape[0])
    idx = indeces[:np.random.randint(min_nimg, max_nimg+1)]
    return idx


def fix_light_sampling(L, LightNum):
    indeces = np.random.permutation(L.shape[0])
    idx = indeces[:LightNum]
    return idx

def crop_index(u, v, w, h, psize):
    p = psize//2
    urange = range(u - p + 1, u + p + 1)
    vrange = range(v - p + 1, v + p + 1)
    uu, vv = np.meshgrid(urange, vrange)
    valid = np.nonzero((uu >= 0) * (uu < w) * (vv >= 0) * (vv < h))
    return vec2ind(uu[valid], vv[valid], w, h).flatten(), vec2ind(valid[1], valid[0], psize, psize).flatten(), vec2ind(psize // 2 - 1, psize // 2 - 1, psize, psize)

def ind2vec(ind, w, h): # u, v [0, w-1];[0,h-1]
    v = ind // w
    u = ind - v * w
    return u, v

def vec2ind(u, v, w, h):
    return v * w + u

def split_random(index, num_split):
    indexlist = np.array_split(np.random.permutation(index),num_split)
    numelements = [len(indexlist[k]) for k in range(num_split)]
    indexlist = [indexlist[k][:min(numelements)] for k in range(num_split)]
    return indexlist
