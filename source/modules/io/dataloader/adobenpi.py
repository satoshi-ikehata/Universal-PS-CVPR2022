import glob
import os
import cv2
import numpy as np
import os


def horizontal_flip(I, N, M): # [? h, w, ? ...]
    I = I[:, ::-1, :, :]
    N = N[:, ::-1, :]
    N[:, :, 0] *= -1
    M = M[:, ::-1, :]
    return I.copy(), N.copy(), M.copy()

def vertical_flip(I, N, M):
    I = I[::-1, :, :, :]
    N = N[::-1, :, :]
    N[:, :, 1] *= -1
    M = M[::-1, :, :]
    return I.copy(), N.copy(), M.copy()

def rotate(I, N, M):
    I = I.transpose(1, 0, 2, 3)
    N = N.transpose(1, 0, 2)
    N = N[:, :, [1,0,2]]
    N[:, :, 0] *= -1
    N[:, :, 1] *= -1
    M = M.transpose(1, 0, 2)
    return I.copy(), N.copy(), M.copy()

def color_swap(I):
    for k in range(I.shape[3]):
        ids = np.random.permutation(3)
        I[:, :, :, k] = I[:, :, ids, k]
    return I.copy()

def blend_augumentation(I):
        # blending
        k = 0.3
        alpha = k + (1-k) * np.random.rand()
        mean_img = np.mean(I, axis=0, keepdims=True)
        I = alpha * I + (1 - alpha) * mean_img
        return I.copy()

def quantize_augumentation(I):
        for k in range(I.shape[3]):
            temp = 255.0 * (I[:, :, :, k] / np.max(I[:, :, :, k]))
            temp = temp.astype(np.uint8)
            I[:, :, :, k] = temp/255.0
        return I.copy()


class dataloader():
    def __init__(self, numberOfImages = None):
        self.numberOfImages = numberOfImages

    def psfcn_normalize(self, imgs): # [NLight, H, W ,C]
        h, w, c = imgs[0].shape
        imgs = [img.reshape(-1, 1) for img in imgs]
        img = np.hstack(imgs)
        norm = np.sqrt((img * img).clip(0.0).sum(1))
        img = img / (norm.reshape(-1,1) + 1e-10)
        imgs = np.split(img, img.shape[1], axis=1)
        imgs = [img.reshape(h, w, -1) for img in imgs]
        print('PSFCN_NORMALIZED')
        return imgs

    def load(self, objlist, objid,  suffix, scale = 1.0, margin = 0):

        self.objname = objlist[objid].split('/')[-1]
        directlist = []
        [directlist.append(p) for p in glob.glob(objlist[objid] + '/%s' % suffix,recursive=True) if os.path.isfile(p)]
        directlist = sorted(directlist)


        if len(directlist) == 0:
            return False
        if os.name == 'posix':
            temp = directlist[0].split("/")
        if os.name == 'nt':
            temp = directlist[0].split("\\")
        img_dir = "/".join(temp[:-1])
        base_path = img_dir + '/baseColor.tif'
        rough_path = img_dir + '/roughness.tif'
        metal_path = img_dir + '/metal.tif'
        depth_path = img_dir + '/depth.exr'
        if os.path.isfile(base_path):
            os.remove(base_path)
        if os.path.isfile(rough_path):
            os.remove(rough_path)
        if os.path.isfile(metal_path):
            os.remove(metal_path)
        if os.path.isfile(depth_path):
            os.remove(depth_path)

        if self.numberOfImages is not None:
            indexset = np.random.permutation(len(directlist))[:self.numberOfImages]
        else:
            indexset = range(len(directlist))

        for i, indexofimage in enumerate(indexset):
            img_path = directlist[indexofimage]
            if i == 0:
                img = cv2.cvtColor(cv2.imread(img_path, flags = cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB)
                h_ = img.shape[0]
                w_ = img.shape[1]
                h0 = int(scale * h_)
                w0 = int(scale * w_)
                img = cv2.resize(img, dsize=None, fx= scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                h = img.shape[0]
                w = img.shape[1]

            else:
                img = cv2.resize(cv2.cvtColor(cv2.imread(img_path, flags = cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB), dsize=None, fx=scale, fy=scale,interpolation=cv2.INTER_NEAREST)


            if img.dtype == 'uint8':
                bit_depth = 255.0
            if img.dtype == 'uint16':
                bit_depth = 65535.0

            img = np.float32(img) / bit_depth

            if i == 0:
                mask = []
                I = np.zeros((len(indexset), h, w, 3), np.float32)
            I[i, :, :, :] = img
            nml_path = img_dir + '/normal.tif'

            if os.path.isfile(nml_path) and i == 0:
                N = np.float32(cv2.resize(cv2.cvtColor(cv2.imread(nml_path, flags = cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB), dsize=None, fx=scale, fy=scale,interpolation=cv2.INTER_NEAREST))/65535.0
                N = 2 * N - 1
                mask = np.abs(1 - np.sqrt(np.sum(N * N, axis=2))) < 1.0e-3

        I = np.reshape(I, (-1, h * w, 3))
        if len(mask) == 0:
            print(nml_path)
        I[:, mask.flatten()==0, :] = 0


        temp = np.mean(I[:, mask.flatten()==1,:], axis=2)
        mean = np.mean(temp, axis=1)
        
        """Normalization of Data"""
        I /= mean.reshape(-1,1,1)
        I = np.transpose(I, (1, 2, 0))
        I = I.reshape(h, w, 3, self.numberOfImages)
        mask = (mask.reshape(h, w, 1)).astype(np.float32) # h, w, w

        h = h0
        w = w0

        prob = 0.5
        if np.random.rand() > prob:
            I, N, mask = horizontal_flip(I, N, mask)
        if np.random.rand() > prob:
            I, N, mask = vertical_flip(I, N, mask)
        if np.random.rand() > prob:
            I, N, mask = rotate(I, N, mask)
        if np.random.rand() > prob:
            I = color_swap(I)



        h = mask.shape[0]
        w = mask.shape[1]
        self.h = h
        self.w = w
        self.I = I
        self.N = N
        self.mask = mask