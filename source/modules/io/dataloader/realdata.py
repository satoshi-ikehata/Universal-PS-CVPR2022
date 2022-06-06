import glob
import os
import numpy as np
import cv2
import re


class dataloader():
    def __init__(self, numberOfImages = None, outdir = '.'):
        self.numberOfImages = numberOfImages
        self.outdir = outdir

    def img_tile(self, imgs, rows, cols, outdir): # [N, h, w, c]
        n, h, w, c = np.shape(imgs)
        print(f'Imge num = {n}')
        if rows * cols <= n:
            img_tiled = []
            for i in range(cols):
                temp = np.reshape(imgs[rows*i:rows*i+rows,:,:,:], (-1, w, 3))
                img_tiled.append(temp)
            img_tiled = np.concatenate(img_tiled, axis = 1)            
            os.makedirs(outdir, exist_ok=True)
            cv2.imwrite(f'{outdir}/tiled.png', (255 * img_tiled[:,:,::-1]).astype(np.uint8))
    def merge_img(self, imgs, merge_num): # [N, h*w, 3]
        imgs_merged = np.zeros(imgs.shape, np.float32)
        for k in range(imgs.shape[0]):
            ids = np.random.permutation(imgs.shape[0])[:merge_num]
            img = imgs[ids, :, :]
            img = np.sum(img, axis=0)
            imgs_merged[k, :, :] = img
        return imgs_merged



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


    def load(self, objlist, objid,  prefix,  margin = 0, loader_imgsize = 256):

        self.objname = re.split(r'\\|/',objlist[objid])[-1]
        self.data_workspace = f'{self.outdir}/{self.objname}'
        os.makedirs(self.data_workspace, exist_ok=True)


        directlist = []
        [directlist.append(p) for p in glob.glob(objlist[objid] + '/%s' % prefix,recursive=True) if os.path.isfile(p)]
        directlist = sorted(directlist)
        distort = True


        if len(directlist) == 0:
            return False
        if os.name == 'posix':
            temp = directlist[0].split("/")
        if os.name == 'nt':
            temp = directlist[0].split("\\")
        img_dir = "/".join(temp[:-1])

        if self.numberOfImages is not None:
            indexset = np.random.permutation(len(directlist))[:self.numberOfImages]
        else:
            indexset = range(len(directlist))
        numberOfImages = np.min([len(indexset), self.numberOfImages])

        imgsize = loader_imgsize
        for i, indexofimage in enumerate(indexset):
            img_path = directlist[indexofimage]
            mask_path = img_dir + '/mask.png'
            # print(img_path)
            img = cv2.cvtColor(cv2.imread(img_path, flags = cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB)

            if i == 0:
                h_ = img.shape[0]
                w_ = img.shape[1]
                h0 = int(h_)
                w0 = int(w_)
                h = imgsize
                w = imgsize
                margin = 4

                N = np.zeros((h, w, 3), np.float32)
                if os.path.isfile(mask_path) and i == 0:
                    mask = (cv2.imread(mask_path, flags = cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) > 0).astype(np.float32)
                    if len(mask.shape) == 3:
                        mask = mask[:, :, 0]
                    rows, cols = np.nonzero(mask)



                    rowmin = np.min(rows)
                    rowmax = np.max(rows)
                    row = rowmax - rowmin
                    colmin = np.min(cols)
                    colmax = np.max(cols)
                    col = colmax - colmin

                    if rowmin - margin <= 0 or rowmax + margin > img.shape[0] or colmin - margin <= 0 or colmax + margin > img.shape[1]:
                        mask = np.float32(cv2.resize(mask, dsize=(h, w),interpolation=cv2.INTER_NEAREST))
                        flag = False
                    else:
                        flag = True



                    if row > col and flag:
                        mask = mask[rowmin-margin:rowmax+margin, np.max([colmin- int(0.5 * (row-col))-margin,0]):np.min([colmax+int(0.5 * (row-col))+margin,img.shape[1]])]
                    elif col >= row and flag:
                        mask = mask[np.max([rowmin-int(0.5*(col-row))-margin,0]):np.min([rowmax+int(0.5*(col-row))+margin, img.shape[0]]), colmin-margin:colmax+margin]
                    if flag == True:
                        mask = np.float32(cv2.resize(mask, dsize=(h, w),interpolation=cv2.INTER_NEAREST))
                elif os.path.isfile(mask_path) == False and i == 0:
                    mask = np.ones((h, w), np.float32)
                    flag = False
                    rowmin = 0
                    rowmax = h
                    colmin = 0
                    colmax = w
                    row = 0
                    col = 0

            if row > col and flag:
                img = img[rowmin-margin:rowmax+margin, np.max([colmin- int(0.5 * (row-col))-margin,0]):np.min([colmax+int(0.5 * (row-col))+margin,img.shape[1]]), :]
            elif col >= row and flag:
                img = img[np.max([rowmin-int(0.5*(col-row))-margin,0]):np.min([rowmax+int(0.5*(col-row))+margin, img.shape[0]]), colmin-margin:colmax+margin, :]


            img = cv2.resize(img, dsize=(h, w),interpolation=cv2.INTER_NEAREST)



            if img.dtype == 'uint8':
                bit_depth = 255.0
            if img.dtype == 'uint16':
                bit_depth = 65535.0

            img = np.float32(img) / bit_depth

            if i == 0:
                I = np.zeros((len(indexset), h, w, 3), np.float32) # [N, h, w, c]

            I[i, :, :, :] = img


        self.img_tile(I, 3, 3, self.data_workspace)


        I = np.reshape(I, (-1, h * w, 3))

        temp = np.mean(I[:, mask.flatten()==1,:], axis=2)
        mean = np.mean(temp, axis=1)
        """Normalization of Data"""
        I /= mean.reshape(-1, 1, 1)


        #
        I = np.transpose(I, (1, 2, 0))
        I = I.reshape(h, w, 3, numberOfImages)
        mask = (mask.reshape(h, w, 1)).astype(np.float32) # 1, h, w
        I = I * mask[:, :, :, np.newaxis]


        h = h0
        w = w0


        h = mask.shape[0]
        w = mask.shape[1]

        self.h = h
        self.w = w
        self.I = I
        self.N = N
        self.mask = mask
        print('Test Data Statistics:')
        print(f'Objname is {self.objname}')
        print(f'Loaded Image Size is {(h, w)}')
        print(f'Number of Images is {I.shape[3]}')
