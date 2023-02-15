import torch
import numpy as np
from tqdm import tqdm
from modules.model import model
import datetime
import cv2



class builder():
    def __init__(self, args, conf, device):
        self.img_channels = conf.img_channels
        self.device = device
        self.net = model.Net(args, device)
        if args.pretrained is not None:
            self.net.load_models(args.pretrained)
            print(f'Pretrained Model Loaded! initial LR is {self.net.print_lr()}')

    def run(self, mode, epoch = 0, writer = None, steps_per_test = 200,
            traindata = None, train_batch_size = 3, train_shuffle = True, train_loader_imgsize = None, train_encoder_imgsize = None, train_decoder_imgsize = None,
            testdata = None, test_batch_size = 3, test_shuffle = False, test_loader_imgsize = None, test_encoder_imgsize = None, test_decoder_imgsize = None):

        if mode == 'Train' or mode == 'TrainAndTest':
            self.net.set_mode('Train')
            traindata.loader_imgsize = train_loader_imgsize
            testdata.loader_imgsize = test_loader_imgsize
            print(f'Train Batch Size is {train_batch_size}')
            train_data_loader = torch.utils.data.DataLoader(traindata, batch_size = train_batch_size, shuffle=train_shuffle, num_workers=4, pin_memory=True)
            test_data_loader = torch.utils.data.DataLoader(testdata, batch_size = test_batch_size, shuffle=test_shuffle, num_workers=4, pin_memory=True)
            losses = 0
            cnt = 0
            for batch in tqdm(train_data_loader, leave=False):
                global_step = epoch * len(train_data_loader) + cnt

                """ test every steps_per_test """
                if np.mod(global_step, steps_per_test) == 0 and mode == 'TrainAndTest' and cnt > 0:
                    self.net.set_mode('Test')
                    for batch_test in test_data_loader:
                        _, output, input = self.net.step(batch_test, decoder_imgsize=test_decoder_imgsize, encoder_imgsize=test_encoder_imgsize) # output = [B, 3, h, w]
                        cv2.imwrite(f'{testdata.data.data_workspace}/input.png', 255 * input[0,:,:,:].transpose(1,2,0)[:,:,::-1])
                        cv2.imwrite(f'{testdata.data.data_workspace}/normal.png', 255 * output[0,:,:,:].transpose(1,2,0)[:,:,::-1])
                    self.net.set_mode('Train')
                    savedir = writer.outdir + '/checkpoint/current'
                    self.net.save_models(savedir)

                # TRAIN STEP
                loss, output, input  = self.net.step(batch, decoder_imgsize=train_decoder_imgsize, encoder_imgsize=train_encoder_imgsize) # output = [B, 3, h, w]
                losses += loss
                cnt += 1

            writer.add('Train Loss', losses/cnt, epoch, 'Scalar')
            writer.add('Learning Rate', self.net.print_lr(), epoch, 'Scalar')
            savedir = writer.outdir + '/checkpoint/' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.net.save_models(savedir)
            self.net.scheduler_step()
            return losses/cnt

        if mode == 'Test':
            cnt = 0
            global_step = epoch
            self.net.set_mode('Test')
            testdata.loader_imgsize = test_loader_imgsize
            test_data_loader = torch.utils.data.DataLoader(testdata, batch_size = test_batch_size, shuffle=test_shuffle, num_workers=0, pin_memory=True)
            for i, batch in enumerate(test_data_loader):
                global_step = epoch * len(test_data_loader) + cnt
                _, output, input = self.net.step(batch, decoder_imgsize=test_decoder_imgsize, encoder_imgsize=test_encoder_imgsize) # output = [B, 3, h, w]
                cv2.imwrite(f'{testdata.data.data_workspace}/input.png', 255 * input[0,:,:,:].transpose(1,2,0)[:,:,::-1])
                cv2.imwrite(f'{testdata.data.data_workspace}/normal.png', 255 * output[0,:,:,:].transpose(1,2,0)[:,:,::-1])
                cnt +=1
