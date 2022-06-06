from .model_utils import *
from ..utils.show_tensor import *
from ..utils.ind2sub import *
import os
import glob
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import kaiming_normal_, trunc_normal_


from .utils import Transformer
from .utils.folked import swin_transformer
from .utils.folked import uper

class PredictionHead(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(PredictionHead, self).__init__()
        modules_regression = []
        modules_regression.append(nn.Linear(dim_input, dim_input//2))
        modules_regression.append(nn.ReLU(inplace=False))
        modules_regression.append(nn.Linear(dim_input//2, dim_output))
        self.regression = nn.Sequential(*modules_regression)

    def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):# or isinstance(m, nn.Linear):
                    kaiming_normal_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()
                if isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.LayerNorm):
                    m.bias.data.zero_()
                    m.weight.data.fill_(1.0)
    def forward(self, x):
        return self.regression(x)

class Encoder(nn.Module):
    def __init__(self, input_nc):
        super(Encoder, self).__init__()
        back = []
        fuse = []

        in_channels = (96, 192, 384, 768)
        back.append(swin_transformer.SwinTransformer(in_chans=input_nc))
        print('Encoder Backbone  is SwinTransformer')

        fuse.append(uper.UPerHead(in_channels = in_channels))
        attn = []
        for i in range(len(in_channels)):
            attn.append(self.attn_block(in_channels[i]))
        self.attn = nn.Sequential(*attn)

        self.backbone = nn.Sequential(*back)
        self.fusion = nn.Sequential(*fuse)

    def init_weights(self, zero = False):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):# or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)

    def attn_block(self, dim, num_attn = 1):
        attn = []
        for k in range(num_attn):
            attn.append(Transformer.SAB(dim, dim, num_heads=8, ln=False, attention_dropout = 0.1, dim_feedforward = 2 * dim))
        return nn.Sequential(*attn)

    def conv_block(self, in_planes, out_planes, kernel_size):
        conv = nn.Sequential(
                nn.Conv2d(in_planes, out_planes,
                            kernel_size=kernel_size, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=False),
            )
        return conv

    def forward(self, x):
        """Standard forward
        INPUT: img [B, N, Cin, H, W]
        OUTPUT: [B, N, Cout, H/4, W/4]
        """

        feats = []
        for k in range(x.shape[1]):
            feats.append(self.backbone(x[:, k, :, :, :]))

        out = [] # layer first
        for l in range(len(feats[0])):
            in_fuse = []
            for k in range(x.shape[1]):
                in_fuse.append(feats[k][l])
            in_fuse = torch.stack(in_fuse, dim=1) # B, N, C, H//4, W//4
            B, N, C, H, W = in_fuse.size()
            in_fuse = in_fuse.permute(0,3,4,1,2).reshape(-1, N, C)
            out_fuse = self.attn[l](in_fuse).reshape(B, H, W, N, C).permute(0,3,4,1,2) # B, N, C, H, W
            out.append(out_fuse)

        feats = []
        for k in range(x.shape[1]):
            feats.append((out[0][:,k,:,:,:], out[1][:,k,:,:,:], out[2][:,k,:,:,:], out[3][:,k,:,:,:]))

        outs = []
        for k in range(x.shape[1]):
            outs.append(self.fusion(feats[k]))
        feats = torch.stack(outs, 1) # [B, N, C, H/4, W/4]
        return feats

class Net():
    def __init__(self, args, device):
        self.device = device
        self.min_nimg = args.min_nimg
        self.num_samples = args.num_samples
        self.model_name = args.session_name
        self.num_agg_enc = args.num_agg_enc
        self.agg_type = args.agg_type
        lr = args.lr
        stype = args.lr_scheduler

        self.encoder = Encoder(4).to(self.device)
        [self.encoder, self.optimizer_encoder, self.scheduler_encoder] = optimizer_setup_AdamW(self.encoder, lr = lr, init=True, stype=stype) # ResNet101: lr =AdamW, lr=0.0001
       
        self.aggregation = Transformer.TransformerLayer(dim_input = 256 + 3, num_enc_sab = self.num_agg_enc, num_outputs = 1, dim_hidden=384, dim_feedforward = 1024, num_heads=8, ln=True, attention_dropout=0.1).to(self.device)
        dim_aggout = 384
        [self.aggregation, self.optimizer_aggregation, self.scheduler_aggregation] = optimizer_setup_AdamW(self.aggregation, lr = lr, init=True, stype=stype)


        self.prediction = PredictionHead(dim_aggout, 3).to(self.device) # No urcainty
        [self.prediction, self.optimizer_prediction, self.scheduler_prediction] = optimizer_setup_AdamW(self.prediction, lr = lr, init=True, stype=stype)
        self.criterionL2 = nn.MSELoss(reduction = 'sum').to(self.device)
        print(f'Session {self.model_name}: #images >= {self.min_nimg}, #samples = {self.num_samples}')

    def set_mode(self, mode):
        if  mode in 'Train':
            print(f'{self.model_name}, TrainMode')
            self.mode = 'Train'
            mode_change(self.encoder, True)
            mode_change(self.aggregation, True)
            mode_change(self.prediction, True)

        elif mode in 'Test':
            print(f'{self.model_name}, TestMode')
            self.mode = 'Test'
            mode_change(self.encoder, False)
            mode_change(self.aggregation, False)
            mode_change(self.prediction, False)
        else:
            print("Mode must be from [Train, Validation, Test]", file=sys.stderr)
            sys.exit(1)

    def scale_lr(self, scale):
        print('learning rate updated  %.5f -> %.5f' % (self.optimizer_encoder.param_groups[0]['lr'], self.optimizer_encoder.param_groups[0]['lr'] * scale))
        self.optimizer_encoder.param_groups[0]['lr'] *= scale
        self.optimizer_aggregation.param_groups[0]['lr'] *= scale
        self.optimizer_prediction.param_groups[0]['lr'] *= scale

    def print_lr(self):
        return self.optimizer_encoder.param_groups[0]['lr']

    def scheduler_step(self):
        print('current learning rate %.5f' % (self.optimizer_encoder.param_groups[0]['lr']))
        self.scheduler_encoder.step()
        self.scheduler_aggregation.step()
        self.scheduler_prediction.step()
        print('updated learning rate %.5f' % (self.optimizer_encoder.param_groups[0]['lr']))

    def save_models(self, dirpath):
        os.makedirs(dirpath, exist_ok = True)
        savemodel(self.encoder, dirpath + f'/{self.model_name}_enc.pytmodel')
        saveoptimizer(self.optimizer_encoder, dirpath + f'/{self.model_name}_enc.optimizer')
        savescheduler(self.scheduler_encoder, dirpath + f'/{self.model_name}_enc.scheduler')

        savemodel(self.aggregation, dirpath + f'/{self.model_name}_agg.pytmodel')
        saveoptimizer(self.optimizer_aggregation, dirpath + f'/{self.model_name}_agg.optimizer')
        savescheduler(self.scheduler_aggregation, dirpath + f'/{self.model_name}_agg.scheduler')

        savemodel(self.prediction, dirpath + f'/{self.model_name}_pred.pytmodel')
        saveoptimizer(self.optimizer_prediction, dirpath + f'/{self.model_name}_pred.optimizer')
        savescheduler(self.scheduler_prediction, dirpath + f'/{self.model_name}_pred.scheduler')

    def load_models(self, dirpath):
        pytmodel = glob.glob(f'{dirpath}/*_enc.pytmodel')
        self.encoder = loadmodel(self.encoder, pytmodel[0])
        pytmodel = glob.glob(f'{dirpath}/*_agg.pytmodel')
        self.aggregation = loadmodel(self.aggregation, pytmodel[0])
        pytmodel = glob.glob(f'{dirpath}/*_pred.pytmodel')
        self.prediction = loadmodel(self.prediction, pytmodel[0])

        scheduler = glob.glob(f'{dirpath}/*_enc.scheduler')
        self.scheduler_encoder = loadscheduler(self.scheduler_encoder, scheduler[0])
        scheduler = glob.glob(f'{dirpath}/*_agg.scheduler')
        self.scheduler_aggregation = loadscheduler(self.scheduler_aggregation, scheduler[0])
        scheduler = glob.glob(f'{dirpath}/*_pred.scheduler')
        self.scheduler_prediction = loadscheduler(self.scheduler_prediction, scheduler[0])

        optimizer = glob.glob(f'{dirpath}/*_enc.optimizer')
        self.optimizer_encoder = loadoptimizer(self.optimizer_encoder, optimizer[0])
        optimizer = glob.glob(f'{dirpath}/*_agg.optimizer')
        self.optimizer_aggregation= loadoptimizer(self.optimizer_aggregation, optimizer[0])
        optimizer = glob.glob(f'{dirpath}/*_pred.optimizer')
        self.optimizer_prediction = loadoptimizer(self.optimizer_prediction, optimizer[0])

    def step(self, batch, decoder_imgsize, encoder_imgsize=None):

        img = batch[0].permute(0, 4, 1, 2, 3).to(self.device)# B N C H W
        nml = batch[1].to(self.device)
        mask = batch[2].to(self.device) # B 1 H W

        min_nimg = self.min_nimg
        if self.mode in 'Train' and img.shape[1] >= min_nimg:
            numI = np.random.randint(img.shape[1]-min_nimg+1)+min_nimg
            imgid = np.random.permutation(range(img.shape[1]))[:numI]
            img = img[:, imgid, :, :, :]


        """ Feature Encoding Stage"""
        B = img.shape[0]
        N = img.shape[1]
        C = img.shape[2]
        H = img.shape[3]
        W = img.shape[4]
        if encoder_imgsize is None:
            data = torch.cat([img * mask.unsqueeze(1).expand(-1, img.shape[1], -1, -1, -1), mask.unsqueeze(1).expand(-1, img.shape[1], -1, -1, -1)], dim=2)
        else:
            img_ = img.reshape(-1, C, H, W)
            img_ = F.interpolate(img_, size=encoder_imgsize, mode='bilinear', align_corners=False).reshape(B, N, C, encoder_imgsize[0], encoder_imgsize[1])
            mask_ = F.interpolate(mask, size=encoder_imgsize, mode='nearest')
            data = torch.cat([img_ * mask_.unsqueeze(1).expand(-1, img.shape[1], -1, -1, -1), mask_.unsqueeze(1).expand(-1, img.shape[1], -1, -1, -1)], dim=2)

        feats = self.encoder(data) # torch.Size([B, N, 256, H/4, W/4]) [img, mask]

        """Process at Canonical Resolution"""
        B = feats.shape[0]
        N = feats.shape[1]
        C = feats.shape[2]
        H = feats.shape[3]
        W = feats.shape[4]

        img_ = img.reshape(-1, img.shape[2], img.shape[3], img.shape[4])
        img_ = F.interpolate(img_, size= (H, W), mode='bilinear', align_corners=False).reshape(img.shape[0], img.shape[1], img.shape[2], H, W)
        m = F.interpolate(mask, size = (H, W), mode='nearest')
        n = F.normalize(F.interpolate(nml, size = (H, W), mode='bilinear', align_corners=False), p=2, dim=1)

        loss = 0
        nout = torch.zeros(B, H * W, 3).to(self.device)

        for b in range(B):
            m_ = m[b, :, :, :].reshape(-1, H * W).permute(1,0)
            n_ = n[b, :, :, :].reshape(-1, H * W).permute(1,0)
            ids = np.nonzero(m_>0)[:,0]
            f = feats[b, :, :, :, :].reshape(-1, C, H * W).permute(2, 0, 1)
            f = f[ids, :, :]
            n_ = n_[ids, :]
            o = img_[b, :, :, :, :].reshape(-1, 3, H * W).permute(2, 0, 1)
            o = o[ids, :, :]
            x = torch.cat([o, f], dim=2) 
            feat_gg = self.aggregation(x)
            out_nml = self.prediction(feat_gg)
            nout_ = F.normalize(out_nml[:, :3],dim=1, p=2)
            nout[b, ids, :] = nout_            
            loss += self.criterionL2(nout_, n_) / len(ids)
        nout_low = nout.permute(0, 2, 1).reshape(B, 3, H, W)
        mask_low = m

        """Prediction at Original Resolution"""
        img_ = img.reshape(-1, img.shape[2], img.shape[3], img.shape[4])
        img_ = F.interpolate(img_, size= decoder_imgsize, mode='bilinear', align_corners=False).reshape(img.shape[0], img.shape[1], img.shape[2], decoder_imgsize[0], decoder_imgsize[1])
        m = F.interpolate(mask, size = decoder_imgsize, mode='nearest')
        n = F.normalize(F.interpolate(nml, size = decoder_imgsize, mode='bilinear', align_corners=False), p=2, dim=1)

        B = img.shape[0]
        N = img.shape[1]
        C = feats.shape[2] + img.shape[2]
        H = decoder_imgsize[0]
        W = decoder_imgsize[1]


        if self.mode in 'Train':
            nout = torch.zeros(B, H * W, 3).to(self.device)
            numMaxSamples = self.num_samples
            for b in range(B):
                m_ = m[b, :, :, :].reshape(-1, H * W).permute(1,0)
                n_ = n[b, :, :, :].reshape(-1, H * W).permute(1,0)

                ids = np.nonzero(m_>0)[:,0]
                if len(ids) > numMaxSamples:
                    ids = ids[np.random.permutation(len(ids))][:numMaxSamples]

                coords = ind2coords((H, W), ids)

                feat = feats[b, :, :, :, :]
                n_ = n_[ids, :]

                x = []
                for k in range(N):
                    f = F.grid_sample(feat[[k], :, :, :], coords.to(self.device), mode='bilinear', align_corners=False).squeeze().permute(1,0)
                    o = img_[b, k, :, :, :]
                    o = o.reshape(o.shape[0], o.shape[1] * o.shape[2]).permute(1,0)
                    o = o[ids, :]
                    x.append(torch.cat([o, f], dim=1))
                x = torch.stack(x, 1)

                feat_gg = self.aggregation(x)
                out_nml = self.prediction(feat_gg)
                nout_ = F.normalize(out_nml[:, :3],dim=1, p=2)
                nout[b, ids, :] = nout_

                loss += self.criterionL2(nout_, n_) / len(ids)
           

            self.optimizer_encoder.zero_grad()
            self.optimizer_aggregation.zero_grad()
            self.optimizer_prediction.zero_grad()
            loss.backward()
            self.optimizer_encoder.step()
            self.optimizer_aggregation.step()
            self.optimizer_prediction.step()
            nout_high = nout.permute(0, 2, 1).reshape(B, 3, H, W)
            mask_high = m

        if self.mode in 'Test':
            nout = torch.zeros(B, H * W, 3).to(self.device)
            loss = torch.Tensor([0])
            numMaxSamples = 10000
            for b in range(B):
                m_ = m[b, :, :, :].reshape(-1, H * W).permute(1,0)
                n_ = n[b, :, :, :].reshape(-1, H * W).permute(1,0)
                ids = np.nonzero(m_>0)[:,0].cpu()
                if len(ids) > 10000:
                    num_split = len(ids) // 10000
                    ids = np.array_split(ids, num_split)
                else:
                    ids = [ids]
                feat = feats[b, :, :, :, :]
                for p in range(len(ids)):
                    x = []
                    coords = ind2coords((H, W), ids[p])
                    for k in range(N):
                        f = F.grid_sample(feat[[k], :, :, :], coords.to(self.device), mode='bilinear', align_corners=False).squeeze().permute(1,0)
                        o = img_[b, k, :, :, :]
                        o = o.reshape(o.shape[0], o.shape[1] * o.shape[2]).permute(1,0)
                        o = o[ids[p], :]
                        x.append(torch.cat([o, f], dim=1))
                    x = torch.stack(x, 1)
                    feat_gg = self.aggregation(x)
                    out_nml = self.prediction(feat_gg)
                    nout_ = F.normalize(out_nml[:, :3],dim=1, p=2)
                    nout[b, ids[p], :] = nout_
            nout_high = nout.permute(0, 2, 1).reshape(B, 3, H, W)
            mask_high = m

        output_low = F.interpolate(0.5*(nout_low+1) * mask_low, scale_factor = 4 * decoder_imgsize[0]/encoder_imgsize[0], mode='bilinear', align_corners=False)
        output = torch.cat([output_low, 0.5*(nout_high+1)* mask_high], dim=3)
        output = F.interpolate(output, size=None, scale_factor = 2.0, mode='bilinear', align_corners=False)
        input = F.relu(img.permute(0, 2, 3, 1, 4).reshape(-1, img.shape[2], img.shape[3], img.shape[4] * img.shape[1]))
        input[input > 1] = 1

        return loss.cpu().detach().numpy(), output.cpu().detach().numpy(), input.cpu().detach().numpy()
