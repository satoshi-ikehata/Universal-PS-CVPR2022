import os
import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP

def loadmodel(model, filename):
    params = torch.load('%s' % filename)
    model.load_state_dict(params,strict=False)
    print('Load %s' % filename)
    return model

def loadoptimizer(optimizer, filename):
    params = torch.load('%s' % filename)
    optimizer.load_state_dict(params)
    print('Load %s' % filename)
    return optimizer

def loadscheduler(scheduler, filename):
    params = torch.load('%s' % filename)
    scheduler.load_state_dict(params)
    print('Load %s' % filename)
    return scheduler

def savemodel(model, filename):
    print('Save %s' % filename)
    torch.save(model.state_dict(), filename)

def saveoptimizer(optimizer, filename):
    print('Save %s' % filename)
    torch.save(optimizer.state_dict(), filename)

def savescheduler(scheduler, filename):
    print('Save %s' % filename)
    torch.save(scheduler.state_dict(), filename)

def optimizer_setup_Adam(net, lr = 0.0001, init=True, stype='step'):
    print(f'optimizer (Adam) lr={lr}')
    if init==True:
        net.init_weights()
    net = torch.nn.DataParallel(net)
    optim_params = [{'params': net.parameters(), 'lr': lr},] # confirmed
    optimizer = torch.optim.Adam(optim_params, betas=(0.9, 0.999), weight_decay=0)
    if stype == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30, eta_min=0, last_epoch=-1)
        print('cosine aneealing learning late scheduler')
    if stype == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
        print('step late scheduler x0.8 decay')
    return net, optimizer, scheduler

def optimizer_setup_SGD(net, lr = 0.01, momentum= 0.9, init=True):
    print(f'optimizer (SGD with momentum) lr={lr}')
    if init==True:
        net.init_weights()
    net = torch.nn.DataParallel(net)
    optim_params = [{'params': net.parameters(), 'lr': lr},] # confirmed
    return net, torch.optim.SGD(optim_params, momentum=momentum, weight_decay=1e-4, nesterov=True)

def optimizer_setup_AdamW(net, lr = 0.001, init=True, stype='step'):
    print(f'optimizer (AdamW) lr={lr}')
    if init==True:
        net.init_weights()
    net = torch.nn.DataParallel(net)
    optim_params = [{'params': net.parameters(), 'lr': lr},] # confirmed
    optimizer = torch.optim.AdamW(optim_params, betas=(0.9, 0.999), weight_decay=0.01)
    if stype == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30, eta_min=0, last_epoch=-1)
        print('cosine aneealing learning late scheduler')
    if stype == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
        print('step late scheduler x0.8 decay')
    return net, optimizer, scheduler

def mode_change(net, Training):
    if Training == True:
        for param in net.parameters():
            param.requires_grad = True
        net.train()
    if Training == False:
        for param in net.parameters():
            param.requires_grad = False
        net.eval()


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def loadCheckpoint(path, model, cuda=True):
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

def saveCheckpoint(save_path, epoch=-1, model=None, optimizer=None, records=None, args=None):
    state   = {'state_dict': model.state_dict(), 'model': args.model}
    records = {'epoch': epoch, 'optimizer':optimizer.state_dict(), 'records': records,
            'args': args}
    torch.save(state, os.path.join(save_path, 'checkp_%d.pth.tar' % (epoch)))
    torch.save(records, os.path.join(save_path, 'checkp_%d_rec.pth.tar' % (epoch)))



def masking(img, mask):
    # img [B, C, H, W]
    # mask [B, 1, H, W] [0,1]
    img_masked = img * mask.expand((-1, img.shape[1], -1, -1))
    return img_masked

def print_model_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])


    print('# parameters: %d' % params)


def angular_error(x1, x2, mask = None):

    if mask is not None:
        dot = torch.sum(x1 * x2 * mask, dim=1, keepdim=True)
        dot = torch.max(torch.min(dot, torch.Tensor([1.0-1.0e-12])), torch.Tensor([-1.0+1.0e-12]))
        emap = torch.abs(180 * torch.acos(dot)/np.pi) * mask
        mae = torch.sum(emap) / torch.sum(mask)
        return mae
    if mask is None:
        dot = torch.sum(x1 * x2, dim=1, keepdim=True)
        dot = torch.max(torch.min(dot, torch.Tensor([1.0-1.0e-12])), torch.Tensor([-1.0+1.0e-12]))
        error = torch.abs(180 * torch.acos(dot)/np.pi)
        return error

def write_errors(filepath, error, trainid, numimg, objname = []):
    dt_now = datetime.datetime.now()
    print(filepath)

    if len(objname) > 0:
        with open(filepath, 'a') as f:
            f.write('%s %03d %s %02d %.2f\n' % (dt_now, numimg, objname, trainid, error))
    else:
        with open(filepath, 'a') as f:
            f.write('%s %03d %02d %.2f\n' % (dt_now, numimg, trainid, error))


def save_nparray_as_hdf5(self, a, filename):
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset('dataset_1', data=a)
    h5f.close()
