import torch
from modules.setup.setup import *
from modules.config.config import *
from modules.model.model_utils import *
from modules.utils.parser_utils import *
import sys
sys.path.append('..') # add parent directly for importing
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--session_name', default = 'DefaultSession')
parser.add_argument('--training_dir', default = 'DefaultTraing')
parser.add_argument('--test_dir', default = 'DefaultTest')
parser.add_argument('--mode', default='TrainAndTest', choices=['Train', 'Test', 'TrainAndTest'])
parser.add_argument('--agg_type', default='Transformer', choices=['Transformer', 'Pooling'])
parser.add_argument('--batchsize', type=int, default='3')
parser.add_argument('--outdir', default='output')
parser.add_argument('--pretrained', default=None)
parser.add_argument('--num_agg_enc', type=int, default=3)
parser.add_argument('--min_nimg', type=int, default=2)
parser.add_argument('--num_samples', type=int, default=2500)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--args', default=None)
parser.add_argument('--lr_scheduler', default='step')
parser.add_argument('--lr_init_scale', type=float, default=1.0)
parser.add_argument('--encoder_imgsize', type=int, default=256)


def main():
    args = parser.parse_args()
    args= load_args(args, args.args)
    print(args)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    print(torch.__version__)
    conf = setup_configuration()
    trainObj, trainData, testData, logger = prepare_model_data(args, conf, device)
    if args.lr_init_scale > 0:
        trainObj.net.scale_lr(args.lr_init_scale)

    print("GLC Encoder")
    print_model_parameters(trainObj.net.encoder)
    print("Decoder:Aggregation")
    print_model_parameters(trainObj.net.aggregation)
    print("Decoder:Prediction")
    print_model_parameters(trainObj.net.prediction)
    if args.mode in ('TrainAndTest','Train'):
        epochs = 20
    else:
        epochs = 1
    for epoch in range(epochs):
        print(f'Run {epoch+1}-th epoch')
        trainObj.run(args.mode, epoch=epoch, writer=logger,steps_per_test = 200,\
                    traindata=trainData, train_batch_size=args.batchsize, train_shuffle=True, train_loader_imgsize=(512, 512), train_encoder_imgsize=(args.encoder_imgsize, args.encoder_imgsize), train_decoder_imgsize=(512, 512),\
                    testdata=testData, test_batch_size=1, test_shuffle=False, test_loader_imgsize=(512, 512), test_encoder_imgsize=(256,256), test_decoder_imgsize=(512, 512))

if __name__ == '__main__':
    main()

