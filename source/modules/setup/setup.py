from modules.builder import builder
from modules.io import dataio
from modules.utils.logger import *
from modules.utils.parser_utils import *

def prepare_model_data(args, conf, device):    
    log = logger(args, 'TrainTest')
    trainObj = builder.builder(args, conf, device)
    trainData = dataio.dataio('Train', args, conf, log.outdir)
    testData = dataio.dataio('Test', args, conf, log.outdir)    
    save_args(args, log.outdir + '/checkpoint/current/')
    return trainObj, trainData, testData, log
