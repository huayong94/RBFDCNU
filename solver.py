import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from config import config
from Controllers import BaseController
from DataLoader import (Caseset, CentralCropTensor, CollateGPU, Dataset,
                        RandomAffineTransform, RandomMirrorTensor2D)
from Networks import BaseNetwork
from Utils import EarlyStopping, SaveCheckpoint, LoadCheckpoint


class Solver:
    """[class]
    An entry class of training of testing networks throught network
        constroller and a config file.
    """
    def __init__(self, config=config):
        """init function
        Args:
            config (dict, optional): config dict contianing all params of
                                     solver. Defaults to config.
        """
        self.config = config

        os.environ["CUDA_VISIBLE_DEVICES"] = self.config['GPUNo']
        
        self.getController()

        # if mode is Train, create a new directory to save model
        if self.config['mode'] == 'Train':
            model_name = self.config['network']
            name = self.config['name']
            self.model_save_name = '{}-{}-{}'.format(
                name, self.net.name, time.strftime('%Y%m%d%H%M%S'))
            self.model_save_path = os.path.join(
                self.config['Train']['model_save_dir'], model_name,
                self.model_save_name)
            self.getLogger()
        elif self.config['mode'] == 'Test':
            mode = self.config['mode']
            self.model_save_path = self.config[mode]['model_save_path']
            self.model_save_name = os.path.basename(self.model_save_path)

    def getController(self):
        print('loading controller...')
        name = self.config['network']
        self.net: BaseNetwork = self.config[name]['network'](
            **self.config[name]['params'])
        self.controller: BaseController = self.config[name]['controller'](
            self.net)
        self.controller.cuda()

    def getLogger(self):
        self.logger = SummaryWriter(os.path.join(self.model_save_path, 'log'))

    def getTrainDataloader(self):
        training_list_path = self.config['dataset']['training_list_path']
        pair_dir = self.config['dataset']['pair_dir']

        dataset = Dataset(training_list_path, pair_dir)
        self.train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['Train']['batch_size'],
            shuffle=True,
            num_workers=0,
            collate_fn=CollateGPU(transforms=transforms.Compose([
                RandomAffineTransform(),
                CentralCropTensor((200, 200), (128, 128)),
                RandomMirrorTensor2D()
            ])),
            pin_memory=False)

    def getValidationDataloader(self):
        validation_list_path = self.config['dataset']['validation_list_path']
        pair_dir = self.config['dataset']['pair_dir']

        dataset = Caseset(validation_list_path, pair_dir,
                          self.config['dataset']['resolution_path'])
        self.validation_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False)

    def saveCheckpoint(self, net, epoch):
        if epoch % self.config['Train']['save_checkpoint_step']:
            pass
        else:
            SaveCheckpoint(net, self.model_save_path, epoch)

    def train(self):
        # data loading
        print('loading data...')
        self.getTrainDataloader()
        self.getValidationDataloader()

        params = self.config[self.config['network']]['params']
        for key in params:
            self.logger.add_hparams({key: str(params[key])}, {})

        # early stop
        earlystop = EarlyStopping(
            min_delta=self.config['Train']['earlystop']['min_delta'],
            patience=self.config['Train']['earlystop']['patience'],
            model_save_path=self.model_save_path)

        # train
        print('training...')
        return self.controller.train(
            self.train_dataloader,
            self.validation_dataloader,
            self.saveCheckpoint,
            earlystop,
            self.logger,
            0,
            self.config['Train']['max_epoch'],
            self.config['Train']['lr'],
            self.config['Train']['v_step'],
        )

    def getTestDataloader(self, data_list_path: str):
        """
        used to load test dataloader
        Caseset generate a batch[1,n,c,w,h] containing slices in the same case.
        The sizes of batches are different.
        Thus batch_size of DataLoader is set to 1.
        """
        dataset = Caseset(data_list_path, self.config['dataset']['pair_dir'],
                          self.config['dataset']['resolution_path'])
        # batch_size = 1
        self.test_dataloader = torch.utils.data.DataLoader(dataset,
                                                           batch_size=1,
                                                           shuffle=False,
                                                           num_workers=0)

    def loadCheckpoint(self, net, epoch=None):
        LoadCheckpoint(net, self.model_save_path, epoch)

    def test(self):
        self.getTestDataloader(self.config['dataset']['testing_list_path'])
        self.loadCheckpoint(self.controller.net, self.config['Test']['epoch'])

        res = self.controller.test(self.test_dataloader, self.model_save_name,
                                   self.config['network'],
                                   self.config['Test']['excel_save_path'],
                                   self.config['Test']['verbose'])
        return res

    def run(self):
        if self.config['mode'] == 'Train':
            self.train()
            self.test()
        elif self.config['mode'] == 'Test':
            self.test()


if __name__ == "__main__":
    solver = Solver(config)
    solver.run()
