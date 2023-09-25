import time
import json
import os
import numpy as np
import torch
from Metrics import MetricTest
from Modules.Loss import DiceCoefficientAll
from Networks import BaseRegistraionNetwork
from Networks import NuNetGenerativeNetwork
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Utils import EarlyStopping, ParamsAll

from rich.progress import Progress, BarColumn, TimeRemainingColumn
import matplotlib
from matplotlib import colors
from matplotlib import pyplot as plt
import copy

progress = Progress(
    "[progress.description]{task.description}",
    BarColumn(),
    "[progress.percentage]{task.percentage:3.2f}%",
    "{task.completed:5.0f}",
    "best: {task.fields[best]:.5f}",
    "best_epoch: {task.fields[best_epoch]:5.0f}",
    TimeRemainingColumn(),
)


class BaseController:
    def __init__(self, net: BaseRegistraionNetwork):
        self.net = net

    def cuda(self):
        self.net.cuda()

    def train(self,
              train_dataloader: DataLoader,
              validation_dataloader: DataLoader,
              save_checkpoint,
              earlystop: EarlyStopping,
              logger: SummaryWriter,
              start_epoch=0,
              max_epoch=1000,
              lr=1e-4,
              v_step=50,
              verbose=1):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        earlystop.on_train_begin()

        end = time.perf_counter()
        if verbose == 0:
            task = progress.add_task('Training...',
                                     total=max_epoch * 2,
                                     best=0,
                                     best_epoch=0)
            progress.start()

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1000,gamma = 0.5)
        for e in range(start_epoch, max_epoch * 2):
            start = end
            # learning
            self.net.train()
            train_loss_dict = self.trainIter(train_dataloader, optimizer)
            scheduler.step()
            # validation
            self.net.eval()
            validation_dice = self.validationIter(validation_dataloader)

            # save checkpoint
            if save_checkpoint:
                save_checkpoint(self.net, e + 1)

            # console
            train_loss_mean_str = ''
            for key in train_loss_dict:
                if 'z' not in key:
                    train_loss_mean_str += '%s : %f, ' % (key,
                                                      train_loss_dict[key])
            end = time.perf_counter()

            if verbose:
                print(e + 1, '%.2f' % (end - start), train_loss_mean_str,
                      validation_dice)

            # early stop
            if earlystop.on_epoch_end(e + 1, validation_dice,
                                      self.net) and e >= max_epoch:
                if verbose == 0:
                    progress.update(task,
                                    advance=1,
                                    best=earlystop.best,
                                    best_epoch=earlystop.best_epoch,
                                    refresh=True)
                    time.sleep(0.0001)
                    progress.stop_task(task)
                    progress.remove_task(task)
                break
            if verbose == 0:
                progress.update(task,
                                advance=1,
                                best=earlystop.best,
                                best_epoch=earlystop.best_epoch,
                                refresh=True)
                time.sleep(0.0001)

        return earlystop.best

    def trainIter(self, dataloader: DataLoader,
                  optimizer: torch.optim.Optimizer) -> dict:
        train_loss_dict = {}
        for data in dataloader:
            src = data['src']['img'].cuda()
            tgt = data['tgt']['img'].cuda()
            optimizer.zero_grad()
            # forward to loss
            loss_dict = self.net.objective(src, tgt, data['tgt']['seg'])
            loss = loss_dict['loss'].mean()
            # backward
            loss.backward()
            # update
            optimizer.step()
            for key in loss_dict:
                loss_mean = loss_dict[key].mean().item()
                if key not in train_loss_dict:
                    train_loss_dict[key] = [loss_mean]
                else:
                    train_loss_dict[key].append(loss_mean)

        for key in train_loss_dict:
            train_loss_dict[key] = np.mean(train_loss_dict[key])
        return train_loss_dict

    def validationIter(self, dataloader: DataLoader):
        dice_list = []
        with torch.no_grad():
            # dice_estimator = DiceCoefficient()
            dice_estimator = DiceCoefficientAll()
            for data in dataloader:
                src = data['src'][0].cuda().float()
                tgt = data['tgt'][0].cuda().float()
                # regard all types of segmeant as one
                src_seg = data['src_seg'][0].cuda().float()
                tgt_seg = data['tgt_seg'][0].cuda().int()
                result = self.net.test(src, tgt, data['tgt_seg'][0])
                phi = result[0]
                warped_src_seg = self.net.transformer(src_seg,
                                                      phi,
                                                      mode='nearest')
                # dice = dice_estimator(tgt_seg, warped_src_seg)
                dice = dice_estimator(tgt_seg,
                                      warped_src_seg.int()).unsqueeze(0)
                dice_list.append(dice)
            # statistics
            dice_tensor = torch.cat(dice_list, 0)
            return dice_tensor.mean().item()

    def test(self,
             dataloader: DataLoader,
             name: str = None,
             network: str = None,
             excel_save_path: str = None,
             verbose=2):

        self.net.eval()
        metric_test = MetricTest()

        with torch.no_grad():
            for data in dataloader:
                src = data['src'][0].cuda().float()
                tgt = data['tgt'][0].cuda().float()
                case_no = data['case_no'].item()
                src_seg = data['src_seg'][0].cuda().float()
                tgt_seg = data['tgt_seg'][0].cuda().float()
                slc_idx = data['slice']
                resolution = data['resolution'].item()               
                results_t = self.net.test(src, tgt, data['tgt_seg'][0])
                resultt_s = self.net.test(tgt, src, data['src_seg'][0])
                phis_t = results_t[0]
                phit_s = resultt_s[0]
                warped_src_seg = self.net.transformer(src_seg,
                                                      phis_t,
                                                      mode='nearest')
                warped_tgt_seg = self.net.transformer(tgt_seg,
                                                      phit_s,
                                                      mode='nearest')

                metric_test.testMetrics(src_seg.int(), warped_src_seg.int(),
                                        tgt_seg.int(), warped_tgt_seg.int(),
                                        resolution, case_no, slc_idx)
                metric_test.testFlow(phis_t, phit_s, case_no)
        
        mean = metric_test.mean()
        if verbose >= 2:
            excel_save_path = os.path.join(excel_save_path, network)
            metric_test.saveAsExcel(network, name, excel_save_path)
        if verbose >= 1:
            metric_test.output()
        return mean, metric_test.details


    def generateDictionary(self, detail, savePath, network):
        
        # York Dataset
        dataset = detail['York']
        if dataset:
            labels = ['LvMyo', 'LvBp', 'Lv']
            York_dictionary = self.datasetDictionary(dataset, labels)
            self.saveDictionary(York_dictionary, savePath, 'York', network)

        # MICCAI Dataset
        dataset = detail['MICCAI']
        if dataset:
            labels = ['LvBp']
            MICCAI_dictionary = self.datasetDictionary(dataset, labels)
            self.saveDictionary(MICCAI_dictionary, savePath, 'MICCAI', network)

        # ACDC Dataset
        dataset = detail['ACDC']
        if dataset:
            labels = ['Rv', 'LvMyo', 'LvBp', 'Lv']
            ACDC_dictionary = self.datasetDictionary(dataset, labels)
            self.saveDictionary(ACDC_dictionary, savePath, 'ACDC', network)
        
        # MnMs Dataset
        dataset = detail['M&M']
        if dataset:
            labels = ['Rv', 'LvMyo', 'LvBp', 'Lv']
            MnMs_dictionary = self.datasetDictionary(dataset, labels)
            self.saveDictionary(MnMs_dictionary, savePath, 'MnMs', network)
        
    def saveDictionary(self, dictionary, savePath, datset, network):
        filename = datset + "_" + network + ".json"
        path = os.path.join(savePath, "_boxPlot", "data")
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, filename)
        f = open(path, "w")
        json.dump(dictionary, f)
        f.close()
        
    def datasetDictionary(self, dataset, labels):

        direction = 'ed_to_es'
        metric_1 = ['Dice', 'SymAPD', 'HD']
        metric_2 = ['BE', 'Jacobian']
        dictionary = {}
        for label in labels:
            dictionary[label] = {}
            for metric in metric_1:
                value = []
                for caseNo in dataset:
                    value.append(dataset[caseNo][label][direction][metric])
                value = np.hstack(value).tolist()
                dictionary[label][metric] = value

        dictionary['flow'] = {}
        for metric in metric_2:
            value = []
            for caseNo in dataset:
                value.append(dataset[caseNo]['flow'][direction][metric])
            value = np.hstack(value).tolist()
            dictionary['flow'][metric] = value
        return dictionary


    def estimate(self, case_data: torch.Tensor):
        self.net.eval()
        with torch.no_grad():
            src = case_data['src'].cuda().float()
            tgt = case_data['tgt'].cuda().float()
            src_seg = case_data['src_seg'].cuda().float()
            tgt_seg = case_data['tgt_seg'].cuda().float()
            slc_idx = case_data['slice']
            if "NU" in type(self.net).__name__:
                result = self.net.testForDraw(src, tgt, src_seg)
            else:
                result = self.net.test(src, tgt)
            
            phi = result[0]
            if len(result) == 3:
                cp_loc = None
            else:
                cp_loc = result[3]
            warped_src = self.net.transformer(src, phi)
            warped_src_seg = self.net.transformer(src_seg, phi, mode='nearest')

            res = {
                'src': src.cpu().numpy()[:, 0, :, :],
                'tgt': tgt.cpu().numpy()[:, 0, :, :],
                'src_seg': src_seg.cpu().numpy()[:, 0, :, :],
                'tgt_seg': tgt_seg.cpu().numpy()[:, 0, :, :],
                'phi': phi.cpu().numpy(),
                'warped_src': warped_src.cpu().numpy()[:, 0, :, :],
                'warped_src_seg': warped_src_seg.cpu().numpy()[:, 0, :, :],
                'slc_idx': slc_idx,
                'cp_loc': cp_loc
            }
            return res

