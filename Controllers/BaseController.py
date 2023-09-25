import time

import os
import numpy as np
import optuna
import torch
from Metrics import MetricTest
from Modules.Loss import DiceCoefficient, DiceCoefficientAll
from Networks import BaseRegistraionNetwork
from Networks import RBFDCNUGenerativeNetwork
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Utils import EarlyStopping, ParamsAll

from rich.progress import Progress, BarColumn, TimeRemainingColumn

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

            # Logger for Tensorboard
            if (e + 1) % 20 == 1:
                if "NU" in type(self.net).__name__:
                    logger.add_scalar(tag='train\loss', scalar_value=train_loss_dict['loss'], global_step=(e + 1))
                    logger.add_scalar(tag='train\KL_loss', scalar_value=train_loss_dict['KL_loss'], global_step=(e + 1))
                    logger.add_scalar(tag='train\ELOB', scalar_value=train_loss_dict['ELOB'], global_step=(e + 1))
                    

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
            if isinstance(self.net, RBFDCNUGenerativeNetwork):
                loss_dict = self.net.objective(src, tgt, data['tgt']['seg'])
            else:
                loss_dict = self.net.objective(src, tgt)
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

                if isinstance(self.net, RBFDCNUGenerativeNetwork):
                    result = self.net.test(src, tgt, data['tgt_seg'][0])
                else:
                    result = self.net.test(src, tgt)
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
        extraValue = []
        NLLList = []

        with torch.no_grad():
            for data in dataloader:
                src = data['src'][0].cuda().float()
                tgt = data['tgt'][0].cuda().float()
                case_no = data['case_no'].item()
                src_seg = data['src_seg'][0].cuda().float()
                tgt_seg = data['tgt_seg'][0].cuda().float()
                slc_idx = data['slice']
                resolution = data['resolution'].item()
                
                if isinstance(self.net, RBFDCNUGenerativeNetwork):
                  results_t = self.net.test(src, tgt, data['tgt_seg'][0])
                  resultt_s = self.net.test(tgt, src, data['src_seg'][0])
                else:
                  results_t = self.net.test(src, tgt)
                  resultt_s = self.net.test(tgt, src)

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

                # extraValue.append(results_t[3])
                NLLList.append(torch.tensor(results_t[3]))
        
        mean = torch.mean(torch.cat(NLLList))
        std = torch.std(torch.cat(NLLList))
        var = torch.var(torch.cat(NLLList))
        print(mean, std, var)

        # if len(extraValue) > 0:
        #     ELBO = torch.mean(torch.cat([v[0] for v in extraValue]))
        #     KL = torch.mean(torch.cat([v[1] for v in extraValue]))
        #     sigma_term = torch.mean(torch.cat([v[2] for v in extraValue]))
        #     smooth_term = torch.mean(torch.cat([v[3] for v in extraValue]))
        #     logdet = torch.mean(torch.cat([v[4] for v in extraValue]))
        #     logSigma = torch.mean(torch.cat([v[5] for v in extraValue]))
        #     trSigma = torch.mean(torch.cat([v[6] for v in extraValue]))
        #     print(ELBO.item())
        #     print(KL.item())
        #     print(sigma_term.item())     
        #     print(smooth_term.item())   
        #     print(logdet.item())
        #     print(logSigma.item())
        #     print(trSigma.item())
        
        # mean = metric_test.mean()
        # if verbose >= 2:
        #     excel_save_path = os.path.join(excel_save_path, network)
        #     metric_test.saveAsExcel(network, name, excel_save_path)
        # if verbose >= 1:
        #     metric_test.output()
        # return mean, metric_test.details

    def hyperOpt(self,
                 hyperparams,
                 load_checkpoint,
                 n_trials,
                 train_dataloader: DataLoader,
                 validation_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 earlystop: EarlyStopping,
                 logger: SummaryWriter,
                 max_epoch=500,
                 lr=1e-4):
        def objective(trial: optuna.Trial):
            hyperparams = trial.study.user_attrs['hyperparams']
            params_instance = ParamsAll(trial, hyperparams)
            print(params_instance)
            load_checkpoint(self.net, 0)
            self.net.setHyperparam(**params_instance)

            self.train(train_dataloader,
                       validation_dataloader,
                       None,
                       earlystop,
                       None,
                       0,
                       max_epoch,
                       lr,
                       v_step=0,
                       verbose=0)

            res, _ = self.test(test_dataloader, verbose=0)
            print(res)
            return 1 - res['mean']
        
        self.net.train()
        study = optuna.create_study()
        study.set_user_attr('hyperparams', hyperparams)
        study.optimize(objective, n_trials, n_jobs=1)
        print(study.best_params)
        return study.best_params

    def speedTest(self, dataloader: DataLoader, device_type='gpu'):
        self.net.eval()
        case_time = []
        slice_time = []
        if device_type is 'cpu':
            self.net.cpu()
        with torch.no_grad():
            for data in dataloader:
                if device_type is 'gpu':
                    src = data['src'][0].cuda().float()
                    tgt = data['tgt'][0].cuda().float()
                else:
                    src = data['src'][0].cpu().float()
                    tgt = data['tgt'][0].cpu().float()
                torch.cuda.synchronize()
                start = time.time()
                result = self.net.test(src, tgt)
                torch.cuda.synchronize()
                end = time.time()
                case_time.append(end - start)

                torch.cuda.synchronize()
                for i in range(src.size()[0]):
                    start = time.time()
                    result = self.net.test(src[i:i + 1], tgt[i:i + 1])
                    torch.cuda.synchronize()
                    end = time.time()
                    slice_time.append(end - start)
        case_res = {'mean': np.mean(case_time), 'std': np.std(case_time)}
        slice_res = {'mean': np.mean(slice_time), 'std': np.std(slice_time)}
        print(device_type)
        print('case', '%.3f(%.3f)' % (case_res['mean'], case_res['std']))
        print('slice', '%.3f(%.3f)' % (slice_res['mean'], slice_res['std']))

    def estimate(self, case_data: torch.Tensor):
        self.net.eval()
        with torch.no_grad():
            src = case_data['src'].cuda().float()
            tgt = case_data['tgt'].cuda().float()
            src_seg = case_data['src_seg'].cuda().float()
            tgt_seg = case_data['tgt_seg'].cuda().float()
            slc_idx = case_data['slice']
            if "NU" in type(self.net).__name__:
                result = self.net.testForDraw(src, tgt, tgt_seg)
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

