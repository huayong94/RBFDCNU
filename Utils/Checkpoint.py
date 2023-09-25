import glob
import os

import torch


def LoadCheckpoint(net, model_save_path, epoch=None, verbose=1):
    if epoch is None:
        model_path_list = glob.glob(os.path.join(model_save_path, '*.pt'))
        iter_num_list = [
            int(os.path.basename(p).split('.')[0]) for p in model_path_list
        ]
        iter_num_list.sort(reverse=True)
        epoch = iter_num_list[0]
    if type(net) == dict:
        for key in net:
            model_path = os.path.join(model_save_path,
                                      '{}.{}.pt'.format(key, epoch))
            if verbose:
                print('Loading {}......'.format(model_path))
            net[key].load_state_dict(torch.load(model_path))
            net[key].eval()
    else:
        model_path = os.path.join(model_save_path, '{}.pt'.format(epoch))
        if verbose:
            print('Loading {}......'.format(model_path))
        net.load_state_dict(torch.load(model_path))
        net.eval()


def SaveCheckpoint(net, model_save_path, epoch, verbose=1):
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    if type(net) == dict:
        for key in net:
            save_checkpoint_path = os.path.join(model_save_path,
                                                '{}.{}.pt'.format(key, epoch))
            torch.save(net[key].state_dict(), save_checkpoint_path)
            if verbose:
                print('Saved model checkpoints into {}...'.format(
                    model_save_path))
    else:
        save_checkpoint_path = os.path.join(model_save_path,
                                            '{}.pt'.format(epoch))
        torch.save(net.state_dict(), save_checkpoint_path)
        if verbose:
            print('Saved model checkpoints into {}...'.format(model_save_path))
