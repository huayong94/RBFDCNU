from Controllers import BaseController
from Networks import NuNetGenerativeNetwork

config = {
    'GPUNo': '0',
    'mode': 'Test',
    'network': 'NuNet',
    'name': 'Final-excludeMM',
    'dataset': {
        'training_list_path': 'DataSet\\training_pair.txt',
        'testing_list_path': 'DataSet\\testing_pair.txt',
        'validation_list_path': 'DataSet\\validation_pair.txt',
        'pair_dir': 'DataSet\\data\\',
        'resolution_path': 'DataSet\\resolution.txt'
    },
    'Train': {
        'batch_size': 32,
        'model_save_dir':
        '',
        'lr': 5e-4,
        'max_epoch': 3000,
        'save_checkpoint_step': 500,
        'v_step': 500,
        'earlystop': {
            'min_delta': 0.00001,
            'patience': 1000
        },
    },
    'Test': {
        'epoch': 'best',
        'model_save_path':
        'DataSet\\',
        'excel_save_path': 'DataSet\\',
        'verbose': 2,
    },
    'NuNet': {
        'controller': BaseController,
        'network': NuNetGenerativeNetwork,
        'params': {
            'encoder_param': {
                'dims': [16, 32, 32, 64, 64],
                'num_layers': [2, 2, 2, 2, 2],
                'local_dims': [16, 32, 32, 64],
                'local_num_layers': [2, 2, 2, 2]
            },
            'i_size': [128, 128],
            'c_factor': 2,
            'cpoint_num': 128,
            'nucpoint_num': 64,
            'ucpoint_num': 64,
            'similarity_factor': 240000,
            'similarity_loss': 'LCC',
            'similarity_loss_param': {
                'win': [9, 9]
            }
        },
    },
}
