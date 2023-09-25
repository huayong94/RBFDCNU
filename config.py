import optuna

from Controllers import BaseController
from Networks import RBFDCNUGenerativeNetwork

config = {
    'GPUNo': '0',                   # GPU Id
    'mode': 'Train',                # Train or Test
    'network': 'RBFDCNU',   
    'name': 'Final-excludeAYM',
    # ACDC, York and Miccai2009
    'dataset': {
        'training_list_path': '.../dataset2D_shs/training_pair.txt',
        'testing_list_path': '.../dataset2D_shs/testing_pair_CenterPair.txt',
        'pair_dir': '.../dataset2D_shs/data/',
        'resolution_path': '.../dataset2D_shs/resolution.txt'
    },
    # MM
    # 'dataset': {
    #     'training_list_path': '.../dataset2D_MnMs/training_pair.txt',
    #     'testing_list_path': '.../dataset2D_MnMs/testing_pair_CenterPair.txt',
    #     'validation_list_path': '.../dataset2D_MnMs/validation_pair.txt',
    #     'pair_dir': '.../dataset2D_MnMs/data/',
    #     'resolution_path': '.../dataset2D_MnMs/resolution.txt'
    # },
    'Train': {
        'batch_size': 32,
        'model_save_dir': '.../RBFDCNU/model',
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
        'model_save_path': '.../RBFDCNU/model/RBFDCNU/model/',
        'excel_save_path': '.../RBFDCNU/model/RBFDCNU/TestSave',
        'verbose': 2,
    },
    'SpeedTest': {
        'epoch': 'best',
        'model_save_path': '.../RBFDCNU/model/RBFDCNU/model/',
        'device': 'cpu'
    },
    'Hyperopt': {
        'n_trials': 30,
        'earlystop': {
            'min_delta': 0.00001,
            'patience': 500
        },
        'max_epoch': 800,
        'lr': 1e-4
    },
    'RBFDCNU': 
    {
        'controller': BaseController,
        'network': RBFDCNUGenerativeNetwork,
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
            'similarity_factor': 120000,
            'loss_mode': 0,
            'cropSize': 64,
            # WLCC
            # 'similarity_loss': 'WLCC',
            # 'similarity_loss_param': {
            #     'alpha': 0.02,
            #     'win': [9, 9]
            # }
            # LCC
            'similarity_loss': 'LCC',
            'similarity_loss_param': {
                'win': [9, 9]
            }
        },
        'hyperparams': {
            'factor_list': [
                {
                    'type': 'suggest_int',
                    'params': {
                        'low': 5000,
                        'high': 500000,
                        'step': 5000
                    }
                }, 0, 0
                #  {
                #     'type': 'suggest_int',
                #     'params': {
                #         'low': 0,
                #         'high': 1000,
                #         'step': 50
                #     }
                # }, {
                #     'type': 'suggest_float',
                #     'params': {
                #         'low': 0,
                #         'high': 10,
                #         'step': 0.5
                #     }
                # }
            ],
            #LCC
            'similarity_loss_param': {
                'win': [9, 9]
            }
        }
    }
}
