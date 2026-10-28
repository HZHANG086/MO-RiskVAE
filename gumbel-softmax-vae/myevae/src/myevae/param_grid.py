from torch.nn import LeakyReLU, Tanh

param_grid = {
    'z_dim': [32],
    'lr': [5e-4], 
    'batch_size': [1024],
    'input_types': [['exp','cna','gistic','fish','sbs','ig']],
    'input_types_subtask': [['clin']],
    'layer_dims': [[[32, 4],[16,4],[4,1],[4,1],[4,1],[4,1]]],
    'layer_dims_subtask' : [[4,1]],
    'kl_weight': [0.33],
    'activation': [LeakyReLU(negative_slope=0.01)],
    'subtask_activation': [Tanh()],
    'epochs': [100],
    'burn_in': [20],
    'patience': [5],
    'dropout': [0.0],
    'dropout_subtask': [0.0],
    'K': [4],
    'gamma': [1.0],
    'tau_start': [1.0],
    'tau_min': [0.5],
    'tau_decay': [0.99],
}


# the default hyperparameter grid for debugging uses
# this is not meant to be used for real training, as the search space is only on z_dim
#原来的
# param_grid = {
#     'z_dim': [8,16,32],
#     'lr': [5e-4], 
#     'batch_size': [1024],
#     'input_types': [['exp','cna','gistic','fish','sbs','ig']],
#     'input_types_subtask': [['clin']],
#     'layer_dims': [[[32, 4],[16,4],[4,1],[4,1],[4,1],[4,1]]],
#     'layer_dims_subtask' : [[4,1]],
#     'kl_weight': [1],
#     'activation': [LeakyReLU()],
#     'subtask_activation': [Tanh()],
#     'epochs': [100],
#     'burn_in': [20],
#     'patience': [5],
#     'dropout': [0.3],
#     'dropout_subtask': [0.3]
# }

#完整超参搜索
# from torch.nn import LeakyReLU, Tanh, ReLU, Sigmoid

# # example hyperparameter grid for a full VAE model
# param_grid = {
#     'z_dim': [8, 16, 32, 64],
#     'lr': [5e-4, 1e-4, 5e-5], 
#     'batch_size': [128, 256, 512],
#     'input_types': [['exp','cna', 'gistic', 'fish', 'sbs', 'ig']],
#     'input_types_subtask': [['clin']],
#     'layer_dims': [[[64, 16], [16, 8], [16, 4], [16, 4], [4], [2]], 
#                    [[128, 32], [64, 16], [16, 8], [16, 4], [4], [2]], 
#                    [[256, 64], [128, 32], [32, 8], [16, 4], [4], [2]]],
#     'layer_dims_subtask' : [[4,1], [8,1], [16,1]],
#     'kl_weight': [0.8, 1, 1.2],
#     'activation': [LeakyReLU(), ReLU(), Sigmoid()],
#     'subtask_activation': [Tanh(), Sigmoid()],
#     'dropout': [0.1, 0.2, 0.3],
#     'dropout_subtask': [0.1, 0.2, 0.3],
#     'epochs': [100,200,300],
#     'burn_in': [20,50],
#     'patience': [20],
# }


