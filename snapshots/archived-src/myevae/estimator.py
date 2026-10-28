#HSIC-VAE
import os
import pandas as pd
import numpy as np
import warnings
warnings.formatwarning = lambda msg, *args, **kwargs: f'{msg}\n'
from torch.utils.data import DataLoader
from torch import no_grad, cat as torch_cat, tensor as torch_tensor, save as torch_save,device as torch_device
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from sklearn.base import BaseEstimator
from sklearn.utils.validation import validate_data, check_is_fitted
from sksurv.metrics import concordance_index_censored

from .dataset import Dataset
from .coxphloss import CoxPHLoss
from .hsic import HSICLoss


# from .kldivergence import KLDivergence

# scikit learn compliant estimator class
# for meta-estimators like Pipeline and GridSearchCV
# implements `fit`, `predict`, and `score` functions
# these functions accept dataframe as input
# `score` returns a single np.float
class VAE(BaseEstimator):
    def __init__(self, 
                 input_types:list[str]=[None], 
                 layer_dims:list[list[int]]=[[None]], 
                 input_types_subtask:list[str]=None,
                 layer_dims_subtask:list[int]=None,
                 z_dim:int=None,
                 lr:float=None,
                 batch_size:int=None,
                 epochs:int=None,
                 burn_in:int=None,
                 patience:int=None, 
                 eventcol:str=None, 
                 durationcol:str=None, 
                 kl_weight:float=None,
                 activation:str=None,
                 subtask_activation:str=None,
                 dropout:float=None,
                 dropout_subtask:float=None,
                 device:str=None):
        self.input_types = input_types
        self.layer_dims = layer_dims 
        self.input_types_subtask = input_types_subtask 
        self.layer_dims_subtask = layer_dims_subtask        
        self.z_dim = z_dim
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.burn_in = burn_in
        self.patience = patience
        self.eventcol = eventcol
        self.durationcol = durationcol
        self.kl_weight = kl_weight
        self.activation = activation
        self.subtask_activation = subtask_activation
        self.dropout = dropout
        self.dropout_subtask = dropout_subtask
        self.device = device
        self.hsic_loss_func = HSICLoss(sigma_x=1.0, sigma_y=1.0)
        self.hsic_weight = 0.1   # γ，建议先小一点

    @property
    def device_(self):
        """获取实际的 torch.device 对象"""
        if self.device is None:
            return torch_device("cuda" if torch.cuda.is_available() else "cpu")
        return torch_device(self.device)

    
    def fit(self, X:pd.DataFrame, y=None, verbose:bool=False, SHAP:bool=False):
        """
        X: dataframe with named columns according to the input type. Also contains event and duration columns.
        y: ignored, because the event and duration columns should be in X.
        verbose: whether to print loss at every epoch
        SHAP: uses ShapMultiModalVAE for the model, where __call__() returns riskpreds only instead of (xs,mu,logvar,riskpreds)
        """
        assert isinstance(X, pd.DataFrame)
        # Check that X has correct shape, set n_features_in_, etc.
        X = pd.DataFrame(validate_data(self, X, y), index=X.index, columns=X.columns)
        self.genes = None
        self.X_ = X
        self.y_ = y
        # assign non-parameter attributes to estimator
        self.input_types_all = self.input_types + self.input_types_subtask
        # convert X into a multi omics dataset
        print(f"[VAE] Using device: {self.device_}")
        dataset = Dataset(X,self.input_types_all,event_indicator_col=self.eventcol,event_time_col=self.durationcol,device=self.device_)
        # determine input dims lazily
        self.input_dims = [getattr(dataset, f"X_{input_type}").shape[1] for input_type in self.input_types]
        # determine subtask input dims lazily
        self.input_dims_subtask = [getattr(dataset, f"X_{input_type}").shape[1] for input_type in self.input_types_subtask]
        
        if SHAP:
            # forward returns just riskpreds
            from .nn import ShapMultiModalVAE as Model
        else:
            # forward returns (reconstructed_xs, mu, logvar, riskpreds)
            from .nn import MultiModalVAE as Model
        
        self.model = Model(input_types = self.input_types, 
                           input_dims = self.input_dims, 
                           layer_dims = self.layer_dims, 
                           input_types_subtask = self.input_types_subtask,
                           input_dims_subtask = self.input_dims_subtask, 
                           layer_dims_subtask = self.layer_dims_subtask, 
                           z_dim = self.z_dim,
                           activation = self.activation,
                           subtask_activation = self.subtask_activation,
                           dropout = self.dropout,
                           dropout_subtask = self.dropout_subtask,)
        self.model = self.model.to(self.device_)                   
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        self.survival_loss_func = CoxPHLoss()
        # self.kl_loss_func = KLDivergence()
        self.mmd_loss_func = MMDLoss(sigma=1.0)
        self.reconstruction_loss_funcs = [MSELoss(reduction='mean') for _ in self.model.input_types_vae]
        self.model.train()
        self.optimizer.zero_grad()
        best_loss = np.inf
        epochs_since_best = 0
        trainloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(1,1+self.epochs):
            current_loss = 0 # survival loss summed across batches
            for _, data in enumerate(trainloader):
                inputs_vae = [data[f'X_{input_type}'].to(self.device_) for input_type in self.model.input_types_vae]
                inputs_task = [data[f'X_{input_type}'].to(self.device_) for input_type in self.model.input_types_subtask]
                # 将 event_indicator 和 event_time 转为张量并移动到设备
                ei = data['event_indicator']
                et = data['event_time']
                # 处理不同类型的输入（numpy array, list, tensor 等）
                if isinstance(ei, torch.Tensor):
                    event_indicator = ei.to(dtype=torch.float64, device=self.device_)
                else:
                    event_indicator = torch_tensor(ei, dtype=torch.float64, device=self.device_)
                if isinstance(et, torch.Tensor):
                    event_time = et.to(dtype=torch.float64, device=self.device_)
                else:
                    event_time = torch_tensor(et, dtype=torch.float64, device=self.device_)

                outputs, mu, logvar, riskpred = self.model.forward((inputs_vae, inputs_task))
                assert len(inputs_vae)==len(outputs)
                eps = torch.randn_like(mu)
                z = mu + eps * torch.exp(0.5 * logvar)

                batch_mmd_loss = self.kl_weight * self.mmd_loss_func(z)
                assert not batch_mmd_loss.isnan().any.item()
                # batch_kl_loss = self.kl_weight * self.kl_loss_func(mu, logvar)
                # assert not batch_kl_loss.isnan().any().item()
                batch_reconstruction_losses = [
                    f(output, input_vae) for f, output, input_vae in zip(self.reconstruction_loss_funcs, outputs, inputs_vae)
                ]
                for brl in batch_reconstruction_losses:
                    assert not brl.isnan().any().item()
                batch_survival_loss = self.survival_loss_func(event_indicator, event_time, riskpred.flatten())
                assert not batch_survival_loss.isnan().any().item()
                X_hsic = torch.cat(inputs_vae, dim=1)
                # 这里用 event_indicator 作为 y（最简单、可跑）
                y_surv = event_indicator.unsqueeze(1)
                residual = y_surv - riskpred.view(-1, 1)
                batch_hsic_loss = self.hsic_weight * self.hsic_loss_func(X_hsic, residual)
                assert not batch_hsic_loss.isnan().any().item()
                batch_loss = batch_mmd_loss + batch_survival_loss + sum(batch_reconstruction_losses) + batch_hsic_loss
                # batch_loss = batch_mmd_loss + batch_survival_loss + sum(batch_reconstruction_losses)
                # batch_loss = batch_kl_loss + batch_survival_loss + sum(batch_reconstruction_losses)
                batch_loss.backward()
                self.optimizer.step()
                current_loss += batch_loss.item()
            
            if epoch <= int(self.burn_in):
                pass
            elif current_loss < best_loss:
                epochs_since_best = 0
                best_loss = current_loss
                if verbose:
                    print(f'Epoch {epoch}: {current_loss}')
            elif epochs_since_best == int(self.patience):
                print(f'Early stopping at epoch {epoch - self.patience - 1}')
                return self
            else:
                epochs_since_best += 1
        if verbose:
            warnings.warn(f'Early stopping not triggered. patience: {self.patience}, best_loss: {best_loss}, epochs_since_best: {epochs_since_best}')
        return self
    
    def eval(self)->None:
        self.model.eval()
        
    def predict(self, X:pd.DataFrame)->torch_tensor:
        # check if fit has been called
        check_is_fitted(self)
        # input validation
        X = pd.DataFrame(validate_data(self, X, reset=False), index=X.index, columns=X.columns)
        
        self.model.eval()
        dataloader = DataLoader(Dataset(X, self.input_types_all, event_indicator_col=self.eventcol,event_time_col=self.durationcol, device=self.device_), batch_size=1024, shuffle=False)
        estimates = []
        for _, data in enumerate(dataloader):
            with no_grad():
                inputs_vae = [data[f'X_{input_type}'] for input_type in self.model.input_types_vae]
                inputs_task = [data[f'X_{input_type}'] for input_type in self.model.input_types_subtask]
                _, _, _, riskpred = self.model.forward((inputs_vae, inputs_task))
                estimates.append(riskpred.flatten().cpu())
        estimates = torch_cat(estimates)
        return estimates

    def score(self, X:pd.DataFrame, y=None)->float:
        assert isinstance(X, pd.DataFrame)
        X = pd.DataFrame(validate_data(self, X, reset=False),index=X.index,columns=X.columns)
        estimates = self.predict(X)
        event = X[self.eventcol].values.astype(bool)
        duration = X[self.durationcol].values
        metric = concordance_index_censored(event, duration, estimates)[0].item()
        return metric
    
    def save(self, pth_path):
        torch_save(self.model.state_dict(), pth_path)
        return

    def __call__(self, X:torch_tensor)->tuple:
        return self.model.__call__(X)
# #MMD-VAE
# import os
# import pandas as pd
# import numpy as np
# import warnings
# warnings.formatwarning = lambda msg, *args, **kwargs: f'{msg}\n'
# from torch.utils.data import DataLoader
# from torch import no_grad, cat as torch_cat, tensor as torch_tensor, save as torch_save,device as torch_device
# import torch
# from torch.nn import MSELoss
# from torch.optim import Adam
# from sklearn.base import BaseEstimator
# from sklearn.utils.validation import validate_data, check_is_fitted
# from sksurv.metrics import concordance_index_censored

# from .dataset import Dataset
# from .coxphloss import CoxPHLoss
# from .mmd import MMDLoss

# # from .kldivergence import KLDivergence

# # scikit learn compliant estimator class
# # for meta-estimators like Pipeline and GridSearchCV
# # implements `fit`, `predict`, and `score` functions
# # these functions accept dataframe as input
# # `score` returns a single np.float
# class VAE(BaseEstimator):
#     def __init__(self, 
#                  input_types:list[str]=[None], 
#                  layer_dims:list[list[int]]=[[None]], 
#                  input_types_subtask:list[str]=None,
#                  layer_dims_subtask:list[int]=None,
#                  z_dim:int=None,
#                  lr:float=None,
#                  batch_size:int=None,
#                  epochs:int=None,
#                  burn_in:int=None,
#                  patience:int=None, 
#                  eventcol:str=None, 
#                  durationcol:str=None, 
#                  kl_weight:float=None,
#                  activation:str=None,
#                  subtask_activation:str=None,
#                  dropout:float=None,
#                  dropout_subtask:float=None,
#                  device:str=None):
#         self.input_types = input_types
#         self.layer_dims = layer_dims 
#         self.input_types_subtask = input_types_subtask 
#         self.layer_dims_subtask = layer_dims_subtask        
#         self.z_dim = z_dim
#         self.lr = lr
#         self.batch_size = batch_size
#         self.epochs = epochs
#         self.burn_in = burn_in
#         self.patience = patience
#         self.eventcol = eventcol
#         self.durationcol = durationcol
#         self.kl_weight = kl_weight
#         self.activation = activation
#         self.subtask_activation = subtask_activation
#         self.dropout = dropout
#         self.dropout_subtask = dropout_subtask
#         self.device = device
#     @property
#     def device_(self):
#         """获取实际的 torch.device 对象"""
#         if self.device is None:
#             return torch_device("cuda" if torch.cuda.is_available() else "cpu")
#         return torch_device(self.device)

    
#     def fit(self, X:pd.DataFrame, y=None, verbose:bool=False, SHAP:bool=False):
#         """
#         X: dataframe with named columns according to the input type. Also contains event and duration columns.
#         y: ignored, because the event and duration columns should be in X.
#         verbose: whether to print loss at every epoch
#         SHAP: uses ShapMultiModalVAE for the model, where __call__() returns riskpreds only instead of (xs,mu,logvar,riskpreds)
#         """
#         assert isinstance(X, pd.DataFrame)
#         # Check that X has correct shape, set n_features_in_, etc.
#         X = pd.DataFrame(validate_data(self, X, y), index=X.index, columns=X.columns)
#         self.genes = None
#         self.X_ = X
#         self.y_ = y
#         # assign non-parameter attributes to estimator
#         self.input_types_all = self.input_types + self.input_types_subtask
#         # convert X into a multi omics dataset
#         print(f"[VAE] Using device: {self.device_}")
#         dataset = Dataset(X,self.input_types_all,event_indicator_col=self.eventcol,event_time_col=self.durationcol,device=self.device_)
#         # determine input dims lazily
#         self.input_dims = [getattr(dataset, f"X_{input_type}").shape[1] for input_type in self.input_types]
#         # determine subtask input dims lazily
#         self.input_dims_subtask = [getattr(dataset, f"X_{input_type}").shape[1] for input_type in self.input_types_subtask]
        
#         if SHAP:
#             # forward returns just riskpreds
#             from .nn import ShapMultiModalVAE as Model
#         else:
#             # forward returns (reconstructed_xs, mu, logvar, riskpreds)
#             from .nn import MultiModalVAE as Model
        
#         self.model = Model(input_types = self.input_types, 
#                            input_dims = self.input_dims, 
#                            layer_dims = self.layer_dims, 
#                            input_types_subtask = self.input_types_subtask,
#                            input_dims_subtask = self.input_dims_subtask, 
#                            layer_dims_subtask = self.layer_dims_subtask, 
#                            z_dim = self.z_dim,
#                            activation = self.activation,
#                            subtask_activation = self.subtask_activation,
#                            dropout = self.dropout,
#                            dropout_subtask = self.dropout_subtask,)
#         self.model = self.model.to(self.device_)                   
#         self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
#         self.survival_loss_func = CoxPHLoss()
#         # self.kl_loss_func = KLDivergence()
#         self.mmd_loss_func = MMDLoss(sigma=1.0)
#         self.reconstruction_loss_funcs = [MSELoss(reduction='mean') for _ in self.model.input_types_vae]
#         self.model.train()
#         self.optimizer.zero_grad()
#         best_loss = np.inf
#         epochs_since_best = 0
#         trainloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
#         for epoch in range(1,1+self.epochs):
#             current_loss = 0 # survival loss summed across batches
#             for _, data in enumerate(trainloader):
#                 inputs_vae = [data[f'X_{input_type}'].to(self.device_) for input_type in self.model.input_types_vae]
#                 inputs_task = [data[f'X_{input_type}'].to(self.device_) for input_type in self.model.input_types_subtask]
#                 # 将 event_indicator 和 event_time 转为张量并移动到设备
#                 ei = data['event_indicator']
#                 et = data['event_time']
#                 # 处理不同类型的输入（numpy array, list, tensor 等）
#                 if isinstance(ei, torch.Tensor):
#                     event_indicator = ei.to(dtype=torch.float64, device=self.device_)
#                 else:
#                     event_indicator = torch_tensor(ei, dtype=torch.float64, device=self.device_)
#                 if isinstance(et, torch.Tensor):
#                     event_time = et.to(dtype=torch.float64, device=self.device_)
#                 else:
#                     event_time = torch_tensor(et, dtype=torch.float64, device=self.device_)

#                 outputs, mu, logvar, riskpred = self.model.forward((inputs_vae, inputs_task))
#                 assert len(inputs_vae)==len(outputs)
#                 eps = torch.randn_like(mu)
#                 z = mu + eps * torch.exp(0.5 * logvar)

#                 batch_mmd_loss = self.kl_weight * self.mmd_loss_func(z)
#                 assert not batch_mmd_loss.isnan().any.item()
#                 # batch_kl_loss = self.kl_weight * self.kl_loss_func(mu, logvar)
#                 # assert not batch_kl_loss.isnan().any().item()
#                 batch_reconstruction_losses = [
#                     f(output, input_vae) for f, output, input_vae in zip(self.reconstruction_loss_funcs, outputs, inputs_vae)
#                 ]
#                 for brl in batch_reconstruction_losses:
#                     assert not brl.isnan().any().item()
#                 batch_survival_loss = self.survival_loss_func(event_indicator, event_time, riskpred.flatten())
#                 assert not batch_survival_loss.isnan().any().item()
#                 batch_loss = batch_mmd_loss + batch_survival_loss + sum(batch_reconstruction_losses)
#                 # batch_loss = batch_kl_loss + batch_survival_loss + sum(batch_reconstruction_losses)
#                 batch_loss.backward()
#                 self.optimizer.step()
#                 current_loss += batch_loss.item()
            
#             if epoch <= int(self.burn_in):
#                 pass
#             elif current_loss < best_loss:
#                 epochs_since_best = 0
#                 best_loss = current_loss
#                 if verbose:
#                     print(f'Epoch {epoch}: {current_loss}')
#             elif epochs_since_best == int(self.patience):
#                 print(f'Early stopping at epoch {epoch - self.patience - 1}')
#                 return self
#             else:
#                 epochs_since_best += 1
#         if verbose:
#             warnings.warn(f'Early stopping not triggered. patience: {self.patience}, best_loss: {best_loss}, epochs_since_best: {epochs_since_best}')
#         return self
    
#     def eval(self)->None:
#         self.model.eval()
        
#     def predict(self, X:pd.DataFrame)->torch_tensor:
#         # check if fit has been called
#         check_is_fitted(self)
#         # input validation
#         X = pd.DataFrame(validate_data(self, X, reset=False), index=X.index, columns=X.columns)
        
#         self.model.eval()
#         dataloader = DataLoader(Dataset(X, self.input_types_all, event_indicator_col=self.eventcol,event_time_col=self.durationcol, device=self.device_), batch_size=1024, shuffle=False)
#         estimates = []
#         for _, data in enumerate(dataloader):
#             with no_grad():
#                 inputs_vae = [data[f'X_{input_type}'] for input_type in self.model.input_types_vae]
#                 inputs_task = [data[f'X_{input_type}'] for input_type in self.model.input_types_subtask]
#                 _, _, _, riskpred = self.model.forward((inputs_vae, inputs_task))
#                 estimates.append(riskpred.flatten().cpu())
#         estimates = torch_cat(estimates)
#         return estimates

#     def score(self, X:pd.DataFrame, y=None)->float:
#         assert isinstance(X, pd.DataFrame)
#         X = pd.DataFrame(validate_data(self, X, reset=False),index=X.index,columns=X.columns)
#         estimates = self.predict(X)
#         event = X[self.eventcol].values.astype(bool)
#         duration = X[self.durationcol].values
#         metric = concordance_index_censored(event, duration, estimates)[0].item()
#         return metric
    
#     def save(self, pth_path):
#         torch_save(self.model.state_dict(), pth_path)
#         return

#     def __call__(self, X:torch_tensor)->tuple:
#         return self.model.__call__(X)



# #beta-VAE
# import os
# import pandas as pd
# import numpy as np
# import warnings
# warnings.formatwarning = lambda msg, *args, **kwargs: f'{msg}\n'
# from torch.utils.data import DataLoader
# from torch import no_grad, cat as torch_cat, tensor as torch_tensor, save as torch_save
# from torch.nn import MSELoss
# from torch.optim import Adam
# from sklearn.base import BaseEstimator
# from sklearn.utils.validation import validate_data, check_is_fitted
# from sksurv.metrics import concordance_index_censored

# from .dataset import Dataset
# from .coxphloss import CoxPHLoss
# from .kldivergence import KLDivergence

# # scikit learn compliant estimator class
# # for meta-estimators like Pipeline and GridSearchCV
# # implements `fit`, `predict`, and `score` functions
# # these functions accept dataframe as input
# # `score` returns a single np.float
# class VAE(BaseEstimator):
#     def __init__(self, 
#                  input_types:list[str]=[None], 
#                  layer_dims:list[list[int]]=[[None]], 
#                  input_types_subtask:list[str]=None,
#                  layer_dims_subtask:list[int]=None,
#                  z_dim:int=None,
#                  lr:float=None,
#                  batch_size:int=None,
#                  epochs:int=None,
#                  burn_in:int=None,
#                  patience:int=None, 
#                  eventcol:str=None, 
#                  durationcol:str=None, 
#                  kl_weight:float=None,
#                  #这里设置了β的范围从（0，1）且步数为0.05，在跑的时候，可以把步数beta_step改的更小，让最后最好的β更加精确
#                  beta_start:float = 0,
#                  beta_end:float = 1.0,
#                  beta_step:float =0.05,
#                  activation:str=None,
#                  subtask_activation:str=None,
#                  dropout:float=None,
#                  dropout_subtask:float=None):
#         self.input_types = input_types
#         self.layer_dims = layer_dims 
#         self.input_types_subtask = input_types_subtask 
#         self.layer_dims_subtask = layer_dims_subtask        
#         self.z_dim = z_dim
#         self.lr = lr
#         self.batch_size = batch_size
#         self.epochs = epochs
#         self.burn_in = burn_in
#         self.patience = patience
#         self.eventcol = eventcol
#         self.durationcol = durationcol
#         self.kl_weight = kl_weight
#         self.beta_start = beta_start
#         self.beta_end = beta_end
#         self.beta_step = beta_step
#         self.activation = activation
#         self.subtask_activation = subtask_activation
#         self.dropout = dropout
#         self.dropout_subtask = dropout_subtask
    
#     def fit(self, X:pd.DataFrame, y=None, verbose:bool=False, SHAP:bool=False):
#         """
#         X: dataframe with named columns according to the input type. Also contains event and duration columns.
#         y: ignored, because the event and duration columns should be in X.
#         verbose: whether to print loss at every epoch
#         SHAP: uses ShapMultiModalVAE for the model, where __call__() returns riskpreds only instead of (xs,mu,logvar,riskpreds)
#         """
#         assert isinstance(X, pd.DataFrame)
#         # Check that X has correct shape, set n_features_in_, etc.
#         X = pd.DataFrame(validate_data(self, X, y), index=X.index, columns=X.columns)
#         self.genes = None
#         self.X_ = X
#         self.y_ = y
#         # assign non-parameter attributes to estimator
#         self.input_types_all = self.input_types + self.input_types_subtask
#         # convert X into a multi omics dataset
#         dataset = Dataset(X,self.input_types_all,event_indicator_col=self.eventcol,event_time_col=self.durationcol)
#         # determine input dims lazily
#         self.input_dims = [getattr(dataset, f"X_{input_type}").shape[1] for input_type in self.input_types]
#         # determine subtask input dims lazily
#         self.input_dims_subtask = [getattr(dataset, f"X_{input_type}").shape[1] for input_type in self.input_types_subtask]
        
#         if SHAP:
#             # forward returns just riskpreds
#             from .nn import ShapMultiModalVAE as Model
#         else:
#             # forward returns (reconstructed_xs, mu, logvar, riskpreds)
#             from .nn import MultiModalVAE as Model
        
#         self.model = Model(input_types = self.input_types, 
#                            input_dims = self.input_dims, 
#                            layer_dims = self.layer_dims, 
#                            input_types_subtask = self.input_types_subtask,
#                            input_dims_subtask = self.input_dims_subtask, 
#                            layer_dims_subtask = self.layer_dims_subtask, 
#                            z_dim = self.z_dim,
#                            activation = self.activation,
#                            subtask_activation = self.subtask_activation,
#                            dropout = self.dropout,
#                            dropout_subtask = self.dropout_subtask,)
#         self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
#         self.survival_loss_func = CoxPHLoss()
#         self.kl_loss_func = KLDivergence()
#         self.reconstruction_loss_funcs = [MSELoss(reduction='mean') for _ in self.model.input_types_vae]
#         self.model.train()
#         self.optimizer.zero_grad()
#         best_loss = np.inf
#         epochs_since_best = 0
#         trainloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
#         current_beta = self.beta_start
#         for epoch in range(1,1+self.epochs):
#             current_loss = 0 # survival loss summed across batches
#             for _, data in enumerate(trainloader):
#                 inputs_vae = [data[f'X_{input_type}'] for input_type in self.model.input_types_vae]
#                 inputs_task = [data[f'X_{input_type}'] for input_type in self.model.input_types_subtask]
#                 outputs, mu, logvar, riskpred = self.model.forward((inputs_vae, inputs_task))
#                 assert len(inputs_vae)==len(outputs)
#                 batch_kl_loss = current_beta * self.kl_weight * self.kl_loss_func(mu, logvar)
#                 assert not batch_kl_loss.isnan().any().item()
#                 batch_reconstruction_losses = [
#                     f(output, input_vae) for f, output, input_vae in zip(self.reconstruction_loss_funcs, outputs, inputs_vae)
#                 ]
#                 for brl in batch_reconstruction_losses:
#                     assert not brl.isnan().any().item()
#                 batch_survival_loss = self.survival_loss_func(data['event_indicator'], data['event_time'], riskpred.flatten())
#                 assert not batch_survival_loss.isnan().any().item()
#                 batch_loss = batch_kl_loss + batch_survival_loss + sum(batch_reconstruction_losses)
#                 batch_loss.backward()
#                 self.optimizer.step()
#                 current_loss += batch_loss.item()

#              # 训练结束后更新 β
#             current_beta = min(current_beta + self.beta_step,self.beta_end) 

#             if epoch <= int(self.burn_in):
#                 pass
#             elif current_loss < best_loss:
#                 epochs_since_best = 0
#                 best_loss = current_loss
#                 if verbose:
#                     print(f'Epoch {epoch}: {current_loss}')
#             elif epochs_since_best == int(self.patience):
#                 print(f'Early stopping at epoch {epoch - self.patience - 1}')
#                 return self
#             else:
#                 epochs_since_best += 1
#         if verbose:
#             warnings.warn(f'Early stopping not triggered. patience: {self.patience}, best_loss: {best_loss}, epochs_since_best: {epochs_since_best}')
#         return self
    
#     def eval(self)->None:
#         self.model.eval()
        
#     def predict(self, X:pd.DataFrame)->torch_tensor:
#         # check if fit has been called
#         check_is_fitted(self)
#         # input validation
#         X = pd.DataFrame(validate_data(self, X, reset=False), index=X.index, columns=X.columns)
        
#         self.model.eval()
#         dataloader = DataLoader(Dataset(X, self.input_types_all, event_indicator_col=self.eventcol,event_time_col=self.durationcol), batch_size=1024, shuffle=False)
#         estimates = []
#         for _, data in enumerate(dataloader):
#             with no_grad():
#                 inputs_vae = [data[f'X_{input_type}'] for input_type in self.model.input_types_vae]
#                 inputs_task = [data[f'X_{input_type}'] for input_type in self.model.input_types_subtask]
#                 _, _, _, riskpred = self.model.forward((inputs_vae, inputs_task))
#                 estimates.append(riskpred.flatten())
#         estimates = torch_cat(estimates)
#         return estimates

#     def score(self, X:pd.DataFrame, y=None)->float:
#         assert isinstance(X, pd.DataFrame)
#         X = pd.DataFrame(validate_data(self, X, reset=False),index=X.index,columns=X.columns)
#         estimates = self.predict(X)
#         event = X[self.eventcol].values.astype(bool)
#         duration = X[self.durationcol].values
#         metric = concordance_index_censored(event, duration, estimates)[0].item()
#         return metric
    
#     def save(self, pth_path):
#         torch_save(self.model.state_dict(), pth_path)
#         return

#     def __call__(self, X:torch_tensor)->tuple:
#         return self.model.__call__(X)