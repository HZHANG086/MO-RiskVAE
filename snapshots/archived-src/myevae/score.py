import os
import pandas as pd
from argparse import ArgumentParser
from json import load as json_load
from torch import load as torch_load
from torch.nn import * # for activation functions 

from .estimator import VAE
from .params import Params

def score_main(endpoint:str, 
         model_name:str, 
         outputdir:str,
         inputdir:str) -> None:
    
    params = Params(model_name=model_name, endpoint=endpoint, outputdir=outputdir)
    
    predict_features_file=f'{inputdir}/valid_features_{endpoint}_processed.csv'
    predict_labels_file=f'{inputdir}/valid_labels.csv'

    assert os.path.exists(predict_features_file), f"{predict_features_file} not found"
    assert os.path.exists(predict_labels_file), f"{predict_labels_file} not found"

    predict_features=pd.read_csv(predict_features_file,index_col=0)
    predict_labels=pd.read_csv(predict_labels_file,index_col=0)[[params.eventcol,params.durationcol]]
    predict_dataframe=pd.concat([predict_labels,predict_features],axis=1)
    
    resultspath = f'{params.resultsprefix}.json'
    results = json_load(open(resultspath,'r'))
    best_params = results['best_epoch']['params']
    print(f'Loaded hyperparameters from {params.resultsprefix}.json')

    # initialize model with best hyperparameters
    best_estimator = \
        VAE(input_types=eval(best_params['input_types']),
            layer_dims=eval(best_params['layer_dims']),
            input_types_subtask=eval(best_params['input_types_subtask']),
            layer_dims_subtask=eval(best_params['layer_dims_subtask']),
            z_dim=eval(best_params['z_dim']),
            lr=eval(best_params['lr']),
            epochs=0, # do not fit
            burn_in=eval(best_params['burn_in']),
            patience=eval(best_params['patience']),
            batch_size=eval(best_params['batch_size']),
            eventcol=results['params_fixed']['eventcol'],
            durationcol=results['params_fixed']['durationcol'],
            kl_weight=eval(best_params['kl_weight']),
            activation=eval(best_params['activation']),
            subtask_activation=eval(best_params['subtask_activation']),
            )
    # call fit to create the nn.Module inside the estimator
    best_estimator.fit(predict_dataframe,verbose=False,SHAP=False)
    # extract the nn Module from the estimator
    # load stored weights
    state_dict = torch_load(f'{params.resultsprefix}.pth')
    best_estimator.model.load_state_dict(state_dict)
    print(f'Loaded weights from {params.resultsprefix}.pth')
    # obtain C-index score
    print(f'Scoring...')
    cindex = best_estimator.score(predict_dataframe,predict_labels)
    # display score
    print(f'C-index: {cindex}')


if __name__ == "__main__":
    parser = ArgumentParser(description='Predict on test set using a trained VAE model.')
    parser.add_argument('-e', '--endpoint', type=str, choices=['pfs','os'], default='both', help='Survival endpoint (pfs or os or both)')
    parser.add_argument('-m', '--name', type=str, help='An experiment name for the model')
    parser.add_argument('-i', '--inputdir', type=str, help='Input data directory.')
    parser.add_argument('-o', '--outputdir', type=str, help='Output directory for model results')
    args = parser.parse_args()
    score_main(args.name, args.endpoint, args.outputdir, args.inputdir)