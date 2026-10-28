import os
import numpy as np
import pandas as pd
import warnings
from argparse import ArgumentParser
from json import load as json_load
import torch
from torch import load as torch_load
from torch.utils.data import DataLoader
from torch.nn import * # for activation functions 
import shap
from .estimator import VAE
from .params import Params
from .dataset import Dataset
from .plotshap import summary_plot

def shap_main(endpoint:str, 
         model_name:str, 
         outputdir:str,
         inputdir:str) -> None:
    
    params = Params(model_name=model_name, endpoint=endpoint, outputdir=outputdir)
    
    background_features_file=f'{inputdir}/train_features_{endpoint}_processed.csv'
    background_labels_file=f'{inputdir}/train_labels.csv'
    shap_features_file=f'{inputdir}/valid_features_{endpoint}_processed.csv'
    shap_labels_file=f'{inputdir}/valid_labels.csv'

    assert os.path.exists(background_features_file), f"{background_features_file} not found" 
    assert os.path.exists(background_labels_file), f"{background_labels_file} not found"
    assert os.path.exists(shap_features_file), f"{shap_features_file} not found"
    assert os.path.exists(shap_labels_file), f"{shap_labels_file} not found"

    background_features=pd.read_csv(background_features_file,index_col=0)
    background_labels=pd.read_csv(background_labels_file,index_col=0)[[params.eventcol,params.durationcol]]
    background_dataframe=pd.concat([background_labels,background_features],axis=1)

    shap_features=pd.read_csv(shap_features_file,index_col=0)
    shap_labels=pd.read_csv(shap_labels_file,index_col=0)[[params.eventcol,params.durationcol]]
    shap_dataframe=pd.concat([shap_labels,shap_features],axis=1)
    
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
            dropout=eval(best_params.get('dropout', '0.0')),
            dropout_subtask=eval(best_params.get('dropout_subtask', '0.0')),
            )
    # call fit to create the nn.Module inside the estimator
    # instantiate the nn.Module in SHAP mode 
    best_estimator.fit(background_dataframe,verbose=False,SHAP=True)
    # load stored weights
    state_dict = torch_load(f'{params.resultsprefix}.pth')
    best_estimator.model.load_state_dict(state_dict)
    print(f'Loaded weights from {params.resultsprefix}.pth')
    
    input_types_vae = eval(results['params_search']['input_types'])[0]
    input_types_subtask = eval(results['params_search']['input_types_subtask'])[0]
    input_types_all = input_types_vae + input_types_subtask
    eventcol=results['params_fixed']['eventcol']
    durationcol=results['params_fixed']['durationcol']
    
    # prepare background data
    background_dataset = Dataset(background_dataframe, input_types_all, event_indicator_col=eventcol,event_time_col=durationcol)
    background_data = [getattr(background_dataset,f"X_{dtype}") for dtype in input_types_all]
    
    # prepare validation data
    shap_dataset = Dataset(shap_dataframe, input_types_all, event_indicator_col=eventcol,event_time_col=durationcol)
    shap_data = [getattr(shap_dataset,f"X_{dtype}") for dtype in input_types_all]
    
    # move data to same device as model (DeepExplainer requires matching device)
    device = next(best_estimator.model.parameters()).device
    background_data = [t.to(device) for t in background_data]
    shap_data = [t.to(device) for t in shap_data]
    
    # get SHAP scores
    print(f'Performing deep integration with background samples...')
    shap_explainer = shap.DeepExplainer(best_estimator.model, background_data)
    
    print(f'Approximating SHAP values of validation samples (may take 1-2 minutes)...')
    shap_values = shap_explainer.shap_values(shap_data)

    # we will use this to index the shap data later during generation of summary plots
    exp_idx = 0
    cna_idx = 1
    gistic_idx = 2
    fish_idx = 3
    sbs_idx = 4
    ig_idx = 5
    clin_idx = 6

    # For Gene expression, gene CN
    # convert ENSG ID to canonical gene names for easier readability
    exp_features = background_dataframe.filter(regex='Feature_exp_ENSG').columns.str.extract('(ENSG.*)$').iloc[:,0]
    try:
        from importlib_resources import files, as_file
        # annotfile = os.path.join(os.path.dirname(__file__), 'data', 'ensembl_ref.csv')
        import myevae.data
        source = files(myevae.data).joinpath('ensembl_ref.csv')
        with as_file(source) as annotfile:
            annot = pd.DataFrame(annotfile)\
                .query("`Gene type` == 'protein_coding'")\
                ['Gene name'].drop_duplicates()
        exp_featurenames = np.array([gname if not pd.isna(gname) else gid 
                                for (gname, gid) 
                                in zip(annot.reindex(exp_features).values, exp_features)])
        exp_notna = np.where(~pd.isna(annot.reindex(exp_features)))[0]
    except:
        warnings.warn('Annotation file for gene names and types (ensembl_ref.csv) not found. Defaulting to using ENSG IDs for gene expression and gene CN.')
        exp_featurenames = exp_features
        exp_notna = exp_featurenames.index

    print(f'Generating SHAP plot for gene expression...')
    summary_plot(shap_values[exp_idx], shap_data[exp_idx], exp_notna, exp_featurenames, outputdir, 'Gene expression', f'shap_{endpoint}_exp.png')

    cna_features = background_dataframe.filter(regex='Feature_CNA_ENSG').columns.str.extract('(ENSG.*)$').iloc[:,0]
    try:
        cna_featurenames = np.array([gname if not pd.isna(gname) else gid 
                                for (gname, gid) 
                                in zip(annot.reindex(cna_features).values, cna_features)])
        cna_notna = np.where(~pd.isna(annot.reindex(cna_features)))[0]
    except:
        cna_featurenames = cna_features
        cna_notna = cna_featurenames.index

    print(f'Generating SHAP plot for gene CN...')
    summary_plot(shap_values[cna_idx], shap_data[cna_idx], cna_notna, cna_featurenames, outputdir, 'Gene copy number', f'shap_{endpoint}_cna.png')

    print(f'Generating SHAP plot for GISTIC CN...')
    gistic_features = background_dataframe.filter(regex='Feature_CNA_(Amp|Del)').columns.str.extract('CNA_(.*)$').iloc[:,0]
    summary_plot(shap_values[gistic_idx], shap_data[gistic_idx], gistic_features.index, gistic_features, outputdir, 'GISTIC copy number', f'shap_{endpoint}_gistic.png')

    print(f'Generating SHAP plot for FISH CN...')
    fish_features = background_dataframe.filter(regex='Feature_fish').columns.str.extract('fish_SeqWGS_Cp_(.*)$').iloc[:,0]
    summary_plot(shap_values[fish_idx], shap_data[fish_idx], fish_features.index, fish_features, outputdir, 'FISH copy number', f'shap_{endpoint}_fish.png')
    
    print(f'Generating SHAP plot for SBS...')
    sbs_features = background_dataframe.filter(regex='Feature_SBS').columns.str.extract('Feature_(SBS.*)$').iloc[:,0]
    summary_plot(shap_values[sbs_idx], shap_data[sbs_idx], sbs_features.index, sbs_features, outputdir, 'Mutational signatures', f'shap_{endpoint}_sbs.png')

    print(f'Generating SHAP plot for IGH translocation...')
    ig_features = background_dataframe.filter(regex='Feature_SeqWGS').columns.str.extract('SeqWGS_(.*)_CALL$').iloc[:,0]
    summary_plot(shap_values[ig_idx], shap_data[ig_idx], ig_features.index, ig_features, outputdir, 'IGH translocation', f'shap_{endpoint}_ig.png')

    print(f'Generating SHAP plot for clinical features...')
    clin_features = background_dataframe.filter(regex='Feature_clin').columns.str.extract('clin_D_PT_(.*)$').iloc[:,0]
    summary_plot(shap_values[clin_idx], shap_data[clin_idx], clin_features.index, clin_features, outputdir, 'Clinical features', f'shap_{endpoint}_clin.png')


if __name__ == "__main__":
    parser = ArgumentParser(description='Predict on test set using a trained VAE model.')
    parser.add_argument('-e', '--endpoint', type=str, choices=['pfs','os'], default='both', help='Survival endpoint (pfs or os or both)')
    parser.add_argument('-m', '--name', type=str, help='An experiment name for the model')
    parser.add_argument('-i', '--inputdir', type=str, help='Input data directory.')
    parser.add_argument('-o', '--outputdir', type=str, help='Output directory for model results')
    args = parser.parse_args()
    shap_main(args.name, args.endpoint, args.outputdir, args.inputdir)