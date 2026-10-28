import argparse
from .preprocess import preprocess_main
from .train import train_main
from .score import score_main
from .shap import shap_main

def myevae_main():
    parser = argparse.ArgumentParser(description='Scripts for training and evaluation of MyeVAE. MyeVAE is a multi-modal variational autoencoder for risk prediction in newly diagnosed multiple myeloma. It is developed from the MMRF CoMMpass dataset. Use the mode flag to determine which action to perform (preprocessing, training, scoring, SHAP).')
    parser.add_argument('mode', type=str, choices=['preprocess', 'train', 'score', 'shap'], help='Mode to run the pipeline in. Preprocess will scale, select, and impute missing features. Train performs hyperparameter tuning. Score performs C-index scoring. Shap creates SHAP summary plots for each input type.')
    parser.add_argument('endpoint', type=str, choices=['pfs','os'], help='Survival endpoint to predict risk against. os = overall survival; pfs = progression-free survival.')
    parser.add_argument('-n', '--name', default='unnamed_model', type=str, help='A name for the model. Defaults to "unnamed_model". e.g. all-omics, exp-only, cna-only, etc. depending on the input modalities used.')
    parser.add_argument('-i', '--inputdir', type=str, default=None, help='Input data directory. csv files of training and validation data should be placed here. e.g. train_features_os_processed.csv, train_labels.csv.')
    parser.add_argument('-o', '--outputdir', default='./output', type=str, help='Output directory for model results. Model weights, json of tuned parameters, and SHAP plots will be saved here.')
    parser.add_argument('-p', '--paramdir', type=str, help='Applies only to mode = train. Folder with hyperparameter file named param_grid.py, specifying a dictionary named param_grid. If not specified, the default hyperparameter grid will be used, from the myevae/param_grid.py')
    parser.add_argument('-t', '--nthreads', default=1, type=int, help='For training - Number of cpu threads to use during grid search. Defaults to 1. Requires pip install myevae[parallel] (with the parallel keyword).')
    args = parser.parse_args()
    
    try:
        assert args.mode in ['preprocess', 'train', 'score', 'shap']
        assert args.endpoint in ['os','pfs']
        if args.mode == "preprocess":
            preprocess_main(endpoint=args.endpoint, 
                            inputdir=args.inputdir)
        elif args.mode == "train":
            train_main(endpoint=args.endpoint, 
                       model_name=args.name, 
                       inputdir=args.inputdir, 
                       param_grid_dir=args.paramdir, 
                       nthreads=args.nthreads, 
                       outputdir=args.outputdir)
        elif args.mode == "score":
            score_main(endpoint=args.endpoint, 
                       model_name=args.name, 
                       inputdir=args.inputdir, 
                       outputdir=args.outputdir)
        elif args.mode == "shap":
            shap_main(endpoint=args.endpoint, 
                      model_name=args.name, 
                      inputdir=args.inputdir, 
                      outputdir=args.outputdir)
        else:
            raise ValueError("Invalid mode or no mode specified. Choose from 'preprocess', 'train', 'score', or 'shap'. Use -h or --help for more information.")
    
    except AssertionError as e:
        print(e)
        parser.print_help()