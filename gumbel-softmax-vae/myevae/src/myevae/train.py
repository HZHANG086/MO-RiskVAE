import os
from argparse import ArgumentParser
from json import dump as json_dump
import pandas as pd
from warnings import warn
from sklearn.model_selection import GridSearchCV

from .params import Params
from .estimator import VAE


def train_main(endpoint: str,
               model_name: str,
               outputdir: str,
               inputdir: str,
               param_grid_dir: str,
               nthreads: int) -> None:

    params = Params(model_name=model_name,
                    endpoint=endpoint,
                    outputdir=outputdir)

    train_features_file = f'{inputdir}/train_features_{endpoint}_processed.csv'
    valid_features_file = f'{inputdir}/valid_features_{endpoint}_processed.csv'
    train_labels_file = f'{inputdir}/train_labels.csv'
    valid_labels_file = f'{inputdir}/valid_labels.csv'

    assert os.path.exists(train_features_file), f"{train_features_file} not found"
    assert os.path.exists(train_labels_file), f"{train_labels_file} not found"
    assert os.path.exists(valid_features_file), f"{valid_features_file} not found"
    assert os.path.exists(valid_labels_file), f"{valid_labels_file} not found"

    train_features = pd.read_csv(train_features_file, index_col=0)
    valid_features = pd.read_csv(valid_features_file, index_col=0)
    train_labels = pd.read_csv(train_labels_file, index_col=0)[[params.eventcol, params.durationcol]]
    valid_labels = pd.read_csv(valid_labels_file, index_col=0)[[params.eventcol, params.durationcol]]

    train_dataframe = pd.concat([train_labels, train_features], axis=1)
    valid_dataframe = pd.concat([valid_labels, valid_features], axis=1)

    base_estimator = VAE(eventcol=params.eventcol,
                         durationcol=params.durationcol)

    # -------- Load param grid --------
    if param_grid_dir is None:
        warn("No user-specified hyperparameter grid .py file. Defaulting to src/param_grid.py.")
        from .param_grid import param_grid
    else:
        print(f'Using user-specified param_grid.py in {param_grid_dir}')
        import sys
        sys.path.append(param_grid_dir)
        from param_grid import param_grid

    # -------- Grid search --------
    grid_search = GridSearchCV(
        base_estimator,
        param_grid,
        cv=3,          # 显式指定
        n_jobs=1       # 不让 sklearn 自己并行（你用 dask 控制）
    )

    if nthreads > 1:
        import joblib
        from dask.distributed import Client, LocalCluster
        cluster = LocalCluster()
        client = Client(cluster)
        with joblib.parallel_config("dask", n_jobs=nthreads):
            grid_search.fit(train_dataframe)
    else:
        grid_search.fit(train_dataframe)

    # -------- Update params --------
    for k, v in grid_search.best_params_.items():
        setattr(params, k, v)

    results = {}

    results['params_fixed'] = {
        k: v for k, v in vars(params).items()
        if not k.startswith('_') and k not in param_grid.keys()
    }

    results['params_search'] = {
        k: str(v) for k, v in param_grid.items()
    }

    results['best_epoch'] = {}
    results['best_epoch']['params'] = {
        k: str(v) for k, v in grid_search.best_params_.items()
    }

    valid_metric = grid_search.score(valid_dataframe)
    results['best_epoch']['valid_metric'] = valid_metric

    os.makedirs(os.path.dirname(params.resultsprefix), exist_ok=True)

    with open(f'{params.resultsprefix}.json', 'w') as f:
        json_dump(results, f, indent=4)

    grid_search.best_estimator_.save(f'{params.resultsprefix}.pth')


if __name__ == "__main__":
    parser = ArgumentParser(
        description='Tune hyperparameters of VAE model using grid search.')

    parser.add_argument('-e', '--endpoint',
                        type=str,
                        choices=['pfs', 'os'],
                        required=True)

    parser.add_argument('-m', '--name',
                        type=str,
                        required=True)

    parser.add_argument('-i', '--inputdir',
                        type=str,
                        required=True)

    parser.add_argument('-p', '--paramdir',
                        type=str,
                        default=None)

    parser.add_argument('-t', '--nthreads',
                        type=int,
                        default=1)

    parser.add_argument('-o', '--outputdir',
                        type=str,
                        required=True)

    args = parser.parse_args()

    train_main(endpoint=args.endpoint,
               model_name=args.name,
               outputdir=args.outputdir,
               inputdir=args.inputdir,
               param_grid_dir=args.paramdir,
               nthreads=args.nthreads)



# import os
# from argparse import ArgumentParser
# from json import dump as json_dump
# import pandas as pd
# from warnings import warn
# from sklearn.model_selection import GridSearchCV

# from .params import Params
# from .estimator import VAE

# # tune the hyperparameters of one model on one dataset (either full dataset, or train+valid dataset)
# # saves model results like C-index as a json file specified by `params.results_prefix`.json
# def train_main(endpoint:str, 
#          model_name:str, 
#          outputdir:str,
#          inputdir:str,
#          param_grid_dir:str,
#          nthreads:int) -> None:
    
#     params = Params(model_name=model_name, endpoint=endpoint, outputdir=outputdir)
    
#     train_features_file=f'{inputdir}/train_features_{endpoint}_processed.csv'
#     valid_features_file=f'{inputdir}/valid_features_{endpoint}_processed.csv'
#     train_labels_file=f'{inputdir}/train_labels.csv'
#     valid_labels_file=f'{inputdir}/valid_labels.csv'

#     assert os.path.exists(train_features_file), f"{train_features_file} not found" 
#     assert os.path.exists(train_labels_file), f"{train_labels_file} not found"
#     assert os.path.exists(valid_features_file), f"{valid_features_file} not found"
#     assert os.path.exists(valid_labels_file), f"{valid_labels_file} not found"

#     train_features=pd.read_csv(train_features_file,index_col=0)
#     valid_features=pd.read_csv(valid_features_file,index_col=0)
#     train_labels=pd.read_csv(train_labels_file,index_col=0)[[params.eventcol,params.durationcol]]
#     valid_labels=pd.read_csv(valid_labels_file,index_col=0)[[params.eventcol,params.durationcol]]
#     train_dataframe=pd.concat([train_labels,train_features],axis=1)
#     valid_dataframe=pd.concat([valid_labels,valid_features],axis=1)

#     base_estimator = VAE(eventcol=params.eventcol,durationcol=params.durationcol)

#     # load the param grid
#     if param_grid_dir is None:
#         warn( "No user-specified hyperparamter grid .py file. Defaulting to src/param_grid.py.")
#         from .param_grid import param_grid
#     else:
#         print(f'Using user-specified param_grid.py in {param_grid_dir}')
#         import sys
#         sys.path.append(param_grid_dir)
#         from param_grid import param_grid
    
#     grid_search = GridSearchCV(base_estimator, param_grid)

#     if nthreads > 1:
#         # use dask for parallelization
#         import joblib
#         from dask.distributed import Client, LocalCluster
#         cluster = LocalCluster()
#         client = Client(cluster)
#         with joblib.parallel_config("dask", n_jobs=nthreads):
#             grid_search.fit(train_dataframe)
#     else:
#         # do not use parallelization
#         grid_search.fit(train_dataframe)

#     # update params with best params
#     for k,v in grid_search.best_params_.items():
#         setattr(params,k,v)

#     results = {}
#     # keeping track of genes used
#     results['params_fixed'] = {k: v for k, v in vars(params).items() if not k.startswith('_') and k not in param_grid.keys()}
#     results['params_search'] = {k: v.__str__() for k, v in param_grid.items() } # save activation as string
#     results['best_epoch'] = {}
#     results['best_epoch']['params'] = {k:v.__str__() for k, v in grid_search.best_params_.items()} # save activation as string

#     valid_metric = grid_search.score(valid_dataframe)
#     results['best_epoch']['valid_metric'] = valid_metric
            
#     os.makedirs(os.path.dirname(params.resultsprefix),exist_ok=True)

#     # save results of hyperparameter tuning
#     with open(f'{params.resultsprefix}.json', 'w') as f:
#         json_dump(results, f, indent=4)

#     # save model state dict
#     grid_search.best_estimator_.save(f'{params.resultsprefix}.pth')

# if __name__ == "__main__":
#     parser = ArgumentParser(description='Tune hyperparameters of VAE model using grid search cross-validation. For adjusting hyperparameters, modify params.py and param_grid.py')
#     parser.add_argument('-e', '--endpoint', type=str, choices=['pfs','os'], default='both', help='Survival endpoint (pfs or os or both)')
#     parser.add_argument('-m', '--name', type=str, help='An experiment name for the model')
#     parser.add_argument('-i', '--inputdir', type=str, help='Input data directory.')
#     parser.add_argument('-p', '--paramdir', type=str, help='For training - folder with hyperparameter file named param_grid.py.')
#     parser.add_argument('-t', '--nthreads', type=int, default=1, help='For training - Number of cpu cores to use. Requires dask[distributed] to be installed.')
#     parser.add_argument('-o', '--outputdir', type=str, help='Output directory for model results')
#     args = parser.parse_args()
#     train_main(args.endpoint, args.name, args.outputdir, args.paramdir, args.nthreads)
    