class Params():
    """
    Base parameter class to hold parameters.
    Instantiated when specific model info is not needed, such as when loading data
    @scale_method: for transforming the microarray GEO datasets. One of 'std', 'robust', 'rank', or 'none'.
    """
    def __init__(self,model_name:str,endpoint:str,outputdir):
        # experiment name for the model. it will have its own output directory. usually name of the omics used.
        self.model_name = model_name
        self.endpoint = endpoint
        self.durationcol = self.endpoint+'cdy'
        self.eventcol = 'cens'+self.endpoint
        self.resultsprefix = f"{outputdir}/{self.model_name}_{self.endpoint}"