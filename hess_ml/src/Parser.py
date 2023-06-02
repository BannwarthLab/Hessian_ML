from argparse import ArgumentParser
import numpy as np
import tomli 
import os

class Parser:
    def __init__(self) -> None:

        self.runtype_target = {'hessian' : 'hessian'}

        self.parse_toml()

        self.cwd = os.getcwd()

    def parse_toml(self):

        """
        Parser for the .toml to generate a dictionoary for the input information
        """

        parser = ArgumentParser()

        parser.add_argument("-i", "--input", type=str, default='input.toml',
                                help="INPUT must be a .toml file")

        parser.add_argument("-to", "--toml", action='store_true',
                                help="generates a basic example .toml file")

        parser.add_argument("-dto", "--dtoml", action='store_true',
                                help="generates a detailed example .toml file")

        with open(parser.parse_args().input, mode="rb") as fp:
            self.config = tomli.load(fp)

        try:

            self.config['runtype']

        except LookupError as exc:

            print("got exception", repr(exc))
    
            print(r"A 'runtype' must be set in order to run this program.")
            print(r"For more information use the flag '--help'.")

        self.threads  = self.config.get('threads',1)
        return 
    

    def parse_general(self): 

        """
        Sets the general parameter needed for:
            - the generation of features
            - the training 
            - the prediction
        """

        self.file_feature  = self.config['general'].get('feature','ml_feature.csv')

        self.file_target   = self.config['general'].get('target',self.runtype_target[self.config['runtype']])

        self.file_coord    = self.config['general'].get('coord','xtbopt.xyz')

        self.file_gradient = self.config['general'].get('grad','gradient')

        self.folder = self.config['general'].get('folder',None)

        self.subfolder = self.config['general'].get('subfolder',False)

        self.rnd_seed    = self.config['general'].get('random_seed',np.random.randint(0,1000))


        return 
    

    def parse_feature(self): 

        """
        Sets the parameter needed for the generation of features
        """

        self.feature_gen = self.config['feature'].get('generate',True)

        #Implement tblite api for generation of the basic features 
        #self.tblite = self.config['feature'].get('tblite',False)
        #If true they are generated with tbltie in advance 


        return 

    def parse_training(self):


        """
        Sets the parameter needed for the training of the ML Model
        """


        self.train_size = self.config['training'].get('train_size',0.75)

        if type(self.train_size) == list:
            train_max = max(self.train_size)
        else:
            train_max = self.train_size

        self.test_size  = self.config['training'].get('test_size',1-train_max)

        self.method     = self.config['training'].get('method','ETR')

        self.SearchCV     = self.config['training'].get('SearchCV',None)

        self.testing        = self.config['training'].get('test',False)

        if self.SearchCV == 'Random':

            self.n_iter_search     = self.config['training'].get('n_iter',25)    

        self.model_name    = self.config['training'].get('model_name',f"{self.runtype_target[self.config['runtype']]}_model")

        try:

            self.config['training']['parameter']

        except:

            print('No parameters for the heteronuclear model are specified. Default parameters are set')

            self.config['training']['parameter'] = {}

        return 
    

    def parse_predict(self): #Set parameters for prediction

        """
        Sets the parameter needed for the prediction of a set of systems
        """

        self.predict_folder = self.config['predict'].get('folder',False)


        if self.config.get('training',False):
            self.predict_model = self.config['predict'].get('model',self.config['training'].get('model_name',f"{self.runtype_target[self.config['runtype']]}_model"))
        else:
            self.predict_model = self.config['predict'].get('model',f"{self.runtype_target[self.config['runtype']]}_model")


        self.predict_model_folder = self.config['predict'].get('model_folder',False)

        self.predict_subfolder = self.config['general'].get('subfolder',False)

        self.predict_files = self.config['predict'].get('files', None)

        self.predict_data_gen  = self.config['predict'].get('generate',True)

        self.normalizer_name = self.config['predict'].get('normalizer',None)

        self.selector_name = self.config['predict'].get('selector',None)


        if self.selector_name == None:

            self.selection = False 
        else:
            self.selection = True 


        if self.normalizer_name == None:

            self.normalization = False
        else:
            self.normalization = True


        if self.predict_folder == self.folder:

            print('You chose the same folder for prediction as you chose for training and testing. This is not recommended.')

        return 