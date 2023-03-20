from argparse import ArgumentParser
import numpy as np
import tomli 
import os

class Parser:
    def __init__(self) -> None:
        self.runtype_target = {'hessian' : 'hessian'}
        self.parse_toml()
        self.cwd = os.getcwd()

    def parse_toml(self): #Parses the .toml input and generates a dict.
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

    def parse_feature_generation(self): #Set parameters for features
        self.train_test = False

        if not(self.config['feature_generation'].get('feature_aa',False)):
            self.feature_gen = True

            self.file_feature  = self.config['feature_generation'].get('feature','ml_feature.csv')
            self.file_target   = self.config['feature_generation'].get('target',self.runtype_target[self.config['runtype']])
            self.folder_data   = self.config['feature_generation'].get('binary','')
            self.output_file   = f"{self.config['feature_generation'].get('output_file','systems')}*.json"
            self.file_dipm     = self.config['feature_generation'].get('dipm','xyz_dipm.csv')
            self.file_coord    = self.config['feature_generation'].get('coord','xtbopt.xyz')
            self.subfolder     = bool(self.config['feature_generation'].get('subfolder',False))
            self.diag          = self.config['feature_generation'].get('diag','DTR')
            
            if self.diag == 'GNN':
                self.file_wbo      = self.config['feature_generation'].get('wbo','wbo')

        else:
            self.feature_gen = False

            self.file_feature_aa = self.config['feature_generation'].get('feature_aa','Feature_Vector_AA')
            self.file_feature_ab = self.config['feature_generation'].get('feature_ab','Feature_Vector_AB')

            self.file_target_ab = self.config['feature_generation'].get('target_ab','Target_Vector_AB')
            self.file_target_ab = self.config['feature_generation'].get('target_ab','Target_Vector_AB')

        return 

    def parse_train_test_parameter(self): #Set parameters for features
        self.rnd_seed    = self.config['training_testing'].get('random_seed',np.random.randint(0,1000))

        try: 
            self.config['training_testing'].get('total_geometries')
        except:
            print('The total numbers of geometries must be known for the split into test and train set.')

        self.total_geometries = self.config['training_testing'].get('total_geometries')
        self.train_size = self.config['training_testing'].get('train_size',0.75)
        self.test_size  = self.config['training_testing'].get('test_size',0.25)
        self.train_test  = self.config['training_testing'].get('test',True)
        self.only_hom  = self.config['training_testing'].get('only_hom','False')
        self.method     = self.config['training_testing'].get('method','ETR')    
        self.SearchCV     = self.config['training_testing'].get('SearchCV',None)    

        if self.only_hom == 'False':
            self.only_hom = False
        elif self.only_hom == 'True':
            self.only_hom = True 

        self.model_name    = self.config['training_testing'].get('model_name',f"{self.runtype_target[self.config['runtype']]}_model")

        if self.config['runtype'] == 'hessian':

            try:
                self.config['training_testing']['homo']

            except:
                print('No parameters for the homonuclear model are specified. Default parameters are set.')
                self.ml_parameter =[{'n_estimators': 150, 'max_depth': 30}]

            else:
                self.ml_parameter = self.config['training_testing']['homo']

        if self.config['runtype'] == 'hessian' and not(self.only_hom):

            try:
                self.config['training_testing']['hetero']

            except:
                print('No parameters for the heteronuclear model are specified. Default parameters are set')
                self.ml_parameter =[{'n_estimators': 175, 'max_depth': 25}]

            else:
                self.ml_parameter.append(self.config['training_testing']['hetero'])
        return 