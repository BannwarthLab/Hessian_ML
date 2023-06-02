from hess_ml.src.Parser import Parser
from hess_ml.src.DataGeneration import DataGeneration
from hess_ml.src.Training import Training
from hess_ml.src.Predicting import Predicting
from hess_ml.src.IO import Input
import os 

class Environment(DataGeneration,Parser,Training,Predicting,Input):
    def __init__(self):
        super().__init__      #initializes all parent classes
        Parser.__init__(self) #initializes only the toml_parser class
        DataGeneration.__init__(self)

        return
    
    def parse(self):

        """
        Parses through the possible .toml configurations
        """

        if self.config.get('general',False):
            self.parse_general()

        if self.config.get('feature',False):
            self.parse_feature()

        if self.config.get('training',False):
            self.parse_training()

        if self.config.get('predict',False):
            self.parse_predict()

        self.config.get('model',{})

        return 
    

    def get_folders(self):

        if self.config.get('feature',False) or self.config.get('training',False):

            self.parse_folders(self.folder,self.subfolder)

            self.do_preparation_split()

        elif self.config.get('predict',False):
            
            self.parse

        return
    

    def gen_features(self):

        if self.config.get('feature',False):
            
            if self.feature_gen:

                self.generate_data(idx=self.train_idx)

        
        #one could add different features that will be imported
        
        return