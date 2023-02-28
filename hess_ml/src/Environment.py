from src.Parser import Parser
from src.DataGenerationJOBLIB import DataGeneration
from src.Training import Training
from src.Testing import Testing

import os 

class Environment(DataGeneration,Parser,Training,Testing):
    def __init__(self):
        super().__init__       #initializes all parent classes 
        Parser.__init__(self) #initializes only the toml_parser class
        

    def generate_data(self):
        DataGeneration.__init__
        if self.feature_gen:
            if self.subfolder:
                if self.diag == 'DTR':
                    self.mol_geo_idx = []
                    self.generate_feature_target_sf_dtr() 
                if self.diag == 'GNN':
                    self.mol_geo_idx = []
                    self.generate_feature_target_sf_gnn()
        return 