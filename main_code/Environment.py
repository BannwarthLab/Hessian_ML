from Parser import Parser
from DataGenerationJOBLIB import DataGeneration
from Training import Training
import os 
from mpi4py import MPI 

class Environment(DataGeneration,Parser,Training):
    def __init__(self):
        #super().__init__       #initializes all parent classes 
        Parser.__init__(self) #initializes only the toml_parser class
        

    def generate_data(self):
        DataGeneration.__init__
        if self.feature_gen:
            if self.subfolder:
                self.mol_geo_idx = []
                self.generate_feature_target_sf() 
        return 