from __future__ import annotations

import os
import time
import sys 
import glob 

import numpy as np
from sklearn.model_selection import train_test_split
from hess_ml.src2.utilities.decorator import initProcess
from hess_ml.src2.ml_models.sklearn.model_template import ActiveTraining, Training
from hess_ml.src2.utilities.parser import Parser
from hess_ml.src2.utilities.reader import rd_txt_file
from hess_ml.src2.ml_models.sklearn.predicting import HessianPredictor,HessianPMPredictor
from hess_ml.src2.governance.config import Configurations
import hess_ml.src2.governance.globals as globals 
from hess_ml.src2.utilities.writer import list_to_txt

from hess_ml.src2.molecule.molecule import Molecule,NuclearHessian,NuclearHessianPM
from hess_ml.src2.molecule.tblite.training_feature import Feature as TrainFeature
from hess_ml.src2.molecule.tblite.training_feature import ReducedFeature as TrainReducedFeature
from hess_ml.src2.molecule.tblite.training_feature import CustomFeature as TrainCustomFeature


class Environment(Parser):
    def __init__(self):
        Parser.__init__(self)

        self.parse()

        self.parse_toml()

        self.set_config()

        self.print_config()

        self.set_device()
        #self.producte = HessianProducer(self.config) -> generates the data etc.

        match self.config.general.runtype.lower():
            case 'hessian_pm':
                self.predictor = HessianPMPredictor(self.config)
            case 'hessian':
                self.predictor = HessianPredictor(self.config)

    def set_config(self):
        """
        Set config from the input toml and fills up the missing information with the default configs
        """
        self.config = Configurations(self.parsed_config)

    def set_device(self):
        if not(self.config.general.gpu):
            globals.DEVICE = "cpu"

    def print_config(self):
        print("Used config is:")
        print("")
        config_classes = ["general","molecule","train","predict"]
        for config_class in config_classes:
            print(f"{config_class.capitalize()} configurations:")
            class_vars:dict = vars(getattr(self.config,config_class))

            for name in class_vars:

                if isinstance(class_vars[name], dict):
                    print("")
                    print(name)
                    for key in class_vars[name]:
                        print(key, ":",  class_vars[name][key])
                else:
                    print(name, ":", class_vars[name])

            print("")

    @initProcess
    def train_procedure(self):
        self.config.internal.train = True

        test_size_threshold = 1e-8

        train_size = self.config.train.train_size

        self.rnd_states = self.config.general.random_state

        self.model_name = self.config.train.model_name

        self.runtype = self.config.general.runtype

        i = None

        if isinstance(train_size, list):

            train_size = sorted(train_size)[::]

            self.shuffle_idx = np.arange(len(self.Features))

            self.shuffle_idx, validation_idx = train_test_split(
                self.shuffle_idx,
                train_size=train_size[-1],
                test_size=self.config.train.validation_size,
                random_state=self.config.general.random_state,
            )

            for i in range(len(train_size)):

                rnd_seed = self.rnd_states

                print(
                    f"Percentage of data set used for training: {train_size[i]*100} %",
                )

                temp_time_old = time.time()

                train_size_temp = train_size[i] / train_size[-1]
                print(train_size_temp)
                if train_size_temp < 1.0 - 1e-8:
                    shuffle_idx_temp,_  = train_test_split(
                        self.shuffle_idx,
                        train_size=train_size_temp,
                        test_size=self.config.train.validation_size,
                        random_state=self.config.general.random_state,
                    )

                else:
                    shuffle_idx_temp = self.shuffle_idx

                print(f"Total training points:{len(shuffle_idx_temp)}")

                if self.config.train.active:
                    model_trainer = ActiveTraining(self.config.train)
                else:
                    model_trainer = Training(self.config.train)

                model_trainer.set_rnd_seed(rnd_seed)
                model_trainer.set_split(i)
                model_trainer.build_model()
                model_trainer.training(features=self.Features,
                                            targets=self.Targets,
                                            shuffle_idx=shuffle_idx_temp,
                                            )

                pred_vals = model_trainer.complete_model.predict(self.Features[validation_idx])

                rmsd = np.sqrt(np.mean(self.Targets[validation_idx]-pred_vals)**2)


                print(f"Validation statistics")
                print(f"RndState: {self.config.general.random_state} Size: {train_size[i]} RMSD: {rmsd}")

                if self.config.train.test_size >= test_size_threshold:
                    self.train_size = train_size[i]
                    self.rnd_seed = self.config.general.random_state
                    self.predictor.model = model_trainer.complete_model
                    rmsd = self.predictor.test_array(self.test_geo)
                    print("Test statistics")
                    print(f"RndState: {self.config.general.random_state} Size: {train_size[i]} RMSD: {rmsd}")


                del model_trainer

        else:
            print(
                f"Percentage of data set used for training: {train_size*100} %",
            )

            self.shuffle_idx = np.arange(len(self.Features))

            print(len(self.shuffle_idx))

            temp_time_old = time.time()


            if self.config.train.active:
                model_trainer = ActiveTraining(self.config.train)
            else:
                model_trainer = Training(self.config.train)

            model_trainer.set_rnd_seed(self.rnd_states)
            model_trainer.set_split(i)
            model_trainer.build_model()
            model_trainer.training(features=self.Features,
                                        targets=self.Targets,
                                        shuffle_idx=self.shuffle_idx)
            temp_time_new = time.time()

            print(f"Training was done in {round(temp_time_new - temp_time_old)} s")

            if self.config.train.test_size >= test_size_threshold and len(self.test_geo) > 0:
                temp_time_old = time.time()

                self.train_size = self.config.train.test_size
                self.rnd_seed = self.config.general.random_state
                self.predictor.model = model_trainer.complete_model
                self.predictor.predict_array(self.test_geo)

                temp_time_new = time.time()

                print(
                    f"Testing was done in {temp_time_new - temp_time_old: 0.2f} s",
                )

    @initProcess
    def prediction_procedure(self):
        self.config.internal.train = False

        self.folders = []

        if self.config.predict.folder is not None:
            self.parse_data_set(self.config.predict.folder)

        if self.config.predict.folder_list is not None:
            files = rd_txt_file(self.config.predict.folder_list)

            self.folders.extend(files)

        self.model_name = self.config.predict.model_name

        print(f"Starting prediction of {len(self.folders)} files")

        temp_time_old = time.time()
        self.predictor.predict_array(self.folders)
        temp_time_new = time.time()

        print(f"Prediction was done in {round(temp_time_new - temp_time_old)} s")


    def import_data(self):
        if self.config.molecule.feature.lower() in ["tblite","complete","reduced","custom"]:
            self.folders = []
            
            if not(os.path.isdir(globals.PROCESSED_DATA_FOLDER)):
                os.mkdir(globals.PROCESSED_DATA_FOLDER)

            if self.config.molecule.folder is not None:
                self.parse_data_set(self.config.molecule.folder)
            elif self.config.molecule.files is not None:
                files = rd_txt_file(self.config.molecule.files)
                self.folders.extend(files)

            else:
                print("Neither a folder or files specified.")

            self.do_preparation_split()

            self.Targets = []
            self.Features = []

            self.generate_data(self.train_idx)

            self.Targets = np.array(self.Targets)
            self.Features = np.array(self.Features).astype(np.float32)

            if not self.splitted and self.config.general.wrt_feature:

                with open(os.path.join(globals.PROCESSED_DATA_FOLDER,"Features.npy"),"wb") as f:
                    np.save(f,self.Features)
                    f.close()

                with open(os.path.join(globals.PROCESSED_DATA_FOLDER,"Targets.npy"),"wb") as f:
                    np.save(f,self.Targets)
                    f.close()

        elif self.config.molecule.feature.lower() == "numpy":

            print("Features and Targets are import from .npy files.")

            # self.Features = np.loadtxt("Features.txt",dtype=np.float32)
            # self.Targets = np.loadtxt("Targets.txt")


            globals.PROCESSED_DATA_FOLDER = self.get_process_folder()

            with open(os.path.join(globals.PROCESSED_DATA_FOLDER,"Features.npy"),"rb") as f:
                self.Features = np.load(f)
                f.close()

            with open(os.path.join(globals.PROCESSED_DATA_FOLDER,"Targets.npy"),"rb") as f:
                self.Targets = np.load(f)
                f.close()

            if os.path.isfile("test_files.txt"):
                self.test_geo =  rd_txt_file("test_files.txt")
            else:
                self.test_geo = []


        elif self.config.molecule.feature.lower() == "numpy_split":

            print("Features and Targets are import from .npy files.")

            feature_files = glob.glob(os.path.join(self.config.molecule.folder,"Features**.npy"))
            target_files = glob.glob(os.path.join(self.config.molecule.folder,"Targets**.npy"))

            feature_files = sorted(feature_files)
            target_files = sorted(target_files)

            assert len(feature_files) == len(target_files), "Missmatch of number of target and feature files." 

            for ffeature,ftarget in zip(feature_files,target_files):
                print(ffeature,ftarget)
                assert ffeature[-8:-4] == ftarget[-8:-4], "Missmath of names of feature and target files."

            with open(feature_files[0], 'rb') as f:
                features = np.load(f)
                f.close()

            with open(target_files[0], 'rb') as f:
                targets = np.load(f)
                f.close()

            for f_feature,f_target in zip(feature_files[1:],target_files[1:]):

                with open(f_feature, 'rb') as f:
                    features = np.append(features,np.load(f),axis=0)
                    f.close()

                with open(f_target, 'rb') as f:
                    targets = np.append(targets,np.load(f),axis=0)
                    f.close()
            
            self.Targets = targets
            self.Features = features


        else:
            print("Feature generation must be specified.")
        # one could add different features that will be imported

    def get_process_folder(self):

        if self.config.molecule.folder is not None:
                if os.path.isfile(os.path.join(self.config.molecule.folder,"Features.npy")):
                    globals.PROCESSED_DATA_FOLDER = self.config.molecule.folder
        return globals.PROCESSED_DATA_FOLDER

    def do_preparation_split(self):
        """
        Does a split of the geometry file directories into train and test sets.
        Saves the information in txt files
        """

        train_size=self.config.train.train_size
        test_size=self.config.train.test_size
        rnd_seed=self.config.general.random_state

        max_train_size = 1.0
        total_structures = len(self.folders)
        geo_idx = np.arange(0, total_structures)

        train_size_temp = max(train_size) if isinstance(train_size,list) else train_size

        if train_size_temp == max_train_size:
            self.train_idx = geo_idx
            self.test_idx = []

        else:
            self.train_idx, self.test_idx = train_test_split(
                geo_idx,
                test_size=test_size,
                train_size=train_size_temp,
                random_state=rnd_seed,
            )

            self.comp_idx = np.concatenate((self.train_idx, self.test_idx), axis=None)

            geo_idx = geo_idx[self.comp_idx]

        self.test_geo = []

        for i in self.test_idx:
            self.test_geo.append(self.folders[i])

        self.train_geo = []

        for i in self.train_idx:
            self.train_geo.append(self.folders[i])

        list_to_txt(self.test_geo, os.path.join("", "test_files.txt"))

        list_to_txt(self.train_geo, os.path.join("", "train_files.txt"))


    def generate_data(self:Environment, idx=None):
        self.wall_time0 = time.time()

        print("Starting Data Generation for Features...")

        # ________Parallelized Feature Generation___________

        # if idx == None:
        #    idx = np.arange(0,len(self.geo_dir))
        self.not_considered = []

        for program in self.config.molecule.program:
            if len(self.config.molecule.program) == 1:
                if program.lower()  == "orca":
                    print("More than one program not implemented yet!")
                elif program.lower()  == "xtb":
                    if self.config.molecule.feature.lower() == 'reduced':
                        self.feature_class = TrainReducedFeature
                    elif self.config.molecule.feature.lower() == 'custom':
                        self.feature_class = TrainCustomFeature
                    else:
                        self.feature_class = TrainFeature
            else:
                print("More than one program not implemented yet!")

        
        match self.config.general.runtype.lower():
            case 'hessian':
                self.hess_type = NuclearHessian
            case 'hessian_pm':
                self.hess_type = NuclearHessianPM



        self.n_data =  0
        self.splitted = False
        n_split = 0

        max_n_data = 5e5

        for geo in idx:

            self.GenerateData(dir=self.folders[geo])

            print(f"Number of DataPoints {self.n_data}")

            if self.n_data > max_n_data and self.config.general.split_feature:

                self.Targets = np.array(self.Targets)
                self.Features = np.array(self.Features).astype(np.float32)

                print(f"Length of features:{self.Features.shape}")

                with open(os.path.join(globals.PROCESSED_DATA_FOLDER,f"Features{n_split:04d}.npy"),"wb") as f:
                    np.save(f,self.Features)
                    f.close()

                with open(os.path.join(globals.PROCESSED_DATA_FOLDER,f"Targets{n_split:04d}.npy"),"wb") as f:
                    np.save(f,self.Targets)
                    f.close()

                self.Features = []
                self.Targets = []

                self.n_data =  0
                n_split += 1
                self.splitted = True

        if self.splitted:

            self.Targets = np.array(self.Targets)
            self.Features = np.array(self.Features).astype(np.float32)

            with open(f"Features{n_split}.npy","wb") as f:
                np.save(f,self.Features)
                f.close()

            with open(f"Targets{n_split}.npy","wb") as f:
                np.save(f,self.Targets)
                f.close()

            print("""Due to the large size of the Features and Targets, the data was split.
                  No training and predicting is performed as the data is only partially stored in the RAM.""")
            self.config.general.train = False
            self.config.general.predict = False
            sys.exit()


        print("")
        print(
            f"Features and Targets of {len(idx)} structures "
            f"were generated in {round(time.time() - self.wall_time0)} s\n",
        )

        outputfile_name = "not_considered"
        with open(outputfile_name, "w") as outfile:
            outfile.write("\n".join(str(i) for i in self.not_considered))
        outfile.close()

    def GenerateData(self, dir):


        print(f"Path: {dir}")

        mol = Molecule(dir,self.config.molecule.xyz_file)
        mol.hessian = self.hess_type
        mol.ml_hessian = self.hess_type

        print(f"Number of atoms: {mol.nat}")
        
        mol.feature = self.feature_class
        mol.read_hessian(self.config.molecule.target_file)

        self.n_data += mol.nat*(mol.nat-1)/2
        print(f"Feature shape: {mol.feature.processed_features.shape}")

        if mol.calc_succeeded:
            if np.isnan(np.sum(mol.feature.processed_features)):
                print(mol.feature.processed_features)

            if np.isnan(np.sum(mol.feature.processed_target)):
                print(mol.feature.processed_target)
                print("Some feature is NaN.")
                sys.exit()

            self.Features.extend(mol.feature.processed_features)
            self.Targets.extend(mol.feature.processed_target)

        else:
            self.not_considered.append(
                os.path.join(
                    self.config.molecule.folder,
                    self.config.molecule.xyz_file,
                ),
            )