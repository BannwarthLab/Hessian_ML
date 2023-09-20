
from joblib import load
import pickle as pickle

import numpy as np
import glob as glob
import os

from hess_ml.src.Observables import Observables
from hess_ml.src.IO import Input
from hess_ml.src.IO import Output
from hess_ml.src.Geometry import Geometry
from joblib import Parallel, delayed
import time as time

class Predicting(Geometry,Input,Output,Observables):
    def __init__(self) -> None:
        super().__init__
        pass

    def predict(self,files):
        
        try:
            self.predict_model
            self.model = load(os.path.join('',f'{self.predict_model}.joblib'))
        except:
            self.model = load(f'{self.model_name}.joblib')
            pass

        if self.normalization:

            pathname = f'{self.model_name}_transformer.joblib'

            self.transformer = load(pathname)

            pathname = f'{self.model_name}_transformer_target.joblib'
            
            self.target_transformer = load(pathname)


        if self.selection:

            pathname = f'{self.model_name}_selector.joblib'

            self.selector = load(pathname)

        self.not_considered = []

        Parallel(n_jobs=1)(delayed(self.predict_hessian)(file=files[file]) for file in range(len(files)))
    
        with open('not_considered_pred', 'w') as outfile:
            outfile.write('\n'.join(str(i) for i in self.not_considered))
        outfile.close
        
        return       
    def comp_test_observables(self):

        freq_pred_list = []      
        ZPE_pred = []
        Z_pred = []

        freq_true_list = []
        ZPE_true  = []
        Z_true  = []

        with open('pred_structures_final.json','rb') as f:

            i = 0

            while True:
                i +=1 
                try:
                    temp_obj = pickle.load(f)
                    self.R_MI_APF_mat = temp_obj.get('R_MI_APF_mat')
                    self.xyz = temp_obj.get('xyz')
                    self.transpose_list = temp_obj.get('transpose_list')

                    self.N_atoms = temp_obj.get('N_atoms')

                    hess_vec_ab = np.array(temp_obj.get('pred_target_AB'))

                    freq = self.get_Frequencies(hess_vec_ab)
                    
                    ZPE = self.get_ZPE(freq)

                    Z = self.get_partition_func(freq)

                    #ZPE_harm = self.get_harmonic_ZPE(freq)

                    Z_pred.append(Z)

                    #ZPE_harm_pred.append(ZPE_harm)
                    
                    ZPE_pred.append(ZPE)

                    freq_pred_list.extend(freq)

                    hess_vec_aa = np.array(temp_obj.get('Target_AA'))
                    hess_vec_ab = np.array(temp_obj.get('Target_AB'))

                    freq = self.get_Frequencies(hess_vec_ab,hess_vec_aa)

                    ZPE = self.get_ZPE(freq)
                    Z = self.get_partition_func(freq)

                    #ZPE_harm = self.get_harmonic_ZPE(freq)
                    Z_true.append(Z)

                    #ZPE_harm_true.append(ZPE_harm)
                    ZPE_true.append(ZPE)
                    freq_true_list.extend(freq)

                except EOFError:
                        break        
        
        np.savetxt('pred_frequencies.txt',freq_pred_list)
        np.savetxt('true_frequencies.txt',freq_true_list)

        np.savetxt('pred_ZPEs.txt',ZPE_pred)
        np.savetxt('true_ZPEs.txt',ZPE_true)
        
        np.savetxt('pred_Z.txt',Z_pred)
        np.savetxt('true_Z.txt',Z_true)

        print('done')

        return
    #obj = self.import_pickle_obj(file)

    def predict_hess_depracted(self):
        #_____reads heteronuclear model and predicts for each structure the heteronuclear blocks____

        self.truncate_file('PredData.json')

        het_model = load(f'{self.model_name}.joblib')

        if self.normalization:
            pathname = f'{self.model_name}_transformer.joblib'
            transformer = load(pathname)

        if self.selection:
            pathname = f'{self.model_name}_selector.joblib'
            selector = load(pathname)


        test_files = glob.glob('TestData_*.json')
        for file in test_files:

            with open(f'{file}','rb') as f:

                while True:

                    try:
                        temp_obj = pickle.load(f)

                        if self.normalization:
                            H_hetero = het_model.predict(transformer.transform(np.array(temp_obj.get('Feature'))))
                        
                        elif self.selection:
                            H_hetero = het_model.predict(selector.transform(np.array(temp_obj.get('Feature'))))

                        else:
                            H_hetero = het_model.predict((np.array(temp_obj.get('Feature'))))

                        temp_obj['pred_target_AB'] = H_hetero

                        with open('PredData.json','ab') as g:
                            pickle.dump(temp_obj,g)

                    except EOFError:
                            break
            g.close()
            f.close()

        del het_model


        return

    def predict_hessian(self,file):

        self.do_calc = True 

        self.gen_data(file,1)

        #with open(os.path.join(file,f'Hessian_ML_Data.json'),'rb') as f:

        #    Dict = pickle.load(f)

        #f.close()
        if self.do_calc:
            cur_time = time.time()
            
            if self.selection:

                H_hetero = self.model.predict(self.selector.transform(np.array(self.Feature_AB)))

            if self.normalization:

                H_hetero = self.model.predict(self.transformer.transform(np.array(self.Feature_AB)))

                H_hetero = self.target_transformer.inverse_transform(H_hetero)
            
            else:

                H_hetero = self.model.predict((np.array(self.Feature_AB)))

            transpose_list = self.transpose_list

            R_MI_APF_mat = self.R_MI_APF_mat

            N_atoms = self.N_atoms
            
            predHess = self.gen_hess_from_vec_pred(H_hetero,N_atoms,R_MI_APF_mat,transpose_list)

            self.hessian_to_xtb(os.path.join(file,f'MLhesssian'),predHess)
            
            print('Prediction:',round(time.time()- cur_time,5),'s')

            del H_hetero
            del transpose_list
            del N_atoms
            del R_MI_APF_mat

        return