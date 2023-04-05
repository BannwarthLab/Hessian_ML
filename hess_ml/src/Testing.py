from sklearn.ensemble import ExtraTreesRegressor
from joblib import dump,load
import pickle as pickle
from src.ReadWrite import ReadWrite
import numpy as np
from src.SaveDat import PickleData
import glob as glob
import os 
from src.Observables import Observables


class Testing(ReadWrite,Observables):
    def __init__(self) -> None:
        super().__init__
        pass

    def test(self):
        self.predict_hess()
        self.comp_test_observables()
        return

    def comp_test_observables(self):
        print('Computing Observables ...',end="")
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
                    self.lamb_len = temp_obj.get('lamb_len')

                    hess_vec_ab = np.array(temp_obj.get('pred_target_AB'))
                    hess_vec_aa = None

                    hessian_dftdt4 = temp_obj.get('hessian_dftd4')

                    freq = self.gen_Frequencies(hess_vec_aa,hess_vec_ab,hess_d4=hessian_dftdt4,pred=True)
                    
                    ZPE = self.get_ZPE(freq)
                    Z = self.get_partition_func(freq)
                    #ZPE_harm = self.get_harmonic_ZPE(freq)
                    Z_pred.append(Z)
                    #ZPE_harm_pred.append(ZPE_harm)
                    ZPE_pred.append(ZPE)

                    freq_pred_list.extend(freq)

                    hess_vec_aa = np.array(temp_obj.get('Target_AA'))
                    hess_vec_ab = np.array(temp_obj.get('Target_AB'))

                    freq = self.gen_Frequencies(hess_vec_aa,hess_vec_ab,hess_d4=hessian_dftdt4,pred=False)
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

    def predict_hess(self):
        #_____reads heteronuclear model and predicts for each structure the heteronuclear blocks____

        self.truncate_file('pred_structures.json')
        self.truncate_file('pred_structures_final.json')


        het_model = load(f'{self.model_name}_hetero.joblib')

        if self.normalization:
            pathname = f'{self.model_name}_hetero_transformer.joblib'
            transformer = load(pathname)

        if self.selection:
            pathname = f'{self.model_name}_hetero_selector.joblib'
            selector = load(pathname)


        test_files = glob.glob('test_structures*.json')
        for file in test_files:
            with open(f'{file}','rb') as f:
                while True:
                    try:
                        temp_obj = pickle.load(f)

                        if self.normalization:
                            H_hetero = het_model.predict(transformer.transform(np.array(temp_obj.get('Feature_AB'))))#[::9,:]
                        
                        elif self.selection:
                            H_hetero = het_model.predict(selector.transform(np.array(temp_obj.get('Feature_AB'))))

                        else:
                            H_hetero = het_model.predict((np.array(temp_obj.get('Feature_AB'))))#[::9,:]
                        temp_obj['pred_target_AB'] = H_hetero#.reshape(len(H_hetero)*9)
                        with open('pred_structures_final.json','ab') as g:
                            pickle.dump(temp_obj,g)

                    except EOFError:
                            break
            g.close()
            f.close()

        del het_model


        return
