import numpy as np
import pandas as pd
from hess_ml.src.Rotation_func import Rotation_Functions
import numpy as np
import pickle as pickle 
import os 
import json as json 

class Input():
      def __init__(self):
            pass 

      def import_coord(self,file):
            with open(file) as myfile:
                  head = [next(myfile) for x in range(2)]

            coord_var = pd.read_csv(file,sep = '\s+',skiprows = 2,header = None)
            coord_var.columns= ['atoms','x','y','z']
            return coord_var,head

      def import_dipm(self,file):
            coord_var = pd.read_csv(file,sep = ',')
            #coord_var.columns= ['atoms','x','y','z']
            return coord_var

      def import_hessian(self,file,coord_var):
            LineList = []
            
            with open (file,'r') as fd:
                  Lines = [line.rstrip('\n') for line in fd]
                  for line in Lines[1:]:
                        LineList += line.split()

            hess = np.zeros([len(coord_var['atoms'])*3,len(coord_var['atoms'])*3])

            i = 0
            for k in range(len(hess[0,:])):
                  for l in range(len(hess[:,0])):
                        hess[k,l] = float(LineList[i])
                        i+=1
            return hess

      def import_hessian_dftd4(self,file,coord):
            nat3 = len(coord['atoms'])*3

            file_path = os.path.join(self.geo_working_dir,'dftd4.json')

            with open(file_path) as f:
                  egh = json.load(f)

            hess_dftd4 = np.array(egh.get('hessian')).reshape(nat3,nat3)

            return hess_dftd4
      
      def import_gradient(self,file):

            with open(file , 'rb') as f:
                  f.close()

            self.gradient = np.genfromtxt(file,skip_header=2+self.N_atoms,skip_footer=1,loose=True)
            #gradient = gradient.flatten()

            return


      def import_ml_features(self,file):
            GFN2_quantities = pd.read_csv(f'{file}')
            self.CN = np.array(GFN2_quantities.loc[:,['coordination number','delta coordination number']].values.tolist())

            self.q_atom = np.array(GFN2_quantities.loc[:,['atomic partial charges','delta partial charges']].values.tolist())

            self.dipm_atom = np.array(GFN2_quantities.loc[:,['dipm_atom_x','dipm_atom_y','dipm_atom_z']].values.tolist())
            self.dipm_delta = np.array(GFN2_quantities.loc[:,['dipm_delta_x','dipm_delta_y','dipm_delta_z']].values.tolist())

            self.dipm_only_mull = np.array(GFN2_quantities.loc[:,['delta dipm only mull x','delta dipm only mull y','delta dipm only mull z']].values.tolist())
            self.dipm_only_Z = np.array(GFN2_quantities.loc[:,['delta dipm only Z x','delta dipm only Z y','delta dipm only Z z']].values.tolist())

            self.qm_atom = Rotation_Functions.qm_matrix(np.array(GFN2_quantities.loc[:,['qm_atom_xx','qm_atom_yy', 'qm_atom_zz','qm_atom_xy','qm_atom_xz','qm_atom_yz']].values.tolist()))
            self.qm_delta = Rotation_Functions.qm_matrix(np.array(GFN2_quantities.loc[:,['qm_delta_xx','qm_delta_yy', 'qm_delta_zz','qm_delta_xy','qm_delta_xz','qm_delta_yz']].values.tolist()))

            self.qm_delta_only_mull = np.array(GFN2_quantities.loc[:,['delta qm only mull x','delta qm only mull y','delta qm only mull z']].values.tolist())
            self.qm_delta_only_Z = np.array(GFN2_quantities.loc[:,['delta qm only Z x','delta qm only Z y','delta qm only Z z']].values.tolist())


            self.energy_based = np.array(GFN2_quantities.loc[:,['chem pot','HOAO_a (eV)','LUAO_a (eV)','HOAO_b (eV)','LUAO_b (eV)',
                                    'E_repulsion','E_EHT',' E_disp_2','E_disp_3','E_ies_ixc','E_aes','E_tot',
                                    'E_axc',' chem_pot_ext','e_gap_ext','ehoao_ext','eluao_ext']].values.tolist())
            
            self.names = GFN2_quantities.columns.tolist()
            
            return 
      
      def import_wbo(self,file):
            wbo = pd.read_csv(file,names=['at1','at2','wbo'],sep='\s+')
            return wbo


      def import_pickle_FT(self,file):
            feature = []
            target = []

            i = 0
            with open(f'{file}','rb') as f:
                  while True:
                        try:
                              i+=1
                              temp_obj = pickle.load(f)
                              feature.extend(temp_obj['Feature'])
                              target.extend(temp_obj['Target_AB'])
                        except EOFError:
                              #print(f'Features and Targets of a total of {i-1} structures are used.\n')
                              break
                            
            return feature,target

      def truncate_file(self,file):
            if os.path.isfile(file):
                  with open(file,'wb+') as f:
                        f.truncate(0)
                  f.close()
            return