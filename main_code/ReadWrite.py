import numpy as np
import pandas as pd
from Rotation_func import Rotation_Functions

import pickle as pickle
class ReadWrite():
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

      def import_gradient(self,file,coord_var):

            gradient = np.genfromtxt(file,skip_header=2+len(coord_var['atoms']),skip_footer=1)
            gradient = gradient.flatten()

            return gradient


      def import_ml_features(self,file):
            GFN2_quantities = pd.read_csv(f'{file}')
            self.CN = np.array(GFN2_quantities.loc[:,['coordination number','delta coordination number']].values.tolist())
            self.dipm_atom = np.array(GFN2_quantities.loc[:,['dipm_atom_x','dipm_atom_y','dipm_atom_z']].values.tolist())
            self.dipm_delta = np.array(GFN2_quantities.loc[:,['dipm_delta_x','dipm_delta_y','dipm_delta_z']].values.tolist())
            self.dipm_only_mull = np.array(GFN2_quantities.loc[:,['delta dipm only mull x','delta dipm only mull y','delta dipm only mull z']].values.tolist())
            self.qm_atom = Rotation_Functions.qm_matrix(np.array(GFN2_quantities.loc[:,['qm_atom_xx','qm_atom_yy', 'qm_atom_zz','qm_atom_xy','qm_atom_xz','qm_atom_yz']].values.tolist()))
            self.qm_delta = Rotation_Functions.qm_matrix(np.array(GFN2_quantities.loc[:,['qm_delta_xx','qm_delta_yy', 'qm_delta_zz','qm_delta_xy','qm_delta_xz','qm_delta_yz']].values.tolist()))
            self.energy_based = np.array(GFN2_quantities.loc[:,['gap (eV)','chem.pot (eV)','HOAO (eV)','LUAO (eV)',
                                    'E_repulsion','E_EHT',' E_disp_2','E_disp_3','E_ies_ixc','E_aes',' E_tot',
                                    'E_axc',' chem_pot_ext','e_gap_ext','ehoao_ext','eluao_ext']].values.tolist())
            self.names = GFN2_quantities.columns.tolist()
            return 

      def import_pickle_FT(self,file):

            objects = []
            with open(file,'rb') as f:
                  for _ in range(len(self.train_idx)):
                        objects.extend(pickle.load(f))
                        
            return objects