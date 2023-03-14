
class FTHomo:
    def __init__(self,some_class,mol,geom,dir) -> None:

        self.dict = {}
        self.dict['dir'] =  dir

        
        self.dict['xyz'] = some_class.xyz
        self.dict['N_atoms'] = some_class.N_atoms
        self.dict['Feature'] = some_class.Feature_AA
        self.dict['Target'] = some_class.Target_AA

        self.dict['init_R_MI'] = some_class.init_R_MI
        self.dict['R_MI_APF_mat'] = some_class.R_MI_APF_mat

        return 

    def add_idx(self,idx):
        self.idx = idx
        return

class FTHetero:
    def __init__(self,some_class,mol,geom,dir) -> None:

        self.dict = {}
        self.dict['dir'] =  dir

        self.dict['xyz'] = some_class.xyz
        self.dict['N_atoms'] = some_class.N_atoms

        self.dict['transpose_list'] = some_class.transpose_list

        self.dict['Feature'] = some_class.Feature_AB
        self.dict['Target'] = some_class.Target_AB

        self.dict['init_R_MI'] = some_class.init_R_MI
        self.dict['R_MI_APF_mat'] = some_class.R_MI_APF_mat
        return 

    def add_idx(self,idx):
        self.idx = idx
        return

class PickleData:
    def __init__(self,some_class,mol,geom,dir) -> None:

        self.dict = {}

        self.dict['dir'] =  dir

        if not(some_class.diag == 'GNN'):
            self.dict['transpose_list'] = some_class.transpose_list
        
        self.dict['xyz'] = some_class.xyz
        self.dict['N_atoms'] = some_class.N_atoms

        self.dict['Feature_AA'] = some_class.Feature_AA
        self.dict['Feature_AB'] = some_class.Feature_AB

        self.dict['Target_AA'] = some_class.Target_AA
        self.dict['Target_AB'] = some_class.Target_AB

        self.dict['init_R_MI'] = some_class.init_R_MI
        self.dict['R_MI_APF_mat'] = some_class.R_MI_APF_mat

        return 

    def add_idx(self,idx):
        self.idx = idx
        return

    def add_pred_target_AB(self,pred_target_AB):
        self.pred_target_AB = pred_target_AB
        return
    
    def add_pred_target_AA(self,pred_target_AA):
        self.pred_target_AA = pred_target_AA
        return
    


class PickleDict:
    def __init__(self,some_class,mol,geom,dir) -> None:
        
        self.dict = {}

        self.dict['dir'] =  dir

        if not(some_class.diag == 'GNN'):
            self.dict['transpose_list'] = some_class.transpose_list
        
        self.dict['xyz'] = some_class.xyz
        self.dict['N_atoms'] = some_class.N_atoms
        self.dict['Feature_AA'] = some_class.Feature_AA
        self.dict['Target_AA'] = some_class.Target_AA

        self.dict['init_R_MI'] = some_class.init_R_MI
        self.dict['R_MI_APF_mat'] = some_class.R_MI_APF_mat
        self.dict['wbo'] = some_class.wbo
        return 

    def add_idx(self,idx):
        self.dict['idx'] = idx
        return

    def add_pred_target_AB(self,pred_target_AB):
        self.dict['pred_Target_AB'] = pred_target_AB
        return
    
    def add_pred_target_AA(self,pred_target_AA):
        self.dict['pred_target_AA'] = pred_target_AA
        return
    