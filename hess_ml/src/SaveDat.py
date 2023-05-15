
class PickleData:
    def __init__(self,some_class,mol,geom,dir) -> None:

        self.dict = {}

        self.dict['dir'] =  dir

        self.dict['transpose_list'] = some_class.transpose_list
        
        self.dict['xyz'] = some_class.xyz
        self.dict['N_atoms'] = some_class.N_atoms

        self.dict['Feature'] = some_class.Feature_AB

        self.dict['Target_AA'] = some_class.Target_AA
        self.dict['Target_AB'] = some_class.Target_AB

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
