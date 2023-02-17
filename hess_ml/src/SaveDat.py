
class FTHomo:
    def __init__(self,some_class,mol,geom) -> None:

        self.mol = mol
        self.geo = geom

        self.xyz = some_class.xyz
        self.N_at = some_class.N_atoms

        self.Feature = some_class.Feature_AA
        self.Target = some_class.Target_AA

        self.init_R_MI = some_class.init_R_MI
        self.R_MI_APF_mat = some_class.R_MI_APF_mat

        return 

    def add_idx(self,idx):
        self.idx = idx
        return

class FTHetero:
    def __init__(self,some_class,mol,geom) -> None:

        self.mol = mol
        self.geo = geom

        self.xyz = some_class.xyz
        self.N_at = some_class.N_atoms

        self.Feature = some_class.Feature_AB
        self.Target = some_class.Target_AB

        self.init_R_MI = some_class.init_R_MI
        self.R_MI_APF_mat = some_class.R_MI_APF_mat

        return 

    def add_idx(self,idx):
        self.idx = idx
        return

class PickleData:
    def __init__(self,some_class,mol,geom) -> None:

        self.mol = mol
        self.geo = geom

        self.transpose_list = some_class.transpose_list

        self.xyz = some_class.xyz
        self.N_atoms = some_class.N_atoms

        self.Feature_AA = some_class.Feature_AA
        self.Feature_AB = some_class.Feature_AB

        self.Target_AA = some_class.Target_AA
        self.Target_AB = some_class.Target_AB

        self.init_R_MI = some_class.init_R_MI
        self.R_MI_APF_mat = some_class.R_MI_APF_mat

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
    