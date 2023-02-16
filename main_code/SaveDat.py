class FTHomo:
    def __init__(self,some_class) -> None:

        self.mol = some_class.mol
        self.geo = some_class.geo

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
    def __init__(self,some_class) -> None:

        self.mol = some_class.mol
        self.geo = some_class.geo

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
