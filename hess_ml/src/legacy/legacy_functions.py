import os 

def parse_folders(self,folder,subfolder):

    print(f'Gathering Folders from {folder}.')

    if subfolder:

        self.gather_subfolders(folder)

    else:
        
        self.gather_folders(folder)
    
    return 

def gather_subfolders(self,folder):

    self.total_structures = 0

    self.geo_dir = []

    molecule_dir = sorted([mol for mol in os.listdir(f'{folder}') if os.path.isdir(os.path.join(f'{folder}',mol))])

    for mol in range(len(molecule_dir)):
        
        data_dir = os.path.join(f'{folder}',molecule_dir[mol])

        temp_dir = sorted([os.path.join(data_dir,geo) for geo in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,geo))])

        self.geo_dir.extend(temp_dir)
        
        self.total_structures += len(temp_dir)

    return


def gather_folders(self,folder):

    molecule_dir = sorted([mol for mol in os.listdir(f'{folder}') if os.path.isdir(os.path.join(f'{folder}',mol))])

    self.geo_dir = (molecule_dir)

    self.total_structures = len(self.geo_dir)

    return 


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