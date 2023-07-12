from hess_ml.src.Environment import Environment
import time as time 

def main() -> None:

    env = Environment()

    env.parse()



    #____________Choosing runtype______________
    
    if env.config['runtype'] == 'hessian':

        env.get_folders()

        env.gen_features()
        
        if env.config.get('training',False):

            env.train(train_conf = env.config['training']['parameter'],runtype=env.runtype_target)

            if env.testing:

                temp_time_old = time.time()

                env.predict(env.test_geo)
                
                temp_time_new = time.time()
                
                print(f'Testing was done in {round(temp_time_new - temp_time_old)} s')

        if env.config.get('predict',False):

            if env.predict_folder:
                
                env.parse_folders(env.predict_folder,env.predict_subfolder)

            if env.predict_files:
                
                files = env.rd_txt_file(env.predict_files)

                try:

                    env.geo_dir.append(files)

                except:

                    env.geo_dir = files

            print(f'Starting prediction of {len(env.geo_dir)} files')

            temp_time_old = time.time()                

            env.predict(env.geo_dir)

            temp_time_new = time.time()

            print(f'Prediction was done in {round(temp_time_new - temp_time_old)} s')


if __name__ == '__main__':
    main()
