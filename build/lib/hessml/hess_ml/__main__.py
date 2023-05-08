from hess_ml.src.Environment import Environment
import time as time 

def main() -> None:
    env = Environment()

    if env.config.get('feature_generation',False):
        env.parse_feature_generation()

    if env.config.get('training_testing',False):
        env.parse_train_test_parameter()
    #__________________________________________
    #____________Choosing runtype______________
    if env.config['runtype'] == 'hessian':


        if env.config.get('feature_generation',False):
            env.generate_data()
             
        if env.config.get('training_testing',False):

            temp_time_old = time.time()

                        
            env.train(train_conf = env.config['training_testing'],mode='hetero')


            temp_time_new = time.time()

            print(f'Training was done in {round(temp_time_new - temp_time_old)} s' )

            temp_time_old = time.time()
            env.test()
            temp_time_new = time.time()
            print(f'Testing was done in {round(temp_time_new - temp_time_old)} s')

if __name__ == '__main__':
    main()
