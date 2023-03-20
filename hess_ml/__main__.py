## gfeldmann/Documents/GitLab/hessian_ml/venv/bin/python3
from src.Environment import Environment

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
            
            if not(env.only_hom):
                env.train(train_conf = env.config['training_testing'],mode='hetero')

            env.train(train_conf = env.config['training_testing'],mode='homo')

            
            env.test(env.only_hom)

if __name__ == '__main__':
    main()
