##/usr/bin/env python #for later
from Environment import Environment

def main():
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

        env.train(mode='hetero')
        env.train(mode='homo')


if __name__ == '__main__':
    main()
