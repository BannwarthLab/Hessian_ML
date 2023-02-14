##/usr/bin/env python #for later
from Environment import Environment
from mpi4py import MPI
from DataGeneration import DataGeneration

def main():
    env = Environment()

    if env.config.get('feature_generation',False):
        env.parse_feature_generation()

    if env.config.get('training_testing',False):
        env.parse_train_test_parameter()

    #__________________________________________
    #____________Choosing runtype______________
    if env.config['runtype'] == 'hessian':
        #if env.config.get('feature_gen',False):
        env.generate_data()
        #MPI.COMM_WORLD.Barrier()
        #MPI.COMM_WORLD.Disconnect()

        env.train(mode='homo')
        env.train(mode='hetero')


if __name__ == '__main__':
    main()
