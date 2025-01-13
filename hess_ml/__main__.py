#!/usr/bin/env python3

from __future__ import annotations

from tblite.interface import Calculator
from hess_ml.src2.governance.environment import Environment
import time 
import torch 
import numpy as np 

def main() -> None:

    time_0 = time.time()
    env = Environment()
    
    if not isinstance(env.config.general.random_state,list):

        random_states = [env.config.general.random_state]

    else:
        random_states = env.config.general.random_state

    for random_state in random_states:

        env.config.general.random_state = random_state

        torch.random.manual_seed(random_state)
        np.random.seed(random_state)

        if env.config.general.feature:
            env.import_data()

        if env.config.general.train:
            env.train_procedure()

        if env.config.general.predict:
            env.prediction_procedure()

        # if env.config.general.optimization:
        #     env.optimization(env.config.predict.folder)

    print(f"Total Wall Time: {time.time() - time_0:4.2F} s")

if __name__ == "__main__":
    main()

