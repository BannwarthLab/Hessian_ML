from __future__ import annotations

from hess_ml.src.environment import Environment

def main() -> None:
    env = Environment()

    env.parse()

    env.parse_toml()

    env.set_config()

    env.print_config()

    if not isinstance(env.config.general.random_state,list):
        
        random_states = [env.config.general.random_state]

    else: 
        random_states = env.config.general.random_state
        
    for random_state in random_states: 

        env.config.general.random_state = random_state

        if env.config.general.feature:
            env.import_data()

        if env.config.general.train:
            env.train_procedure()

        if env.config.general.predict:
            env.prediction_procedure()

if __name__ == "__main__":
    main()

