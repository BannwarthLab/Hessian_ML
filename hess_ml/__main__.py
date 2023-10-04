from hess_ml.src.Environment import Environment
from hess_ml.src.Parser import Parser
import time as time 

def main() -> None:

    env = Environment()

    env.parse()

    env.parse_toml()

    env.print_config()
    
    env.set_general_config()

    env.import_data()

    env.do_train()

    env.do_prediction()

if __name__ == '__main__':
    main()
