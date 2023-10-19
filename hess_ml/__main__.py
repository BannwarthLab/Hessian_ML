import time

from hess_ml.src.environment import Environment
from hess_ml.src.parser import Parser


def main() -> None:
    env = Environment()

    env.parse()

    env.parse_toml()

    env.set_general_config()

    env.print_config()

    env.import_data()

    env.train_procedure()

    env.prediction_procedure()


if __name__ == "__main__":
    main()

# 0.20309255655459282
# 0.04129
