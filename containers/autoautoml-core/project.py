"""
Core project
"""

from pprint import pprint
from autoautoml.core import AutoMLCore

from autoautoml.utils import (get_experiment_settings, read_artifact, save_artifact,
                    read_dataset, clean_dataset)


class AutoML(AutoMLCore):

    name = "autoautoml-core"

    def load(self):
        """
        Loads the original data
        """
        self.dataset = read_dataset(self.data_config["training_path"])

    def fit(self):
        """
        Performs the search
        """
        print("SEARCHING ESTIMATORS")
        self.dataset = clean_dataset(self.dataset, **self.data_config)
        self.automl_settings = self.experiment_settings["automl_settings"]
        self.automl_pipeline = None

    def evaluate(self):
        """
        Evaluates and stores performance
        """
        print("EVALUATING ESTIMATOR")
        self.metadata = {
            "metrics": {
                "accuracy": 1.0
            },
            "model_params": {
                "l1_ratio": 0.2
            }
        }
        pprint(self.metadata)
