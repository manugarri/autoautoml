"""
Core AutoML class
"""
from pprint import pprint

from autoautoml.utils import (
        get_experiment_settings, infer_categoricals,
        read_artifact, save_artifact,
        read_dataset, clean_dataset, evaluate
)


class AutoMLCore:
    name = "core"

    def __init__(self):
        print("INITIALIZING SEARCH")
        self.experiment_settings = get_experiment_settings()
        self.data_config = self.experiment_settings["data"]
        self.problem_type = self.data_config["problem_type"]
        self.target = self.data_config["target_column"]
        self.automl_settings = self.experiment_settings["automl_settings"]
        self.artifact_config = self.experiment_settings.get("artifacts", {})

    def load(self):
        """
        Loads the original data
        """
        self.dataset = read_dataset(self.data_config["training_path"])
        self.dataset = clean_dataset(self.dataset, **self.data_config)
        drop_columns = self.data_config.get("drop_columns", [])
        self.feature_columns = self.data_config.get(
                "feature_columns",
                list(self.dataset.drop(columns=[self.target] + drop_columns).columns)
        )
        self.X = self.dataset[self.feature_columns]
        self.y = self.dataset[self.target]

    def fit(self):
        """
        Performs the search
        """
        print("SEARCHING ESTIMATORS")

    def evaluate(self):
        """
        Evaluates and stores performance
        """
        print("EVALUATING ESTIMATOR")

    def save(self):
        """
        Refits and saves final estimator
        """
        if "metadata" in self.artifact_config:
            print("SAVING METADATA")
            save_artifact(self.metadata, self.artifact_config["metadata_path"])
        if "model_path" in self.artifact_config:
            print("SAVING MODEL")
            save_artifact(self.automl_pipeline, self.artifact_config["model_path"])

    def run(self):
        """
        End to end process
        """
        self.load()
        self.fit()
        self.evaluate()
        self.save()
