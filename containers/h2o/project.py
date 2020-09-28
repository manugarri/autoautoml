"""
AutoML using H2o
https://github.com/h2oai/h2o-3
"""
from pprint import pprint

import h2o
from h2o.automl import H2OAutoML

from autoautoml.core import AutoMLCore
from autoautoml.utils import evaluate, save_artifact, write


class AutoML(AutoMLCore):

    name = "h2o"

    def load(self):
        """
        Loads the original data
        """
        super().load()
        h2o.init()
        dataset = self.dataset[self.feature_columns + [self.target]]
        h2o_dataset = h2o.H2OFrame(dataset)
        self.train, self.test = h2o_dataset.split_frame(ratios=[.9])
        if self.problem_type == "classification":
            self.train[self.target] = self.train[self.target].asfactor()

    def build_pipeline(self):
        """
        Makes a pipeline based on data_config
        """
        return H2OAutoML(**self.automl_settings)

    def fit(self):
        """
        Performs the search
        """
        print("SEARCHING ESTIMATORS")
        self.automl_pipeline = self.build_pipeline()
        self.automl_pipeline.train(
            x=self.feature_columns,
            y=self.target,
            training_frame=self.train,
        )

        self.automl_pipeline.leaderboard.head()

    def evaluate(self):
        """
        Evaluates and stores performance
        """
        print("EVALUATING ESTIMATOR")
        train_preds = self.automl_pipeline.predict(self.train).as_data_frame().predict
        test_preds = self.automl_pipeline.predict(self.test).as_data_frame().predict
        y_train = self.train[self.target].as_data_frame()[self.target]
        y_test = self.test[self.target].as_data_frame()[self.target]
        train_score = evaluate(y_train, train_preds, self.problem_type)
        test_score = evaluate(y_test, test_preds, self.problem_type)

        self.automl_pipeline.leader.model_performance(self.test).show()

        self.metadata = {
            "metrics": {
                "test": test_score,
                "train": train_score
            },
            "experiment_settings": self.experiment_settings
        }
        pprint(self.metadata)

    def save(self):
        print("Saving artifacts")
        if "metadata" in self.artifact_config:
            print("SAVING METADATA")
            save_artifact(self.metadata, self.artifact_config["metadata_path"])
        if "model_path" in self.artifact_config:
            model_artifact_name = h2o.save_model(self.automl_pipeline.leader, path=".")
            with open(model_artifact_name, "rb") as fname:
                serialized_model = fname.read()
                write(serialized_model, self.artifact_config["model_path"])
