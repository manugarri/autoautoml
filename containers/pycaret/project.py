"""
AutoML using PyCaret
https://github.com/pycaret/pycaret
"""

from pprint import pprint

from sklearn.model_selection import train_test_split

from autoautoml.core import AutoMLCore
from autoautoml.utils import (get_experiment_settings, read_dataset, save_artifact, evaluate,
                   infer_categoricals, clean_dataset)


class AutoML(AutoMLCore):

    name = "pycaret"

    def fit(self):
        """
        Performs the search
        """
        self.train, self.test = train_test_split(self.dataset, test_size=0.1)
        self.X_train = self.train.drop(columns=self.target)
        self.y_train = self.train[self.target]
        self.X_test = self.test.drop(columns=self.target)
        self.y_test = self.test[self.target]
        if self.problem_type == "classification":
            from pycaret.classification import automl, compare_models, setup
        else:
            from pycaret.regression import automl, compare_models, setup
        experiment = setup(data=self.train, target=self.target, silent=True, html=False)
        compare_models(**self.automl_settings)
        self.automl_pipeline = automl()

    def evaluate(self):
        """
        Evaluates and stores performance
        """
        print("EVALUATING ESTIMATOR")
        train_preds = self.automl_pipeline.predict(self.X_train)
        test_preds = self.automl_pipeline.predict(self.X_test)
        train_score = evaluate(self.y_train, train_preds, self.problem_type)
        test_score = evaluate(self.y_test, test_preds, self.problem_type)
        self.metadata = {
            "metrics": {
                "test_score": test_score,
                "train_score": train_score
            },
            "experiment_settings": self.experiment_settings
        }
        pprint(self.metadata)
