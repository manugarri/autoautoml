"""
AutoML using AutoKeras
https://autokeras.com
"""

from pprint import pprint
import warnings
warnings.simplefilter("ignore")

import autokeras as ak
from category_encoders.one_hot import OneHotEncoder
from mlxtend.feature_selection import ColumnSelector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union

from autoautoml.core import AutoMLCore
from autoautoml.utils import evaluate


class AutoML(AutoMLCore):

    name = "autokeras"

    def build_pipeline(self):
        """
        Makes a pipeline based on data_config
        """
        if self.problem_type == "classification":
            automl_pipeline = ak.StructuredDataClassifier(**self.automl_settings)
        elif self.problem_type == "regression":
            automl_pipeline = ak.StructuredDataRegressor(**self.automl_settings)
        return automl_pipeline

    def fit(self):
        """
        Performs the search
        """
        super().fit()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.1
        )
        self.automl_pipeline = self.build_pipeline()
        self.automl_pipeline.fit(self.X_train, self.y_train)


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
