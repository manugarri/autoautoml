"""
AutoML using AutoSklearn
https://automl.github.io/auto-sklearn/master/
"""

from pprint import pprint

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from category_encoders.one_hot import OneHotEncoder
from mlxtend.feature_selection import ColumnSelector
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union

from autoautoml.core import AutoMLCore
from autoautoml.utils import (
        get_experiment_settings, infer_categoricals,
        read_artifact, save_artifact,
        read_dataset, clean_dataset, evaluate
)


class AutoML(AutoMLCore):

    name = "autosklearn"

    def load(self):
        """
        Loads the original data
        """
        super().load()
        self.feat_type = ['categorical' if x.name == 'object' else 'numerical' for x in self.X.dtypes]

    def build_pipeline(self):
        """
        Makes a pipeline based on data_config
        This is because autosklearn does not perform automatic data encoding
        """
        categorical_list = infer_categoricals(self.X)
        preprocessing_steps = []
        if self.data_config.get("text_columns"):
            print("Applying TFIDF to text columns: {data_config.get('text_columns')}")
            preprocessing_steps.append(make_pipeline(
                ColumnSelector(cols=data_config.get("text_columns"), drop_axis=True),
                TfidfVectorizer()
            ))
            categorical_list = [c for c in categorical_list if c not in data_config["text_columns"]]
        if categorical_list:
            print(f"Applying One Hot Encoding to categorical columns: {categorical_list}")
            preprocessing_steps.append(make_pipeline(
                ColumnSelector(cols=categorical_list),
                OneHotEncoder(handle_unknown="impute")
            ))
        if preprocessing_steps:
            preprocessing_steps = make_union(*preprocessing_steps)
            preprocessing_steps = make_pipeline(preprocessing_steps, SimpleImputer())
        else:
            preprocessing_steps = SimpleImputer()
        if self.problem_type == "classification":
            automl = AutoSklearnClassifier(**self.automl_settings)
        else:
            automl = AutoSklearnRegressor(**self.automl_settings)
        automl_pipeline = make_pipeline(
            preprocessing_steps,
            automl
        )
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
        print(self.automl_pipeline.steps[-1][1].sprint_statistics())

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
