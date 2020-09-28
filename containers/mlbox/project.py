"""
AutoML using MLBox
https://mlbox.readthedocs.io
"""

from pprint import pprint

from mlbox.preprocessing import Drift_thresholder, Reader
from mlbox.optimisation import Optimiser
from mlbox.prediction import Predictor
from sklearn.model_selection import train_test_split

from autoautoml.core import AutoMLCore
from autoautoml.utils import (get_experiment_settings, read_dataset, evaluate, save_artifact,
                   clean_dataset)


class AutoML(AutoMLCore):

    name = "mlbox"

    def fit(self):
        """
        Performs the search
        """
        super().fit()
        self.train, self.test = train_test_split(self.dataset, test_size=0.1
        )
        self.data_dict = {
            "train": self.train,
            "test": self.test,
            "target": self.train[self.target]
        }
        if self.problem_type == "classification":
            optimization_scorer = "accuracy"
        else:
            self.optimization_scorer = "mean_squared_error"
        opt = Optimiser(scoring=optimization_scorer, n_folds=3)
        opt.evaluate(None, self.data_dict)
        space = {'ne__numerical_strategy': {"search": "choice",
                                            "space": [0,
                                                      "mean",
                                                      "median",
                                                      "most_frequent"]
                                            },
                 'ce__strategy': {"search": "choice",
                                  "space": ["label_encoding",
                                            "random_projection",
                                            "entity_embedding",
                                            "dummification"]},
                 'fs__threshold': {"search": "uniform",
                                   "space": [0.01, 0.3]},
                 'est__max_depth': {"search": "choice",
                                    "space": [3, 4, 5, 6, 7, 10, 15, 20]},
                 'est__n_estimators': {"search": "choice",
                                    "space": [50, 100, 200, 300, 400, 600, 800, 1000]},
                 }
        best = opt.optimise(space, self.data_dict, 15)
        self.automl = Predictor().fit_predict(best, self.data_dict)

    def evaluate(self):
        """
        Evaluates and stores performance
        """
        print("EVALUATING ESTIMATOR")
        train_preds = self.automl.predict(self.data_dict["train"])
        test_preds = self.automl.predict(self.data_dict["test"])
        train_score = evaluate(y_train, train_preds, problem_type)
        test_score = evaluate(y_test, test_preds, problem_type)
        self.metadata = {
            "metrics": {
                "test_score": test_score,
                "train_score": train_score
            },
            "experiment_settings": experiment_settings
        }
        pprint(self.metadata)


