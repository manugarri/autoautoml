"""
AutoML using Uber's Ludwig
https://github.com/uber/ludwig
"""
import os
from pprint import pprint
import zipfile

from ludwig.api import LudwigModel
from sklearn.model_selection import train_test_split

from autoautoml.core import AutoMLCore
from autoautoml.utils import (get_experiment_settings, read_dataset, evaluate, 
                              write, save_artifact, infer_categoricals, clean_dataset)

import warnings
warnings.simplefilter("ignore")


class AutoML(AutoMLCore):

    name = "ludwig"

    def load(self):
        """
        Loads the original data
        """
        super().load()
        self.feat_type = ['categorical' if x.name == 'object' else 'numerical' for x in self.X.dtypes]
        if "model_definition" not in self.automl_settings:
            categorical_list = infer_categoricals(self.dataset.drop(columns=self.target))
            input_features = [
                {"name": col, "type": "category"}
                for col in categorical_list
            ]
            input_features.extend([
                {"name": col, "type": "numerical"}
                for col in self.feature_columns
                if col not in categorical_list
            ])
            if self.problem == "classification":
                output_features = [
                    {
                        "name": self.target,
                        "type": "category"
                    }
                ]
            else:
                output_features = [
                    {
                        "name": self.target,
                        "type": "numerical"
                    }
                ]
            self.automl_settings["model_definition"] = {"input_features": input_features}

    def build_pipeline(self):
        return LudwigModel(**self.automl_settings)

    def fit(self):
        """
        Performs the search
        """
        super().fit()
        self.train_df, self.test_df = train_test_split(self.dataset, test_size=0.1)
        self.automl_pipeline = self.build_pipeline()
        self.automl_pipeline.train(
          data_train_df=self.train_df,
        )

    def evaluate(self):
        """
        Evaluates and stores performance
        """
        print("EVALUATING ESTIMATOR")
        if self.problem_type == "classification":
            metric = "accuracy"
        else:
            metric = "mean_squared_error"
        train_score = self.automl_pipeline.test(data_df=self.train_df)[1][self.target][metric]
        test_score = self.automl_pipeline.test(data_df=self.test_df)[1][self.target][metric]
        self.metadata = {
            "metrics": {
                "test_score": test_score,
                "train_score": train_score
            },
            "experiment_settings": self.experiment_settings
        }
        pprint(self.metadata)

    def save(self):
        """
        Refits and saves final estimator
        """
        if "metadata" in self.artifact_config:
            print("SAVING METADATA")
            save_artifact(self.metadata, self.artifact_config["metadata_path"])
        if "model_path" in self.artifact_config:
            print("SAVING MODEL")
            tmp_model_path = "/tmp/model"
            # ludwig saves in a folder, so we zip it as a single file for saving
            self.automl_pipeline.save(tmp_model_path)
            zipfolder('ludwig_model.zip', tmp_model_path)
            with open('ludwig_model.zip', "rb") as fname:
                serialized_model = fname.read()
                write(serialized_model, self.artifact_config["model_path"])

def zipfolder(output_fname, target_dir):            
    zipobj = zipfile.ZipFile(output_fname, 'w', zipfile.ZIP_DEFLATED)
    rootlen = len(target_dir) + 1
    for base, dirs, files in os.walk(target_dir):
        for file in files:
            fn = os.path.join(base, file)
            zipobj.write(fn, fn[rootlen:])



