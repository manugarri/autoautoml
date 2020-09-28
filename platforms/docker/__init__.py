import json

import docker

CONTAINERS = [
    'autoautoml-core',
    'autokaggle',
    'autokeras',
    'autosklearn',
    'auto-viml',
    'h2o',
    'tpot',
    'ludwig',
    'mlbox',
    'pycaret',
]


def run_docker_container(container_settings):
    """
    runs an experiment inside a container
    Custom docker container with the setting platform_settings
    """
    client = docker.from_env()
    container_name = container_settings["name"]
    if container_name not in CONTAINERS:
        raise Exception('container {} is not a valid container, valid containers are {}'.format(container_name, CONTAINERS))
    experiment_settings = container_settings["settings"]
    experiment_settings["name"] = container_settings["name"]
    platform_settings = container_settings["settings"]["platform_settings"]
    environment = container_settings["environment"]
    environment.update({"EXPERIMENT_SETTINGS": json.dumps(experiment_settings)})
    container = client.containers.run(
        container_name,
        detach=True,
        environment=environment,
        **platform_settings
    )
    for line in container.logs(stream=True):
        print(line.strip().decode())


def run_docker_job(job_settings):
    """
    Runs an automl job, a job can run different containers (different automl libraries)
    """
    for container in job_settings["containers"]:
        experiment_settings = {}
        experiment_settings.update(container)
        experiment_settings.update({
            "data": job_settings["data"],
            "artifacts": job_settings.get("artifacts"),
            "job_dir": job_settings.get("job_dir"),
            "platform_settings": job_settings.get("platform_settings", {})
        })
        container_settings = {
            "environment": job_settings.get("environment", {}),
            "name": container["name"],
            "settings": experiment_settings
        }
        run_docker_container(container_settings)


if __name__ == "__main__":
    sample_experiment_settings = {"data": {"target_column": "Survived", "problem_type": "classification", "feature_columns": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"], "test_path": "/tests/data/titanic/titanic_test.csv", "categorical_columns": ["Sex", "Embarked"], "training_path": "/tests/data/titanic/titanic_train.csv"}, "automl_settings": {"time_left_for_this_task": 60}, "artifacts": {"model_path": "/tests/data/titanic/autosklearn/pipeline.pkl", "metadata_path": "tests/data/titanic/autosklearn/metadata.json", "test_predictions": "/tests/data/titanic/autosklearn/test_predictions.csv"}}
    experiment_settings = {
        "name": "autokaggle-core",
        "settings": json.dumps(sample_experiment_settings),
        "environment": {}
    }

    run_container(experiment_settings)
