import json
import os

import boto3

CONTAINERS = [
    'autokaggle-core',
    'autosklearn',
    'h2o',
    'tpot',
    'ludwig',
    'mlbox',
]
MAX_DURATION = 3600 * 6


def run_awsbatch_container(container_settings):
    """
    runs an aws batch job
    Custom containerOverrides parameters can be added with platform_settings
    """
    container_name = container_settings["name"]
    if container_name not in CONTAINERS:
        raise Exception('container {} is not a valid container, valid containers are {}'.format(container_name, CONTAINERS))
    container_settings['settings']['name'] = container_name
    platform_settings = container_settings["settings"].get("platform_settings", {})
    client = boto3.client('batch', region_name=os.environ['AWS_REGION'])
    container_extras.update({
        'environment': [
            {
                'name': 'EXPERIMENT_SETTINGS',
                'value': json.dumps(container_settings['settings'])
            },
        ],
    })
    response = client.submit_job(
        jobName='automl_job_{}'.format(container_name),
        jobQueue='automl',
        jobDefinition=container_name,
        containerOverrides=platform_settings,
        timeout={
            'attemptDurationSeconds': MAX_DURATION
        }
    )
    print(response)


def run_awsbatch_job(job_settings):
    """
    Runs an automl job using aws batch, a job can run different containers (different automl libraries)
    """
    for container in job_settings["containers"]:
        experiment_settings = {}
        experiment_settings.update(container)
        experiment_settings.update({
            "data": job_settings["data"],
            "artifacts": job_settings.get("artifacts"),
            "job_dir": job_settings.get("job_dir")
        })
        container_settings = {
            "environment": job_settings.get("environment", {}),
            "name": container["name"],
            "settings": experiment_settings
        }
        run_awsbatch_container(container_settings)


"""
{
    "jobDefinitionName": "generate_matches",
    "type": "container",
    "parameters": {},
    "containerProperties": {
        "image": "535821280149.dkr.ecr.eu-west-1.amazonaws.com/product-matching",
        "vcpus": 4,
        "memory": 1000,
        "command": [
            "python",
            "/tmp/generate_matches.py"
        ]
        'resourceRequirements': [
            {
                'value': 'string',
                'type': 'GPU'
            },
        ]
    }
}
"""
