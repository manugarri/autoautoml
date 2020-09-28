"""
Creates an AWS Batch Compute environment, a job queue and a job definition
"""
import os
import time

import boto3

AUTOML_ENVNAME = 'automl'
AWS_REGION = os.environ['AWS_REGION']
CONTAINERS = [
    'autokaggle-core',
    'autosklearn',
    'h2o',
    'tpot',
    'ludwig',
    'mlbox',
]
AWS_ACCOUNT_ID = boto3.client('sts').get_caller_identity().get('Account')
batch = boto3.client('batch', region_name=AWS_REGION)


def create_compute_environment():
    environments = batch.describe_compute_environments()["computeEnvironments"]
    environment_names = [e["computeEnvironmentName"] for e in environments]
    if AUTOML_ENVNAME not in environment_names:
        print("CREATING AWS BATCH COMPUTE ENVIRONMENT")
        ec2 = boto3.client('ec2', region_name=os.environ["AWS_REGION"])
        default_vpc = ec2.describe_vpcs(Filters=[{"Name": "isDefault", 'Values': ['true']}])["Vpcs"][0]["VpcId"]
        default_subnets = ec2.describe_subnets(Filters=[{"Name": "vpcId", 'Values': [default_vpc]}])["Subnets"]
        subnets = [s["SubnetId"] for s in default_subnets]
        sec_groups = [g["GroupId"] for g in ec2.describe_security_groups()["SecurityGroups"] if g["VpcId"] == default_vpc]
        # go here to get a spotfleet role:
        # https://console.aws.amazon.com/iam/home?region=us-east-1#roles/aws-ec2-spot-fleet-tagging-role
        response = batch.create_compute_environment(
            computeEnvironmentName=AUTOML_ENVNAME,
            type='MANAGED',
            state='ENABLED',
            computeResources={
                'type': 'SPOT',
                'minvCpus': 0,
                'desiredvCpus': 0,
                'maxvCpus': 64,
                'instanceTypes': [
                    'optimal',
                    'm3.medium',
                    'p2.xlarge'
                ],
                'instanceRole': 'automl_instance_role',
                'bidPercentage': 60,
                'spotIamFleetRole': f'arn:aws:iam::{AWS_ACCOUNT_ID}:role/aws-ec2-spot-fleet-tagging-role',
                'subnets': subnets,
                'securityGroupIds': sec_groups,
                'tags': {
                    'Name': 'automl'
                },
            },
            serviceRole=f'arn:aws:iam::{AWS_ACCOUNT_ID}:role/service-role/AWSBatchServiceRole'
        )
        print(response)
    else:
        print("Compute environment {} already exists".format(AUTOML_ENVNAME))


def create_job_queue():
    queues = batch.describe_job_queues()["jobQueues"]
    queue_names = [q["jobQueueName"] for q in queues]
    if AUTOML_ENVNAME not in queue_names:
        print("CREATING JOB QUEUE")
        time.sleep(30)
        response = batch.create_job_queue(
            jobQueueName=AUTOML_ENVNAME,
            state='ENABLED',
            priority=100,
            computeEnvironmentOrder=[
                {
                    'order': 1,
                    'computeEnvironment': 'automl'
                },
            ]
        )
        print(response)
    else:
        print("Job Queue {} already exists".format(AUTOML_ENVNAME))


def setup_job_definitions():
    for container in CONTAINERS:
        print("Creating Job Definition for container {}".format(container))
        container_image = "{}.dkr.ecr.{}.amazonaws.com/{}".format(AWS_ACCOUNT_ID, AWS_REGION, container)
        response = batch.register_job_definition(
            jobDefinitionName=container,
            type='container',
            containerProperties={
                 'image': container_image,
                 'vcpus': 1,
                 'memory': 1000,
                 'privileged': True,
            }
        )
        print(response)


if __name__ == "__main__":
    create_compute_environment()
    create_job_queue()
    setup_job_definitions()
