#!/usr/bin/env python3
import base64
import os
import sys

from botocore.exceptions import ClientError
import boto3
import docker

image_name = sys.argv[1]

docker_client = docker.from_env(version='1.24')
ecr_client = boto3.client('ecr', region_name=os.environ["AWS_REGION"])

try:
    response = ecr_client.create_repository(repositoryName=image_name)
except ClientError as e:
    if e.response['Error']['Code'] == 'RepositoryAlreadyExistsException':
        print("Repository already exists")
    else:
        print("Unexpected error: %s" % e)

token = ecr_client.get_authorization_token()
username, password = base64.b64decode(token['authorizationData'][0]['authorizationToken']).decode().split(':')
registry = token['authorizationData'][0]['proxyEndpoint']

docker_client.login(username, password, registry=registry)
print("PUSHING IMAGE {} TO AWS ECR".format(image_name))


img = docker_client.images.get(image_name)
assert img.tag(os.path.join(registry.strip("https://"), image_name))
for line in docker_client.images.push(os.path.join(registry.strip("https://"), image_name),
        stream=True,
        decode=True,
        tag="latest"):
    print(line)
