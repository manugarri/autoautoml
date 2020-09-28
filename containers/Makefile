.ONESHELL:
.SILENT:
SHELL := bash

build:
	bash ./bin/build_image.sh

push_ecr:
	python ./bin/push_image_ecr.py $(CONTAINER)

deploy_ecr: build push_ecr

test_interactive:
	bash ./bin/test_docker_local_interactive.sh
