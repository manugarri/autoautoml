.ONESHELL:
.SILENT:
SHELL := bash
build:
	make -C containers build

push_ecr:
	make -C containers push_ecr

deploy_ecr: build push_ecr

test_interactive:
	make -C containers test_interactive

test:
	python ./bin/run_manual.py ./containers/$(CONTAINER)/test_$(CONTAINER).json
