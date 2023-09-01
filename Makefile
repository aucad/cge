SHELL := /bin/bash

ITERS = 2 5
ATTACKS = hsj pgd zoo cpgd
CONFIGS = $(shell find config/$(cat) -type f -iname '*.yaml' ! -name 'default.yaml'  ! -name 'test.yaml')

all: exp

dev: test lint

exp:
	$(foreach f, $(CONFIGS), $(foreach a, $(ATTACKS), $(foreach i, $(ITERS), \
	python3 -m exp $(f) $($(v)) -i $(i) -a $(a) -v ; )))

test:
	pytest --cov-report term-missing --cov=./exp test

lint:
	flake8 ./exp --count --show-source --statistics

clean:
	@rm -fr .pytest_cache/
	@rm -fr .eggs/
	@rm -fr .coverage
	@rm -fr result
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name '*~' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -fr {} +


.PHONY: exp test