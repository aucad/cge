SHELL := /bin/bash

ITERS = 2 5
ATTACKS = hsj pgd zoo
VALID = V_DURING V_AFTER
V_DURING := -v
V_AFTER :=
CONFIGS = $(shell find config/$(cat) -type f -iname '*.yaml' ! -name 'default.yaml')

all: exp

dev: test lint

.PHONY: exp
exp:
	@$(foreach f, $(CONFIGS), $(foreach a, $(ATTACKS), \
	$(foreach v, $(VALID), $(foreach i, $(ITERS), \
	python3 -m exp $(f) $($(v)) -i $(i) -a $(a) ; ))))

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
