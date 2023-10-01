SHELL := /bin/bash

TIMES = 1 2 3 4 5
CLASSIFIERS = dnn xgb
ATTACKS = hsj pgd zoo cpgd
ATTK_CONF = $(shell find config/$(cat) -type f -iname '*.yaml' ! -name 'default.yaml'  ! -name '*_prf*.yaml')
PERF_CONF = $(shell find config/$(cat) -type f -iname '*_prf*.yaml')
ALL__CONF = $(shell find config/$(cat) -type f -iname '*.yaml' ! -name 'default.yaml' )

all: graphs attacks

dev: test lint

attacks:
	$(foreach f, $(ATTK_CONF), $(foreach a, $(ATTACKS), \
	$(foreach c, $(CLASSIFIERS), \
	python3 -m exp $(f) -a $(a) -c $(c) -v --out result/attacks ; )))

graphs:
	$(foreach f, $(ALL__CONF), python3 -m exp $(f) --graph; )

time:
	$(foreach f, $(PERF_CONF), $(foreach t, $(TIMES), \
	python3 -m exp $(f) --fn $(t) ; ))

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


.PHONY: test