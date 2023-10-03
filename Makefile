SHELL := /bin/bash

ifndef $DIR
DIR:=result
endif

TIMES = 1 2 3 4 5
CLASSIFIERS = dnn xgb
ATTACKS = hsj pgd zoo cpgd
ALL_CONF  = $(shell find config/$(cat) -type f -iname '*.yaml' ! -name 'default.yaml' )
ATTK_CONF = $(shell find config/$(cat) -type f -iname '*.yaml' ! -name 'default.yaml'  ! -name '*_prf*.yaml')
PERF_CONF = $(shell find config/$(cat) -type f -iname '*_prf*.yaml')
RES_DIRS  = $(shell find $(DIR) -type d -maxdepth 1 )


all: graphs attacks

dev: test lint

attacks:
	$(foreach f, $(ATTK_CONF), $(foreach a, $(ATTACKS), \
	$(foreach c, $(CLASSIFIERS), \
	python3 -m exp $(f) -a $(a) -c $(c) -v --out result/attacks ; )))

original:
	$(foreach f, $(ATTK_CONF), $(foreach a, $(ATTACKS), \
	$(foreach c, $(CLASSIFIERS), \
	python3 -m exp $(f) -a $(a) -c $(c) --out result/original ; )))

plots:
	$(foreach d, $(RES_DIRS), python3 -m exp $(d) --plot ; )

graphs:
	$(foreach f, $(ALL_CONF), python3 -m exp $(f) --graph; )

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