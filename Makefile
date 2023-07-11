.PHONY: test exp

all: test lint

test:
	pytest --cov-report term-missing --cov=./exp test

lint:
	flake8 ./exp --count --show-source --statistics

exp:
	@$(foreach f, $(shell find config/$(cat) -type f -iname '*.yaml'), \
 		python3 -m exp $(f) ; )

clean:
	@rm -fr .pytest_cache/
	@rm -fr .eggs/
	@rm -fr .coverage
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name '*~' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -fr {} +
