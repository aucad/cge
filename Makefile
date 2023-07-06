.PHONY: test

all: test lint

test:
	pytest --cov-report term-missing --cov=./exp test

lint:
	flake8 ./exp --count --show-source --statistics

clean:
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name '*~' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -fr {} +