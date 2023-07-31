.PHONY: test exp

VALIDATE = VALID_DURING VALID_AFTER
VALID_DURING := --validate
VALID_AFTER :=

all: test lint

test:
	pytest --cov-report term-missing --cov=./exp test

lint:
	flake8 ./exp --count --show-source --statistics

exp:
	@$(foreach f, $(shell find config/$(cat) -type f -iname '*.yaml'), \
	$(foreach v, $(VALIDATE), \
 		python3 -m exp $(f) $($(v)) ; ))

clean:
	@rm -fr .pytest_cache/
	@rm -fr .eggs/
	@rm -fr .coverage
	@rm -fr result
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name '*~' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -fr {} +
