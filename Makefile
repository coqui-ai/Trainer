.DEFAULT_GOAL := help
.PHONY: test system-deps dev-deps deps style lint install help docs

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

target_dirs := tests trainer

test_all:	## run tests and don't stop on an error.
	coverage run -m pytest trainer tests

test:	## run tests.
	coverage run -m pytest -x trainer tests

test_failed:  ## only run tests failed the last time.
	coverage run -m pytest --ff trainer tests

style:	## update code style.
	black ${target_dirs}
	isort ${target_dirs}

lint:	## run pylint linter.
	pylint ${target_dirs}

dev-deps:  ## install development deps
	pip install -r requirements.dev.txt

doc-deps:  ## install docs dependencies
	pip install -r docs/requirements.txt

build-docs: ## build the docs
	cd docs && make clean && make build

deps:	## install 🐸 requirements.
	pip install -r requirements.txt

install:	## install 🐸 Trainer for development.
	pip install -e .[all]

docs:	## build the docs
	$(MAKE) -C docs clean && $(MAKE) -C docs html
