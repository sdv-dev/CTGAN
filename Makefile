.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

.PHONY: help
help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)


# CLEAN TARGETS

.PHONY: clean-build
clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

.PHONY: clean-pyc
clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

.PHONY: clean-coverage
clean-coverage: ## remove coverage artifacts
	rm -f .coverage
	rm -f .coverage.*
	rm -fr htmlcov/

.PHONY: clean-test
clean-test: ## remove test artifacts
	rm -fr .tox/
	rm -fr .pytest_cache

.PHONY: clean
clean: clean-build clean-pyc clean-test clean-coverage ## remove all build, test, coverage and Python artifacts


# INSTALL TARGETS

.PHONY: install
install: clean-build clean-pyc ## install the package to the active Python's site-packages
	pip install .

.PHONY: install-test
install-test: clean-build clean-pyc ## install the package and test dependencies
	pip install .[test]

.PHONY: install-develop
install-develop: clean-build clean-pyc ## install the package in editable mode and dependencies for development
	pip install -e .[dev]

.PHONY: install-readme
install-readme: clean-build clean-pyc ## install the package in editable mode and readme dependencies for developement
	pip install -e .[readme]

# LINT TARGETS

.PHONY: lint
lint:
	invoke lint

.PHONY: fix-lint
fix-lint:
	invoke fix-lint


# TEST TARGETS

.PHONY: test-unit
test-unit: ## run unit tests using pytest
	invoke unit

.PHONY: test-integration
test-integration: ## run integration tests using pytest
	invoke integration

.PHONY: test-readme
test-readme: ## run the readme snippets
	invoke readme

.PHONY: check-dependencies
check-dependencies: ## test if there are any broken dependencies
	pip check

.PHONY: test
test: test-unit test-integration test-readme ## test everything that needs test dependencies

.PHONY: test-devel
test-devel: lint ## test everything that needs development dependencies


.PHONY: coverage
coverage: ## check code coverage quickly with the default Python
	coverage run --source ctgan -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html


# RELEASE TARGETS

.PHONY: dist
dist: clean ## builds source and wheel package
	python -m build --wheel --sdist
	ls -l dist

.PHONY: publish-confirm
publish-confirm:
	@echo "WARNING: This will irreversibly upload a new version to PyPI!"
	@echo -n "Please type 'confirm' to proceed: " \
		&& read answer \
		&& [ "$${answer}" = "confirm" ]

.PHONY: publish-test
publish-test: dist publish-confirm ## package and upload a release on TestPyPI
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

.PHONY: publish
publish: dist publish-confirm ## package and upload a release
	twine upload dist/*

.PHONY: git-merge-main-stable
git-merge-main-stable: ## Merge main into stable
	git checkout stable || git checkout -b stable
	git merge --no-ff main -m"make release-tag: Merge branch 'main' into stable"

.PHONY: git-merge-stable-main
git-merge-stable-main: ## Merge stable into main
	git checkout main
	git merge stable

.PHONY: git-push
git-push: ## Simply push the repository to github
	git push

.PHONY: git-push-tags-stable
git-push-tags-stable: ## Push tags and stable to github
	git push --tags origin stable

.PHONY: bumpversion-release
bumpversion-release: ## Bump the version to the next release
	bump-my-version bump release --no-tag

.PHONY: bumpversion-patch
bumpversion-patch: ## Bump the version to the next patch
	bump-my-version bump --no-tag patch

.PHONY: bumpversion-candidate
bumpversion-candidate: ## Bump the version to the next candidate
	bump-my-version bump candidate --no-tag

.PHONY: bumpversion-minor
bumpversion-minor: ## Bump the version the next minor skipping the release
	bump-my-version bump --no-tag minor

.PHONY: bumpversion-major
bumpversion-major: ## Bump the version the next major skipping the release
	bump-my-version bump --no-tag major

.PHONY: bumpversion-revert
bumpversion-revert: ## Undo a previous bumpversion-release
	git tag --delete $(shell git tag --points-at HEAD)
	git checkout main
	git branch -D stable

CLEAN_DIR := $(shell git status --short | grep -v ??)
CURRENT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD 2>/dev/null)
CURRENT_VERSION := $(shell grep "^current_version" pyproject.toml | grep -o "dev[0-9]*")
CHANGELOG_LINES := $(shell git diff HEAD..origin/stable HISTORY.md 2>&1 | wc -l)

.PHONY: check-clean
check-clean: ## Check if the directory has uncommitted changes
ifneq ($(CLEAN_DIR),)
	$(error There are uncommitted changes)
endif

.PHONY: check-main
check-main: ## Check if we are in main branch
ifneq ($(CURRENT_BRANCH),main)
	$(error Please make the release from main branch\n)
endif

.PHONY: check-candidate
check-candidate: ## Check if a release candidate has been made
ifeq ($(CURRENT_VERSION),dev0)
	$(error Please make a release candidate and test it before atempting a release)
endif

.PHONY: check-history
check-history: ## Check if HISTORY.md has been modified
ifeq ($(CHANGELOG_LINES),0)
	$(error Please insert the release notes in HISTORY.md before releasing)
endif

.PHONY: check-deps
check-deps: # Dependency targets
	$(eval allow_list='numpy=|pandas=|tqdm=|torch=|rdt=')
	pip freeze | grep -v "CTGAN.git" | grep -E $(allow_list) > $(OUTPUT_FILEPATH)

.PHONY: check-release
check-release: check-clean check-candidate check-main check-history ## Check if the release can be made
	@echo "A new release can be made"

.PHONY: release
release: check-release git-merge-main-stable bumpversion-release git-push-tags-stable \
	git-merge-stable-main bumpversion-patch git-push

.PHONY: release-test
release-test: check-release git-merge-main-stable bumpversion-release bumpversion-revert

.PHONY: release-candidate
release-candidate: check-main publish bumpversion-candidate git-push

.PHONY: release-candidate-test
release-candidate-test: check-clean check-main publish-test