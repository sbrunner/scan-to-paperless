export DOCKER_BUILDKIT=1

TAG = $(shell git describe --abbrev=0 --tags)

.PHONY: build
build:
	docker build --build-arg=VERSION=${TAG}+$(shell git rev-list --count ${TAG}) --target=base --tag=sbrunner/scan-to-paperless .

.PHONY: build-all
build-all:
	docker build --build-arg=VERSION=${TAG}+$(shell git rev-list --count ${TAG}) --target=all --tag=sbrunner/scan-to-paperless:latest-all .

.PHONY: build-tests
build-tests:
	docker build --build-arg=VERSION=${TAG}+$(shell git rev-list --count ${TAG}) --target=tests --tag=sbrunner/scan-to-paperless-tests .

.PHONY: prospector
prospector: build-tests
	docker run --rm sbrunner/scan-to-paperless-tests prospector --die-on-tool-error --output=pylint

.PHONY: prospector-fast
prospector-fast:
	docker run --rm  --volume=$$(pwd):/opt/ sbrunner/scan-to-paperless-tests prospector --die-on-tool-error --output=pylint

DOCKER_RUN_TESTS = docker run --rm --env=PYTHONPATH=/opt/ --volume=$$(pwd)/results:/results --volume=$$(pwd)/tests:/tests --volume=$$(pwd)/scan_to_paperless:/opt/scan_to_paperless
DOCKER_RUN_TESTS_IMAGE = $(DOCKER_RUN_TESTS) sbrunner/scan-to-paperless-tests
.PHONY: pytest
pytest: build-tests
	$(DOCKER_RUN_TESTS_IMAGE) bash -c 'cd /tests && pytest --durations=0 --verbose --color=yes'

.PHONY: pytest-last-failed
pytest-last-failed:
	$(DOCKER_RUN_TESTS_IMAGE) bash -c 'cd /tests && pytest --durations=0 --verbose --color=yes --last-failed'

.PHONY: pytest-exitfirst
pytest-exitfirst:
	$(DOCKER_RUN_TESTS_IMAGE) bash -c 'cd /tests && pytest --durations=0 --verbose --color=yes --exitfirst'

.PHONY: pytest-failedfirst-exitfirst
pytest-failedfirst-exitfirst:
	$(DOCKER_RUN_TESTS_IMAGE) bash -c 'cd /tests && pytest --durations=0 --verbose --color=yes --failed-first --exitfirst'

.PHONY: pytest-c2cwsgiutils-debug
pytest-c2cwsgiutils-debug:
	$(DOCKER_RUN_TESTS) --volume=$$(pwd)/../c2cwsgiutils/c2cwsgiutils:/usr/local/lib/python3.10/dist-packages/c2cwsgiutils \
	sbrunner/scan-to-paperless-tests bash -c 'cd /tests && pytest --durations=0 --verbose --color=yes'
