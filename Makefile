export DOCKER_BUILDKIT=1

.PHONY: jsonschema
jsonschema:
	jsonschema2md scan_to_paperless/config_schema.json config.md
	jsonschema2md scan_to_paperless/process_schema.json process.md
	jsonschema-gentypes

.PHONY: build
build:
	docker build --target=base --tag=sbrunner/scan-to-paperless .

.PHONY: build-all
build-all:
	docker build --target=all --tag=sbrunner/scan-to-paperless:latest-all .

.PHONY: build-tests
build-tests:
	docker build --target=tests --tag=tests .

.PHONY: prospector
prospector: build-test
	docker run --rm tests prospector -X --output=pylint

.PHONY: pytest
pytest: build-test
	docker run --rm --env=PYTHONPATH=/opt/ --volume=$$(pwd)/results:/results --volume=$$(pwd)/tests:/tests --volume=$$(pwd)/scan_to_paperless:/opt/scan_to_paperless tests bash -c 'cd /tests && pytest --durations=0 --verbose --color=yes'

.PHONY: pytest-last-failed
pytest-last-failed:
	docker run --rm --env=PYTHONPATH=/opt/ --volume=$$(pwd)/results:/results --volume=$$(pwd)/tests:/tests --volume=$$(pwd)/scan_to_paperless:/opt/scan_to_paperless tests bash -c 'cd /tests && pytest --durations=0 --verbose --color=yes --last-failed'

.PHONY: pytest-exitfirst
pytest-exitfirst:
	docker run --rm --env=PYTHONPATH=/opt/ --volume=$$(pwd)/results:/results --volume=$$(pwd)/tests:/tests --volume=$$(pwd)/scan_to_paperless:/opt/scan_to_paperless tests bash -c 'cd /tests && pytest --durations=0 --verbose --color=yes --exitfirst'

.PHONY: pytest-failedfirst-exitfirst
pytest-failedfirst-exitfirst:
	docker run --rm --env=PYTHONPATH=/opt/ --volume=$$(pwd)/results:/results --volume=$$(pwd)/tests:/tests --volume=$$(pwd)/scan_to_paperless:/opt/scan_to_paperless tests bash -c 'cd /tests && pytest --durations=0 --verbose --color=yes --failed-first --exitfirst'
