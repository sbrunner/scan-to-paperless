
.PHONY: jsonschema
jsonschema:
	jsonschema2md scan_to_paperless/config_schema.json config.md
	jsonschema2md scan_to_paperless/process_schema.json process.md
	jsonschema-gentypes

.PHONY: build-test
build-test:
	docker build --target=tests --tag=tests .

.PHONY: prospector
prospector: build-test
	docker run --rm tests prospector --output=pylint

.PHONY: pytest
pytest: build-test
	docker run --rm --env=PYTHONPATH=/opt/ --volume=$$(pwd)/results:/results --volume=$$(pwd)/tests:/tests tests bash -c 'cd /tests && pytest --durations=0 --verbose --color=yes'
