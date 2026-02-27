The project should fully be in async mode.
`pathlib` must not be used.
Adding an `async` to a non-async function is completely possible.
`aiofiles` must not be used, use `anyio` instead.

All disk operation must be done in async mode, libraries like `pikepdf` and not `opencv` which does not support async operations should be used only for in-memory operations.

In the `scan_to_paperless/jupyter.py` file, a jupyter notebook is built, verify that he reflect the changes done in the rest of the codebase.

## Bash

Use the long parameter names for clarity and maintainability.

## Documentation

The user documentation in the `README.md` file should be updated to reflect the changes in the codebase.

## Tests

The new functionalities should be reasonably tested in the `tests/` folder.

Test files in `tests/` may not follow the rules concerning `async` requirements, as there are no performance requirements.
