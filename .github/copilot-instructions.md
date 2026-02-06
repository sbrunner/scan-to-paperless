The project should fully be in async mode.
`pathlib` must not be used.
Adding an `async` to a non-async function is completely possible.
`aiofiles` must not be used, use `anyio` instead.

In the `scan_to_paperless/jupyter.py` file, a jupyter notebook is built, verify that he reflect the changes done in the rest of the codebase.

The user Documentation in the README.md file should be updated to reflect the changes in the codebase.

The new functionalities should be reasonably tested in the `tests/` folder.
