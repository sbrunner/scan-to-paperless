#!/usr/bin/env python3

"""Get the status of current scan."""

import argparse
import asyncio
import subprocess  # nosec

import argcomplete
from anyio import Path
from ruamel.yaml.main import YAML

import scan_to_paperless.process_schema
from scan_to_paperless import CONFIG_PATH, get_config


def _print_status(folder: Path, message: str, error: bool = False) -> None:
    print(f"{'[ERROR]' if error else ''} {message} - {folder}")


async def _check_error_status(folder: Path, yaml: YAML) -> bool:
    """Check if folder has an error status. Returns True if error found."""
    if not await (folder / "error.yaml").exists():
        return False

    async with await (folder / "error.yaml").open(encoding="utf-8") as error_file:
        error = yaml.load(await error_file.read())
        if error is not None and "error" in error:
            _print_status(folder, "Job in error", error=True)
            print(error["error"])
            if isinstance(error["error"], subprocess.CalledProcessError):
                print(error["error"].output.decode())
                if error["error"].stderr:
                    print(error["error"].stderr)
            if "traceback" in error:
                print("\n".join(error["traceback"]))
            return True
        _print_status(folder, "Job in unknown error", error=True)
        print(error)
        return True


async def _is_already_processed(
    folder: Path, job_config: scan_to_paperless.process_schema.Configuration
) -> bool:
    """Check if job has already been processed."""
    if "steps" not in job_config or not job_config["steps"]:
        return False

    for img in job_config["steps"][-1]["sources"]:
        img_path = folder / Path(img).name
        if not await img_path.exists():
            return False
    return True


async def _check_completion_status(folder: Path, in_progress: bool) -> None:
    """Check completion status."""
    if await (folder / "REMOVE_TO_CONTINUE").exists():
        _print_status(folder, "To be validated")
        return
    if await (folder / "DONE").exists():
        _print_status(folder, "Process finish")
        return
    if in_progress:
        _print_status(folder, "In progress")


async def _main_async() -> None:
    """Get the status of current scan."""
    parser = argparse.ArgumentParser("Process the scanned documents.")
    parser.add_argument("--in-progress", action="store_true", help="Also show the in progress process.")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    config = await get_config(CONFIG_PATH)
    scan_folder = await Path(config["scan_folder"]).expanduser()
    yaml = YAML(typ="safe")
    yaml.default_flow_style = False

    async for folder in scan_folder.iterdir():
        if not await folder.is_dir():
            continue
        if not await (folder / "config.yaml").exists():
            _print_status(folder, "Missing config")
            continue

        async with await (folder / "config.yaml").open(encoding="utf-8") as config_file:
            job_config: scan_to_paperless.process_schema.Configuration = yaml.load(await config_file.read())

        if job_config is None:
            _print_status(folder, "Empty config", error=True)
            continue

        if await _check_error_status(folder, yaml):
            continue

        if await _is_already_processed(folder, job_config):
            await _check_completion_status(folder, args.in_progress)
            continue

        if args.in_progress:
            _print_status(folder, "In progress")
            continue


def main() -> None:
    """Get the status of current scan."""
    asyncio.run(_main_async())
