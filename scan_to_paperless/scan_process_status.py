#!/usr/bin/env python3

"""Get the status of current scan."""

import argparse
import subprocess  # nosec
from pathlib import Path

import argcomplete
from ruamel.yaml.main import YAML

import scan_to_paperless.process_schema
from scan_to_paperless import CONFIG_PATH, get_config


def _print_status(folder: Path, message: str, error: bool = False) -> None:
    print(f"{'[ERROR]' if error else ''} {message} - {folder}")


def main() -> None:
    """Get the status of current scan."""
    parser = argparse.ArgumentParser("Process the scanned documents.")
    parser.add_argument("--in-progress", action="store_true", help="Also show the in progress process.")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    config = get_config(CONFIG_PATH)
    for folder in Path(config["scan_folder"]).expanduser().glob("*"):
        if not folder.is_dir():
            continue
        if not (folder / "config.yaml").exists():
            _print_status(folder, "Missing config")
        else:
            yaml = YAML(typ="safe")
            yaml.default_flow_style = False
            with (folder / "config.yaml").open(encoding="utf-8") as config_file:
                job_config: scan_to_paperless.process_schema.Configuration = yaml.load(config_file.read())

            if job_config is None:
                _print_status(folder, "Empty config", error=True)
                continue

            if (folder / "error.yaml").exists():
                with (folder / "error.yaml").open(encoding="utf-8") as error_file:
                    error = yaml.load(error_file.read())
                    if error is not None and "error" in error:
                        _print_status(folder, "Job in error", error=True)
                        print(error["error"])
                        if isinstance(error["error"], subprocess.CalledProcessError):
                            print(error["error"].output.decode())
                            if error["error"].stderr:
                                print(error["error"].stderr)
                        if "traceback" in error:
                            print("\n".join(error["traceback"]))
                        continue
                    _print_status(folder, "Job in unknown error", error=True)
                    print(error)
                    continue
            else:
                already_proceed = True
                if "steps" not in job_config or not job_config["steps"]:
                    already_proceed = False
                else:
                    for img in job_config["steps"][-1]["sources"]:
                        img_path = folder / Path(img).name
                        if not img_path.exists():
                            already_proceed = False
                if already_proceed:
                    if (folder / "REMOVE_TO_CONTINUE").exists():
                        _print_status(folder, "To be validated")
                        continue
                    if (folder / "DONE").exists():
                        _print_status(folder, "Process finish")
                        continue
                    if args.in_progress:
                        _print_status(folder, "In progress")
                    continue
                if args.in_progress:
                    _print_status(folder, "In progress")
                continue
