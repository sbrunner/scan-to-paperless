#!/usr/bin/env python3

"""Get the status of current scan."""

import argparse
import glob
import os
import subprocess  # nosec

import argcomplete
from ruamel.yaml.main import YAML

import scan_to_paperless.process_schema
from scan_to_paperless import CONFIG_PATH, get_config


def _print_status(folder: str, message: str, error: bool = False) -> None:
    print(f"{'[ERROR]' if error else ''} {message} - {folder}")


def main() -> None:
    """Get the status of current scan."""
    parser = argparse.ArgumentParser("Process the scanned documents.")
    parser.add_argument("--in-progress", action="store_true", help="Also show the in progress process.")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    config = get_config(CONFIG_PATH)
    for folder in glob.glob(os.path.join(os.path.expanduser(config["scan_folder"]), "*")):
        if not os.path.isdir(folder):
            continue
        if not os.path.exists(os.path.join(folder, "config.yaml")):
            _print_status(folder, "Missing config")
        else:
            yaml = YAML(typ="safe")
            yaml.default_flow_style = False
            with open(os.path.join(folder, "config.yaml"), encoding="utf-8") as config_file:
                job_config: scan_to_paperless.process_schema.Configuration = yaml.load(config_file.read())

            if job_config is None:
                _print_status(folder, "Empty config", True)
                continue

            if os.path.exists(os.path.join(folder, "error.yaml")):
                with open(os.path.join(folder, "error.yaml"), encoding="utf-8") as error_file:
                    error = yaml.load(error_file.read())
                    if error is not None and "error" in error:
                        _print_status(folder, "Job in error", True)
                        print(error["error"])
                        if isinstance(error["error"], subprocess.CalledProcessError):
                            print(error["error"].output.decode())
                            if error["error"].stderr:
                                print(error["error"].stderr)
                        if "traceback" in error:
                            print("\n".join(error["traceback"]))
                        continue
                    else:
                        _print_status(folder, "Job in unknown error", True)
                        print(error)
                        continue
            else:
                already_proceed = True
                if "steps" not in job_config or not job_config["steps"]:
                    already_proceed = False
                else:
                    for img in job_config["steps"][-1]["sources"]:
                        img = os.path.join(folder, os.path.basename(img))
                        if not os.path.exists(img):
                            already_proceed = False
                if already_proceed:
                    if os.path.exists(os.path.join(folder, "REMOVE_TO_CONTINUE")):
                        _print_status(folder, "To be validated")
                        continue
                    if os.path.exists(os.path.join(folder, "DONE")):
                        _print_status(folder, "Process finish")
                        continue
                    else:
                        if args.in_progress:
                            _print_status(folder, "In progress")
                        continue
                else:
                    if args.in_progress:
                        _print_status(folder, "In progress")
                    continue
