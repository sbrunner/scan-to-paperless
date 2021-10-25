#!/usr/bin/env python3

"""Get the status of current scan."""

import glob
import os
import re
import subprocess  # nosec

from ruamel.yaml.main import YAML

import scan_to_paperless.process_schema
from scan_to_paperless import CONFIG_PATH, get_config


def main() -> None:
    """Get the status of current scan."""
    config = get_config(CONFIG_PATH)
    for folder in glob.glob(os.path.join(os.path.expanduser(config["scan_folder"]), "*")):
        print(re.sub(r".", "-", folder))
        print(folder)

        if not os.path.exists(os.path.join(folder, "config.yaml")):
            print("No config")
        else:
            yaml = YAML(typ="safe")
            yaml.default_flow_style = False
            with open(os.path.join(folder, "config.yaml"), encoding="utf-8") as config_file:
                job_config: scan_to_paperless.process_schema.Configuration = yaml.load(config_file.read())

            if os.path.exists(os.path.join(folder, "error.yaml")):
                with open(os.path.join(folder, "error.yaml"), encoding="utf-8") as error_file:
                    error = yaml.load(error_file.read())
                    if error is not None and "error" in error:
                        print(error["error"])
                        if isinstance(error["error"], subprocess.CalledProcessError):
                            print(error["error"].output.decode())
                            if error["error"].stderr:
                                print(error["error"].stderr)
                        if "traceback" in error:
                            print("\n".join(error["traceback"]))
                    else:
                        print("Unknown error")
                        print(error)
            else:
                allready_proceed = True
                if "transformed_images" not in job_config:
                    allready_proceed = False
                else:
                    for img in job_config["transformed_images"]:
                        img = os.path.join(folder, os.path.basename(img))
                        if not os.path.exists(img):
                            allready_proceed = False
                if allready_proceed:
                    if os.path.exists(os.path.join(folder, "REMOVE_TO_CONTINUE")):
                        print("To be validated")
                    if os.path.exists(os.path.join(folder, "DONE")):
                        print("Process finish")
                    else:
                        print("Waiting to be imported")
                else:
                    print("Not ready")
