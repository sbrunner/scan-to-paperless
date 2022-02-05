"""Scan a new document."""

import argparse
import glob
import os
import random
import re
import subprocess  # nosec
import sys
from typing import Any, List, Optional, cast

import argcomplete
import numpy as np
from ruamel.yaml.main import YAML
from skimage import io

from scan_to_paperless import CONFIG_PATH, get_config

if sys.version_info.minor >= 8:
    from scan_to_paperless import config as stp_config
else:
    from scan_to_paperless import config_old as stp_config  # type: ignore


def call(cmd: List[str], cmd2: Optional[List[str]] = None, **kwargs: Any) -> None:
    """Verbose implementation of check_call."""
    del cmd2
    print(" ".join(cmd) if isinstance(cmd, list) else cmd)
    try:
        subprocess.check_call(cmd, **kwargs)  # nosec
    except subprocess.CalledProcessError as exception:
        print(exception)
        sys.exit(1)


def output(cmd: List[str], cmd2: Optional[List[str]] = None, **kwargs: Any) -> bytes:
    """Verbose implementation of check_output."""
    del cmd2
    print(" ".join(cmd) if isinstance(cmd, list) else cmd)
    try:
        return cast(bytes, subprocess.check_output(cmd, **kwargs))  # nosec
    except subprocess.CalledProcessError as exception:
        print(exception)
        sys.exit(1)


def main() -> None:
    """Scan a new document."""
    parser = argparse.ArgumentParser()

    presets = [e[len(CONFIG_PATH) - 4 : -5] for e in glob.glob(f"{CONFIG_PATH[:-5]}-*.yaml")]  # noqa

    parser.add_argument(
        "--mode",
        choices=("adf", "one", "multi", "double"),
        default="adf",
        help="The scan mode: 'adf': use Auto Document Feeder (Default), "
        "one: Scan one page, multi: scan multiple pages, double: scan double sided document using the ADF",
    )
    parser.add_argument(
        "--preset",
        choices=presets,
        help="Use an alternate configuration",
    )
    parser.add_argument(
        "--append-credit-card",
        action="store_true",
        help="Append vertically the credit card",
    )
    parser.add_argument("--assisted-split", action="store_true", help="Split operation, see help")
    parser.add_argument(
        "--config",
        action="store_true",
        help="Print the configuration and exit",
    )
    parser.add_argument(
        "--set-config",
        nargs=2,
        action="append",
        default=[],
        help="Set a configuration option",
    )

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    config_filename = CONFIG_PATH if args.preset is None else f"{CONFIG_PATH[:-5]}-{args.preset}.yaml"
    config: stp_config.Configuration = get_config(config_filename)

    if args.config:
        yaml = YAML()
        yaml.default_flow_style = False
        print("Config from file: " + config_filename)
        yaml.dump(config, sys.stdout)
        sys.exit()

    dirty = False
    for conf in args.set_config:
        config[conf[0]] = conf[1]  # type: ignore
        dirty = True
    if dirty:
        yaml = YAML()
        yaml.default_flow_style = False
        with open(config_filename, "w", encoding="utf-8") as config_file:
            config_file.write(
                "# yaml-language-server: $schema=https://raw.githubusercontent.com/sbrunner/scan-to-paperless"
                "/master/scan_to_paperless/config_schema.json\n\n"
            )
            yaml.dump(config, config_file)

    if "scan_folder" not in config:
        print(
            """The scan folder isn't set, use:
    scan --set-settings scan_folder <a_folder>
    This should be shared with the process container in 'source'."""
        )
        sys.exit(1)

    rand_int = str(random.randint(0, 999999))  # nosec
    base_folder = os.path.join(os.path.expanduser(config["scan_folder"]), rand_int)
    while os.path.exists(base_folder):
        rand_int = str(random.randint(0, 999999))  # nosec
        base_folder = os.path.join(os.path.expanduser(config["scan_folder"]), rand_int)

    destination = f"/destination/{rand_int}.pdf"

    root_folder = os.path.join(base_folder, "source")
    os.makedirs(root_folder)

    try:
        scanimage: List[str] = [config.get("scanimage", "scanimage")]
        scanimage += config.get("scanimage_arguments", ["--format=png", "--mode=color", "--resolution=300"])
        scanimage += [f"--batch={root_folder}/image-%d.png"]
        if args.mode in ("adf", "double"):
            scanimage += ["--source=ADF"]
        if args.mode == "multi":
            scanimage += ["--batch-prompt"]
        if args.mode == "one":
            scanimage += ["--batch-count=1"]

        if args.mode == "double":
            call(scanimage + ["--batch-start=1", "--batch-increment=2"])
            odd = os.listdir(root_folder)
            input("Put your document in the automatic document feeder for the other side, and press enter.")
            call(
                scanimage
                + [
                    f"--batch-start={len(odd) * 2}",
                    "--batch-increment=-2",
                    f"--batch-count={len(odd)}",
                ]
            )
            for img in os.listdir(root_folder):
                if img not in odd:
                    path = os.path.join(root_folder, img)
                    image = io.imread(path)
                    image = np.rot90(image, 2)
                    io.imsave(path, image.astype(np.uint8))
        else:
            call(scanimage)

        args_: stp_config.Arguments = {}
        args_.update(config.get("default_args", {}))
        args_cmd = dict(args._get_kwargs())  # pylint: disable=protected-access
        del args_cmd["mode"]
        del args_cmd["preset"]
        del args_cmd["config"]
        del args_cmd["set_config"]
        args_.update(cast(stp_config.Arguments, args_cmd))

    except subprocess.CalledProcessError as exception:
        print(exception)
        sys.exit(1)

    print(root_folder)
    subprocess.call([config.get("viewer", "eog"), root_folder])  # nosec

    images = []
    for img in os.listdir(root_folder):
        if not img.startswith("image-"):
            continue
        images.append(os.path.join("source", img))

    regex = re.compile(r"^source\/image\-([0-9]+)\.png$")

    def image_match(image_name: str) -> int:
        match = regex.match(image_name)
        assert match
        return int(match.group(1))

    images = sorted(images, key=image_match)
    if images:
        process_config = {
            "images": images,
            "destination": destination,
            "args": args_,
        }
        yaml = YAML(typ="safe")
        yaml.default_flow_style = False
        with open(
            os.path.join(os.path.dirname(root_folder), "config.yaml"), "w", encoding="utf-8"
        ) as process_file:
            process_file.write(
                "# yaml-language-server: $schema=https://raw.githubusercontent.com/sbrunner/scan-to-paperless"
                "/master/scan_to_paperless/process_schema.json\n\n"
            )
            yaml.dump(process_config, process_file)
    else:
        os.rmdir(root_folder)
        os.rmdir(base_folder)
