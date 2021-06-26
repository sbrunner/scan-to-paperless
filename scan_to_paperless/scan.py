#!/usr/bin/env python3


import argparse
import os
import random
import re
import subprocess  # nosec
import sys
from typing import Any, List, Optional, cast

import argcomplete
import numpy as np
from argcomplete.completers import ChoicesCompleter
from ruamel.yaml.main import YAML
from skimage import io

from scan_to_paperless import CONFIG_PATH, get_config

if sys.version_info.minor >= 8:
    from scan_to_paperless import config as stp_config
else:
    from scan_to_paperless import config_old as stp_config  # type: ignore


def call(cmd: List[str], cmd2: Optional[List[str]] = None, **kwargs: Any) -> None:
    del cmd2
    print(" ".join(cmd) if isinstance(cmd, list) else cmd)
    try:
        subprocess.check_call(cmd, **kwargs)  # nosec
    except subprocess.CalledProcessError as exception:
        print(exception)
        sys.exit(1)


def output(cmd: List[str], cmd2: Optional[List[str]] = None, **kwargs: Any) -> bytes:
    del cmd2
    print(" ".join(cmd) if isinstance(cmd, list) else cmd)
    try:
        return cast(bytes, subprocess.check_output(cmd, **kwargs))  # nosec
    except subprocess.CalledProcessError as exception:
        print(exception)
        sys.exit(1)


def main() -> None:
    config: stp_config.Configuration = get_config()

    parser = argparse.ArgumentParser()

    def add_argument(name: str, choices: Optional[List[str]] = None, **kwargs: Any) -> None:
        arg = parser.add_argument(name, **kwargs)
        if choices is not None:
            arg.completer = ChoicesCompleter(choices)  # type: ignore

    add_argument("--no-adf", dest="adf", action="store_false", help="Don't use ADF")
    add_argument(
        "--double-sided",
        action="store_true",
        help="Number of pages in double sided mode",
    )
    add_argument(
        "--append-credit-card",
        action="store_true",
        help="Append vertically the credit card",
    )
    add_argument("--assisted-split", action="store_true", help="Split operation, se help")
    add_argument(
        "--set-config",
        nargs=2,
        action="append",
        default=[],
        help="Set a configuration option",
    )

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    dirty = False
    for conf in args.set_config:
        config[conf[0]] = conf[1]  # type: ignore
        dirty = True
    if dirty:
        yaml = YAML()
        yaml.default_flow_style = False
        with open(CONFIG_PATH, "w", encoding="utf-8") as config_file:
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
        scanimage += [
            f"--batch={root_folder}/image-%d.png",
            "--source=ADF" if args.adf else "--batch-prompt",
        ]

        if args.double_sided:
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
                    image = np.rot90(image, 2)  # type: ignore
                    io.imsave(path, image.astype(np.uint8))
        else:
            call(scanimage)

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
        args_: stp_config.Arguments = {}
        args_.update(config.get("default_args", {}))
        args_cmd = dict(args._get_kwargs())  # pylint: disable=protected-access
        del args_cmd["adf"]
        del args_cmd["double_sided"]
        del args_cmd["set_config"]
        args_.update(cast(stp_config.Arguments, args_cmd))
        process_config = {
            "images": images,
            "destination": destination,
            "args": args_,
        }
        yaml = YAML(typ="safe")
        yaml.default_flow_style = False
        with open(os.path.join(os.path.dirname(root_folder), "config.yaml"), "w") as config_file:
            config_file.write(
                "# yaml-language-server: $schema=https://raw.githubusercontent.com/sbrunner/scan-to-paperless"
                "/master/scan_to_paperless/process_schema.json\n\n"
            )
            yaml.dump(process_config, config_file)

    except subprocess.CalledProcessError as exception:
        print(exception)
        sys.exit(1)

    print(root_folder)
    subprocess.call([config.get("viewer", "eog"), root_folder])  # nosec
