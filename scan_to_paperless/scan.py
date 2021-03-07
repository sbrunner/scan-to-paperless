#!/usr/bin/env python3


import argparse
import datetime
import os
import random
import re
import subprocess
import sys

import argcomplete
import yaml
from argcomplete.completers import ChoicesCompleter

from scan_to_paperless import CONFIG_FOLDER, CONFIG_PATH, get_config


def call(cmd, cmd2=None, **kwargs):
    del cmd2
    print(" ".join(cmd) if isinstance(cmd, list) else cmd)
    try:
        subprocess.check_call(cmd, **kwargs)
    except subprocess.CalledProcessError as exception:
        print(exception)
        sys.exit(1)


def output(cmd, cmd2=None, **kwargs):
    del cmd2
    print(" ".join(cmd) if isinstance(cmd, list) else cmd)
    try:
        return subprocess.check_output(cmd, **kwargs)
    except subprocess.CalledProcessError as exception:
        print(exception)
        sys.exit(1)


def main():
    config = get_config()

    parser = argparse.ArgumentParser()

    def add_argument(name, choices=None, **kwargs):
        arg = parser.add_argument(name, **kwargs)
        if choices is not None:
            arg.completer = ChoicesCompleter(choices)

    add_argument("--no-adf", dest="adf", action="store_false", help="Don't use ADF")
    add_argument(
        "--no-level",
        dest="level",
        action="store_false",
        help="Don't use level correction",
    )
    add_argument("title", nargs="*", choices=["No title"], help="The document title")
    add_argument(
        "--date",
        choices=[datetime.date.today().strftime("%Y%m%d")],
        help="The document date",
    )
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

    import numpy as np
    from skimage import io

    dirty = False
    for conf in args.set_config:
        config[conf[0]] = conf[1]
        dirty = True
    if dirty:
        with open(CONFIG_PATH, "w", encoding="utf-8") as config_file:
            config_file.write(yaml.safe_dump(config, default_flow_style=False))

    if "scan_folder" not in config:
        print(
            """The scan folder isn't set, use:
    scan --set-settings scan_folder <a_folder>
    This should be shared with the process container in 'source'."""
        )
        sys.exit(1)

    title = None
    full_name = None
    rand_int = str(random.randint(0, 999999))
    base_folder = os.path.join(os.path.expanduser(config["scan_folder"]), rand_int)
    while os.path.exists(base_folder):
        rand_int = str(random.randint(0, 999999))
        base_folder = os.path.join(os.path.expanduser(config["scan_folder"]), rand_int)

    if args.title:
        title = " ".join(args.title)
        full_name = title
        if args.date is not None:
            full_name = f"{args.date}Z - {full_name}"
        if "/" in full_name:
            print("The name can't contains some '/' in the title.")
            sys.exit(1)
        destination = f"/destination/{full_name}.pdf"
    elif args.date is not None:
        destination = f"/destination/{args.date}Z - {rand_int}.pdf"
    else:
        destination = f"/destination/{rand_int}.pdf"

    root_folder = os.path.join(base_folder, "source")
    os.makedirs(root_folder)

    try:
        scanimage = (
            ["scanimage"]
            + config.get(
                "scanimage_arguments",
                ["--format=png", "--mode=color", "--resolution=300"],
            )
            + [
                f"--batch={root_folder}/image-%d.png",
                "--source=ADF" if args.adf else "--batch-prompt",
            ]
        )

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
                    image = np.rot90(image, 2)
                    io.imsave(path, image.astype(np.uint8))
        else:
            call(scanimage)

        images = []
        for img in os.listdir(root_folder):
            if not img.startswith("image-"):
                continue
            images.append(os.path.join("source", img))

        regex = re.compile(r"^source\/image\-([0-9]+)\.png$")
        images = sorted(images, key=lambda e: int(regex.match(e).group(1)))
        args_ = {}
        args_.update(config.get("default_args", {}))
        args_.update(dict(args._get_kwargs()))
        config = {
            "images": images,
            "title": title,
            "full_name": full_name,
            "destination": destination,
            "args": args_,
        }
        with open(os.path.join(os.path.dirname(root_folder), "config.yaml"), "w") as config_file:
            config_file.write(yaml.safe_dump(config, default_flow_style=False))

    except subprocess.CalledProcessError as exception:
        print(exception)
        sys.exit(1)

    print(root_folder)
    subprocess.call([config.get("viewer", "eog"), root_folder])
