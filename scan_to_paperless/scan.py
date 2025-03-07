#!/usr/bin/env python

"""Scan a new document."""

import argparse
import datetime
import math
import os
import re
import subprocess  # nosec
import sys
import time
from pathlib import Path
from typing import Any, cast

import argcomplete
import PIL.Image
import pyperclip
from ruamel.yaml.main import YAML

from scan_to_paperless import CONFIG_FILENAME, CONFIG_FOLDER, CONFIG_PATH, get_config
from scan_to_paperless import config as schema

from .config import VIEWER_DEFAULT


def call(cmd: list[str], cmd2: list[str] | None = None, **kwargs: Any) -> None:
    """Verbose implementation of check_call."""
    del cmd2
    print(" ".join(cmd) if isinstance(cmd, list) else cmd)
    try:
        subprocess.check_call(cmd, **kwargs)  # noqa: S603
    except subprocess.CalledProcessError as exception:
        print(exception)
        sys.exit(1)


def output(cmd: list[str], cmd2: list[str] | None = None, **kwargs: Any) -> bytes:
    """Verbose implementation of check_output."""
    del cmd2
    print(" ".join(cmd) if isinstance(cmd, list) else cmd)
    try:
        return cast(bytes, subprocess.check_output(cmd, **kwargs))  # noqa: S603
    except subprocess.CalledProcessError as exception:
        print(exception)
        sys.exit(1)


def convert_clipboard() -> None:
    """Convert clipboard code from the PDF."""
    original = pyperclip.paste()
    new = "\n".join(["" if e == "|" else e for e in original.split("\n")])
    if new != original:
        pyperclip.copy(new)
        print(new)


def main() -> None:
    """Scan a new document."""
    parser = argparse.ArgumentParser()

    presets = [
        e.stem[len(str(CONFIG_FILENAME)) - 4 :] for e in CONFIG_FOLDER.glob(f"{CONFIG_PATH.stem}-*.yaml")
    ]

    parser.add_argument(
        "--mode",
        choices=("adf", "one", "multi", "double"),
        default="adf",
        help="The scan mode: 'adf': use Auto Document Feeder (Default), "
        "one: Scan one page, multi: scan multiple pages, double: scan double sided document using the ADF, "
        "the default used configuration is, "
        "adf: {scanimage_arguments: [--source=ADF]}, "
        "multi: {scanimage_arguments: [--batch-prompt]}, "
        "one: {scanimage_arguments: [--batch-count=1]}, "
        "double: {scanimage_arguments: [--source=ADF], auto_bash: true, rotate_even: true}",
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
        metavar=("KEY", "VALUE"),
        action="append",
        default=[],
        help="Set a configuration option",
    )
    parser.add_argument(
        "--convert-clipboard",
        action="store_true",
        help="Wait and convert clipboard content, used to fix the newlines in the copied codes, "
        "see requirement: https://pypi.org/project/pyperclip/",
    )
    parser.add_argument(
        "--no-remove-to-continue",
        action="store_true",
        default=False,
        help="Don't wait for REMOVE_TO_CONTINUE's deletion before uploading to Paperless.",
    )

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    config_filename = (
        CONFIG_PATH if args.preset is None else Path(f"{str(CONFIG_PATH)[:-5]}-{args.preset}.yaml")
    )
    config: schema.Configuration = get_config(config_filename)

    if args.config:
        yaml = YAML()
        yaml.default_flow_style = False
        print(f"Config from file: {config_filename}")
        yaml.dump(config, sys.stdout)
        sys.exit()

    if args.convert_clipboard:
        print("Wait for clipboard content to be converted, press Ctrl+C to stop")
        convert_clipboard()
        try:
            previous = pyperclip.paste()
            while True:
                current = pyperclip.paste()
                if current != previous:
                    previous = current
                    convert_clipboard()
                time.sleep(0.1)
        except KeyboardInterrupt:
            print()
            sys.exit()

    dirty = False
    for conf in args.set_config:
        config[conf[0]] = conf[1]  # type: ignore[literal-required]
        dirty = True
    if dirty:
        yaml = YAML()
        yaml.default_flow_style = False
        with config_filename.open("w", encoding="utf-8") as config_file:
            config_file.write(
                "# yaml-language-server: $schema=https://raw.githubusercontent.com/sbrunner/scan-to-paperless"
                "/master/scan_to_paperless/config_schema.json\n\n",
            )
            yaml.dump(config, config_file)

    if "scan_folder" not in config:
        print(
            """The scan folder isn't set, use:
    scan --set-settings scan_folder <a_folder>
    This should be shared with the process container in 'source'.""",
        )
        sys.exit(1)
    now = datetime.datetime.now(datetime.timezone.utc)
    base_folder = Path(config["scan_folder"]).expanduser() / now.strftime("%Y%m%d-%H%M%S")
    while base_folder.exists():
        now += datetime.timedelta(seconds=1)
        base_folder = Path(config["scan_folder"]).expanduser() / now.strftime("%Y%m%d-%H%M%S")

    root_folder = base_folder / "source"
    root_folder.mkdir(parents=True)

    try:
        scanimage: list[str] = [config.get("scanimage", schema.SCANIMAGE_DEFAULT)]
        scanimage += config.get("scanimage_arguments", schema.SCANIMAGE_ARGUMENTS_DEFAULT)
        scanimage += [f"--batch={root_folder}/image-%d.{config.get('extension', schema.EXTENSION_DEFAULT)}"]
        mode_config = config.get("modes", {}).get(args.mode, {})
        mode_default = cast(schema.Mode, schema.MODES_DEFAULT.get(args.mode, {}))
        scanimage += mode_config.get("scanimage_arguments", mode_default.get("scanimage_arguments", []))

        if mode_config.get("auto_bash", mode_default.get("auto_bash", schema.AUTO_BASH_DEFAULT)):
            call([*scanimage, "--batch-start=1", "--batch-increment=2"])
            odd = os.listdir(root_folder)
            input("Put your document in the automatic document feeder for the other side, and press enter.")
            call(
                [
                    *scanimage,
                    f"--batch-start={len(odd) * 2}",
                    "--batch-increment=-2",
                    f"--batch-count={len(odd)}",
                ],
            )
            if mode_config.get("rotate_even", mode_default.get("rotate_even", schema.ROTATE_EVEN_DEFAULT)):
                for img in os.listdir(root_folder):
                    if img not in odd:
                        path = root_folder / img
                        with PIL.Image.open(path) as image:
                            image.rotate(180).save(path, dpi=image.info["dpi"])
        else:
            call(scanimage)

        args_: schema.Arguments = {
            "append_credit_card": args.append_credit_card,
            "assisted_split": args.assisted_split,
        }
        args_.update(config.get("default_args", {}))

    except subprocess.CalledProcessError as exception:
        print(exception)
        sys.exit(1)

    for img in os.listdir(root_folder):
        if not img.startswith("image-"):
            continue
        if "dpi" not in args_:
            with PIL.Image.open(root_folder / img) as image:
                if "dpi" in image.info:
                    args_["dpi"] = math.sqrt(
                        sum(float(e) * e for e in image.info["dpi"]) / len(image.info["dpi"]),
                    )

    subprocess.call([config.get("viewer", VIEWER_DEFAULT), root_folder])  # noqa: S603

    images = []
    for img in os.listdir(root_folder):
        if not img.startswith("image-"):
            continue
        images.append(Path("source") / img)

    regex = re.compile(rf"^source\/image\-([0-9]+)\.{config.get('extension', schema.EXTENSION_DEFAULT)}$")

    def image_match(image_path: Path) -> int:
        match = regex.match(str(image_path))
        assert match
        return int(match.group(1))

    images = sorted(images, key=image_match)
    if images:
        process_config = {
            "images": images,
            "args": args_,
        }
        yaml = YAML()
        yaml.default_flow_style = False
        with (root_folder.parent / "config.yaml").open("w", encoding="utf-8") as process_file:
            process_file.write(
                "# yaml-language-server: $schema=https://raw.githubusercontent.com/sbrunner/scan-to-paperless"
                "/master/scan_to_paperless/process_schema.json\n\n",
            )
            yaml.dump(process_config, process_file)
    else:
        root_folder.rmdir()
        base_folder.rmdir()


if __name__ == "__main__":
    main()
