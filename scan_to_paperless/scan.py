#!/usr/bin/env python

"""Scan a new document."""

import asyncio
import datetime
import io
import math
import os
import re
import shlex
import subprocess  # nosec
import sys
import time
from enum import StrEnum
from typing import Annotated, Any, cast

import PIL.Image
import pyperclip
import typer
from anyio import Path
from ruamel.yaml.main import YAML

from scan_to_paperless import CONFIG_FILENAME, CONFIG_FOLDER, CONFIG_PATH, get_config
from scan_to_paperless import config as schema

from .config import VIEWER_DEFAULT


async def call(cmd: list[str], **kwargs: Any) -> None:
    """Verbose implementation of check_call."""
    print(shlex.join(cmd))
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            **kwargs,
        )
        returncode = await process.wait()
        if returncode != 0:
            print(f"Command returned with code {returncode}")
            sys.exit(returncode)
    except subprocess.CalledProcessError as exception:
        print(exception)
        sys.exit(1)


async def output(cmd: list[str], **kwargs: Any) -> bytes:
    """Verbose implementation of check_output."""
    print(shlex.join(cmd))
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **kwargs,
        )
        stdout, _ = await process.communicate()
        if process.returncode != 0:
            print(f"Command returned with code {process.returncode}")
            sys.exit(process.returncode)
    except subprocess.CalledProcessError as exception:
        print(exception)
        sys.exit(1)
    else:
        return stdout


def do_convert_clipboard() -> None:
    """Convert clipboard code from the PDF."""
    original = pyperclip.paste()
    new = "\n".join(["" if e == "|" else e for e in original.split("\n")])
    if new != original:
        pyperclip.copy(new)
        print(new)


app = typer.Typer(rich_markup_mode=None)


def available_presets() -> list[str]:
    """Return the list of available presets."""
    # Use synchronous os.listdir for Typer autocompletion callback
    config_folder_str = str(CONFIG_FOLDER)
    # Extract stem from filename (remove .yaml extension)
    config_stem = (
        str(CONFIG_FILENAME).rsplit(".", 1)[0] if "." in str(CONFIG_FILENAME) else str(CONFIG_FILENAME)
    )

    try:
        files = os.listdir(config_folder_str)  # noqa: PTH208
        return [
            f.rsplit(".", 1)[0][len(config_stem) + 1 :]  # Remove prefix and .yaml extension
            for f in files
            if f.startswith(f"{config_stem}-") and f.endswith(".yaml")
        ]
    except (FileNotFoundError, PermissionError):
        return []


@app.command(name="config", help="Print the configuration.")
async def main_config(
    preset: Annotated[
        str,
        typer.Option(help="Use an alternate configuration", autocompletion=available_presets),
    ],
) -> None:
    """Print the configuration."""
    config_filename = CONFIG_PATH if preset is None else Path(f"{str(CONFIG_PATH)[:-5]}-{preset}.yaml")
    config: schema.Configuration = await get_config(config_filename)

    yaml = YAML()
    yaml.default_flow_style = False
    print(f"Config from file: {config_filename}")
    yaml.dump(config, sys.stdout)
    sys.exit()


@app.command(help="Set a configuration option.")
async def set_config(
    key: Annotated[str, typer.Option(help="Configuration key to set")],
    value: Annotated[str, typer.Option(help="Configuration value to set")],
    preset: Annotated[
        str,
        typer.Option(help="Use an alternate configuration", autocompletion=available_presets),
    ],
) -> None:
    """Set a configuration option."""
    config_filename = CONFIG_PATH if preset is None else Path(f"{str(CONFIG_PATH)[:-5]}-{preset}.yaml")
    config: schema.Configuration = await get_config(config_filename)

    config[key] = value  # type: ignore[literal-required]

    yaml = YAML()
    yaml.default_flow_style = False
    async with await config_filename.open("w", encoding="utf-8") as config_file:
        await config_file.write(
            "# yaml-language-server: $schema=https://raw.githubusercontent.com/sbrunner/scan-to-paperless"
            "/master/scan_to_paperless/config_schema.json\n\n",
        )
        yaml.dump(config, config_file)


@app.command(
    help="Wait and convert clipboard content, used to fix the newlines in the copied codes, "
    "see requirement: https://pypi.org/project/pyperclip/",
)
def convert_clipboard() -> None:
    """
    Convert clipboard content.
    """
    print("Wait for clipboard content to be converted, press Ctrl+C to stop")
    do_convert_clipboard()
    try:
        previous = pyperclip.paste()
        while True:
            current = pyperclip.paste()
            if current != previous:
                previous = current
                do_convert_clipboard()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
        sys.exit()


class _Mode(StrEnum):
    ADF = "adf"
    MULTI = "multi"
    ONE = "one"
    DOUBLE = "double"


app_scan = typer.Typer(rich_markup_mode=None)


@app.command(help="Scan a new document.")
@app_scan.command(help="Scan a new document.")
async def scan(
    mode: Annotated[
        _Mode,
        typer.Option(
            help="\n\n".join(  # noqa: FLY002
                [
                    "The scan mode: ",
                    "'adf': use Auto Document Feeder (Default) (default used arguments: --source=ADF)",
                    "'one': scan one page (default used arguments: --batch-count=1)",
                    "'multi': scan multiple pages (default used arguments: --batch-prompt)",
                    "'double': scan double sided document using the ADF (default used arguments: --source=ADF, auto_bash: true, rotate_even: true)",
                ],
            ),
        ),
    ] = _Mode.ADF,
    preset: Annotated[
        str | None,
        typer.Option(
            help="Use an alternate configuration",
            autocompletion=available_presets,
        ),
    ] = None,
    append_credit_card: Annotated[
        bool,
        typer.Option(
            help="Append vertically the credit card",
        ),
    ] = False,
    assisted_split: Annotated[
        bool,
        typer.Option(
            help="Split operation, see help",
        ),
    ] = False,
) -> None:
    """Scan a new document."""
    config_filename = CONFIG_PATH if preset is None else Path(f"{str(CONFIG_PATH)[:-5]}-{preset}.yaml")
    config: schema.Configuration = await get_config(config_filename)

    scan_folder = await _validate_scan_folder(config)
    base_folder = await _get_base_folder(scan_folder)
    root_folder = base_folder / "source"
    await root_folder.mkdir(parents=True)

    try:
        scanimage_cmd = _build_scanimage_command(config, root_folder, mode)
        await _perform_scan(scanimage_cmd, root_folder, config, mode)

        args_: schema.Arguments = {
            "append_credit_card": append_credit_card,
            "assisted_split": assisted_split,
        }
        args_.update(config.get("default_args", {}))

        await _detect_image_dpi(args_, root_folder)

    except subprocess.CalledProcessError as exception:
        print(exception)
        sys.exit(1)

    viewer_cmd = [config.get("viewer", VIEWER_DEFAULT), str(root_folder)]
    process = await asyncio.create_subprocess_exec(*viewer_cmd)
    await process.wait()

    extension = config.get("extension", schema.EXTENSION_DEFAULT)
    images = await _get_sorted_images(root_folder, extension)
    await _save_process_config(root_folder, images, args_)


async def _validate_scan_folder(config: schema.Configuration) -> Path:
    """Validate and return the scan folder path."""
    if "scan_folder" not in config:
        print(
            """The scan folder isn't set, use:
    scan --set-settings scan_folder <a_folder>
    This should be shared with the process container in 'source'.""",
        )
        sys.exit(1)
    return await Path(config["scan_folder"]).expanduser()


async def _get_base_folder(scan_folder: Path) -> Path:
    """Get a unique base folder for the scan."""
    now = datetime.datetime.now(datetime.UTC)
    base_folder = scan_folder / now.strftime("%Y%m%d-%H%M%S")
    while await base_folder.exists():
        now += datetime.timedelta(seconds=1)
        base_folder = scan_folder / now.strftime("%Y%m%d-%H%M%S")
    return base_folder


async def _scan_adf_mode(
    scanimage_cmd: list[str],
    root_folder: Path,
) -> None:
    """Scan using ADF mode."""
    del root_folder
    await call([*scanimage_cmd])


async def _scan_double_mode(
    scanimage_cmd: list[str],
    root_folder: Path,
) -> None:
    """Scan double-sided document using ADF."""
    await call([*scanimage_cmd, "--batch-start=1", "--batch-increment=2"])
    odd = [p async for p in root_folder.iterdir()]
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None, input, "Put your document in the automatic document feeder for the other side, and press enter."
    )
    await call(
        [
            *scanimage_cmd,
            f"--batch-start={len(odd) * 2}",
            "--batch-increment=-2",
            f"--batch-count={len(odd)}",
        ],
    )
    async for img in root_folder.iterdir():
        if img not in odd:
            path = root_folder / img
            with PIL.Image.open(path) as image:
                image.rotate(180).save(path, dpi=image.info["dpi"])


def _build_scanimage_command(
    config: schema.Configuration,
    root_folder: Path,
    mode: _Mode,
) -> list[str]:
    """Build the scanimage command based on configuration and mode."""
    scanimage: list[str] = [config.get("scanimage", schema.SCANIMAGE_DEFAULT)]
    scanimage += config.get("scanimage_arguments", schema.SCANIMAGE_ARGUMENTS_DEFAULT)
    extension = config.get("extension", schema.EXTENSION_DEFAULT)
    scanimage += [f"--batch={root_folder}/image-%d.{extension}"]

    mode_config = config.get("modes", {}).get(mode.value, {})
    mode_default = cast("schema.Mode", schema.MODES_DEFAULT.get(mode.value, {}))
    scanimage += mode_config.get("scanimage_arguments", mode_default.get("scanimage_arguments", []))

    return scanimage


async def _perform_scan(
    scanimage_cmd: list[str],
    root_folder: Path,
    config: schema.Configuration,
    mode: _Mode,
) -> None:
    """Perform the actual scan based on mode."""
    mode_config = config.get("modes", {}).get(mode.value, {})
    mode_default = cast("schema.Mode", schema.MODES_DEFAULT.get(mode.value, {}))

    if mode_config.get("auto_bash", mode_default.get("auto_bash", schema.AUTO_BASH_DEFAULT)):
        if mode == _Mode.DOUBLE:
            await _scan_double_mode(scanimage_cmd, root_folder)
        else:
            await call([*scanimage_cmd])
    else:
        await _scan_adf_mode(scanimage_cmd, root_folder)


async def _detect_image_dpi(args_: schema.Arguments, root_folder: Path) -> None:
    """Detect DPI from images if not already set."""
    if "dpi" in args_:
        return

    async for img in root_folder.iterdir():
        if not img.name.startswith("image-"):
            continue
        with PIL.Image.open(root_folder / img) as image:
            if "dpi" in image.info:
                args_["dpi"] = math.sqrt(
                    sum(float(e) * e for e in image.info["dpi"]) / len(image.info["dpi"]),
                )
                return


async def _get_sorted_images(root_folder: Path, extension: str) -> list[Path]:
    """Get sorted list of images from root folder."""
    images = []
    async for img in root_folder.iterdir():
        if not img.name.startswith("image-"):
            continue
        images.append(Path(img).relative_to(root_folder.parent))

    regex = re.compile(rf"source/image\-([0-9]+)\.{extension}$")

    def image_match(image_path: Path) -> int:
        match = regex.match(str(image_path))
        assert match
        return int(match.group(1))

    return sorted(images, key=image_match)


async def _save_process_config(
    root_folder: Path,
    images: list[Path],
    args_: schema.Arguments,
) -> None:
    """Save the process configuration file."""
    if not images:
        await root_folder.rmdir()
        await root_folder.parent.rmdir()
        return

    process_config = {
        "images": [str(image) for image in images],
        "args": args_,
    }
    yaml = YAML()
    yaml.default_flow_style = False
    async with await (root_folder.parent / "config.yaml").open("w", encoding="utf-8") as process_file:
        await process_file.write(
            "# yaml-language-server: $schema=https://raw.githubusercontent.com/sbrunner/scan-to-paperless"
            "/master/scan_to_paperless/process_schema.json\n\n",
        )
        out = io.StringIO()
        yaml.dump(process_config, out)
        await process_file.write(out.getvalue())


if __name__ == "__main__":
    app_scan()
