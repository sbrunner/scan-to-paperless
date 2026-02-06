#!/usr/bin/env python

"""Scan a new document."""

import asyncio
import datetime
import math
import re
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
        return cast("bytes", subprocess.check_output(cmd, **kwargs))  # noqa: S603
    except subprocess.CalledProcessError as exception:
        print(exception)
        sys.exit(1)


def do_convert_clipboard() -> None:
    """Convert clipboard code from the PDF."""
    original = pyperclip.paste()
    new = "\n".join(["" if e == "|" else e for e in original.split("\n")])
    if new != original:
        pyperclip.copy(new)
        print(new)


app = typer.Typer(rich_markup_mode=None)


async def available_presets() -> list[str]:
    """Return the list of available presets."""
    return [
        e.stem[len(str(CONFIG_FILENAME)) - 4 :]
        async for e in CONFIG_FOLDER.glob(f"{CONFIG_PATH.stem}-*.yaml")
    ]


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

    if "scan_folder" not in config:
        print(
            """The scan folder isn't set, use:
    scan --set-settings scan_folder <a_folder>
    This should be shared with the process container in 'source'.""",
        )
        sys.exit(1)
    now = datetime.datetime.now(datetime.UTC)
    base_folder = (await Path(config["scan_folder"]).expanduser()) / now.strftime("%Y%m%d-%H%M%S")
    while await base_folder.exists():
        now += datetime.timedelta(seconds=1)
        base_folder = (await Path(config["scan_folder"]).expanduser()) / now.strftime("%Y%m%d-%H%M%S")

    root_folder = base_folder / "source"
    await root_folder.mkdir(parents=True)

    try:
        scanimage: list[str] = [config.get("scanimage", schema.SCANIMAGE_DEFAULT)]
        scanimage += config.get("scanimage_arguments", schema.SCANIMAGE_ARGUMENTS_DEFAULT)
        scanimage += [f"--batch={root_folder}/image-%d.{config.get('extension', schema.EXTENSION_DEFAULT)}"]
        mode_config = config.get("modes", {}).get(mode.value, {})
        mode_default = cast("schema.Mode", schema.MODES_DEFAULT.get(mode.value, {}))
        scanimage += mode_config.get("scanimage_arguments", mode_default.get("scanimage_arguments", []))

        if mode_config.get("auto_bash", mode_default.get("auto_bash", schema.AUTO_BASH_DEFAULT)):
            call([*scanimage, "--batch-start=1", "--batch-increment=2"])
            odd = [p async for p in root_folder.iterdir()]
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
                async for img in root_folder.iterdir():
                    if img not in odd:
                        path = root_folder / img
                        with PIL.Image.open(path) as image:
                            image.rotate(180).save(path, dpi=image.info["dpi"])
        else:
            call(scanimage)

        args_: schema.Arguments = {
            "append_credit_card": append_credit_card,
            "assisted_split": assisted_split,
        }
        args_.update(config.get("default_args", {}))

    except subprocess.CalledProcessError as exception:
        print(exception)
        sys.exit(1)

    async for img in root_folder.iterdir():
        if not img.name.startswith("image-"):
            continue
        if "dpi" not in args_:
            with PIL.Image.open(root_folder / img) as image:
                if "dpi" in image.info:
                    args_["dpi"] = math.sqrt(
                        sum(float(e) * e for e in image.info["dpi"]) / len(image.info["dpi"]),
                    )

    viewer_cmd = [config.get("viewer", VIEWER_DEFAULT), str(root_folder)]
    process = await asyncio.create_subprocess_exec(*viewer_cmd)
    await process.wait()

    images = []
    async for img in root_folder.iterdir():
        if not img.name.startswith("image-"):
            continue
        images.append(Path(img).relative_to(root_folder.parent))

    regex = re.compile(rf"source/image\-([0-9]+)\.{config.get('extension', schema.EXTENSION_DEFAULT)}$")

    def image_match(image_path: Path) -> int:
        match = regex.match(str(image_path))
        assert match
        return int(match.group(1))

    images = sorted(images, key=image_match)
    if images:
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
            yaml.dump(process_config, process_file)
    else:
        await root_folder.rmdir()
        await base_folder.rmdir()


if __name__ == "__main__":
    app_scan()
