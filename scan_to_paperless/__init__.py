"""The scan to Paperless main module."""

import os
import pathlib
from typing import cast

from anyio import Path
from deepmerge.merger import Merger
from ruamel.yaml.main import YAML

from scan_to_paperless import config as schema

CONFIG_FILENAME = "scan-to-paperless.yaml"

if "APPDATA" in os.environ:
    CONFIG_FOLDER = Path(os.environ["APPDATA"])
elif "XDG_CONFIG_HOME" in os.environ:
    CONFIG_FOLDER = Path(os.environ["XDG_CONFIG_HOME"])
else:
    CONFIG_FOLDER = Path(pathlib.Path("~/.config").expanduser())

CONFIG_PATH = CONFIG_FOLDER / CONFIG_FILENAME


class ScanToPaperlessError(Exception):
    """Base exception for this module."""


async def get_config(config_filename: Path) -> schema.Configuration:
    """Get the configuration."""
    if await config_filename.exists():
        yaml = YAML()
        yaml.default_flow_style = False
        async with await config_filename.open(encoding="utf-8") as config_file:
            file_content = await config_file.read()
            config = cast("schema.Configuration", yaml.load(file_content))
            if "extends" in config:
                base_config = await get_config(
                    await (await (config_filename.parent / config["extends"]).expanduser()).resolve(),
                )

                strategies_config = config.get("merge_strategies", cast("schema.MergeStrategies", {}))
                merger = Merger(
                    [
                        (list, strategies_config.get("for_list", ["override"])),
                        (dict, strategies_config.get("for_dict", ["merge"])),
                    ],
                    strategies_config.get("fallback", ["override"]),
                    strategies_config.get("type_conflict", ["override"]),
                )
                config = merger.merge(base_config, config)
            return config
    print(f"Missing config file: {config_filename}")
    return {}
