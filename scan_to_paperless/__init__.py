"""The scan to Paperless main module."""

import os
import pathlib
from typing import Any, cast

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


async def get_config(config_filename: Path, verbose: bool = True) -> schema.Configuration:
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
                    verbose=verbose,
                )

                strategies_config = config.get("merge_strategies", cast("schema.MergeStrategies", {}))
                for_list: list[Any] = strategies_config.get("for_list", ["override"])
                for_dict: list[Any] = strategies_config.get("for_dict", ["merge"])
                fallback: list[Any] = strategies_config.get("fallback", ["override"])
                type_conflict: list[Any] = strategies_config.get("type_conflict", ["override"])
                merger = Merger(
                    [
                        (list, for_list),
                        (dict, for_dict),
                    ],
                    fallback,
                    type_conflict,
                )
                config = merger.merge(base_config, config)
            return config
    if verbose:
        print(f"Missing config file: {config_filename}")
    return {}
