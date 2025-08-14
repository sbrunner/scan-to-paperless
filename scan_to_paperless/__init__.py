"""The scan to Paperless main module."""

import os.path
from pathlib import Path
from typing import cast

from deepmerge.merger import Merger
from ruamel.yaml.main import YAML

from scan_to_paperless import config as schema

CONFIG_FILENAME = "scan-to-paperless.yaml"

if "APPDATA" in os.environ:
    CONFIG_FOLDER = Path(os.environ["APPDATA"])
elif "XDG_CONFIG_HOME" in os.environ:
    CONFIG_FOLDER = Path(os.environ["XDG_CONFIG_HOME"])
else:
    CONFIG_FOLDER = Path("~/.config").expanduser()

CONFIG_PATH = CONFIG_FOLDER / CONFIG_FILENAME


class ScanToPaperlessError(Exception):
    """Base exception for this module."""


def get_config(config_filename: Path) -> schema.Configuration:
    """Get the configuration."""
    if config_filename.exists():
        yaml = YAML()
        yaml.default_flow_style = False
        with config_filename.open(encoding="utf-8") as config_file:
            config = cast("schema.Configuration", yaml.load(config_file))
            if "extends" in config:
                base_config = get_config(
                    (config_filename.parent / config["extends"]).expanduser().resolve(),
                )

                strategies_config = cast("schema.MergeStrategies", config.get("strategies", {}))
                merger = Merger(
                    [
                        (list, strategies_config.get("list", ["override"])),
                        (dict, strategies_config.get("dict", ["merge"])),
                    ],
                    strategies_config.get("fallback", ["override"]),
                    strategies_config.get("type_conflict", ["override"]),
                )
                config = merger.merge(base_config, config)
            return config
    print(f"Missing config file: {config_filename}")
    return {}
