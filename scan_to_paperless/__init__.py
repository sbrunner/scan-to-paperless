"""The scan to Paperless main module."""
import os.path
import sys
from typing import cast

from deepmerge import Merger
from ruamel.yaml.main import YAML

if sys.version_info.minor >= 8:
    from scan_to_paperless import config as schema
else:
    from scan_to_paperless import config_old as schema  # type: ignore

CONFIG_FILENAME = "scan-to-paperless.yaml"

if "APPDATA" in os.environ:
    CONFIG_FOLDER = os.environ["APPDATA"]
elif "XDG_CONFIG_HOME" in os.environ:
    CONFIG_FOLDER = os.environ["XDG_CONFIG_HOME"]
else:
    CONFIG_FOLDER = os.path.expanduser("~/.config")

CONFIG_PATH = os.path.join(CONFIG_FOLDER, CONFIG_FILENAME)


def get_config(config_filename: str) -> schema.Configuration:
    """Get the configuration."""
    if os.path.exists(config_filename):
        yaml = YAML()
        yaml.default_flow_style = False
        with open(config_filename, encoding="utf-8") as config_file:
            config = cast(schema.Configuration, yaml.load(config_file))
            if "extends" in config:
                base_config = get_config(
                    os.path.normpath(
                        os.path.join(os.path.dirname(config_filename), os.path.expanduser(config["extends"]))
                    )
                )

                strategies_config = cast(schema.MergeStrategies, config.get("strategies", {}))
                merger = Merger(
                    [
                        (list, strategies_config.get("list", ["override"])),
                        (dict, strategies_config.get("dict", ["merge"])),
                    ],
                    strategies_config.get("fallback", ["override"]),
                    strategies_config.get("type_conflict", ["override"]),
                )
                config = cast(schema.Configuration, merger.merge(base_config, config))
            return config
    print(f"Missing config file: {config_filename}")
    return {}
