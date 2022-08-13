"""The scan to Paperless main module."""
import os.path
import sys
from typing import cast

from deepmerge import Merger
from ruamel.yaml.main import YAML

if sys.version_info.minor >= 8:
    from scan_to_paperless import config as stp_config
else:
    from scan_to_paperless import config_old as stp_config  # type: ignore

CONFIG_FILENAME = "scan-to-paperless.yaml"

if "APPDATA" in os.environ:
    CONFIG_FOLDER = os.environ["APPDATA"]
elif "XDG_CONFIG_HOME" in os.environ:
    CONFIG_FOLDER = os.environ["XDG_CONFIG_HOME"]
else:
    CONFIG_FOLDER = os.path.expanduser("~/.config")

CONFIG_PATH = os.path.join(CONFIG_FOLDER, CONFIG_FILENAME)


def get_config(config_filename: str) -> stp_config.Configuration:
    """Get the configuration."""
    if os.path.exists(config_filename):
        yaml = YAML()
        yaml.default_flow_style = False
        with open(config_filename, encoding="utf-8") as config_file:
            config = cast(stp_config.Configuration, yaml.load(config_file))
            if "exteds" in config:

                base_config = get_config(
                    os.path.normpath(
                        os.path.join(os.path.dirname(config_filename), os.path.expanduser(config["extends"]))
                    )
                )

                strategies_config = cast(stp_config.MergeStrategies, config.get("strategies", {}))
                merger = Merger(
                    [
                        (list, strategies_config.get("list", ["override"])),
                        (dict, strategies_config.get("dict", ["merge"])),
                    ],
                    strategies_config.get("fallback", ["merge"]),
                    strategies_config.get("type_conflict", ["merge"]),
                )
                config = cast(stp_config.Configuration, merger.merge(base_config, config))
            return config
    print(f"Missing config file: {config_filename}")
    return {}
