import pytest

from scan_to_paperless import config as schema
from scan_to_paperless import scan


def test_get_mode_config_default() -> None:
    mode_config = scan._get_mode_config({}, "adf")
    assert mode_config == schema.MODES_DEFAULT["adf"]


def test_get_mode_config_custom() -> None:
    config = {
        "modes": {
            "custom": {
                "scanimage_arguments": ["--source=flatbed"],
                "auto_bash": True,
                "rotate_even": True,
            },
        },
    }
    mode_config = scan._get_mode_config(config, "custom")
    assert mode_config == config["modes"]["custom"]


def test_get_mode_config_merge() -> None:
    config = {
        "modes": {
            "double": {
                "rotate_even": False,
            },
        },
    }
    mode_config = scan._get_mode_config(config, "double")
    assert mode_config["scanimage_arguments"] == schema.MODES_DEFAULT["double"]["scanimage_arguments"]
    assert mode_config["auto_bash"] is True
    assert mode_config["rotate_even"] is False


def test_get_mode_config_unknown(capsys: pytest.CaptureFixture[str]) -> None:
    config = {"modes": {"custom": {}}}
    with pytest.raises(SystemExit) as exc_info:
        scan._get_mode_config(config, "missing")
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Unknown scan mode" in captured.out
    assert "custom" in captured.out
    assert "adf" in captured.out
