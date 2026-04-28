import pytest
from anyio import Path

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


@pytest.mark.asyncio
async def test_get_config_sources_and_default_args_sources(tmp_path) -> None:
    base = Path(str(tmp_path)) / "base.yaml"
    child = Path(str(tmp_path)) / "child.yaml"
    grand_child = Path(str(tmp_path)) / "grand-child.yaml"

    await base.write_text(
        """default_args:
  cut_white: 210
  cut_black: 10
""",
        encoding="utf-8",
    )
    await child.write_text(
        """extends: base.yaml
default_args:
  cut_white: 205
  dpi: 300
""",
        encoding="utf-8",
    )
    await grand_child.write_text(
        """extends: child.yaml
default_args:
  dpi: 400
""",
        encoding="utf-8",
    )

    sources, default_args_sources = await scan._get_config_sources_and_default_args_sources(grand_child)

    assert [source.name for source in sources] == ["base.yaml", "child.yaml", "grand-child.yaml"]
    assert default_args_sources["cut_black"].endswith("base.yaml")
    assert default_args_sources["cut_white"].endswith("child.yaml")
    assert default_args_sources["dpi"].endswith("grand-child.yaml")
