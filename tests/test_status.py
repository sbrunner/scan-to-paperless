import os
from pathlib import Path

import pytest
from c2cwsgiutils.acceptance.image import check_screenshot

from scan_to_paperless import status


@pytest.mark.asyncio
@pytest.mark.flaky(reruns=3)
async def test_status() -> None:
    os.environ["SCAN_CODES_FOLDER"] = "./codes"
    os.environ["SCAN_FINAL_FOLDER"] = "./consume"
    os.environ["SCAN_SOURCE_FOLDER"] = "./scan"
    old_cwd = Path.cwd()
    status_dir = Path(__file__).parent / "status"
    os.chdir(status_dir)
    status_instance = status.Status()
    await status_instance.init()
    status_instance.set_current_folder(Path("7"))
    status_instance.write()
    os.chdir(old_cwd)

    parent_path = Path(__file__).parent
    check_screenshot(
        f"file://{parent_path / 'status' / 'scan' / 'status.html'}",
        parent_path / "status",
        parent_path / "status" / "status.expected.png",
        generate_expected_image=False,
        width=850,
        height=1800,
        sleep=1000,
    )
