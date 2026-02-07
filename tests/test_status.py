import asyncio
import os

import anyio
import pytest
from c2cwsgiutils.acceptance.image import check_screenshot

from scan_to_paperless import status


@pytest.mark.asyncio
@pytest.mark.flaky(reruns=3)
async def test_status() -> None:
    os.environ["SCAN_CODES_FOLDER"] = "./codes"
    os.environ["SCAN_FINAL_FOLDER"] = "./consume"
    os.environ["SCAN_SOURCE_FOLDER"] = "./scan"
    old_cwd = os.getcwd()  # noqa: PTH109
    status_dir = anyio.Path(__file__).parent / "status"
    os.chdir(str(status_dir))
    status_instance = status.Status()
    await status_instance.init()
    await asyncio.sleep(0.1)
    await status_instance.set_current_folder("7")
    await status_instance.write()
    os.chdir(old_cwd)

    parent_path = anyio.Path(__file__).parent
    print(f"Checking screenshot for file://{parent_path / 'status' / 'scan' / 'status.html'}")
    async with await anyio.open_file(parent_path / "status" / "scan" / "status.html", encoding="utf-8") as f:
        content = await f.read()
        print(content)
    check_screenshot(
        f"file://{parent_path / 'status' / 'scan' / 'status.html'}",
        str(parent_path / "status"),
        str(parent_path / "status" / "status.expected.png"),
        generate_expected_image=False,
        width=850,
        height=1800,
        sleep=1000,
    )
