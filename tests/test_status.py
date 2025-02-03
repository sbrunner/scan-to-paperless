import os
from pathlib import Path

import pytest
from c2cwsgiutils.acceptance.image import check_screenshot

from scan_to_paperless import status


@pytest.mark.flaky(reruns=3)
def test_status() -> None:
    os.environ["SCAN_CODES_FOLDER"] = "./codes"
    os.environ["SCAN_FINAL_FOLDER"] = "./consume"
    os.environ["SCAN_SOURCE_FOLDER"] = "./scan"
    old_cwd = os.getcwd()
    os.chdir(os.path.join(os.path.dirname(__file__), "status"))
    status_instance = status.Status()
    status_instance.set_current_folder(Path("7"))
    status_instance.write()
    os.chdir(old_cwd)

    check_screenshot(
        f"file://{os.path.join(os.path.dirname(__file__), 'status', 'scan', 'status.html')}",
        os.path.join(os.path.dirname(__file__), "status"),
        os.path.join(os.path.dirname(__file__), "status", "status.expected.png"),
        generate_expected_image=False,
        width=850,
        height=1800,
        sleep=1000,
    )
