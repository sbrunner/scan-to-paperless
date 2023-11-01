import os
import subprocess

from c2cwsgiutils.acceptance.image import check_image_file

from scan_to_paperless import status
from scan_to_paperless.scan import output


def test_status():
    os.environ["SCAN_CODES_FOLDER"] = "./codes"
    os.environ["SCAN_FINAL_FOLDER"] = "./consume"
    os.environ["SCAN_SOURCE_FOLDER"] = "./scan"
    old_cwd = os.getcwd()
    os.chdir(os.path.join(os.path.dirname(__file__), "status"))
    status_instance = status.Status()
    status_instance.set_current_folder("7")
    status_instance.write()
    os.chdir(old_cwd)

    subprocess.run(
        [
            "node",
            "screenshot.js",
            f'--url=file://{os.path.join(os.path.dirname(__file__), "status", "scan", "status.html")}',
            f'--output={os.path.join(os.path.dirname(__file__), "status", "status.current.png")}',
            "--width=850",
            "--height=1800",
        ],
        check=True,
        cwd="/opt",
    )

    check_image_file(
        os.path.join(os.path.dirname(__file__), "status"),
        os.path.join(os.path.dirname(__file__), "status", "status.current.png"),
        os.path.join(os.path.dirname(__file__), "status", "status.expected.png"),
        generate_expected_image=False,
    )
