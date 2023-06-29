"""Manage the status file of the progress."""

import datetime
import glob
import html
import os.path
from typing import Dict, NamedTuple, Optional

import jinja2
from ruamel.yaml.main import YAML

WAITING_STATUS_NAME = "Waiting validation"
WAITING_STATUS_DESCRIPTION = """You should validate that the generate images are correct.<br>
    If the result is correct remove the <code>REMOVE_TO_CONTINUE</code> file.<br>
    If not you can:<br>
    <ul>
        <li>Edit the generated image Then remove the <code>REMOVE_TO_CONTINUE</code> file.</li>
        <li>Edit the <code>config.yaml</code> file to change the parameters, the remove the generated files to force the regeneration.</li>
        <li>Edit the source image the remove the corresponding generated file to force the regeneration.</li>
    </ul>"""


class _Folder(NamedTuple):
    status: str
    details: str


class Status:
    """Manage the status file of the progress."""

    def __init__(self, no_write: bool = False) -> None:
        """Construct."""

        self.no_write = no_write
        self._file = os.path.join(os.environ.get("SCAN_SOURCE_FOLDER", "/source"), "status.html")
        self._status: Dict[str, _Folder] = {}
        self._global_status = "Starting..."
        self._global_status_update = datetime.datetime.utcnow().replace(microsecond=0)
        self._start_time = datetime.datetime.utcnow().replace(microsecond=0)
        self._last_scan = datetime.datetime.utcnow()
        self._current_folder: Optional[str] = None
        self.scan()

    def set_global_status(self, status: str) -> None:
        """Set the global status."""

        if self._global_status != status:
            self._global_status = status
            self._global_status_update = datetime.datetime.utcnow().replace(microsecond=0)

        self.write()

    def set_current_folder(self, name: Optional[str]) -> None:
        """Set the current folder."""

        self._current_folder = name

    def set_current_config(self, name: str) -> None:
        """Set the current config file."""

        if name.endswith("/config.yaml"):
            name = os.path.basename(os.path.dirname(name))
        self._current_folder = name

    def set_status(self, name: str, status: str, details: str = "") -> None:
        """Set the status of a folder."""

        # Config file name
        if name.endswith("/config.yaml"):
            name = os.path.basename(os.path.dirname(name))
        self._status[name] = _Folder(html.escape(status), details)

        if self.no_write:
            print(f"{name}: {status}")

        if not self.no_write:
            self.write()

    def scan(self) -> None:
        """Scan for changes for waiting documents."""

        self._last_scan = datetime.datetime.utcnow()

        for name in self._status:  # pylint: disable=consider-using-dict-items
            if name != self._current_folder:
                if os.path.isdir(os.path.join(os.environ.get("SCAN_SOURCE_FOLDER", "/source"), name)):
                    self._update_status(name)
                else:
                    del self._status[name]

        for folder_name in glob.glob(os.path.join(os.environ.get("SCAN_SOURCE_FOLDER", "/source"), "*")):
            if os.path.isdir(folder_name):
                name = os.path.basename(folder_name)
                if name not in self._status:
                    self._update_status(name)

    def _update_status(self, name: str) -> None:
        yaml = YAML(typ="safe")
        if os.path.exists(os.path.join(os.environ.get("SCAN_SOURCE_FOLDER", "/source"), name, "error.yaml")):
            with open(
                os.path.join(os.environ.get("SCAN_SOURCE_FOLDER", "/source"), name, "error.yaml"),
                encoding="utf-8",
            ) as error_file:
                error = yaml.load(error_file)

            self.set_status(
                name, "Error: " + error["error"], "<code>" + "<br />".join(error["traceback"]) + "</code>"
            )
            return

        if os.path.exists(os.path.join(os.environ.get("SCAN_SOURCE_FOLDER", "/source"), name, "DONE")):
            self.set_status(name, "Done")
            return

        if not os.path.exists(
            os.path.join(os.environ.get("SCAN_SOURCE_FOLDER", "/source"), name, "config.yaml")
        ):
            len_folder = (
                len(os.path.join(os.environ.get("SCAN_SOURCE_FOLDER", "/source"), name).rstrip("/")) + 1
            )
            files = [
                f[len_folder:]
                for f in glob.glob(
                    os.path.join(os.environ.get("SCAN_SOURCE_FOLDER", "/source"), name, "**"),
                    recursive=True,
                )
                if os.path.isfile(f)
            ]
            self.set_status(name, "Missing config", ", ".join(files))
            return

        with open(
            os.path.join(os.environ.get("SCAN_SOURCE_FOLDER", "/source"), name, "config.yaml"),
            encoding="utf-8",
        ) as config_file:
            config = yaml.load(config_file)

        if config is None:
            self.set_status(name, "Empty config")
            return

        if os.path.exists(
            os.path.join(os.environ.get("SCAN_SOURCE_FOLDER", "/source"), name, "REMOVE_TO_CONTINUE")
        ):
            rerun = False
            for image in config["steps"][-1]["sources"]:
                if not os.path.exists(image):
                    rerun = True

            if rerun:
                if len(config["steps"]) >= 2:
                    self.set_status(name, "Waiting to " + config["steps"][-2])
                else:
                    self.set_status(name, "Waiting to transform")
            else:
                self.set_status(name, WAITING_STATUS_NAME, WAITING_STATUS_DESCRIPTION)
        else:
            if len(config["steps"]) >= 1:
                self.set_status(name, "Waiting to " + config["steps"][-1]["name"])
            else:
                self.set_status(name, "Waiting to transform")

    def write(self) -> None:
        """Write the status file."""

        if self._last_scan < datetime.datetime.utcnow() - datetime.timedelta(minutes=1):
            self.scan()

        if self.no_write:
            return

        with open(self._file, "w", encoding="utf-8") as status_file:
            env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(os.path.dirname(__file__)),
                autoescape=jinja2.select_autoescape(),
            )
            template = env.get_template("status.html")
            status_file.write(
                template.render(
                    global_status=self._global_status,
                    global_status_update=self._global_status_update,
                    start_time=self._start_time,
                    datetime=datetime,
                    status=self._status,
                )
            )
