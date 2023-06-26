"""Manage the status file of the progress."""

import datetime
import glob
import html
import os.path
from typing import Dict, NamedTuple, Optional

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

        if self._global_status == status:
            return

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

        for name in self._status:
            if name != self._current_folder:
                self._update_status(name)

        names = []
        for config_file_name in glob.glob(
            os.path.join(os.environ.get("SCAN_SOURCE_FOLDER", "/source"), "*/config.yaml")
        ):
            name = os.path.basename(os.path.dirname(config_file_name))
            names.append(name)
            if name not in self._status:
                self._update_status(name, force=True)

        for folder_name in glob.glob(os.path.join(os.environ.get("SCAN_SOURCE_FOLDER", "/source"), "*")):
            name = os.path.basename(folder_name)

            if name not in self._status:
                names.append(name)

                self.set_status(
                    name,
                    "Missing config",
                    ", ".join(
                        glob.glob(
                            os.path.join(os.environ.get("SCAN_SOURCE_FOLDER", "/source"), folder_name, "**"),
                            recursive=True,
                        )
                    ),
                )

        for name in self._status:  # pylint: disable=consider-using-dict-items
            if name not in names:
                del self._status[name]

        self._last_scan = datetime.datetime.utcnow()

    def _update_status(self, name: str, force: bool = False) -> None:
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

        with open(
            os.path.join(os.environ.get("SCAN_SOURCE_FOLDER", "/source"), name, "config.yaml"),
            encoding="utf-8",
        ) as config_file:
            config = yaml.load(config_file)

        if config is None:
            self.set_status(name, "Empty config")
            return

        if os.path.exists(os.path.join(os.environ.get("SCAN_SOURCE_FOLDER", "/source"), name, "DONE")):
            self.set_status(name, "Done")

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
            elif force:
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
            status_file.write(
                f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Scan to Paperless status</title>

    <link
      rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.2.3/css/bootstrap.min.css"
      integrity="sha512-SbiR/eusphKoMVVXysTKG/7VseWii+Y3FdHrt0EpKgpToZeemhqHeZeLWLhJutz/2ut2Vw1uQEj2MbRF+TVBUA=="
      crossorigin="anonymous"
      referrerpolicy="no-referrer"
    />
    <link
      rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap-icons/1.10.5/font/bootstrap-icons.min.css"
      integrity="sha512-ZnR2wlLbSbr8/c9AgLg3jQPAattCUImNsae6NHYnS9KrIwRdcY9DxFotXhNAKIKbAXlRnujIqUWoXXwqyFOeIQ=="
      crossorigin="anonymous"
      referrerpolicy="no-referrer"
    />
    <link
      rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap-table/1.21.4/bootstrap-table.min.css"
      integrity="sha512-vaPSEKPBIWEWK+pGIwdLnPzw7S2Tr6rYVT05v+KN89YVpEJavFiY1dPzT+e1ZeyizjEPBicVxJ5QixXZw0Nopw=="
      crossorigin="anonymous"
      referrerpolicy="no-referrer"
    />
  </head>
  <body class="px-5 py-4">
    <h1>Scan to Paperless status</h1>
    <p>{self._global_status} since <script>
    window.document.write(new Date('{self._global_status_update.isoformat()}Z').toLocaleString());
    </script></p>
    <p>Started at: <script>
    window.document.write(new Date('{self._start_time.isoformat()}Z').toLocaleString());
    </script></p>
    <p>Generated at: <script>
    window.document.write(new Date('{datetime.datetime.utcnow().replace(microsecond=0).isoformat()}Z').toLocaleString());
    </script></p>
    <h2>Jobs</h2>
    <table data-toggle="table">
      <thead>
        <tr>
          <th data-sortable="true">Folder</th>
          <th data-sortable="true">Status</th>
          <th>Details</th>
        </tr>
      </thead>
      <tbody>
"""
            )
            for name, folder in self._status.items():
                tr_attributes = ' class="alert alert-info"' if name == self._current_folder else ""
                status_file.write(
                    f"""        <tr{tr_attributes}>
          <td><a href="./{name}" target="_blank">{name}</a></td>
          <td>{folder.status}</td>
          <td>{folder.details}</td>
        </tr>
"""
                )
            status_file.write(
                """      </tbody>
    </table>

    <script
      src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.7.0/jquery.min.js"
      integrity="sha512-3gJwYpMe3QewGELv8k/BX9vcqhryRdzRMxVfq6ngyWXwo03GFEzjsUm8Q7RZcHPHksttq7/GFoxjCVUjkjvPdw=="
      crossorigin="anonymous"
      referrerpolicy="no-referrer"
    ></script>
    <script
      src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/js/bootstrap.bundle.min.js"
      integrity="sha512-VK2zcvntEufaimc+efOYi622VN5ZacdnufnmX7zIhCPmjhKnOi9ZDMtg1/ug5l183f19gG1/cBstPO4D8N/Img=="
      crossorigin="anonymous"
      referrerpolicy="no-referrer"
    ></script>
    <script
      src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap-table/1.21.4/bootstrap-table.min.js"
      integrity="sha512-rZAhvMayqW5e/N+xdp011tYAIdxgMMJtKxUXx7scO4iBPSUXAKdkrKIPRu6tLr0O9V6Bs9QujJF3MqmgSNfYPA=="
      crossorigin="anonymous"
      referrerpolicy="no-referrer"
    ></script>
  </body>
</html>"""
            )
