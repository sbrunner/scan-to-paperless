"""Manage the status file of the progress."""

import datetime
import glob
import os.path
from typing import Dict, NamedTuple


class _Folder(NamedTuple):
    status: str
    details: str


class Status:
    """Manage the status file of the progress."""

    def __init__(self, no_write: bool = False) -> None:
        """Construct."""

        self.no_write = no_write
        self._file = os.path.join(os.environ.get("SCAN_SOURCE_FOLDER", "status.html"))
        self._status: Dict[str, _Folder] = {}
        self._global_status = "Starting..."

    def set_global_status(self, status: str) -> None:
        """Set the global status."""

        self._global_status = status
        self.update()

    def set_status(self, name: str, status: str, details: str = "") -> None:
        """Set the status of a folder."""

        # Config file name
        if name.endswith("/config.yaml"):
            name = os.path.basename(os.path.dirname(name))
        self._status[name] = _Folder(status, details)

        if self.no_write:
            print(f"{name}: {status}")

        if not self.no_write:
            self.update()

    def update(self) -> None:
        """Update the status list."""

        names = []
        for config_file_name in glob.glob(
            os.path.join(os.environ.get("SCAN_SOURCE_FOLDER", "/source"), "*/config.yaml")
        ):
            name = os.path.basename(os.path.dirname(config_file_name))
            names.append(name)
            if name not in self._status:
                self._status[name] = _Folder("Not started", "")
        for name in self._status:  # pylint: disable=consider-using-dict-items
            if name not in names:
                del self._status[name]

        if not self.no_write:
            self.write()

    def write(self) -> None:
        """Write the status file."""

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
  <body>
    <h1>Scan to Paperless status</h1>
    <p>{self._global_status}</p>
    <p>Generated at: <script>
    window.document.write(new Date('{datetime.datetime.utcnow().replace(microsecond=0).isoformat()}Z').toLocaleString());
    </script></p>
    <h2>Jobs</h2>
    <table data-toggle="table">
      <thead>
        <tr>
          <th>Folder</th>
          <th>Status</th>
          <th>Details</th>
        </tr>
      </thead>
      <tbody>
"""
            )
            for name, folder in self._status.items():
                status_file.write(
                    f"""        <tr>
          <td data-sortable="true">{name}</td>
          <td data-sortable="true">{folder.status}</td>
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
