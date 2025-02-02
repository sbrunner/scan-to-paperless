"""Manage the status file of the progress."""

import asyncio
import datetime
import glob
import html
import os.path
import traceback
from enum import Enum
from typing import NamedTuple

import asyncinotify
import jinja2
from ruamel.yaml.main import YAML

from scan_to_paperless import process_schema

_WAITING_STATUS_NAME = "Waiting validation"
_WAITING_STATUS_DESCRIPTION = """<div class="sidebar-box"><p>You should validate that the generate images are correct ({generated_images}).<br />
    If the result is correct remove the <a href="./{name}/REMOVE_TO_CONTINUE" target="_blank"><code>REMOVE_TO_CONTINUE</code></a> file.</p>
<p>If not you can:</p>
<ul>
    <li>Edit the generated image, then remove the <a href="./{name}/REMOVE_TO_CONTINUE" target="_blank"><code>REMOVE_TO_CONTINUE</code></a> file.</li>
    <li>Edit the <a href="./{name}/config.yaml" target="_blank"><code>config.yaml</code></a> file to change the parameters, then remove the generated files ({generated_images}) to force the regeneration.</li>
    <li>Edit the source images ({source_images}) then remove the corresponding generated files ({generated_images}) to force the regeneration.</li>
</ul>
<p>In the <a href="./{name}/source" target="_blank"><code>source</code></a> folder you can also find some images postfixed by <code>-skew-corrected</code> that the source image where the skew correction is applied.</p>
<p>In the <a href="./{name}/histogram" target="_blank"><code>histogram</code></a> folder yu can find tow histogram of the source images, one of them use a logarithm scale.</p>
<p>In the <a href="./{name}/crop" target="_blank"><code>crop</code></a> folder you can find the images with the detected block used by the automatic crop.</p>
<p>In the <a href="./{name}/skew" target="_blank"><code>skew</code></a> folder you can find some images that represent the the shew detection will be based on.</p>
<p>In the <a href="./{name}/jupyter" target="_blank"><code>jupyter</code></a> folder you can find a Jupyter notebook that can help you to optimize the configuration.</p>
<p class="read-more"><a href="javascript:void(0)" class="button">Read More</a></p></div>"""
_WAITING_ASSISTED_SPLIT_DESCRIPTION = """<div class="sidebar-box"><p>You are in assisted split mode, in the step where you should choose should the splitting,</p>
<p>For that you should open the <a href="./{name}/config.yaml" target="_blank"><code>config.yaml</code></a> file,
in the section <code>assisted_split</code> you can find the list of the images that will be split, with a configuration like that.</p>
<pre><code>assisted_split:
- source: /source/z-563313-h/assisted-split/image-1.png
  destinations:
  - 4
  - 1
  limits:
  - name: VC
    type: image center
    value: 1777
    vertical: true
    margin: 0
</code></pre>
<ul>
<li><code>source</code> is the path (on the server) of the image to split</li>
<li><code>destinations</code> is the list of the destination pages of the split images</li>
<li><code>limits</code> is the list of the limits used to split the image</li>
<li><code>name</code> is the name of the auto detected limit that we can found on the generated image on the <a href="./{name}/" target="_blank">base</a> folder</li>
<li><code>type</code> is the type of detection</li>
<li><code>value</code> is the position where we will to split</li>
<li><code>vertical</code> is a boolean that indicate if the limit is vertical or horizontal</li>
<li><code>margin</code> is the margin of the limit, if it set we will "lost" the part of the image around the split position</li>
</ul>
<p>For each image we will have all the detected limits, you can choose the limit that you want to use for the split (you can modify it if needed), and remove the other entries.</p>
<p>When you have finished to choose the limits, you should save the edited <code>config.yaml</code> file, then remove the <a href="./{name}/REMOVE_TO_CONTINUE" target="_blank"><code>REMOVE_TO_CONTINUE</code></a> file to continue the process.</p>
<p class="read-more"><a href="javascript:void(0)" class="button">Read More</a></p></div>"""
_WAITING_TO_STATUS = "Waiting to {}"
STATUS_TRANSFORM = "transform"
STATUS_ASSISTED_SPLIT = "split"
STATUS_FINALIZE = "finalize"
_WAITING_TO_TRANSFORM_STATUS = _WAITING_TO_STATUS.format(STATUS_TRANSFORM)
_WAITING_TO_ASSISTED_SPLIT_STATUS = _WAITING_TO_STATUS.format(STATUS_ASSISTED_SPLIT)
_DONE_STATUS = "Done"
_WAITING_TO_FINALIZE_STATUS = _WAITING_TO_STATUS.format(STATUS_FINALIZE)


class _Folder(NamedTuple):
    nb_images: int
    status: str
    details: str
    step: process_schema.Step | None


class JobType(Enum):
    """The type of job."""

    NONE = "None"
    TRANSFORM = "transform"
    ASSISTED_SPLIT = "assisted-split"
    FINALIZE = "finalize"
    DOWN = "down"
    CODE = "code"


class Status:
    """Manage the status file of the progress."""

    def __init__(self, no_write: bool = False) -> None:
        """Construct."""
        self.no_write = no_write
        self._file = os.path.join(os.environ.get("SCAN_SOURCE_FOLDER", "/source"), "status.html")
        self._status: dict[str, _Folder] = {}
        self._codes: list[str] = []
        self._consume: list[str] = []
        self._global_status = "Starting..."
        self._global_status_update = datetime.datetime.utcnow().replace(microsecond=0)
        self._start_time = datetime.datetime.utcnow().replace(microsecond=0)
        self._current_folder: str | None = None

        self._codes_folder = os.environ.get("SCAN_CODES_FOLDER", "/scan-codes")
        if self._codes_folder[-1] != "/":
            self._codes_folder += "/"
        self._consume_folder = os.environ.get("SCAN_FINAL_FOLDER", "/destination")
        if self._consume_folder[-1] != "/":
            self._consume_folder += "/"
        self._source_folder = os.environ.get("SCAN_SOURCE_FOLDER", "/source")

        self._init()

    def set_global_status(self, status: str) -> None:
        """Set the global status."""
        if self._global_status != status:
            print(status)
            self._global_status = status
            self._global_status_update = datetime.datetime.utcnow().replace(microsecond=0)

            self.write()

    def set_current_folder(self, name: str | None) -> None:
        """Set the current folder."""
        if name is None:
            if self._current_folder is not None:
                old_name = self._current_folder
                self._current_folder = None
                self._update_source_error(old_name)
                self.write()
            return

        if name.endswith("/config.yaml"):
            name = os.path.basename(os.path.dirname(name))

        write = self._current_folder != name
        self._current_folder = name

        if write:
            self.write()

    def set_status(
        self,
        name: str,
        nb_images: int,
        status: str,
        details: str = "",
        step: process_schema.Step | None = None,
        write: bool = False,
    ) -> None:
        """Set the status of a folder."""
        # Config file name
        if name.endswith("/config.yaml"):
            name = os.path.basename(os.path.dirname(name))
        if nb_images <= 0 and name in self._status:
            nb_images = self._status[name].nb_images
        self._status[name] = _Folder(nb_images, html.escape(status), details, step)

        if self.no_write:
            print(f"{name}: {status}")

        if write:
            self.write()

    def _init(self) -> None:
        """Scan for changes for waiting documents."""
        self.update_scan_codes()
        self.write()
        self._update_consume()
        self.write()

        for folder_name in glob.glob(os.path.join(self._source_folder, "*")):
            if os.path.isdir(folder_name):
                name = os.path.basename(folder_name)
                self._update_source_error(name)
                self.write()

    def _update_status(self, name: str) -> None:
        yaml = YAML(typ="safe")
        if os.path.exists(os.path.join(self._source_folder, name, "error.yaml")):
            with open(
                os.path.join(self._source_folder, name, "error.yaml"),
                encoding="utf-8",
            ) as error_file:
                error = yaml.load(error_file)

            self.set_status(
                name,
                -1,
                "Error: " + error["error"],
                "<p>Stacktrace:</p><p><code>"
                + "<br />".join(error["traceback"])
                + f'</code></p><p>Remove the <a href="./{name}/error.yaml" target="_blank"><code>error.yaml</code></a> file to retry.</p>',
            )
            return

        if os.path.exists(os.path.join(self._source_folder, name, "DONE")):
            self.set_status(name, -1, _DONE_STATUS)
            return

        if not os.path.exists(os.path.join(self._source_folder, name, "config.yaml")):
            len_folder = len(os.path.join(self._source_folder, name).rstrip("/")) + 1
            files = [
                f[len_folder:]
                for f in glob.glob(
                    os.path.join(self._source_folder, name, "**"),
                    recursive=True,
                )
                if os.path.isfile(f)
            ]
            files = [f'<a href="./{name}/{f}" target="_blank"><code>{f}</code></a>' for f in files]
            self.set_status(name, -1, "Missing config", ", ".join(files))
            return

        with open(
            os.path.join(self._source_folder, name, "config.yaml"),
            encoding="utf-8",
        ) as config_file:
            config = yaml.load(config_file)

        if config is None:
            self.set_status(name, -1, "Empty config")
            return

        run_step: process_schema.Step | None = None
        rerun = False
        for step in reversed(config.get("steps", [])):
            all_present = True
            for source in step["sources"]:
                if not os.path.exists(source):
                    rerun = True
                    all_present = False
                    break
            if all_present:
                run_step = step
                break
        if run_step is None:
            run_step = {
                "sources": config["images"],
                "name": "transform",
            }
        nb_images = len(run_step["sources"])

        if (
            rerun
            or not os.path.exists(os.path.join(self._source_folder, name, "REMOVE_TO_CONTINUE"))
            or len(config.get("steps", [])) == 0
        ):
            self.set_status(name, nb_images, _WAITING_TO_STATUS.format(run_step["name"]), step=run_step)
        else:
            len_folder = len(os.path.join(self._source_folder, name).rstrip("/")) + 1
            source_images = (
                config["steps"][-2]["sources"] if len(config.get("steps", [])) >= 2 else config["images"]
            )
            generated_images = [f[len_folder:] for f in config["steps"][-1]["sources"]]
            self.set_status(
                name,
                nb_images,
                _WAITING_STATUS_NAME,
                (
                    _WAITING_ASSISTED_SPLIT_DESCRIPTION
                    if config["steps"][-1]["name"] == "split"
                    else _WAITING_STATUS_DESCRIPTION
                ).format(
                    name=name,
                    source_images=", ".join(
                        [
                            f'<a href="./{name}/{f}" target="_blank"><code>{f}</code></a>'
                            for f in source_images
                        ]
                    ),
                    generated_images=", ".join(
                        [
                            f'<a href="./{name}/{f}" target="_blank"><code>{f}</code></a>'
                            for f in generated_images
                        ]
                    ),
                ),
            )

    def write(self) -> None:
        """Write the status file."""
        if self.no_write:
            return

        import natsort  # pylint: disable=import-outside-toplevel

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
                    current_folder=self._current_folder,
                    start_time=self._start_time,
                    datetime=datetime,
                    status=self._status,
                    sorted_status_key=natsort.natsorted(self._status.keys()),
                    codes=self._codes,
                    consume=self._consume,
                )
            )

    def get_next_job(self) -> tuple[str | None, JobType, process_schema.Step | None]:
        """Get the next job to do."""
        job_types = [
            (JobType.TRANSFORM, _WAITING_TO_TRANSFORM_STATUS),
            (JobType.ASSISTED_SPLIT, _WAITING_TO_ASSISTED_SPLIT_STATUS),
            (JobType.FINALIZE, _WAITING_TO_FINALIZE_STATUS),
        ]
        if os.environ.get("PROGRESS", "FALSE") != "TRUE":
            job_types.append((JobType.DOWN, _DONE_STATUS))

        for job_type, waiting_status in job_types:
            for name, folder in self._status.items():
                if folder.status == waiting_status:
                    return name, job_type, folder.step

        if self._codes:
            return self._codes[0], JobType.CODE, None

        return None, JobType.NONE, None

    def update_scan_codes(self) -> None:
        """Update the list of files witch one we should scan the codes."""
        self._codes = [
            f[len(self._codes_folder) :]
            for f in glob.glob(os.path.join(self._codes_folder, "**"), recursive=True)
            if os.path.isfile(f)
        ]

    async def _watch_scan_codes_debug(self) -> None:
        print(f"Start watching {self._codes_folder}")
        with asyncinotify.Inotify() as inotify:
            inotify.add_watch(
                self._codes_folder,
                asyncinotify.Mask.ACCESS
                | asyncinotify.Mask.ATTRIB
                | asyncinotify.Mask.CLOSE
                | asyncinotify.Mask.CREATE
                | asyncinotify.Mask.DELETE
                | asyncinotify.Mask.DELETE_SELF
                | asyncinotify.Mask.MODIFY
                | asyncinotify.Mask.MOVE
                | asyncinotify.Mask.MOVE_SELF
                | asyncinotify.Mask.OPEN,
            )
            async for event in inotify:
                print(f"Watch event on folder {self._codes_folder}: {event.path} - {repr(event.mask)}")

    async def _watch_scan_codes(self) -> None:
        with asyncinotify.Inotify() as inotify:
            inotify.add_watch(
                self._codes_folder,
                asyncinotify.Mask.CLOSE_WRITE | asyncinotify.Mask.DELETE | asyncinotify.Mask.MOVE,
            )
            async for _ in inotify:
                self.update_scan_codes()
                self.write()

    def _update_consume(self) -> None:
        self._consume = [
            f[len(self._consume_folder) :]
            for f in glob.glob(os.path.join(self._consume_folder, "**"), recursive=True)
            if os.path.isfile(f)
        ]

    async def _watch_destination_debug(self) -> None:
        print(f"Start watching {self._consume_folder}")
        with asyncinotify.Inotify() as inotify:
            inotify.add_watch(
                self._consume_folder,
                asyncinotify.Mask.ACCESS
                | asyncinotify.Mask.ATTRIB
                | asyncinotify.Mask.CLOSE
                | asyncinotify.Mask.CREATE
                | asyncinotify.Mask.DELETE
                | asyncinotify.Mask.DELETE_SELF
                | asyncinotify.Mask.MODIFY
                | asyncinotify.Mask.MOVE
                | asyncinotify.Mask.MOVE_SELF
                | asyncinotify.Mask.OPEN,
            )
            async for event in inotify:
                print(f"Watch event on folder {self._consume_folder}: {event.path} - {repr(event.mask)}")

    async def _watch_destination(self) -> None:
        with asyncinotify.Inotify() as inotify:
            inotify.add_watch(
                self._consume_folder,
                asyncinotify.Mask.CLOSE_WRITE | asyncinotify.Mask.DELETE | asyncinotify.Mask.MOVE,
            )
            async for _ in inotify:
                self._update_consume()
                self.write()

    def _update_source_error(self, name: str) -> bool:
        if name != self._current_folder:
            if os.path.isdir(os.path.join(self._source_folder, name)):
                try:
                    self._update_status(name)
                except Exception as exception:
                    trace = traceback.format_exc().split("\n")
                    self.set_status(
                        name,
                        -1,
                        f"Error: {exception}",
                        f"<p>Stacktrace:</p><p><code>{'<br />'.join(trace)}</code></p>",
                    )
                return True
            else:
                if name in self._status:
                    del self._status[name]
                    return True
        return False

    async def _watch_sources_debug(self) -> None:
        print(f"Start watching {self._source_folder}")
        with asyncinotify.Inotify() as inotify:
            inotify.add_watch(
                self._source_folder,
                asyncinotify.Mask.ACCESS
                | asyncinotify.Mask.ATTRIB
                | asyncinotify.Mask.CLOSE
                | asyncinotify.Mask.CREATE
                | asyncinotify.Mask.DELETE
                | asyncinotify.Mask.DELETE_SELF
                | asyncinotify.Mask.MODIFY
                | asyncinotify.Mask.MOVE
                | asyncinotify.Mask.MOVE_SELF
                | asyncinotify.Mask.OPEN,
            )
            async for event in inotify:
                print(f"Watch event on folder {self._source_folder}: {event.path} - {repr(event.mask)}")

    async def _watch_sources(self) -> None:
        with asyncinotify.Inotify() as inotify:
            length = len(self._source_folder.rstrip("/")) + 1
            inotify.add_watch(
                self._source_folder,
                asyncinotify.Mask.CLOSE_WRITE | asyncinotify.Mask.DELETE | asyncinotify.Mask.MOVE,
            )
            async for event in inotify:
                name = str(event.path)[length:].split("/", 1)[0]
                if name == "status.html":
                    continue
                print(f"Update source '{name}' from event")
                if self._update_source_error(name):
                    self.write()

    def start_watch(self) -> None:
        """Watch files changes to update status."""
        asyncio.create_task(self._watch_scan_codes(), name="Watch scan codes")
        asyncio.create_task(self._watch_destination(), name="Watch destination")
        asyncio.create_task(self._watch_sources(), name="Watch sources")
        if os.environ.get("DEBUG_INOTIFY", "FALSE") == "TRUE":
            asyncio.create_task(self._watch_scan_codes_debug(), name="Watch scan codes debug")
            asyncio.create_task(self._watch_destination_debug(), name="Watch destination debug")
            asyncio.create_task(self._watch_sources_debug(), name="Watch sources debug")
