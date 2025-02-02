"""Manage the status file of the progress."""

import asyncio
import datetime
import glob
import html
import os.path
import traceback
from collections.abc import AsyncGenerator, Generator
from enum import Enum
from pathlib import Path
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


def _get_directories_recursive(path: Path) -> Generator[Path, None, None]:
    """
    Recursively list all directories under path, Including path itself.

    The path itself is always yielded before its children are iterated, so you
    can pre-process a path (by watching it with inotify) before you get the
    directory listing.

    Passing a non-directory won't raise an error or anything, it'll just yield
    nothing.
    """
    if path.is_dir():
        yield path
        for child in path.iterdir():
            yield from _get_directories_recursive(child)


async def _watch_recursive(path: Path, mask: asyncinotify.Mask) -> AsyncGenerator[asyncinotify.Event, None]:
    watchers: dict[Path, asyncinotify.Watch] = {}
    used_mask = (
        mask
        | asyncinotify.Mask.MOVED_FROM
        | asyncinotify.Mask.MOVED_TO
        | asyncinotify.Mask.CREATE
        | asyncinotify.Mask.DELETE_SELF
        | asyncinotify.Mask.IGNORED
    )
    with asyncinotify.Inotify() as inotify:
        for directory in _get_directories_recursive(path):
            inotify.add_watch(directory, used_mask)

        # Things that can throw this off:
        #
        # * Moving a watched directory out of the watch tree (will still
        #   generate events even when outside of directory tree)
        #
        # * Doing two changes on a directory or something before the program
        #   has a time to handle it (this will also throw off a lot of inotify
        #   code, though)
        #
        # * Moving a watched directory within a watched directory will get the
        #   wrong path.  This needs to use the cookie system to link events
        #   together and complete the move properly, which can still make some
        #   events get the wrong path if you get file events during the move or
        #   something silly like that, since MOVED_FROM and MOVED_TO aren't
        #   guaranteed to be contiguous.  That exercise is left up to the
        #   reader.
        #
        # * Trying to watch a path that doesn't exist won't automatically
        #   create it or anything of the sort.
        #
        # * Deleting and recreating or moving the watched directory won't do
        #   anything special, but it probably should.
        async for event in inotify:
            # Add subdirectories to watch if a new directory is added.  We do
            # this recursively here before processing events to make sure we
            # have complete coverage of existing and newly-created directories
            # by watching before recursing and adding, since we know
            # get_directories_recursive is depth-first and yields every
            # directory before iterating their children, we know we won't miss
            # anything.
            if asyncinotify.Mask.CREATE in event.mask and event.path is not None and event.path.is_dir():
                for directory in _get_directories_recursive(event.path):
                    if directory in watchers:
                        continue
                    print(f"EVENT: Watching {directory}")
                    watchers[directory] = inotify.add_watch(directory, used_mask)
            elif asyncinotify.Mask.DELETE_SELF in event.mask:
                if event.path in watchers:
                    print(f"EVENT: Removing watch (delete self) {event.path}")
                    inotify.rm_watch(watchers[event.path])
                    del watchers[event.path]
            elif asyncinotify.Mask.IGNORED in event.mask:
                if event.path in watchers:
                    print(f"EVENT: Removing watch (ignored) {event.path}")
                    inotify.rm_watch(watchers[event.path])
                    del watchers[event.path]
            elif asyncinotify.Mask.MOVED_FROM in event.mask:
                if event.path in watchers:
                    print(f"EVENT: Removing watch (moved from) {event.path}")
                    inotify.rm_watch(watchers[event.path])
                    del watchers[event.path]
            elif asyncinotify.Mask.MOVED_TO in event.mask and event.path not in watchers:
                if event.path is None or event.path in watchers:
                    continue
                print(f"EVENT: Watching (moved to) {event.path}")
                watchers[event.path] = inotify.add_watch(event.path, used_mask)

            # If there is at least some overlap, assume the user wants this event.
            if event.mask & mask:
                yield event


class Status:
    """Manage the status file of the progress."""

    def __init__(self, no_write: bool = False) -> None:
        """Construct."""
        self.no_write = no_write
        self._file = Path(os.environ.get("SCAN_SOURCE_FOLDER", "/source")) / "status.html"
        self._status: dict[Path, _Folder] = {}
        self._codes: list[Path] = []
        self._consume: list[Path] = []
        self._global_status = "Starting..."
        self._global_status_update = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
        self._start_time = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
        self._current_folder: Path | None = None

        codes_folder = os.environ.get("SCAN_CODES_FOLDER", "/scan-codes")
        if codes_folder[-1] != "/":
            codes_folder += "/"
        self._codes_folder = Path(codes_folder)
        consume_folder = os.environ.get("SCAN_FINAL_FOLDER", "/destination")
        if consume_folder[-1] != "/":
            consume_folder += "/"
        self._consume_folder = Path(consume_folder)
        source_folder = os.environ.get("SCAN_SOURCE_FOLDER", "/source")
        if source_folder[-1] != "/":
            source_folder += "/"
        self._source_folder = Path(source_folder)

        self._init()

    def set_global_status(self, status: str) -> None:
        """Set the global status."""
        if self._global_status != status:
            print(status)
            self._global_status = status
            self._global_status_update = datetime.datetime.utcnow().replace(microsecond=0)

            self.write()

    def set_current_folder(self, name: Path | None) -> None:
        """Set the current folder."""
        if name is None:
            if self._current_folder is not None:
                old_name = self._current_folder
                self._current_folder = None
                self._update_source_error(old_name)
                self.write()
            return

        if name.name == "config.yaml":
            name = name.parent
            if len(name.parents) >= 1:
                name = name.parents[-2]

        write = self._current_folder != name
        self._current_folder = name

        if write:
            self.write()

    def set_status(
        self,
        name: Path,
        nb_images: int,
        status: str,
        details: str = "",
        step: process_schema.Step | None = None,
        write: bool = False,
    ) -> None:
        """Set the status of a folder."""
        # Config file name
        if name.name == "config.yaml":
            name = name.parent
            if len(name.parents) >= 1:
                name = name.parents[-2]
        if nb_images <= 0 and name in self._status:
            nb_images = self._status[name].nb_images
        self._status[Path(name)] = _Folder(nb_images, html.escape(status), details, step)

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

        for folder_name in glob.glob(str(self._source_folder / "*")):
            if os.path.isdir(folder_name):
                name = os.path.basename(folder_name)
                self._update_source_error(Path(name))
                self.write()

    def _update_status(self, name: Path) -> None:
        yaml = YAML(typ="safe")
        if (self._source_folder / name / "error.yaml").exists():
            with open(
                self._source_folder / name / "error.yaml",
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

        if (self._source_folder / name / "DONE").exists():
            self.set_status(name, -1, _DONE_STATUS)
            return

        if not (self._source_folder / name / "config.yaml").exists():
            len_folder = len(str(self._source_folder / name)) + 1
            files = [
                f[len_folder:]
                for f in glob.glob(
                    str(self._source_folder / name / "**"),
                    recursive=True,
                )
                if os.path.isfile(f)
            ]
            files = [f'<a href="./{name}/{f}" target="_blank"><code>{f}</code></a>' for f in files]
            self.set_status(name, -1, "Missing config", ", ".join(files))
            return

        with open(
            self._source_folder / name / "config.yaml",
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
            or not (self._source_folder / name / "REMOVE_TO_CONTINUE").exists()
            or len(config.get("steps", [])) == 0
        ):
            self.set_status(name, nb_images, _WAITING_TO_STATUS.format(run_step["name"]), step=run_step)
        else:
            source_images = (
                config["steps"][-2]["sources"] if len(config.get("steps", [])) >= 2 else config["images"]
            )
            generated_images = [
                str(Path(f).relative_to(self._source_folder / name)) for f in config["steps"][-1]["sources"]
            ]
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

    def get_next_job(self) -> tuple[Path | None, JobType, process_schema.Step | None]:
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
            Path(f).relative_to(self._codes_folder)
            for f in glob.glob(str(self._codes_folder / "**"), recursive=True)
            if os.path.isfile(f)
        ]

    async def _watch_scan_codes_debug(self) -> None:
        print(f"Start watching {self._codes_folder}")
        async for event in _watch_recursive(
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
        ):
            print(f"Watch event on folder {self._codes_folder}: {event.path} - {repr(event.mask)}")

    async def _watch_scan_codes(self) -> None:
        async for _event in _watch_recursive(
            self._codes_folder,
            asyncinotify.Mask.CLOSE_WRITE | asyncinotify.Mask.DELETE | asyncinotify.Mask.MOVE,
        ):
            self.update_scan_codes()
            self.write()

    def _update_consume(self) -> None:
        self._consume = [
            Path(f).relative_to(self._consume_folder)
            for f in glob.glob(str(self._consume_folder / "**"), recursive=True)
            if os.path.isfile(f)
        ]

    async def _watch_destination_debug(self) -> None:
        print(f"Start watching {self._consume_folder}")
        async for event in _watch_recursive(
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
        ):
            print(f"Watch event on folder {self._consume_folder}: {event.path} - {repr(event.mask)}")

    async def _watch_destination(self) -> None:
        async for _event in _watch_recursive(
            self._consume_folder,
            asyncinotify.Mask.CLOSE_WRITE | asyncinotify.Mask.DELETE | asyncinotify.Mask.MOVE,
        ):
            self._update_consume()
            self.write()

    def _update_source_error(self, name: Path) -> bool:
        if name != self._current_folder:
            if (self._source_folder / name).is_dir():
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
        async for event in _watch_recursive(
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
        ):
            print(f"Watch event on folder {self._source_folder}: {event.path} - {repr(event.mask)}")

    async def _watch_sources(self) -> None:
        async for event in _watch_recursive(
            self._source_folder,
            asyncinotify.Mask.CLOSE_WRITE | asyncinotify.Mask.DELETE | asyncinotify.Mask.MOVE,
        ):
            if event.path is None:
                continue
            name = event.path.relative_to(self._source_folder)
            if len(name.parents) > 2:
                name = name.parents[-2]
            if name == Path("status.html"):
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
