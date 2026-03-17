"""Manage the status file of the progress."""

import asyncio
import datetime
import html
import logging
import os
import pathlib
import traceback
from collections.abc import AsyncGenerator
from enum import Enum
from typing import Any, NamedTuple, cast

import anyio
import anyio.abc
import asyncinotify
import jinja2
from ruamel.yaml.main import YAML

from scan_to_paperless import process_schema

_LOGGER = logging.getLogger(__name__)

_CONFIG_FILENAME = "config.yaml"
_STATUS_FILENAME = "status.html"

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


async def _get_directories_recursive(path: anyio.Path) -> AsyncGenerator[anyio.Path, None]:
    """
    Recursively list all directories under path, Including path itself.

    The path itself is always yielded before its children are iterated, so you
    can pre-process a path (by watching it with inotify) before you get the
    directory listing.

    Passing a non-directory won't raise an error or anything, it'll just yield
    nothing.
    """
    if await path.is_dir():
        yield path
        async for child in path.iterdir():
            async for subdir in _get_directories_recursive(child):
                yield subdir


class Status:
    """Manage the status file of the progress."""

    _watch_scan_codes_task: asyncio.Task[None] | None = None
    _watch_destination_task: asyncio.Task[None] | None = None
    _watch_sources_task: asyncio.Task[None] | None = None
    _watch_scan_codes_debug_task: asyncio.Task[None] | None = None
    _watch_destination_debug_task: asyncio.Task[None] | None = None
    _watch_sources_debug_task: asyncio.Task[None] | None = None

    def __init__(self, no_write: bool = False) -> None:
        """Construct."""
        self.no_write = no_write
        self._file = anyio.Path(os.environ.get("SCAN_SOURCE_FOLDER", "/source")) / _STATUS_FILENAME
        self._status: dict[str, _Folder] = {}
        self._codes: list[str] = []
        self._consume: list[str] = []
        self._global_status = "Starting..."
        self._global_status_update = datetime.datetime.now(datetime.UTC).replace(microsecond=0)
        self._start_time = datetime.datetime.now(datetime.UTC).replace(microsecond=0)
        self._current_folder: str | None = None

        codes_folder = os.environ.get("SCAN_CODES_FOLDER", "/scan-codes")
        if codes_folder[-1] != "/":
            codes_folder += "/"
        self._codes_folder = anyio.Path(codes_folder)
        consume_folder = os.environ.get("SCAN_FINAL_FOLDER", "/destination")
        if consume_folder[-1] != "/":
            consume_folder += "/"
        self._consume_folder = anyio.Path(consume_folder)
        source_folder = os.environ.get("SCAN_SOURCE_FOLDER", "/source")
        if source_folder[-1] != "/":
            source_folder += "/"
        self._source_folder = anyio.Path(source_folder)

    async def set_global_status(self, status: str) -> None:
        """Set the global status."""
        if self._global_status != status:
            print(status)
            self._global_status = status
            self._global_status_update = datetime.datetime.now(datetime.UTC).replace(microsecond=0)

            await self.write()

    async def set_current_folder(self, path: anyio.Path | str | None) -> None:
        """Set the current folder."""
        if self._current_folder is not None:
            old_name = self._current_folder
            self._current_folder = None
            await self._update_status(old_name)
            await self.update_scan_codes()
            await self._update_consume()

        if path is None:
            await self.write()
            return

        if isinstance(path, anyio.Path):
            if path.name == _CONFIG_FILENAME:
                path = path.parent
            name = path.name
        else:
            name = path
        write = self._current_folder != name
        self._current_folder = name

        if write:
            await self.write()

    async def set_status(
        self,
        path: anyio.Path | str,
        nb_images: int,
        status: str,
        details: str = "",
        step: process_schema.Step | None = None,
        write: bool = False,
    ) -> None:
        """Set the status of a folder."""
        # Config file name
        if isinstance(path, anyio.Path):
            name = path.parent.name if path.name == _CONFIG_FILENAME else path.name
        else:
            name = path
        if nb_images <= 0 and name in self._status:
            nb_images = self._status[name].nb_images
        self._status[name] = _Folder(nb_images, html.escape(status), details, step)

        if self.no_write:
            print(f"{name}: {status}")

        if write:
            await self.write()

    async def init(self) -> None:
        """Scan for changes for waiting documents."""
        await self.update_scan_codes()
        await self._update_consume()
        await self.write()

        async for folder_path in self._source_folder.iterdir():
            if await folder_path.is_dir():
                name = folder_path.name
                await self._update_source_error(name)
                await self.write()

    async def _update_status(self, name: str) -> None:
        if not await (self._source_folder / name).exists():
            if name in self._status:
                del self._status[name]
            return

        if await (self._source_folder / name / "error.yaml").exists():
            yaml = YAML(typ="safe")
            async with await anyio.open_file(
                self._source_folder / name / "error.yaml",
                encoding="utf-8",
            ) as error_file:
                error = yaml.load(await error_file.read())

            await self.set_status(
                name,
                -1,
                "Error: " + error["error"],
                "<p>Stacktrace:</p><p><code>"
                + "<br />".join(error["traceback"])
                + f'</code></p><p>Remove the <a href="./{name}/error.yaml" target="_blank"><code>error.yaml</code></a> file to retry.</p>',
            )
            return

        if await (self._source_folder / name / "DONE").exists():
            await self.set_status(name, -1, _DONE_STATUS)
            return

        config = await self._get_config(name)
        if config is None:
            return

        run_step, rerun = await self._get_run_step(config)
        nb_images = len(run_step["sources"])

        if not await self._check_missing_sources(name, run_step):
            return

        # Determine if we should wait for next step or validation
        if (
            rerun
            or not await (self._source_folder / name / "REMOVE_TO_CONTINUE").exists()
            or len(config.get("steps", [])) == 0
        ):
            await self._set_waiting_status(name, run_step, nb_images)
        else:
            await self._set_validation_status(name, config, nb_images)

    async def _get_config(self, name: str) -> dict[str, Any] | None:
        """Load and validate configuration file."""
        config_path = self._source_folder / name / _CONFIG_FILENAME
        if not await config_path.exists():
            files = [
                str(f.relative_to(self._source_folder / name))
                async for f in (self._source_folder / name).rglob("*")
                if await f.is_file() and not f.name.startswith(".")
            ]
            files_html = [f'<a href="./{name}/{f}" target="_blank"><code>{f}</code></a>' for f in files]
            await self.set_status(name, -1, "Missing config", ", ".join(files_html))
            return None

        yaml = YAML(typ="safe")
        async with await anyio.open_file(config_path, encoding="utf-8") as config_file:
            config = cast("dict[str, Any]", yaml.load(await config_file.read()))

        if config is None:
            await self.set_status(name, -1, "Empty config")
            return None

        return config

    async def _get_run_step(self, config: dict[str, Any]) -> tuple[process_schema.Step, bool]:
        """Determine which step to run and if it's a rerun."""
        run_step: process_schema.Step | None = None
        rerun = False

        for step in reversed(config.get("steps", [])):
            all_present = True
            for source in step["sources"]:
                if not await anyio.Path(source).exists():
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

        return run_step, rerun

    async def _check_missing_sources(self, name: str, run_step: process_schema.Step) -> bool:
        """Check if all source images are present."""
        for source in run_step["sources"]:
            if not await (self._source_folder / name / source).exists():
                await self.set_status(
                    name,
                    len(run_step["sources"]),
                    "Waiting for sources",
                    f"Missing source image <code>{source}</code>",
                )
                return False
        return True

    async def _set_waiting_status(self, name: str, run_step: process_schema.Step, nb_images: int) -> None:
        """Set status when waiting for next step."""
        await self.set_status(
            name,
            nb_images,
            _WAITING_TO_STATUS.format(run_step["name"]),
            step=run_step,
        )

    async def _set_validation_status(self, name: str, config: dict[str, Any], nb_images: int) -> None:
        """Set status when waiting for user validation."""
        source_images = (
            config["steps"][-2]["sources"] if len(config.get("steps", [])) >= 2 else config["images"]
        )
        generated_images = [
            str(anyio.Path(f).relative_to(self._source_folder / name)) for f in config["steps"][-1]["sources"]
        ]

        description = (
            _WAITING_ASSISTED_SPLIT_DESCRIPTION
            if config["steps"][-1]["name"] == "split"
            else _WAITING_STATUS_DESCRIPTION
        ).format(
            name=name,
            source_images=", ".join(
                [f'<a href="./{name}/{f}" target="_blank"><code>{f}</code></a>' for f in source_images],
            ),
            generated_images=", ".join(
                [f'<a href="./{name}/{f}" target="_blank"><code>{f}</code></a>' for f in generated_images],
            ),
        )

        await self.set_status(
            name,
            nb_images,
            _WAITING_STATUS_NAME,
            description,
        )

    async def write(self) -> None:
        """Write the status file."""
        if self.no_write:
            return

        import natsort  # noqa: PLC0415, RUF100

        async with await self._file.open("w", encoding="utf-8") as status_file:
            env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(str(anyio.Path(__file__).parent)),
                autoescape=jinja2.select_autoescape(),
            )
            template = env.get_template(_STATUS_FILENAME)
            await status_file.write(
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
                ),
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

    async def update_scan_codes(self) -> None:
        """Update the list of files witch one we should scan the codes."""
        self._codes = [
            str(path.relative_to(self._codes_folder))
            async for path in self._codes_folder.rglob("*")
            if await path.is_file()
        ]

    async def _watch_scan_codes_debug(self) -> None:
        print(f"Start watching {self._codes_folder}")
        with asyncinotify.RecursiveInotify() as inotify:
            inotify.add_recursive_watch(
                pathlib.Path(self._codes_folder),
                asyncinotify.Mask.ATTRIB
                | asyncinotify.Mask.CLOSE_WRITE
                | asyncinotify.Mask.CREATE
                | asyncinotify.Mask.DELETE
                | asyncinotify.Mask.DELETE_SELF
                | asyncinotify.Mask.MODIFY
                | asyncinotify.Mask.MOVE
                | asyncinotify.Mask.MOVE_SELF,
            )
            async for event in inotify:
                print(f"Watch event on folder {self._codes_folder}: {event.path} - {event.mask!r}")

    async def _watch_scan_codes(self) -> None:
        with asyncinotify.RecursiveInotify() as inotify:
            inotify.add_recursive_watch(
                pathlib.Path(self._codes_folder),
                asyncinotify.Mask.CLOSE_WRITE | asyncinotify.Mask.DELETE | asyncinotify.Mask.MOVE,
            )
            async for _event in inotify:
                await self.update_scan_codes()
                await self.write()

    async def _update_consume(self) -> None:
        self._consume = [
            str(path.relative_to(self._consume_folder))
            async for path in self._consume_folder.rglob("*")
            if await path.is_file()
        ]

    async def _watch_destination_debug(self) -> None:
        print(f"Start watching {self._consume_folder}")
        with asyncinotify.RecursiveInotify() as inotify:
            inotify.add_recursive_watch(
                pathlib.Path(self._consume_folder),
                asyncinotify.Mask.ATTRIB
                | asyncinotify.Mask.CLOSE_WRITE
                | asyncinotify.Mask.CREATE
                | asyncinotify.Mask.DELETE
                | asyncinotify.Mask.DELETE_SELF
                | asyncinotify.Mask.MODIFY
                | asyncinotify.Mask.MOVE
                | asyncinotify.Mask.MOVE_SELF,
            )
            async for event in inotify:
                print(f"Watch event on folder {self._consume_folder}: {event.path} - {event.mask!r}")

    async def _watch_destination(self) -> None:
        with asyncinotify.RecursiveInotify() as inotify:
            inotify.add_recursive_watch(
                pathlib.Path(self._consume_folder),
                asyncinotify.Mask.CLOSE_WRITE | asyncinotify.Mask.DELETE | asyncinotify.Mask.MOVE,
            )
            async for _event in inotify:
                await self._update_consume()
                await self.write()

    async def _update_source_error(self, name: str) -> bool:
        if name != self._current_folder:
            if await (self._source_folder / name).is_dir():
                try:
                    await self._update_status(name)
                except Exception as exception:  # noqa: BLE001
                    trace = traceback.format_exc().split("\n")
                    await self.set_status(
                        name,
                        -1,
                        f"Error: {exception}",
                        f"<p>Stacktrace:</p><p><code>{'<br />'.join(trace)}</code></p>",
                    )
                return True
            if name in self._status:
                del self._status[name]
                return True
        return False

    async def _watch_sources_debug(self) -> None:
        print(f"Start watching {self._source_folder}")
        with asyncinotify.RecursiveInotify() as inotify:
            inotify.add_recursive_watch(
                pathlib.Path(self._source_folder),
                asyncinotify.Mask.ATTRIB
                | asyncinotify.Mask.CLOSE_WRITE
                | asyncinotify.Mask.CREATE
                | asyncinotify.Mask.DELETE
                | asyncinotify.Mask.DELETE_SELF
                | asyncinotify.Mask.MODIFY
                | asyncinotify.Mask.MOVE
                | asyncinotify.Mask.MOVE_SELF,
            )
            async for event in inotify:
                print(f"Watch event on folder {self._source_folder}: {event.path} - {event.mask!r}")

    async def _watch_sources(self) -> None:
        with asyncinotify.RecursiveInotify() as inotify:
            inotify.add_recursive_watch(
                pathlib.Path(self._source_folder),
                asyncinotify.Mask.CLOSE_WRITE | asyncinotify.Mask.DELETE | asyncinotify.Mask.MOVE,
            )
            async for event in inotify:
                if event.path is None:
                    continue
                path = event.path.relative_to(self._source_folder)
                if len(path.parents) > 1:
                    path = path.parents[-2]
                name = path.name
                if name == _STATUS_FILENAME:
                    continue
                if name.startswith("."):
                    continue
                print(f"Update source '{name}' from event")
                if await self._update_source_error(name):
                    await self.write()

    def start_watch(self) -> None:
        """Watch files changes to update status."""
        self._watch_scan_codes_task = asyncio.create_task(self._watch_scan_codes(), name="Watch scan codes")
        self._watch_destination_task = asyncio.create_task(
            self._watch_destination(),
            name="Watch destination",
        )
        self._watch_sources_task = asyncio.create_task(self._watch_sources(), name="Watch sources")
        if os.environ.get("DEBUG_INOTIFY", "FALSE").lower() in ("true", "1", "yes"):
            self._watch_scan_codes_debug_task = asyncio.create_task(
                self._watch_scan_codes_debug(),
                name="Watch scan codes debug",
            )
            self._watch_destination_debug_task = asyncio.create_task(
                self._watch_destination_debug(),
                name="Watch destination debug",
            )
            self._watch_sources_debug_task = asyncio.create_task(
                self._watch_sources_debug(),
                name="Watch sources debug",
            )
