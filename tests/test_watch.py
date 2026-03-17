"""Integration tests for Status filesystem watching via asyncinotify."""

import asyncio
import io
import os
import sys
import tempfile

import anyio
import pytest
from ruamel.yaml.main import YAML

from scan_to_paperless import status

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="asyncinotify only works on Linux")


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing and configure env vars."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source_dir = anyio.Path(tmpdir) / "source"
        codes_dir = anyio.Path(tmpdir) / "codes"
        consume_dir = anyio.Path(tmpdir) / "consume"

        os.makedirs(str(source_dir))
        os.makedirs(str(codes_dir))
        os.makedirs(str(consume_dir))

        os.environ["SCAN_SOURCE_FOLDER"] = str(source_dir)
        os.environ["SCAN_CODES_FOLDER"] = str(codes_dir)
        os.environ["SCAN_FINAL_FOLDER"] = str(consume_dir)

        yield {
            "source": source_dir,
            "codes": codes_dir,
            "consume": consume_dir,
        }


async def _wait_for_condition(condition_fn, timeout=5.0, interval=0.05):
    """Poll until condition_fn() returns True or the timeout expires."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if condition_fn():
            return True
        await asyncio.sleep(interval)
    return False


async def _cancel_watch_tasks(status_obj):
    """Cancel all watch tasks created by start_watch()."""
    tasks = [
        status_obj._watch_scan_codes_task,
        status_obj._watch_destination_task,
        status_obj._watch_sources_task,
    ]
    for task in tasks:
        if task is not None:
            task.cancel()
    await asyncio.gather(*[t for t in tasks if t is not None], return_exceptions=True)


@pytest.mark.asyncio
async def test_watch_scan_codes_detects_new_file(temp_dirs):
    """When a file is written to the codes folder, _codes is updated."""
    status_obj = status.Status(no_write=True)
    status_obj.start_watch()
    # Let watch tasks initialize their inotify watches.
    await asyncio.sleep(0.1)

    try:
        test_file = anyio.Path(temp_dirs["codes"]) / "test_code.txt"
        async with await anyio.open_file(test_file, "w") as f:
            await f.write("test content")

        result = await _wait_for_condition(lambda: len(status_obj._codes) > 0)

        assert result, "_codes was not updated within the timeout after file creation"
        assert "test_code.txt" in status_obj._codes
    finally:
        await _cancel_watch_tasks(status_obj)


@pytest.mark.asyncio
async def test_watch_destination_detects_new_file(temp_dirs):
    """When a file is written to the consume folder, _consume is updated."""
    status_obj = status.Status(no_write=True)
    status_obj.start_watch()
    await asyncio.sleep(0.1)

    try:
        test_file = anyio.Path(temp_dirs["consume"]) / "document.pdf"
        async with await anyio.open_file(test_file, "w") as f:
            await f.write("pdf content")

        result = await _wait_for_condition(lambda: len(status_obj._consume) > 0)

        assert result, "_consume was not updated within the timeout after file creation"
        assert "document.pdf" in status_obj._consume
    finally:
        await _cancel_watch_tasks(status_obj)


@pytest.mark.asyncio
async def test_watch_sources_detects_new_file_and_updates_status(temp_dirs):
    """When a file is written into a source sub-folder, _status is updated for that folder."""
    source_dir = anyio.Path(temp_dirs["source"])
    folder_path = source_dir / "testjob"
    await folder_path.mkdir(parents=True)

    # Create a minimal valid config with a source image.
    config = {"images": ["image1.png"]}
    yaml = YAML()
    buf = io.StringIO()
    yaml.dump(config, buf)
    async with await anyio.open_file(folder_path / "config.yaml", "w", encoding="utf-8") as f:
        await f.write(buf.getvalue())
    async with await anyio.open_file(folder_path / "image1.png", "w") as f:
        await f.write("dummy image")

    status_obj = status.Status(no_write=True)
    status_obj.start_watch()
    await asyncio.sleep(0.1)

    try:
        # Write any new file inside the job folder to trigger a CLOSE_WRITE event.
        async with await anyio.open_file(folder_path / "trigger.txt", "w") as f:
            await f.write("trigger")

        result = await _wait_for_condition(lambda: "testjob" in status_obj._status)

        assert result, "_status was not updated for 'testjob' within the timeout"
        assert status_obj._status["testjob"].status == status._WAITING_TO_TRANSFORM_STATUS
    finally:
        await _cancel_watch_tasks(status_obj)


@pytest.mark.asyncio
async def test_watch_sources_detects_file_deletion_and_updates_status(temp_dirs):
    """When a source image is deleted, the folder status changes to 'Waiting for sources'."""
    source_dir = anyio.Path(temp_dirs["source"])
    folder_path = source_dir / "deletejob"
    await folder_path.mkdir(parents=True)

    config = {"images": ["image1.png"]}
    yaml = YAML()
    buf = io.StringIO()
    yaml.dump(config, buf)
    async with await anyio.open_file(folder_path / "config.yaml", "w", encoding="utf-8") as f:
        await f.write(buf.getvalue())
    async with await anyio.open_file(folder_path / "image1.png", "w") as f:
        await f.write("dummy image")

    status_obj = status.Status(no_write=True)
    status_obj.start_watch()
    await asyncio.sleep(0.1)

    try:
        # Delete the source image to trigger a DELETE event.
        await (folder_path / "image1.png").unlink()

        result = await _wait_for_condition(
            lambda: (
                status_obj._status.get("deletejob", None) is not None
                and status_obj._status["deletejob"].status == "Waiting for sources"
            )
        )

        assert result, "_status for 'deletejob' was not updated to 'Waiting for sources' within the timeout"
        assert status_obj._status["deletejob"].status == "Waiting for sources"
    finally:
        await _cancel_watch_tasks(status_obj)
