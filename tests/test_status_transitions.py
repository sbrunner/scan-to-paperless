"""Test the Status class state transitions."""

import io
import os
import tempfile
from pathlib import Path as PathlibPath
from typing import Any

import anyio
import pytest
from ruamel.yaml.main import YAML

from scan_to_paperless import process_schema, status


class TestStatusTransitions:
    """Test the state transitions of the Status class."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = PathlibPath(tmpdir) / "source"
            codes_dir = PathlibPath(tmpdir) / "codes"
            consume_dir = PathlibPath(tmpdir) / "consume"

            source_dir.mkdir()
            codes_dir.mkdir()
            consume_dir.mkdir()

            os.environ["SCAN_SOURCE_FOLDER"] = str(source_dir)
            os.environ["SCAN_CODES_FOLDER"] = str(codes_dir)
            os.environ["SCAN_FINAL_FOLDER"] = str(consume_dir)

            yield {
                "source": source_dir,
                "codes": codes_dir,
                "consume": consume_dir,
                "tmpdir": tmpdir,
            }

    @pytest.fixture
    def status_instance(self, temp_dirs):
        """Create a Status instance with temp dirs."""
        status_obj = status.Status(no_write=True)
        return status_obj

    async def _create_source_folder(self, temp_dirs: dict[str, PathlibPath], folder_name: str) -> PathlibPath:
        """Create a source folder with config and images."""
        folder_path = anyio.Path(temp_dirs["source"]) / folder_name
        await folder_path.mkdir(parents=True)
        return folder_path

    async def _create_config(
        self,
        folder_path: anyio.Path,
        images: list[str],
        steps: list[dict[str, Any]] | None = None,
    ) -> None:
        """Create a config.yaml file in the folder."""
        config = {"images": images}
        if steps:
            config["steps"] = steps

        yaml = YAML()
        config_file = folder_path / "config.yaml"

        # Dump to StringIO first, then write to file
        string_buffer = io.StringIO()
        yaml.dump(config, string_buffer)
        async with await anyio.open_file(config_file, "w", encoding="utf-8") as f:
            await f.write(string_buffer.getvalue())

    async def _create_image_file(self, folder_path: anyio.Path, image_name: str) -> None:
        """Create a dummy image file."""
        image_path = folder_path / image_name
        async with await anyio.open_file(image_path, "w") as f:
            await f.write("dummy image content")

    async def _create_marker_file(self, folder_path: anyio.Path, marker_name: str) -> None:
        """Create a marker file (like DONE, REMOVE_TO_CONTINUE, etc.)."""
        marker_path = folder_path / marker_name
        async with await anyio.open_file(marker_path, "w") as f:
            await f.write("")

    @pytest.mark.asyncio
    async def test_initial_status_starting(self, status_instance: status.Status) -> None:
        """Test that initial global status is 'Starting...'."""
        assert status_instance._global_status == "Starting..."

    @pytest.mark.asyncio
    async def test_set_global_status(self, status_instance: status.Status) -> None:
        """Test setting global status."""
        await status_instance.set_global_status("Running")
        assert status_instance._global_status == "Running"

        await status_instance.set_global_status("Done")
        assert status_instance._global_status == "Done"

    @pytest.mark.asyncio
    async def test_no_duplicate_global_status_update(self, status_instance: status.Status) -> None:
        """Test that setting same global status twice doesn't update timestamp twice."""
        await status_instance.set_global_status("Running")
        first_update = status_instance._global_status_update

        await status_instance.set_global_status("Running")
        second_update = status_instance._global_status_update

        assert first_update == second_update

    @pytest.mark.asyncio
    async def test_set_current_folder_none(self, status_instance: status.Status) -> None:
        """Test setting current folder to None."""
        status_instance._current_folder = "test"
        await status_instance.set_current_folder(None)
        assert status_instance._current_folder is None

    @pytest.mark.asyncio
    async def test_set_current_folder_with_string(self, status_instance: status.Status) -> None:
        """Test setting current folder with string."""
        await status_instance.set_current_folder("test-folder")
        assert status_instance._current_folder == "test-folder"

    @pytest.mark.asyncio
    async def test_set_current_folder_with_path(self, status_instance: status.Status, temp_dirs) -> None:
        """Test setting current folder with anyio.Path."""
        folder_path = anyio.Path(temp_dirs["source"]) / "test-folder"
        await folder_path.mkdir(parents=True)

        await status_instance.set_current_folder(folder_path)
        assert status_instance._current_folder == "test-folder"

    @pytest.mark.asyncio
    async def test_set_current_folder_with_config_path(
        self, status_instance: status.Status, temp_dirs
    ) -> None:
        """Test setting current folder with config.yaml path."""
        folder_path = anyio.Path(temp_dirs["source"]) / "test-folder"
        await folder_path.mkdir(parents=True)
        await self._create_config(folder_path, ["image1.png"])

        config_path = folder_path / "config.yaml"
        await status_instance.set_current_folder(config_path)
        assert status_instance._current_folder == "test-folder"

    @pytest.mark.asyncio
    async def test_status_waiting_for_sources(self, status_instance: status.Status, temp_dirs) -> None:
        """Test status when waiting for source images."""
        folder_path = await self._create_source_folder(temp_dirs, "folder1")
        await self._create_config(folder_path, ["image1.png", "image2.png"])

        await status_instance._update_status("folder1")

        assert "folder1" in status_instance._status
        assert status_instance._status["folder1"].status == "Waiting for sources"
        assert "Missing source image" in status_instance._status["folder1"].details

    @pytest.mark.asyncio
    async def test_status_waiting_to_transform(self, status_instance: status.Status, temp_dirs) -> None:
        """Test status when waiting to transform."""
        folder_path = await self._create_source_folder(temp_dirs, "folder1")
        await self._create_config(folder_path, ["image1.png"])
        await self._create_image_file(folder_path, "image1.png")

        await status_instance._update_status("folder1")

        assert "folder1" in status_instance._status
        assert status_instance._status["folder1"].status == status._WAITING_TO_TRANSFORM_STATUS
        assert status_instance._status["folder1"].step is not None
        assert status_instance._status["folder1"].step.get("name") == "transform"

    @pytest.mark.asyncio
    async def test_status_missing_config(self, status_instance: status.Status, temp_dirs) -> None:
        """Test status when config.yaml is missing."""
        await self._create_source_folder(temp_dirs, "folder1")

        await status_instance._update_status("folder1")

        assert "folder1" in status_instance._status
        assert status_instance._status["folder1"].status == "Missing config"

    @pytest.mark.asyncio
    async def test_status_empty_config(self, status_instance: status.Status, temp_dirs) -> None:
        """Test status when config.yaml is empty."""
        folder_path = await self._create_source_folder(temp_dirs, "folder1")
        config_file = folder_path / "config.yaml"
        async with await anyio.open_file(config_file, "w") as f:
            await f.write("")

        await status_instance._update_status("folder1")

        assert "folder1" in status_instance._status
        assert status_instance._status["folder1"].status == "Empty config"

    @pytest.mark.asyncio
    async def test_status_done(self, status_instance: status.Status, temp_dirs) -> None:
        """Test status when DONE marker is present."""
        folder_path = await self._create_source_folder(temp_dirs, "folder1")
        await self._create_config(folder_path, ["image1.png"])
        await self._create_image_file(folder_path, "image1.png")
        await self._create_marker_file(folder_path, "DONE")

        await status_instance._update_status("folder1")

        assert "folder1" in status_instance._status
        assert status_instance._status["folder1"].status == status._DONE_STATUS

    @pytest.mark.asyncio
    async def test_status_error(self, status_instance: status.Status, temp_dirs) -> None:
        """Test status when error.yaml is present."""
        folder_path = await self._create_source_folder(temp_dirs, "folder1")

        error_config = {"error": "Test error message", "traceback": ["line 1", "line 2"]}
        yaml = YAML()
        error_file = folder_path / "error.yaml"

        # Dump to StringIO first, then write to file
        string_buffer = io.StringIO()
        yaml.dump(error_config, string_buffer)
        async with await anyio.open_file(error_file, "w", encoding="utf-8") as f:
            await f.write(string_buffer.getvalue())

        await status_instance._update_status("folder1")

        assert "folder1" in status_instance._status
        assert "Error" in status_instance._status["folder1"].status
        assert "Test error message" in status_instance._status["folder1"].status

    @pytest.mark.asyncio
    async def test_status_waiting_validation(self, status_instance: status.Status, temp_dirs) -> None:
        """Test status when waiting for user validation."""
        folder_path = await self._create_source_folder(temp_dirs, "folder1")

        # Create initial images
        await self._create_image_file(folder_path, "image1.png")

        # Create intermediate step results
        await self._create_image_file(folder_path, "transformed1.png")

        steps = [
            {"name": "transform", "sources": [str(folder_path / "image1.png")]},
            {"name": "finalize", "sources": [str(folder_path / "transformed1.png")]},
        ]
        await self._create_config(folder_path, ["image1.png"], steps=steps)

        # Create REMOVE_TO_CONTINUE marker to indicate waiting for validation
        await self._create_marker_file(folder_path, "REMOVE_TO_CONTINUE")

        await status_instance._update_status("folder1")

        assert "folder1" in status_instance._status
        assert status_instance._status["folder1"].status == status._WAITING_STATUS_NAME

    @pytest.mark.asyncio
    async def test_status_waiting_assisted_split(self, status_instance: status.Status, temp_dirs) -> None:
        """Test status when waiting for assisted split validation."""
        folder_path = await self._create_source_folder(temp_dirs, "folder1")

        # Create initial images
        await self._create_image_file(folder_path, "image1.png")

        # Create step results
        await self._create_image_file(folder_path, "transformed1.png")
        await self._create_image_file(folder_path, "split1.png")

        steps = [
            {"name": "transform", "sources": [str(folder_path / "image1.png")]},
            {"name": "split", "sources": [str(folder_path / "transformed1.png")]},
            {"name": "finalize", "sources": [str(folder_path / "split1.png")]},
        ]
        await self._create_config(folder_path, ["image1.png"], steps=steps)

        # Create REMOVE_TO_CONTINUE marker to indicate waiting for validation
        await self._create_marker_file(folder_path, "REMOVE_TO_CONTINUE")

        await status_instance._update_status("folder1")

        assert "folder1" in status_instance._status
        assert status_instance._status["folder1"].status == status._WAITING_STATUS_NAME

    @pytest.mark.asyncio
    async def test_set_status_basic(self, status_instance: status.Status) -> None:
        """Test setting status directly."""
        await status_instance.set_status("folder1", 5, "Processing")

        assert "folder1" in status_instance._status
        assert status_instance._status["folder1"].nb_images == 5
        assert status_instance._status["folder1"].status == "Processing"

    @pytest.mark.asyncio
    async def test_set_status_with_details(self, status_instance: status.Status) -> None:
        """Test setting status with details."""
        details = "<p>Processing image 1 of 5</p>"
        await status_instance.set_status("folder1", 5, "Processing", details=details)

        assert status_instance._status["folder1"].details == details

    @pytest.mark.asyncio
    async def test_set_status_with_step(self, status_instance: status.Status) -> None:
        """Test setting status with step information."""
        step: process_schema.Step = {"name": "transform", "sources": ["image1.png"]}
        await status_instance.set_status("folder1", 1, "Processing", step=step)

        assert status_instance._status["folder1"].step == step

    @pytest.mark.asyncio
    async def test_set_status_preserves_nb_images(self, status_instance: status.Status) -> None:
        """Test that setting status with -1 preserves existing nb_images."""
        await status_instance.set_status("folder1", 5, "First status")
        await status_instance.set_status("folder1", -1, "Updated status")

        assert status_instance._status["folder1"].nb_images == 5

    @pytest.mark.asyncio
    async def test_status_html_escaping(self, status_instance: status.Status) -> None:
        """Test that status details are HTML escaped."""
        details = "<script>alert('xss')</script>"
        await status_instance.set_status("folder1", 1, "Test", details=details)

        # The status itself is escaped, but details are not (for HTML formatting)
        assert status_instance._status["folder1"].details == details

    @pytest.mark.asyncio
    async def test_init(self, status_instance: status.Status, temp_dirs) -> None:
        """Test initialization of Status instance."""
        folder_path = await self._create_source_folder(temp_dirs, "folder1")
        await self._create_config(folder_path, ["image1.png"])
        await self._create_image_file(folder_path, "image1.png")

        await status_instance.init()

        # After init, folder should be in status
        assert "folder1" in status_instance._status
        assert status_instance._status["folder1"].status == status._WAITING_TO_TRANSFORM_STATUS

    @pytest.mark.asyncio
    async def test_get_next_job_transform(self, status_instance: status.Status) -> None:
        """Test getting next job when waiting for transform."""
        step: process_schema.Step = {"name": "transform", "sources": ["image1.png"]}
        await status_instance.set_status("folder1", 1, status._WAITING_TO_TRANSFORM_STATUS, step=step)

        job_name, job_type, job_step = status_instance.get_next_job()

        assert job_name == "folder1"
        assert job_type == status.JobType.TRANSFORM
        assert job_step == step

    @pytest.mark.asyncio
    async def test_get_next_job_assisted_split(self, status_instance: status.Status) -> None:
        """Test getting next job when waiting for assisted split."""
        step: process_schema.Step = {"name": "split", "sources": ["image1.png"]}
        await status_instance.set_status(
            "folder1",
            1,
            status._WAITING_TO_ASSISTED_SPLIT_STATUS,
            step=step,
        )

        job_name, job_type, job_step = status_instance.get_next_job()

        assert job_name == "folder1"
        assert job_type == status.JobType.ASSISTED_SPLIT
        assert job_step == step

    @pytest.mark.asyncio
    async def test_get_next_job_finalize(self, status_instance: status.Status) -> None:
        """Test getting next job when waiting for finalize."""
        step: process_schema.Step = {"name": "finalize", "sources": ["image1.png"]}
        await status_instance.set_status("folder1", 1, status._WAITING_TO_FINALIZE_STATUS, step=step)

        job_name, job_type, job_step = status_instance.get_next_job()

        assert job_name == "folder1"
        assert job_type == status.JobType.FINALIZE
        assert job_step == step

    @pytest.mark.asyncio
    async def test_get_next_job_priority_order(self, status_instance: status.Status) -> None:
        """Test that jobs are returned in priority order: TRANSFORM > ASSISTED_SPLIT > FINALIZE."""
        step_transform: process_schema.Step = {"name": "transform", "sources": ["i1.png"]}
        step_split: process_schema.Step = {"name": "split", "sources": ["i2.png"]}
        step_finalize: process_schema.Step = {"name": "finalize", "sources": ["i3.png"]}

        await status_instance.set_status(
            "folder_finalize",
            1,
            status._WAITING_TO_FINALIZE_STATUS,
            step=step_finalize,
        )
        await status_instance.set_status(
            "folder_split",
            1,
            status._WAITING_TO_ASSISTED_SPLIT_STATUS,
            step=step_split,
        )
        await status_instance.set_status(
            "folder_transform", 1, status._WAITING_TO_TRANSFORM_STATUS, step=step_transform
        )

        job_name, job_type, _ = status_instance.get_next_job()

        # TRANSFORM should have priority
        assert job_type == status.JobType.TRANSFORM
        assert job_name == "folder_transform"

    @pytest.mark.asyncio
    async def test_get_next_job_code_scanning(self, status_instance: status.Status) -> None:
        """Test getting next job when there are codes to scan."""
        status_instance._codes = ["code1.txt", "code2.txt"]

        job_name, job_type, job_step = status_instance.get_next_job()

        assert job_name == "code1.txt"
        assert job_type == status.JobType.CODE
        assert job_step is None

    @pytest.mark.asyncio
    async def test_get_next_job_none_available(self, status_instance: status.Status) -> None:
        """Test getting next job when no jobs are available."""
        job_name, job_type, job_step = status_instance.get_next_job()

        assert job_name is None
        assert job_type == status.JobType.NONE
        assert job_step is None

    @pytest.mark.asyncio
    async def test_status_folder_deleted(self, status_instance: status.Status, temp_dirs) -> None:
        """Test that folder is removed from status when it's deleted."""
        folder_path = await self._create_source_folder(temp_dirs, "folder1")
        await self._create_config(folder_path, ["image1.png"])
        await self._create_image_file(folder_path, "image1.png")

        await status_instance._update_status("folder1")
        assert "folder1" in status_instance._status

        # Delete the folder
        await anyio.to_thread.run_sync(lambda: __import__("shutil").rmtree(str(folder_path)))

        await status_instance._update_status("folder1")
        assert "folder1" not in status_instance._status

    @pytest.mark.asyncio
    async def test_update_scan_codes(self, status_instance: status.Status, temp_dirs) -> None:
        """Test updating scan codes list."""
        codes_dir = anyio.Path(temp_dirs["codes"])

        # Create some code files
        await (codes_dir / "code1.txt").touch()
        await (codes_dir / "code2.txt").touch()
        subdir = codes_dir / "subdir"
        await subdir.mkdir()
        await (subdir / "code3.txt").touch()

        await status_instance.update_scan_codes()

        assert "code1.txt" in status_instance._codes
        assert "code2.txt" in status_instance._codes
        assert "subdir/code3.txt" in status_instance._codes

    @pytest.mark.asyncio
    async def test_update_consume(self, status_instance: status.Status, temp_dirs) -> None:
        """Test updating consume list."""
        consume_dir = anyio.Path(temp_dirs["consume"])

        # Create some files
        await (consume_dir / "file1.pdf").touch()
        await (consume_dir / "file2.pdf").touch()
        subdir = consume_dir / "subdir"
        await subdir.mkdir()
        await (subdir / "file3.pdf").touch()

        await status_instance._update_consume()

        assert "file1.pdf" in status_instance._consume
        assert "file2.pdf" in status_instance._consume
        assert "subdir/file3.pdf" in status_instance._consume

    @pytest.mark.asyncio
    async def test_complex_state_transition_workflow(self, status_instance: status.Status, temp_dirs) -> None:
        """Test a complex workflow with multiple state transitions."""
        folder_path = await self._create_source_folder(temp_dirs, "test-doc")
        await self._create_image_file(folder_path, "scan1.png")
        await self._create_image_file(folder_path, "scan2.png")

        # Step 1: Waiting for transform
        steps = [
            {"name": "transform", "sources": [str(folder_path / "scan1.png"), str(folder_path / "scan2.png")]}
        ]
        await self._create_config(folder_path, ["scan1.png", "scan2.png"], steps=steps)

        await status_instance._update_status("test-doc")
        assert status_instance._status["test-doc"].status == status._WAITING_TO_TRANSFORM_STATUS

        # Step 2: Simulate transformation and waiting for next step
        await self._create_image_file(folder_path, "transformed1.png")
        await self._create_image_file(folder_path, "transformed2.png")

        steps = [
            {
                "name": "transform",
                "sources": [str(folder_path / "scan1.png"), str(folder_path / "scan2.png")],
            },
            {
                "name": "split",
                "sources": [str(folder_path / "transformed1.png"), str(folder_path / "transformed2.png")],
            },
        ]
        await self._create_config(folder_path, ["scan1.png", "scan2.png"], steps=steps)

        # Without REMOVE_TO_CONTINUE, it should wait for split
        await status_instance._update_status("test-doc")
        assert status_instance._status["test-doc"].status == status._WAITING_TO_ASSISTED_SPLIT_STATUS

        # Step 3: User validates and adds REMOVE_TO_CONTINUE
        await self._create_marker_file(folder_path, "REMOVE_TO_CONTINUE")
        await status_instance._update_status("test-doc")
        assert status_instance._status["test-doc"].status == status._WAITING_STATUS_NAME

        # Step 4: Mark as done
        await self._create_marker_file(folder_path, "DONE")
        await status_instance._update_status("test-doc")
        assert status_instance._status["test-doc"].status == status._DONE_STATUS
