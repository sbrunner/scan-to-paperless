"""Tests for scan.py module."""

import os
import tempfile
from unittest.mock import patch

from scan_to_paperless.scan import available_presets


def test_available_presets_empty():
    """Test available_presets returns empty list when no presets exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("scan_to_paperless.scan.CONFIG_FOLDER", tmpdir):
            with patch("scan_to_paperless.scan.CONFIG_FILENAME", "scan-to-paperless.yaml"):
                result = available_presets()
                assert result == []


def test_available_presets_with_files():
    """Test available_presets returns list of preset names."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test preset files
        open(os.path.join(tmpdir, "scan-to-paperless-test.yaml"), "w").close()
        open(os.path.join(tmpdir, "scan-to-paperless-prod.yaml"), "w").close()
        open(os.path.join(tmpdir, "scan-to-paperless.yaml"), "w").close()  # Should be ignored
        open(os.path.join(tmpdir, "other-file.yaml"), "w").close()  # Should be ignored
        
        with patch("scan_to_paperless.scan.CONFIG_FOLDER", tmpdir):
            with patch("scan_to_paperless.scan.CONFIG_FILENAME", "scan-to-paperless.yaml"):
                result = available_presets()
                assert sorted(result) == ["prod", "test"]


def test_available_presets_missing_folder():
    """Test available_presets handles missing config folder gracefully."""
    with patch("scan_to_paperless.scan.CONFIG_FOLDER", "/nonexistent/folder"):
        with patch("scan_to_paperless.scan.CONFIG_FILENAME", "scan-to-paperless.yaml"):
            result = available_presets()
            assert result == []
