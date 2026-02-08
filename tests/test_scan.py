"""Tests for the scan module."""

from unittest.mock import AsyncMock, patch

import pytest

from scan_to_paperless.scan import _Mode, scan


def test_scan_invokes_async_logic():
    """Test that scan() properly invokes _scan_async via asyncio.run."""
    # Arrange
    mode = _Mode.ADF
    preset = "test-preset"
    append_credit_card = True
    assisted_split = False

    # Mock _scan_async to avoid actual scanning
    with patch("scan_to_paperless.scan._scan_async", new_callable=AsyncMock) as mock_scan_async:
        # Act
        scan(
            mode=mode,
            preset=preset,
            append_credit_card=append_credit_card,
            assisted_split=assisted_split,
        )

        # Assert
        mock_scan_async.assert_called_once_with(mode, preset, append_credit_card, assisted_split)


def test_scan_invokes_async_logic_with_defaults():
    """Test that scan() properly invokes _scan_async with default arguments."""
    # Mock _scan_async to avoid actual scanning
    with patch("scan_to_paperless.scan._scan_async", new_callable=AsyncMock) as mock_scan_async:
        # Act - call with all defaults
        scan()

        # Assert - verify defaults are passed correctly
        mock_scan_async.assert_called_once_with(_Mode.ADF, None, False, False)


def test_scan_does_not_return_unawaited_coroutine():
    """Test that scan() does not accidentally return an unawaited coroutine."""
    # Mock _scan_async to return a simple coroutine
    with patch("scan_to_paperless.scan._scan_async", new_callable=AsyncMock) as mock_scan_async:
        # Act
        result = scan()

        # Assert - should return None, not a coroutine
        assert result is None
        mock_scan_async.assert_called_once()
