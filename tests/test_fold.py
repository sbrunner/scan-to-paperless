# Copyright (c) 2026 Stéphane Brunner

import cv2
import numpy as np
from anyio import Path

from scan_to_paperless import process_utils

_FIXTURE = Path(__file__).parent / "fold-anonymized.png"
_PX_PER_MM = 9.45


def _load_fixture() -> np.ndarray:
    image = cv2.imread(str(_FIXTURE), cv2.IMREAD_UNCHANGED)
    assert image is not None
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    return image


def _page_mask(image: np.ndarray) -> np.ndarray:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[:, 27:2014] = 255
    return mask


def test_detect_fold() -> None:
    detection = process_utils.detect_fold(_load_fixture(), None, {}, px_per_mm=_PX_PER_MM)

    assert detection is not None
    assert 180 <= detection["start"] <= 205
    assert 215 <= detection["end"] <= 235
    assert detection["top"] <= 160
    assert detection["bottom"] >= 225
    assert detection["mean"] > 235


def test_detect_fold_rejects_printed_line() -> None:
    image = _load_fixture()
    printed = image.copy()
    printed[295:300, 100:1900] = 0

    detection = process_utils.detect_fold(printed, _page_mask(printed), {}, px_per_mm=_PX_PER_MM)

    assert detection is not None
    assert detection["start"] < 250


def test_detect_fold_without_fold() -> None:
    image = np.full((400, 1200, 3), 255, dtype=np.uint8)
    for y in range(80, 320, 60):
        cv2.putText(
            image,
            "Lorem ipsum dolor sit amet",
            (120, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
        )

    detection = process_utils.detect_fold(image, None, {}, px_per_mm=_PX_PER_MM)

    assert detection is None


def test_remove_fold_preserves_text() -> None:
    image = _load_fixture()
    detection = process_utils.detect_fold(image, None, {}, px_per_mm=_PX_PER_MM)
    assert detection is not None

    corrected = process_utils.remove_fold(image, detection, None, {}, px_per_mm=_PX_PER_MM)
    assert corrected.shape == image.shape

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corrected_gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    margins = np.concatenate([np.arange(66, 146), np.arange(1894, 1974)])
    profile = np.median(gray[:, margins], axis=1)
    band = slice(detection["top"], detection["bottom"])
    line_row = int(np.argmin(profile[band])) + detection["top"]
    line_region = slice(max(0, line_row - 2), line_row + 3)
    before = gray[line_region, margins].mean()
    after = corrected_gray[line_region, margins].mean()
    assert after > before + 5

    text_region = (slice(165, 210), slice(200, 420))
    text_before = np.count_nonzero(gray[text_region] < 100)
    text_after = np.count_nonzero(corrected_gray[text_region] < 100)
    assert text_after >= text_before * 0.9
