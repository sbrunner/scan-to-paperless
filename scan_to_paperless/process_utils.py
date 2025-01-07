"""Utility functions and context used in the process."""

import logging
import math
import os
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

import cv2
import numpy as np
from PIL import Image

import scan_to_paperless
import scan_to_paperless.jupyter_utils
import scan_to_paperless.status
from scan_to_paperless import process_schema as schema

if TYPE_CHECKING:
    NpNdarrayInt = np.ndarray[np.uint8, Any]
else:
    NpNdarrayInt = np.ndarray

_LOG = logging.getLogger(__name__)


def rotate_image(image: NpNdarrayInt, angle: float, background: int | tuple[int, int, int]) -> NpNdarrayInt:
    """Rotate the image."""
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center: tuple[Any, ...] = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cast(
        NpNdarrayInt,
        cv2.warpAffine(
            image,
            rot_mat,
            (int(round(height)), int(round(width))),
            borderValue=cast(Sequence[float], background),
        ),
    )


def crop_image(
    image: NpNdarrayInt,
    x: int,
    y: int,
    width: int,
    height: int,
    background: tuple[int] | tuple[int, int, int],
) -> NpNdarrayInt:
    """Crop the image."""
    matrix: NpNdarrayInt = np.array([[1.0, 0.0, -x], [0.0, 1.0, -y]])
    return cast(
        NpNdarrayInt,
        cv2.warpAffine(image, matrix, (int(round(width)), int(round(height))), borderValue=background),
    )


class Context:
    """All the context of the current image with his mask."""

    def __init__(
        self,
        config: schema.Configuration,
        step: schema.Step,
        config_file_name: str | None = None,
        root_folder: str | None = None,
        image_name: str | None = None,
    ) -> None:
        """Initialize."""
        self.config = config
        self.step = step
        self.config_file_name = config_file_name
        self.root_folder = root_folder
        self.image_name = image_name
        self.image: NpNdarrayInt | None = None
        self.mask: NpNdarrayInt | None = None
        self.get_index: Callable[
            [NpNdarrayInt], tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...] | None
        ] = lambda image: np.ix_(
            np.arange(0, image.shape[1]),
            np.arange(0, image.shape[1]),
            np.arange(0, image.shape[2]),
        )

        self.process_count = self.step.get("process_count", 0)

    def _get_default_mask_file(self, default_file_name: str) -> str:
        if not self.root_folder:
            return ""
        mask_file = os.path.join(self.root_folder, default_file_name)
        if not os.path.exists(mask_file):
            base_folder = os.path.dirname(self.root_folder)
            assert base_folder
            mask_file = os.path.join(base_folder, default_file_name)
            if not os.path.exists(mask_file):
                return ""
        return mask_file

    def _get_mask(
        self,
        auto_mask_config: schema.AutoMask | None,
        config_section: str,
        mask_file: str | None = None,
    ) -> NpNdarrayInt | None:
        """Init the mask."""
        if auto_mask_config is not None:
            assert self.image is not None
            hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

            lower_val = np.array(
                auto_mask_config.setdefault("lower_hsv_color", schema.LOWER_HSV_COLOR_DEFAULT)
            )
            upper_val = np.array(
                auto_mask_config.setdefault("upper_hsv_color", schema.UPPER_HSV_COLOR_DEFAULT)
            )
            mask = cv2.inRange(hsv, lower_val, upper_val)

            de_noise_size = auto_mask_config.setdefault("de_noise_size", schema.DE_NOISE_SIZE_DEFAULT)
            mask = cv2.copyMakeBorder(
                mask,
                de_noise_size,
                de_noise_size,
                de_noise_size,
                de_noise_size,
                cv2.BORDER_REPLICATE,
            )
            if auto_mask_config.setdefault("de_noise_morphology", schema.DE_NOISE_MORPHOLOGY_DEFAULT):
                mask = cv2.morphologyEx(
                    mask,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (de_noise_size, de_noise_size)),
                )
            else:
                blur = cv2.blur(
                    mask,
                    (de_noise_size, de_noise_size),
                )
                _, mask = cv2.threshold(
                    blur,
                    auto_mask_config.setdefault("de_noise_level", schema.DE_NOISE_LEVEL_DEFAULT),
                    255,
                    cv2.THRESH_BINARY,
                )

            inverse_mask = auto_mask_config.setdefault("inverse_mask", schema.INVERSE_MASK_DEFAULT)
            if not inverse_mask:
                mask = cv2.bitwise_not(mask)

            buffer_size = auto_mask_config.setdefault("buffer_size", schema.BUFFER_SIZE_DEFAULT)
            blur = cv2.blur(mask, (buffer_size, buffer_size))
            _, mask = cv2.threshold(
                blur,
                auto_mask_config.setdefault("buffer_level", schema.BUFFER_LEVEL_DEFAULT),
                255,
                cv2.THRESH_BINARY,
            )

            mask = mask[de_noise_size:-de_noise_size, de_noise_size:-de_noise_size]

            if self.root_folder and mask_file and os.path.exists(mask_file):
                mask = cv2.add(
                    mask,
                    cv2.bitwise_not(
                        cv2.resize(
                            cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE),
                            (mask.shape[1], mask.shape[0]),
                        )
                    ),
                )

            final_mask = cv2.bitwise_not(mask)

            if os.environ.get("PROGRESS", "FALSE") == "TRUE" and self.root_folder:
                self.save_progress_images(config_section.replace("_", "-"), final_mask)
        elif self.root_folder and mask_file:
            final_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if self.image is not None and final_mask is not None:
                return cast(NpNdarrayInt, cv2.resize(final_mask, (self.image.shape[1], self.image.shape[0])))
        return cast(NpNdarrayInt, final_mask)

    def init_mask(self) -> None:
        """Init the mask image used to mask the image on the crop and skew calculation."""
        mask_config = self.config["args"].setdefault(
            "mask", cast(schema.MaskOperation, schema.MASK_OPERATION_DEFAULT)
        )
        self.mask = (
            self._get_mask(
                mask_config.setdefault("auto_mask", {}),
                "mask",
                mask_config.setdefault("additional_filename", self._get_default_mask_file("mask.png")),
            )
            if mask_config.setdefault("enabled", schema.MASK_ENABLED_DEFAULT)
            else None
        )

    def get_background_color(self) -> tuple[int, int, int]:
        """Get the background color."""
        return cast(
            tuple[int, int, int],
            self.config["args"].setdefault("background_color", schema.BACKGROUND_COLOR_DEFAULT),
        )

    def do_initial_cut(self) -> None:
        """Definitively mask the original image."""
        cut_config = self.config["args"].get("cut", cast(schema.CutOperation, schema.CUT_OPERATION_DEFAULT))
        if cut_config.get("enabled", schema.CROP_ENABLED_DEFAULT):
            assert self.image is not None
            mask = self._get_mask(
                cut_config.setdefault("auto_mask", {}),
                "auto_cut",
                cut_config.setdefault("additional_filename", self._get_default_mask_file("cut.png")),
            )
            self.image[mask == 0] = self.get_background_color()

    def get_process_count(self) -> int:
        """Get the step number."""
        try:
            return self.process_count
        finally:
            self.process_count += 1

    def get_masked(self) -> NpNdarrayInt:
        """Get the mask."""
        if self.image is None:
            raise scan_to_paperless.ScanToPaperlessException("The image is None")
        if self.mask is None:
            return self.image.copy()

        image = self.image.copy()
        image[self.mask == 0] = self.get_background_color()
        return image

    def crop(self, x: int, y: int, width: int, height: int) -> None:
        """Crop the image."""
        if self.image is None:
            raise scan_to_paperless.ScanToPaperlessException("The image is None")
        self.image = crop_image(self.image, x, y, width, height, self.get_background_color())
        if self.mask is not None:
            self.mask = crop_image(self.mask, x, y, width, height, (0,))

    def rotate(self, angle: float) -> None:
        """Rotate the image."""
        if self.image is None:
            raise scan_to_paperless.ScanToPaperlessException("The image is None")
        self.image = rotate_image(self.image, angle, self.get_background_color())
        if self.mask is not None:
            self.mask = rotate_image(self.mask, angle, 0)

    def get_px_value(self, value: int | float) -> float:
        """Get the value in px."""
        return value / 10 / 2.51 * self.config["args"].setdefault("dpi", schema.DPI_DEFAULT)

    def is_progress(self) -> bool:
        """Return we want to have the intermediate files."""
        return os.environ.get("PROGRESS", "FALSE") == "TRUE" or self.config.setdefault(
            "progress", schema.PROGRESS_DEFAULT
        )

    def save_progress_images(
        self,
        name: str,
        image: NpNdarrayInt | None = None,
        image_prefix: str = "",
        process_count: int | None = None,
        force: bool = False,
    ) -> str | None:
        """Save the intermediate images."""
        if scan_to_paperless.jupyter_utils.is_ipython():
            if image is None:
                return None

            from IPython.display import (  # pylint: disable=import-outside-toplevel
                display,
            )

            display(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))  # type: ignore[no-untyped-call]
            return None

        if process_count is None:
            process_count = self.get_process_count()
        if (self.is_progress() or force) and self.image_name is not None and self.root_folder is not None:
            name = f"{process_count}-{name}" if self.is_progress() else name
            dest_folder = os.path.join(self.root_folder, name)
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            dest_image = os.path.join(dest_folder, image_prefix + self.image_name)
            if image is not None:
                try:
                    cv2.imwrite(dest_image, image)
                    return dest_image
                except Exception as exception:
                    print(exception)
            else:
                try:
                    assert self.image is not None
                    cv2.imwrite(dest_image, self.image)
                except Exception as exception:
                    print(exception)
                dest_image = os.path.join(dest_folder, "mask-" + self.image_name)
                try:
                    dest_image = os.path.join(dest_folder, "masked-" + self.image_name)
                except Exception as exception:
                    print(exception)
                try:
                    cv2.imwrite(dest_image, self.get_masked())
                except Exception as exception:
                    print(exception)
        return None

    def display_image(self, image: NpNdarrayInt) -> None:
        """Display the image."""
        if scan_to_paperless.jupyter_utils.is_ipython():
            from IPython.display import (  # pylint: disable=import-outside-toplevel
                display,
            )

            display(Image.fromarray(cv2.cvtColor(image[self.get_index(image)], cv2.COLOR_BGR2RGB)))  # type: ignore[no-untyped-call]
