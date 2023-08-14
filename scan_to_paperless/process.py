#!/usr/bin/env python3

"""Process the scanned documents."""

import argparse
import datetime
import json
import logging
import math
import os
import re
import shutil
import subprocess  # nosec
import sys
import tempfile
import time
import traceback
from typing import IO, TYPE_CHECKING, Any, Optional, Protocol, TypedDict, Union, cast

# read, write, rotate, crop, sharpen, draw_line, find_line, find_contour
import cv2
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pikepdf
import ruamel.yaml.compat
from deskew import determine_skew_debug_images
from PIL import Image, ImageDraw, ImageFont
from ruamel.yaml.main import YAML
from skimage.color import rgb2gray, rgba2rgb
from skimage.exposure import histogram as skimage_histogram
from skimage.metrics import structural_similarity
from skimage.util import img_as_ubyte

import scan_to_paperless
import scan_to_paperless.status
from scan_to_paperless import code
from scan_to_paperless import process_schema as schema

if TYPE_CHECKING:
    NpNdarrayInt = np.ndarray[np.uint8, Any]
    CompletedProcess = subprocess.CompletedProcess[str]
else:
    NpNdarrayInt = np.ndarray
    CompletedProcess = subprocess.CompletedProcess

# dither, crop, append, repage
CONVERT = ["gm", "convert"]
_LOG = logging.getLogger(__name__)


class ScanToPaperlessException(Exception):
    """Base exception for this module."""


def rotate_image(
    image: NpNdarrayInt, angle: float, background: Union[int, tuple[int, int, int]]
) -> NpNdarrayInt:
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
        cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background),
    )


def crop_image(  # pylint: disable=too-many-arguments
    image: NpNdarrayInt,
    x: int,
    y: int,
    width: int,
    height: int,
    background: Union[tuple[int], tuple[int, int, int]],
) -> NpNdarrayInt:
    """Crop the image."""

    matrice: NpNdarrayInt = np.array([[1.0, 0.0, -x], [0.0, 1.0, -y]])
    return cast(
        NpNdarrayInt,
        cv2.warpAffine(image, matrice, (int(round(width)), int(round(height))), borderValue=background),
    )


class Context:  # pylint: disable=too-many-instance-attributes
    """All the context of the current image with his mask."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        config: schema.Configuration,
        step: schema.Step,
        config_file_name: Optional[str] = None,
        root_folder: Optional[str] = None,
        image_name: Optional[str] = None,
    ) -> None:
        """Initialize."""

        self.config = config
        self.step = step
        self.config_file_name = config_file_name
        self.root_folder = root_folder
        self.image_name = image_name
        self.image: Optional[NpNdarrayInt] = None
        self.mask: Optional[NpNdarrayInt] = None
        self.index: Optional[tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...]] = None
        self.process_count = self.step.get("process_count", 0)

    def _get_mask(
        self,
        auto_mask_config: Optional[schema.AutoMask],
        config_section: str,
        default_file_name: str,
    ) -> Optional[NpNdarrayInt]:
        """Init the mask."""

        if auto_mask_config is not None:
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

            if self.root_folder:
                mask_file: str = os.path.join(self.root_folder, default_file_name)
                assert mask_file
                if not os.path.exists(mask_file):
                    base_folder = os.path.dirname(self.root_folder)
                    assert base_folder
                    mask_file = os.path.join(base_folder, default_file_name)
                    if not os.path.exists(mask_file):
                        mask_file = ""
                mask_file = auto_mask_config.setdefault("additional_filename", mask_file)
                if mask_file and os.path.exists(mask_file):
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
        elif self.root_folder:
            mask_file = os.path.join(self.root_folder, default_file_name)
            if not os.path.exists(mask_file):
                base_folder = os.path.dirname(self.root_folder)
                assert base_folder
                mask_file = os.path.join(base_folder, default_file_name)
                if not os.path.exists(mask_file):
                    return None

            final_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if self.image is not None and final_mask is not None:
                return cast(NpNdarrayInt, cv2.resize(final_mask, (self.image.shape[1], self.image.shape[0])))
        return cast(NpNdarrayInt, final_mask)

    def init_mask(self) -> None:
        """Init the mask image used to mask the image on the crop and skew calculation."""

        auto_mask_config = self.config["args"].setdefault(
            "auto_mask", cast(schema.AutoMaskOperation, schema.AUTO_MASK_OPERATION_DEFAULT)
        )
        self.mask = (
            self._get_mask(
                auto_mask_config.setdefault("auto_mask", {}),
                "auto_mask",
                "mask.png",
            )
            if auto_mask_config.setdefault("enabled", schema.AUTO_MASK_ENABLED_DEFAULT)
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

        if "auto_cut" in self.config["args"]:
            assert self.image is not None
            mask = self._get_mask(
                self.config["args"]
                .setdefault("auto_cut", cast(schema.AutoCut, schema.AUTO_CUT_DEFAULT))
                .setdefault("auto_mask", {}),
                "auto_cut",
                "cut.png",
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
            raise ScanToPaperlessException("The image is None")
        if self.mask is None:
            return self.image.copy()

        image = self.image.copy()
        image[self.mask == 0] = self.get_background_color()
        return image

    def crop(self, x: int, y: int, width: int, height: int) -> None:
        """Crop the image."""

        if self.image is None:
            raise ScanToPaperlessException("The image is None")
        self.image = crop_image(self.image, x, y, width, height, self.get_background_color())
        if self.mask is not None:
            self.mask = crop_image(self.mask, x, y, width, height, (0,))

    def rotate(self, angle: float) -> None:
        """Rotate the image."""

        if self.image is None:
            raise ScanToPaperlessException("The image is None")
        self.image = rotate_image(self.image, angle, self.get_background_color())
        if self.mask is not None:
            self.mask = rotate_image(self.mask, angle, 0)

    def get_px_value(self, value: Union[int, float]) -> float:
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
        image: Optional[NpNdarrayInt] = None,
        image_prefix: str = "",
        process_count: Optional[int] = None,
        force: bool = False,
    ) -> Optional[str]:
        """Save the intermediate images."""

        if _is_ipython():
            if image is None:
                return None

            from IPython.display import display  # pylint: disable=import-outside-toplevel,import-error

            display(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
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


def add_intermediate_error(
    config: schema.Configuration,
    config_file_name: Optional[str],
    error: Exception,
    traceback_: list[str],
) -> None:
    """Add in the config non fatal error."""

    if config_file_name is None:
        raise ScanToPaperlessException("The config file name is required") from error
    if "intermediate_error" not in config:
        config["intermediate_error"] = []

    old_intermediate_error: list[schema.IntermediateError] = []
    old_intermediate_error.extend(config["intermediate_error"])
    yaml = YAML()
    yaml.default_flow_style = False
    try:
        config["intermediate_error"].append({"error": str(error), "traceback": traceback_})
        with open(config_file_name + "_", "w", encoding="utf-8") as config_file:
            yaml.dump(config, config_file)
    except Exception as exception:
        print(exception)
        config["intermediate_error"] = old_intermediate_error
        config["intermediate_error"].append({"error": str(error), "traceback": traceback_})
        with open(config_file_name + "_", "w", encoding="utf-8") as config_file:
            yaml.dump(config, config_file)
    os.rename(config_file_name + "_", config_file_name)


def call(cmd: Union[str, list[str]], **kwargs: Any) -> None:
    """Verbose version of check_output with no returns."""

    if isinstance(cmd, list):
        cmd = [str(element) for element in cmd]
    print(" ".join(cmd) if isinstance(cmd, list) else cmd)
    sys.stdout.flush()
    kwargs.setdefault("check", True)
    subprocess.run(  # nosec # pylint: disable=subprocess-run-check
        cmd,
        capture_output=True,
        **kwargs,
    )


def run(cmd: Union[str, list[str]], **kwargs: Any) -> CompletedProcess:
    """Verbose version of check_output with no returns."""

    if isinstance(cmd, list):
        cmd = [str(element) for element in cmd]
    print(" ".join(cmd) if isinstance(cmd, list) else cmd)
    sys.stdout.flush()
    return subprocess.run(cmd, stderr=subprocess.PIPE, check=True, **kwargs)  # nosec


def output(cmd: Union[str, list[str]], **kwargs: Any) -> str:
    """Verbose version of check_output."""

    if isinstance(cmd, list):
        cmd = [str(element) for element in cmd]
    print(" ".join(cmd) if isinstance(cmd, list) else cmd)
    sys.stdout.flush()
    return cast(bytes, subprocess.check_output(cmd, stderr=subprocess.PIPE, **kwargs)).decode()  # nosec


def image_diff(image1: NpNdarrayInt, image2: NpNdarrayInt) -> tuple[float, NpNdarrayInt]:
    """Do a diff between images."""

    width = max(image1.shape[1], image2.shape[1])
    height = max(image1.shape[0], image2.shape[0])
    image1 = cv2.resize(image1, (width, height))
    image2 = cv2.resize(image2, (width, height))

    image1 = image1 if len(image1.shape) == 2 else cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = image2 if len(image2.shape) == 2 else cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    score, diff = structural_similarity(image1, image2, full=True)
    diff = (255 - diff * 255).astype("uint8")
    return score, diff


class FunctionWithContextReturnsImage(Protocol):
    """Function with context and returns an image."""

    def __call__(self, context: Context) -> Optional[NpNdarrayInt]:
        """Call the function."""


class FunctionWithContextReturnsNone(Protocol):
    """Function with context and no return."""

    def __call__(self, context: Context) -> None:
        """Call the function."""


class ExternalFunction(Protocol):
    """Function that call an external tool."""

    def __call__(self, context: Context, source: str, destination: str) -> None:
        """Call the function."""


# Decorate a step of the transform
class Process:  # pylint: disable=too-few-public-methods
    """
    Encapsulate a transform function.

    To save the process image when needed.
    """

    def __init__(self, name: str, ignore_error: bool = False, progress: bool = True) -> None:
        """Initialize."""
        self.name = name
        self.ignore_error = ignore_error
        self.progress = progress

    def __call__(self, func: FunctionWithContextReturnsImage) -> FunctionWithContextReturnsNone:
        """Call the function."""

        def wrapper(context: Context) -> None:
            start_time = time.perf_counter()
            if self.ignore_error:
                try:
                    new_image = func(context)
                    if new_image is not None and self.ignore_error:
                        context.image = new_image
                except Exception as exception:
                    print(exception)
                    if not _is_ipython():
                        add_intermediate_error(
                            context.config,
                            context.config_file_name,
                            exception,
                            traceback.format_exc().split("\n"),
                        )
            else:
                new_image = func(context)
                if new_image is not None:
                    context.image = new_image
            elapsed_time = time.perf_counter() - start_time
            if os.environ.get("TIME", "FALSE") == "TRUE":
                print(f"Elapsed time in {self.name}: {int(round(elapsed_time))}s.")

            if self.progress:
                context.save_progress_images(self.name)

        return wrapper


def external(func: ExternalFunction) -> FunctionWithContextReturnsImage:
    """Run an external tool."""

    def wrapper(context: Context) -> Optional[NpNdarrayInt]:
        with tempfile.NamedTemporaryFile(suffix=".png") as source:
            cv2.imwrite(source.name, context.image)
            with tempfile.NamedTemporaryFile(suffix=".png") as destination:
                func(context, source.name, destination.name)
                return cast(NpNdarrayInt, cv2.imread(destination.name))

    return wrapper


def get_contour_to_crop(
    contours: list[tuple[int, int, int, int]], margin_horizontal: int = 0, margin_vertical: int = 0
) -> tuple[int, int, int, int]:
    """Get the contour to crop."""

    content = [
        contours[0][0],
        contours[0][1],
        contours[0][0] + contours[0][2],
        contours[0][1] + contours[0][3],
    ]
    for contour in contours:
        content[0] = min(content[0], contour[0])
        content[1] = min(content[1], contour[1])
        content[2] = max(content[2], contour[0] + contour[2])
        content[3] = max(content[3], contour[1] + contour[3])

    return (
        content[0] - margin_horizontal,
        content[1] - margin_vertical,
        content[2] - content[0] + 2 * margin_horizontal,
        content[3] - content[1] + 2 * margin_vertical,
    )


def crop(context: Context, margin_horizontal: int, margin_vertical: int) -> None:
    """
    Do a crop on an image.

    Margin in px
    """

    image = context.get_masked()
    process_count = context.get_process_count()
    contours = find_contours(
        image,
        context,
        "crop",
        context.config["args"].setdefault("crop", {}).setdefault("contour", {}),
    )

    if contours:
        for contour in contours:
            draw_rectangle(image, contour)
        context.save_progress_images(
            "crop", image[context.index] if _is_ipython() else image, process_count=process_count, force=True
        )

        x, y, width, height = get_contour_to_crop(contours, margin_horizontal, margin_vertical)
        context.crop(x, y, width, height)


def _get_level(context: Context) -> tuple[bool, float, float]:
    level_ = context.config["args"].setdefault("level", schema.LEVEL_DEFAULT)
    min_p100 = 0.0
    max_p100 = 100.0
    if level_ is True:
        min_p100 = schema.MIN_LEVEL_DEFAULT
        max_p100 = schema.MAX_LEVEL_DEFAULT
    elif isinstance(level_, (float, int)):
        min_p100 = 0.0 + level_
        max_p100 = 100.0 - level_
    if level_ is not False:
        min_p100 = context.config["args"].setdefault("min_level", min_p100)
        max_p100 = context.config["args"].setdefault("max_level", max_p100)

    min_ = min_p100 / 100.0 * 255.0
    max_ = max_p100 / 100.0 * 255.0
    return level_ is not False, min_, max_


def _histogram(
    context: Context,
    histogram_data: Any,
    histogram_centers: Any,
    histogram_max: Any,
    process_count: int,
    log: bool,
) -> None:
    _, axes = plt.subplots(figsize=(15, 5))
    axes.set_xlim(0, 255)

    if log:
        axes.semilogy(histogram_centers, histogram_data, lw=1)
    else:
        axes.plot(histogram_centers, histogram_data, lw=1)
    axes.set_title("Gray-level histogram")

    points = []
    level_, min_, max_ = _get_level(context)

    if level_ and min_ > 0:
        points.append(("min_level", min_, histogram_max / 5))

    cut_white = (
        context.config["args"].setdefault("cut_white", schema.CUT_WHITE_DEFAULT) / 255 * (max_ - min_) + min_
    )
    cut_black = (
        context.config["args"].setdefault("cut_black", schema.CUT_BLACK_DEFAULT) / 255 * (max_ - min_) + min_
    )

    if cut_black > 0.0:
        points.append(("cut_black", cut_black, histogram_max / 10))
    if cut_white < 255.0:
        points.append(("cut_white", cut_white, histogram_max / 5))

    if level_ and max_ < 255.0:
        points.append(("max_level", max_, histogram_max / 10))

    for label, value, pos in points:
        if int(round(value)) < len(histogram_data):
            hist_value = histogram_data[int(round(value))]
            axes.annotate(
                label,
                xy=(value, hist_value),
                xycoords="data",
                xytext=(value, hist_value + pos),
                textcoords="data",
                arrowprops={"facecolor": "black", "width": 1},
            )

    plt.tight_layout()
    with tempfile.NamedTemporaryFile(suffix=".png") as file:
        if not _is_ipython():
            plt.savefig(file.name)
            subprocess.run(["gm", "convert", "-flatten", file.name, file.name], check=True)  # nosec
            image = cv2.imread(file.name)
            context.save_progress_images(
                "histogram",
                image,
                image_prefix="log-" if log else "",
                process_count=process_count,
                force=True,
            )


@Process("histogram", progress=False)
def histogram(context: Context) -> None:
    """Create an image with the histogram of the current image."""

    noisy_image = img_as_ubyte(context.image)
    histogram_data, histogram_centers = skimage_histogram(noisy_image)
    histogram_max = max(histogram_data)
    process_count = context.get_process_count()

    _histogram(context, histogram_data, histogram_centers, histogram_max, process_count, False)
    _histogram(context, histogram_data, histogram_centers, histogram_max, process_count, True)


@Process("level")
def level(context: Context) -> NpNdarrayInt:
    """Do the level on an image."""

    img_yuv = cv2.cvtColor(context.image, cv2.COLOR_BGR2YUV)

    if context.config["args"].setdefault("auto_level", schema.AUTO_LEVEL_DEFAULT):
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cast(NpNdarrayInt, cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR))

    _, min_, max_ = _get_level(context)

    chanel_y = img_yuv[:, :, 0]
    mins = np.zeros(chanel_y.shape)
    maxs: NpNdarrayInt = np.zeros(chanel_y.shape) + 255

    values = (chanel_y - min_) / (max_ - min_) * 255
    img_yuv[:, :, 0] = np.minimum(maxs, np.maximum(mins, values))
    return cast(NpNdarrayInt, cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR))


@Process("color-cut")
def color_cut(context: Context) -> None:
    """Set the near white to white and near black to black."""

    assert context.image is not None
    grayscale = cv2.cvtColor(context.image, cv2.COLOR_BGR2GRAY)

    white_mask = cv2.inRange(
        grayscale, context.config["args"].setdefault("cut_white", schema.CUT_WHITE_DEFAULT), 255
    )
    black_mask = cv2.inRange(
        grayscale, 0, context.config["args"].setdefault("cut_black", schema.CUT_BLACK_DEFAULT)
    )
    context.image[white_mask == 255] = (255, 255, 255)
    context.image[black_mask == 255] = (0, 0, 0)


@Process("mask-cut")
def cut(context: Context) -> None:
    """Mask the image with the cut mask."""

    context.do_initial_cut()


@Process("deskew")
def deskew(context: Context) -> None:
    """Deskew an image."""

    images_config = context.config.setdefault("images_config", {})
    image_config = images_config.setdefault(context.image_name, {}) if context.image_name else {}
    image_status = image_config.setdefault("status", {})
    angle = image_config.setdefault("angle", None)
    if angle is None:
        image = context.get_masked()
        image_rgb = rgba2rgb(image) if len(image.shape) == 3 and image.shape[2] == 4 else image
        grayscale = rgb2gray(image_rgb) if len(image_rgb.shape) == 3 else image_rgb

        deskew_configuration = context.config["args"].setdefault("deskew", {})
        skew_angle, debug_images = determine_skew_debug_images(
            grayscale,
            min_angle=deskew_configuration.setdefault("min_angle", schema.DESKEW_MIN_ANGLE_DEFAULT),
            max_angle=deskew_configuration.setdefault("max_angle", schema.DESKEW_MAX_ANGLE_DEFAULT),
            min_deviation=deskew_configuration.setdefault(
                "angle_derivation", schema.DESKEW_ANGLE_DERIVATION_DEFAULT
            ),
            sigma=deskew_configuration.setdefault("sigma", schema.DESKEW_SIGMA_DEFAULT),
            num_peaks=deskew_configuration.setdefault("num_peaks", schema.DESKEW_NUM_PEAKS_DEFAULT),
            angle_pm_90=deskew_configuration.setdefault("angle_pm_90", schema.DESKEW_ANGLE_PM_90_DEFAULT),
        )
        if skew_angle is not None:
            image_status["angle"] = float(skew_angle)
            angle = float(skew_angle)

        if not _is_ipython():
            process_count = context.get_process_count()
            for name, image in debug_images:
                context.save_progress_images("skew", image, name, process_count, True)

    if angle:
        context.rotate(angle)

        if context.image_name is not None:
            image_name_split = os.path.splitext(context.image_name)
            sources = [img for img in context.config.get("images", []) if f"{image_name_split[0]}." in img]
            if len(sources) == 1:
                assert context.root_folder
                image = rotate_image(
                    cv2.imread(os.path.join(context.root_folder, sources[0])),
                    angle,
                    context.get_background_color(),
                )
                source_split = os.path.splitext(sources[0])
                cv2.imwrite(
                    os.path.join(context.root_folder, source_split[0] + "-skew-corrected" + source_split[1]),
                    image,
                )


@Process("docrop")
def docrop(context: Context) -> None:
    """Crop an image."""

    # Margin in mm
    crop_config = context.config["args"].setdefault("crop", {})
    if not crop_config.setdefault("enabled", schema.CROP_ENABLED_DEFAULT):
        return
    margin_horizontal = context.get_px_value(
        crop_config.setdefault("margin_horizontal", schema.MARGIN_HORIZONTAL_DEFAULT)
    )
    margin_vertical = context.get_px_value(
        crop_config.setdefault("margin_vertical", schema.MARGIN_VERTICAL_DEFAULT)
    )
    crop(context, int(round(margin_horizontal)), int(round(margin_vertical)))


@Process("sharpen")
def sharpen(context: Context) -> Optional[NpNdarrayInt]:
    """Sharpen an image."""

    if (
        context.config["args"]
        .setdefault("sharpen", cast(schema.Sharpen, schema.SHARPEN_DEFAULT))
        .setdefault("enabled", schema.SHARPEN_ENABLED_DEFAULT)
        is False
    ):
        return None
    if context.image is None:
        raise ScanToPaperlessException("The image is required")
    image = cv2.GaussianBlur(context.image, (0, 0), 3)
    return cast(NpNdarrayInt, cv2.addWeighted(context.image, 1.5, image, -0.5, 0))


@Process("dither")
@external
def dither(context: Context, source: str, destination: str) -> None:
    """Dither an image."""

    if (
        context.config["args"]
        .setdefault("dither", cast(schema.Dither, schema.DITHER_DEFAULT))
        .setdefault("enabled", schema.DITHER_ENABLED_DEFAULT)
        is False
    ):
        return
    call(CONVERT + ["+dither", source, destination])


@Process("autorotate", True)
def autorotate(context: Context) -> None:
    """
    Auto rotate an image.

    Put the text in the right position.
    """

    auto_rotate_configuration = context.config["args"].setdefault("auto_rotate", {})
    if not auto_rotate_configuration.setdefault("enabled", schema.AUTO_ROTATE_ENABLED_DEFAULT):
        return
    with tempfile.NamedTemporaryFile(suffix=".png") as source:
        cv2.imwrite(source.name, context.get_masked())
        try:
            orientation_lst = output(["tesseract", source.name, "-", "--psm", "0", "-l", "osd"]).splitlines()
            orientation_lst = [e for e in orientation_lst if "Orientation in degrees" in e]
            context.rotate(int(orientation_lst[0].split()[3]))
        except subprocess.CalledProcessError:
            print("Not text found")


def draw_line(  # pylint: disable=too-many-arguments
    image: NpNdarrayInt,
    vertical: bool,
    position: Optional[float],
    value: Optional[int],
    name: str,
    type_: str,
    color: tuple[int, int, int],
    line: Optional[tuple[int, int, int, int]] = None,
) -> schema.Limit:
    """Draw a line on an image."""

    img_len = image.shape[0 if vertical else 1]
    if line is None:
        assert position is not None
        assert value is not None
        if vertical:
            cv2.rectangle(
                image, (int(position) - 1, img_len), (int(position) + 1, img_len - value), color, -1
            )
            cv2.putText(
                image, name, (int(position), img_len - value), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4
            )
        else:
            cv2.rectangle(image, (0, int(position) - 1), (value, int(position) + 1), color, -1)
            cv2.putText(image, name, (value, int(position)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)
    else:
        position = line[0] if vertical else line[1]
        cv2.rectangle(
            image,
            (line[0] - (1 if vertical else 0), line[1] - (0 if vertical else 1)),
            (line[2] + (1 if vertical else 0), line[3] + (0 if vertical else 1)),
            color,
            -1,
        )
        cv2.putText(image, name, (line[0], line[3]), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)

    assert position is not None
    return {"name": name, "type": type_, "value": int(position), "vertical": vertical, "margin": 0}


def draw_rectangle(image: NpNdarrayInt, contour: tuple[int, int, int, int], border: bool = True) -> None:
    """Draw a rectangle on an image."""

    color = (0, 255, 0)
    opacity = 0.1
    x, y, width, height = contour
    x = int(round(x))
    y = int(round(y))
    width = int(round(width))
    height = int(round(height))

    sub_img = image[y : y + height, x : x + width]
    mask_image = np.zeros(sub_img.shape, dtype=np.uint8)
    mask_image[:, :] = color
    opacity_result = cv2.addWeighted(sub_img, 1 - opacity, mask_image, opacity, 1.0)
    if opacity_result is not None:
        image[y : y + height, x : x + width] = opacity_result

    if border:
        cv2.rectangle(image, (x, y), (x + 1, y + height), color, -1)
        cv2.rectangle(image, (x, y), (x + width, y + 1), color, -1)
        cv2.rectangle(image, (x, y + height - 1), (x + width, y + height), color, -1)
        cv2.rectangle(image, (x + width - 1, y), (x + width, y + height), color, -1)


def find_lines(
    image: NpNdarrayInt, vertical: bool, config: schema.LineDetection
) -> list[tuple[int, int, int, int]]:
    """Find the lines on an image."""

    edges = cv2.Canny(
        image,
        config.setdefault("high_threshold", schema.LINE_DETECTION_HIGH_THRESHOLD_DEFAULT),
        config.setdefault("low_threshold", schema.LINE_DETECTION_LOW_THRESHOLD_DEFAULT),
        apertureSize=config.setdefault("aperture_size", schema.LINE_DETECTION_APERTURE_SIZE_DEFAULT),
    )
    lines = cv2.HoughLinesP(
        image=edges,
        rho=config.setdefault("rho", schema.LINE_DETECTION_RHO_DEFAULT),
        theta=np.pi / 2,
        threshold=config.setdefault("threshold", schema.LINE_DETECTION_THRESHOLD_DEFAULT),
        minLineLength=(image.shape[0] if vertical else image.shape[1])
        / 100
        * config.setdefault("min_line_length", schema.LINE_DETECTION_MIN_LINE_LENGTH_DEFAULT),
        maxLineGap=config.setdefault("max_line_gap", schema.LINE_DETECTION_MAX_LINE_GAP_DEFAULT),
    )

    if lines is None:
        return []

    lines = [line for line, in lines if (line[0] == line[2] if vertical else line[1] == line[3])]

    def _key(line: tuple[int, int, int, int]) -> int:
        return line[1] - line[3] if vertical else line[2] - line[0]

    return cast(list[tuple[int, int, int, int]], sorted(lines, key=_key)[:5])


def zero_ranges(values: NpNdarrayInt) -> NpNdarrayInt:
    """Create an array that is 1 where a is 0, and pad each end with an extra 0."""

    is_zero: NpNdarrayInt = np.concatenate([[0], np.equal(values, 0).view(np.int8), [0]])
    abs_diff = np.abs(np.diff(is_zero))
    # Runs start and end where abs_diff is 1.
    ranges = np.where(abs_diff == 1)[0].reshape(-1, 2)
    return cast(NpNdarrayInt, ranges)


def find_limit_contour(
    image: NpNdarrayInt, vertical: bool, contours: list[tuple[int, int, int, int]]
) -> list[int]:
    """Find the contour for assisted split."""

    image_size = image.shape[1 if vertical else 0]

    values = np.zeros(image_size)
    if vertical:
        for x, _, width, height in contours:
            x_int = int(round(x))
            for value in range(x_int, min(x_int + width, image_size)):
                values[value] += height
    else:
        for _, y, width, height in contours:
            y_int = int(round(y))
            for value in range(y_int, min(y_int + height, image_size)):
                values[value] += width

    ranges = zero_ranges(values)

    result: list[int] = []
    for ranges_ in ranges:
        if ranges_[0] != 0 and ranges_[1] != image_size:
            result.append(int(round(sum(ranges_) / 2)))

    return result


def find_limits(
    image: NpNdarrayInt, vertical: bool, context: Context, contours: list[tuple[int, int, int, int]]
) -> tuple[list[int], list[tuple[int, int, int, int]]]:
    """Find the limit for assisted split."""
    contours_limits = find_limit_contour(image, vertical, contours)
    lines = find_lines(
        image, vertical, context.config["args"].setdefault("limit_detection", {}).setdefault("line", {})
    )
    return contours_limits, lines


def fill_limits(
    image: NpNdarrayInt, vertical: bool, contours_limits: list[int], lines: list[tuple[int, int, int, int]]
) -> list[schema.Limit]:
    """Fill the limit for assisted split."""
    third_image_size = int(image.shape[0 if vertical else 1] / 3)
    limits: list[schema.Limit] = []
    prefix = "V" if vertical else "H"
    for index, line in enumerate(lines):
        limits.append(
            draw_line(image, vertical, None, None, f"{prefix}L{index}", "line detection", (255, 0, 0), line)
        )
    for index, contour in enumerate(contours_limits):
        limits.append(
            draw_line(
                image,
                vertical,
                contour,
                third_image_size,
                f"{prefix}C{index}",
                "contour detection",
                (0, 255, 0),
            )
        )
    if not limits:
        half_image_size = image.shape[1 if vertical else 0] / 2
        limits.append(
            draw_line(
                image, vertical, half_image_size, third_image_size, f"{prefix}C", "image center", (0, 0, 255)
            )
        )

    return limits


def find_contours(
    image: NpNdarrayInt,
    context: Context,
    name: str,
    config: schema.Contour,
) -> list[tuple[int, int, int, int]]:
    """Find the contours on an image."""
    block_size = context.get_px_value(
        config.setdefault("threshold_block_size", schema.THRESHOLD_BLOCK_SIZE_DEFAULT)
    )
    threshold_value_c = config.setdefault("threshold_value_c", schema.THRESHOLD_VALUE_C_DEFAULT)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    block_size = int(round(block_size / 2) * 2)

    # Clean the image using method with the inverted binarized image
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size + 1, threshold_value_c
    )
    if context.is_progress() or _is_ipython():
        if _is_ipython():
            print("Threshold")
        context.save_progress_images(
            "threshold", cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)[context.index] if _is_ipython() else thresh
        )

    return _find_contours_thresh(image, thresh, context, name, config)


def _find_contours_thresh(
    image: NpNdarrayInt,
    thresh: NpNdarrayInt,
    context: Context,
    name: str,
    config: schema.Contour,
) -> list[tuple[int, int, int, int]]:
    min_size = context.get_px_value(config.setdefault("min_box_size", schema.MIN_BOX_SIZE_DEFAULT[name]))
    min_black = config.setdefault("min_box_black", schema.MIN_BOX_BLACK_DEFAULT)
    kernel_size = context.get_px_value(
        config.setdefault("contour_kernel_size", schema.CONTOUR_KERNEL_SIZE_DEFAULT)
    )

    kernel_size = int(round(kernel_size / 2))

    # Assign a rectangle kernel size
    kernel: NpNdarrayInt = np.ones((kernel_size, kernel_size), "uint8")
    par_img = cv2.dilate(thresh, kernel, iterations=5)

    contours, _ = cv2.findContours(par_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = []

    for cnt in contours:
        x, y, width, height = cv2.boundingRect(cnt)
        if width > min_size and height > min_size:
            contour_image = crop_image(image, x, y, width, height, context.get_background_color())
            imagergb = (
                rgba2rgb(contour_image)
                if len(contour_image.shape) == 3 and contour_image.shape[2] == 4
                else contour_image
            )
            contour_image = rgb2gray(imagergb) if len(imagergb.shape) == 3 else imagergb
            if (1 - np.mean(contour_image)) * 100 > min_black:
                result.append(
                    (
                        x + kernel_size * 2,
                        y + kernel_size * 2,
                        width - kernel_size * 4,
                        height - kernel_size * 4,
                    )
                )

    return result


def _update_config(config: schema.Configuration) -> None:
    """Convert the old configuration to the new one."""

    old_config = cast(dict[str, Any], config)
    # no_crop => crop.enabled (inverted)
    if "no_crop" in old_config["args"]:
        config["args"].setdefault("crop", {}).setdefault("enabled", not old_config["args"]["no_crop"])
        del old_config["args"]["no_crop"]
    # margin_horizontal => crop.margin_horizontal
    if "margin_horizontal" in old_config["args"]:
        config["args"].setdefault("crop", {}).setdefault(
            "margin_horizontal", old_config["args"]["margin_horizontal"]
        )
        del old_config["args"]["margin_horizontal"]
    # margin_vertical => crop.margin_vertical
    if "margin_vertical" in old_config["args"]:
        config["args"].setdefault("crop", {}).setdefault(
            "margin_vertical", old_config["args"]["margin_vertical"]
        )
        del old_config["args"]["margin_vertical"]
    # crop.min_box_size => crop.contour.min_box_size
    if "min_box_size" in old_config["args"].get("crop", {}):
        config["args"].setdefault("crop", {}).setdefault("contour", {}).setdefault(
            "min_box_size", old_config["args"]["crop"]["min_box_size"]
        )
        del old_config["args"]["crop"]["min_box_size"]
    # crop.min_box_black => crop.contour.min_box_black
    if "min_box_black" in old_config["args"].get("crop", {}):
        config["args"].setdefault("crop", {}).setdefault("contour", {}).setdefault(
            "min_box_black", old_config["args"]["crop"]["min_box_black"]
        )
        del old_config["args"]["crop"]["min_box_black"]
    # crop.contour_kernel_size => crop.contour.contour_kernel_size
    if "contour_kernel_size" in old_config["args"].get("crop", {}):
        config["args"].setdefault("crop", {}).setdefault("contour", {}).setdefault(
            "contour_kernel_size", old_config["args"]["crop"]["contour_kernel_size"]
        )
        del old_config["args"]["crop"]["contour_kernel_size"]
    # crop.threshold_block_size => crop.contour.threshold_block_size
    if "threshold_block_size" in old_config["args"].get("crop", {}):
        config["args"].setdefault("crop", {}).setdefault("contour", {}).setdefault(
            "threshold_block_size", old_config["args"]["crop"]["threshold_block_size"]
        )
        del old_config["args"]["crop"]["threshold_block_size"]
    # crop.threshold_value_c => crop.contour.threshold_value_c
    if "threshold_value_c" in old_config["args"].get("crop", {}):
        config["args"].setdefault("crop", {}).setdefault("contour", {}).setdefault(
            "threshold_value_c", old_config["args"]["crop"]["threshold_value_c"]
        )
        del old_config["args"]["crop"]["threshold_value_c"]
    # empty: null => empty.enabled: false
    if "empty" in old_config["args"] and (
        old_config["args"]["empty"] is True or old_config["args"]["empty"] is False
    ):
        config["args"]["empty"] = {"enabled": old_config["args"]["empty"]}
    if "empty" in old_config["args"] and old_config["args"]["empty"] is None:
        config["args"]["empty"] = {"enabled": False}
    # min_box_size_empty => empty.contour.min_box_size
    if "min_box_size_empty" in old_config["args"]:
        config["args"].setdefault("empty", {}).setdefault("contour", {}).setdefault(
            "min_box_size", old_config["args"]["min_box_size_empty"]
        )
        del old_config["args"]["min_box_size_empty"]
    # min_box_black_empty => empty.contour.min_box_black
    if "min_box_black_empty" in old_config["args"]:
        config["args"].setdefault("empty", {}).setdefault("contour", {}).setdefault(
            "min_box_black", old_config["args"]["min_box_black_empty"]
        )
        del old_config["args"]["min_box_black_empty"]
    # contour_kernel_size_empty => empty.contour.contour_kernel_size
    if "contour_kernel_size_empty" in old_config["args"]:
        config["args"].setdefault("empty", {}).setdefault("contour", {}).setdefault(
            "contour_kernel_size", old_config["args"]["contour_kernel_size_empty"]
        )
        del old_config["args"]["contour_kernel_size_empty"]
    # threshold_block_size_empty => empty.contour.threshold_block_size
    if "threshold_block_size_empty" in old_config["args"]:
        config["args"].setdefault("empty", {}).setdefault("contour", {}).setdefault(
            "threshold_block_size", old_config["args"]["threshold_block_size_empty"]
        )
        del old_config["args"]["threshold_block_size_empty"]
    # threshold_value_c_empty => empty.contour.threshold_value_c
    if "threshold_value_c_empty" in old_config["args"]:
        config["args"].setdefault("empty", {}).setdefault("contour", {}).setdefault(
            "threshold_value_c", old_config["args"]["threshold_value_c_empty"]
        )
        del old_config["args"]["threshold_value_c_empty"]
    # min_box_size_limit => limit_detection.contour.min_box_size
    if "min_box_size_limit" in old_config["args"]:
        config["args"].setdefault("limit_detection", {}).setdefault("contour", {}).setdefault(
            "min_box_size", old_config["args"]["min_box_size_limit"]
        )
        del old_config["args"]["min_box_size_limit"]
    # min_box_black_limit => limit_detection.contour.min_box_black
    if "min_box_black_limit" in old_config["args"]:
        config["args"].setdefault("limit_detection", {}).setdefault("contour", {}).setdefault(
            "min_box_black", old_config["args"]["min_box_black_limit"]
        )
        del old_config["args"]["min_box_black_limit"]
    # contour_kernel_size_limit => limit_detection.contour.contour_kernel_size
    if "contour_kernel_size_limit" in old_config["args"]:
        config["args"].setdefault("limit_detection", {}).setdefault("contour", {}).setdefault(
            "contour_kernel_size", old_config["args"]["contour_kernel_size_limit"]
        )
        del old_config["args"]["contour_kernel_size_limit"]
    # threshold_block_size_limit => limit_detection.contour.threshold_block_size
    if "threshold_block_size_limit" in old_config["args"]:
        config["args"].setdefault("limit_detection", {}).setdefault("contour", {}).setdefault(
            "threshold_block_size", old_config["args"]["threshold_block_size_limit"]
        )
        del old_config["args"]["threshold_block_size_limit"]
    # threshold_value_c_limit => limit_detection.contour.threshold_value_c
    if "threshold_value_c_limit" in old_config["args"]:
        config["args"].setdefault("limit_detection", {}).setdefault("contour", {}).setdefault(
            "threshold_value_c", old_config["args"]["threshold_value_c_limit"]
        )
        del old_config["args"]["threshold_value_c_limit"]
    # auto_mask: null => auto_mask.enabled: false
    if "auto_mask" in old_config["args"] and (
        old_config["args"]["auto_mask"] is True or old_config["args"]["auto_mask"] is False
    ):
        config["args"]["auto_mask"] = {"enabled": old_config["args"]["auto_mask"]}
    if "auto_mask" in old_config["args"] and old_config["args"]["auto_mask"] is None:
        config["args"]["auto_mask"] = {"enabled": False}
    # auto_mask.lower_hsv_color => auto_mask.auto_mask.lower_hsv_color
    if "lower_hsv_color" in old_config["args"].get("auto_mask", {}):
        config["args"].setdefault("auto_mask", {}).setdefault("auto_mask", {}).setdefault(
            "lower_hsv_color",
            old_config["args"]["auto_mask"]["lower_hsv_color"],
        )
        del old_config["args"]["auto_mask"]["lower_hsv_color"]
    # auto_mask.upper_hsv_color => auto_mask.auto_mask.upper_hsv_color
    if "upper_hsv_color" in old_config["args"].get("auto_mask", {}):
        config["args"].setdefault("auto_mask", {}).setdefault("auto_mask", {}).setdefault(
            "upper_hsv_color",
            old_config["args"]["auto_mask"]["upper_hsv_color"],
        )
        del old_config["args"]["auto_mask"]["upper_hsv_color"]
    # auto_mask.de_noise_morphology => auto_mask.auto_mask.de_noise_morphology
    if "de_noise_morphology" in old_config["args"].get("auto_mask", {}):
        config["args"].setdefault("auto_mask", {}).setdefault("auto_mask", {}).setdefault(
            "de_noise_morphology",
            old_config["args"]["auto_mask"]["de_noise_morphology"],
        )
        del old_config["args"]["auto_mask"]["de_noise_morphology"]
    # auto_mask.inverse_mask => auto_mask.auto_mask.inverse_mask
    if "inverse_mask" in old_config["args"].get("auto_mask", {}):
        config["args"].setdefault("auto_mask", {}).setdefault("auto_mask", {}).setdefault(
            "inverse_mask",
            old_config["args"]["auto_mask"]["inverse_mask"],
        )
        del old_config["args"]["auto_mask"]["inverse_mask"]
    # auto_mask.de_noise_size => auto_mask.auto_mask.de_noise_size
    if "de_noise_size" in old_config["args"].get("auto_mask", {}):
        config["args"].setdefault("auto_mask", {}).setdefault("auto_mask", {}).setdefault(
            "de_noise_size",
            old_config["args"]["auto_mask"]["de_noise_size"],
        )
        del old_config["args"]["auto_mask"]["de_noise_size"]
    # auto_mask.de_noise_level => auto_mask.auto_mask.de_noise_level
    if "de_noise_level" in old_config["args"].get("auto_mask", {}):
        config["args"].setdefault("auto_mask", {}).setdefault("auto_mask", {}).setdefault(
            "de_noise_level",
            old_config["args"]["auto_mask"]["de_noise_level"],
        )
        del old_config["args"]["auto_mask"]["de_noise_level"]
    # auto_mask.buffer_size => auto_mask.auto_mask.buffer_size
    if "buffer_size" in old_config["args"].get("auto_mask", {}):
        config["args"].setdefault("auto_mask", {}).setdefault("auto_mask", {}).setdefault(
            "buffer_size",
            old_config["args"]["auto_mask"]["buffer_size"],
        )
        del old_config["args"]["auto_mask"]["buffer_size"]
    # auto_mask.buffer_level => auto_mask.auto_mask.buffer_level
    if "buffer_level" in old_config["args"].get("auto_mask", {}):
        config["args"].setdefault("auto_mask", {}).setdefault("auto_mask", {}).setdefault(
            "buffer_level",
            old_config["args"]["auto_mask"]["buffer_level"],
        )
        del old_config["args"]["auto_mask"]["buffer_level"]
    # auto_mask.additional_filename => auto_mask.auto_mask.additional_filename
    if "additional_filename" in old_config["args"].get("auto_mask", {}):
        config["args"].setdefault("auto_mask", {}).setdefault("auto_mask", {}).setdefault(
            "additional_filename",
            old_config["args"]["auto_mask"]["additional_filename"],
        )
        del old_config["args"]["auto_mask"]["additional_filename"]
    # auto_cut: null => auto_cut.enabled: false
    if "auto_cut" in old_config["args"] and (
        old_config["args"]["auto_cut"] is True or old_config["args"]["auto_cut"] is False
    ):
        config["args"]["auto_cut"] = {"enabled": old_config["args"]["auto_cut"]}
    if "auto_cut" in old_config["args"] and old_config["args"]["auto_cut"] is None:
        config["args"]["auto_cut"] = {"enabled": False}
    # auto_cut.lower_hsv_color => auto_cut.auto_mask.lower_hsv_color
    if "lower_hsv_color" in old_config["args"].get("auto_cut", {}):
        config["args"].setdefault("auto_cut", {}).setdefault("auto_mask", {}).setdefault(
            "lower_hsv_color",
            old_config["args"]["auto_cut"]["lower_hsv_color"],
        )
        del old_config["args"]["auto_cut"]["lower_hsv_color"]
    # auto_cut.upper_hsv_color => auto_cut.auto_mask.upper_hsv_color
    if "upper_hsv_color" in old_config["args"].get("auto_cut", {}):
        config["args"].setdefault("auto_cut", {}).setdefault("auto_mask", {}).setdefault(
            "upper_hsv_color",
            old_config["args"]["auto_cut"]["upper_hsv_color"],
        )
        del old_config["args"]["auto_cut"]["upper_hsv_color"]
    # auto_cut.de_noise_morphology => auto_cut.auto_mask.de_noise_morphology
    if "de_noise_morphology" in old_config["args"].get("auto_cut", {}):
        config["args"].setdefault("auto_cut", {}).setdefault("auto_mask", {}).setdefault(
            "de_noise_morphology",
            old_config["args"]["auto_cut"]["de_noise_morphology"],
        )
        del old_config["args"]["auto_cut"]["de_noise_morphology"]
    # auto_cut.inverse_mask => auto_cut.auto_mask.inverse_mask
    if "inverse_mask" in old_config["args"].get("auto_cut", {}):
        config["args"].setdefault("auto_cut", {}).setdefault("auto_mask", {}).setdefault(
            "inverse_mask",
            old_config["args"]["auto_cut"]["inverse_mask"],
        )
        del old_config["args"]["auto_cut"]["inverse_mask"]
    # auto_cut.de_noise_size => auto_cut.auto_mask.de_noise_size
    if "de_noise_size" in old_config["args"].get("auto_cut", {}):
        config["args"].setdefault("auto_cut", {}).setdefault("auto_mask", {}).setdefault(
            "de_noise_size",
            old_config["args"]["auto_cut"]["de_noise_size"],
        )
        del old_config["args"]["auto_cut"]["de_noise_size"]
    # auto_cut.de_noise_level => auto_cut.auto_mask.de_noise_level
    if "de_noise_level" in old_config["args"].get("auto_cut", {}):
        config["args"].setdefault("auto_cut", {}).setdefault("auto_mask", {}).setdefault(
            "de_noise_level",
            old_config["args"]["auto_cut"]["de_noise_level"],
        )
        del old_config["args"]["auto_cut"]["de_noise_level"]
    # auto_cut.buffer_size => auto_cut.auto_mask.buffer_size
    if "buffer_size" in old_config["args"].get("auto_cut", {}):
        config["args"].setdefault("auto_cut", {}).setdefault("auto_mask", {}).setdefault(
            "buffer_size",
            old_config["args"]["auto_cut"]["buffer_size"],
        )
        del old_config["args"]["auto_cut"]["buffer_size"]
    # auto_cut.buffer_level => auto_cut.auto_mask.buffer_level
    if "buffer_level" in old_config["args"].get("auto_cut", {}):
        config["args"].setdefault("auto_cut", {}).setdefault("auto_mask", {}).setdefault(
            "buffer_level",
            old_config["args"]["auto_cut"]["buffer_level"],
        )
        del old_config["args"]["auto_cut"]["buffer_level"]
    # auto_cut.additional_filename => auto_cut.auto_mask.additional_filename
    if "additional_filename" in old_config["args"].get("auto_cut", {}):
        config["args"].setdefault("auto_cut", {}).setdefault("auto_mask", {}).setdefault(
            "additional_filename",
            old_config["args"]["auto_cut"]["additional_filename"],
        )
        del old_config["args"]["auto_cut"]["additional_filename"]
    # run_optipng => optipng.enabled
    if "run_optipng" in old_config["args"]:
        config["args"].setdefault("optipng", {}).setdefault("enabled", old_config["args"]["run_optipng"])
        del old_config["args"]["run_optipng"]
    # run_pngquant => pngquant.enabled
    if "run_pngquant" in old_config["args"]:
        config["args"].setdefault("pngquant", {}).setdefault("enabled", old_config["args"]["run_pngquant"])
        del old_config["args"]["run_pngquant"]
    # pngquant_options => pngquant.options
    if "pngquant_options" in old_config["args"]:
        config["args"].setdefault("pngquant", {}).setdefault(
            "options", old_config["args"]["pngquant_options"]
        )
        del old_config["args"]["pngquant_options"]
    # run_exiftool => exiftool.enabled
    if "run_exiftool" in old_config["args"]:
        config["args"].setdefault("exiftool", {}).setdefault("enabled", old_config["args"]["run_exiftool"])
        del old_config["args"]["run_exiftool"]
    # run_ps2pdf => ps2pdf.enabled
    if "run_ps2pdf" in old_config["args"]:
        config["args"].setdefault("ps2pdf", {}).setdefault("enabled", old_config["args"]["run_ps2pdf"])
        del old_config["args"]["run_ps2pdf"]
    # jpeg => jpeg.enabled
    if "jpeg" in old_config["args"] and (
        old_config["args"]["jpeg"] is True or old_config["args"]["jpeg"] is False
    ):
        config["args"]["jpeg"] = {"enabled": old_config["args"]["jpeg"]}
    if "jpeg" in old_config["args"] and old_config["args"]["jpeg"] is None:
        config["args"]["jpeg"] = {"enabled": False}
    # jpeg_quality => jpeg.quality
    if "jpeg_quality" in old_config["args"]:
        config["args"].setdefault("jpeg", {}).setdefault("quality", old_config["args"]["jpeg_quality"])
        del old_config["args"]["jpeg_quality"]
    # tesseract => tesseract.enabled
    if "tesseract" in old_config["args"] and (
        old_config["args"]["tesseract"] is True or old_config["args"]["tesseract"] is False
    ):
        config["args"]["tesseract"] = {"enabled": old_config["args"]["tesseract"]}
    if "tesseract" in old_config["args"] and old_config["args"]["tesseract"] is None:
        config["args"]["tesseract"] = {"enabled": False}
    # tesseract_lang => tesseract.lang
    if "tesseract_lang" in old_config["args"]:
        config["args"].setdefault("tesseract", {}).setdefault("lang", old_config["args"]["tesseract_lang"])
        del old_config["args"]["tesseract_lang"]
    # no_auto_rotate= auto_rotate.enabled (inverted)
    if "no_auto_rotate" in old_config["args"]:
        config["args"].setdefault("auto_rotate", {}).setdefault(
            "enabled", not old_config["args"]["no_auto_rotate"]
        )
        del old_config["args"]["no_auto_rotate"]
    # sharpen => sharpen.enabled
    if "sharpen" in old_config["args"] and (
        old_config["args"]["sharpen"] is True or old_config["args"]["sharpen"] is False
    ):
        config["args"]["sharpen"] = {"enabled": old_config["args"]["sharpen"]}
    if "sharpen" in old_config["args"] and old_config["args"]["sharpen"] is None:
        config["args"]["sharpen"] = {"enabled": False}
    # dither => dither.enabled
    if "dither" in old_config["args"] and (
        old_config["args"]["dither"] is True or old_config["args"]["dither"] is False
    ):
        config["args"]["dither"] = {"enabled": old_config["args"]["dither"]}
    if "dither" in old_config["args"] and old_config["args"]["dither"] is None:
        config["args"]["dither"] = {"enabled": False}
    # rule.enable => rule.enabled
    if "enable" in old_config["args"].get("rule", {}):
        config["args"].setdefault("rule", {}).setdefault("enabled", old_config["args"]["rule"]["enable"])
        del old_config["args"]["rule"]["enable"]


def _pretty_repr(value: Any, prefix: str = "") -> str:
    if isinstance(value, dict):
        return "\n".join(
            [
                "{",
                *[
                    f'{prefix}    "{key}": {_pretty_repr(value, prefix + "    ")},'
                    for key, value in value.items()
                ],
                prefix + "}",
            ]
        )

    return repr(value)


def _is_ipython() -> bool:
    try:
        __IPYTHON__  # type: ignore[name-defined] # pylint: disable=pointless-statement
        return True
    except NameError:
        return False


def _create_jupyter_notebook(root_folder: str, context: Context, step: schema.Step) -> None:
    # Jupyter notebook
    dest_folder = os.path.join(root_folder, "jupyter")
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    with open(os.path.join(dest_folder, "README.txt"), "w", encoding="utf-8") as readme_file:
        readme_file.write(
            """# Jupyter notebook

Install dependencies:
pip install scan-to-paperless[process] jupyterlab Pillow

Run:
jupyter lab

Open the notebook file.
"""
        )

    notebook = nbformat.v4.new_notebook()  # type: ignore[no-untyped-call]

    notebook["cells"].append(
        nbformat.v4.new_markdown_cell(  # type: ignore[no-untyped-call]
            """# Scan to Paperless

This notebook show the transformation applied on the images of the document.

At the start of each step, se set some values on the `context.config["args"]` dict,
you can change the values to see the impact on the result,
then yon can all those changes in the `config.yaml` file, in the `args` section."""
        )
    )

    notebook["cells"].append(
        nbformat.v4.new_markdown_cell("Do the required imports.")  # type: ignore[no-untyped-call]
    )
    notebook["cells"].append(
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            """import os
import cv2
import numpy as np
from IPython.display import (  # convert color from CV2 BGR back to RGB
    clear_output,
    display,
)
from PIL import Image

from scan_to_paperless import process"""
        )
    )

    notebook["cells"].append(
        nbformat.v4.new_markdown_cell(  # type: ignore[no-untyped-call]
            """Calculate the base folder of the document."""
        )
    )
    notebook["cells"].append(
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            """import IPython

base_folder = os.path.dirname(os.path.dirname(IPython.extract_module_locals()[1]['__vsc_ipynb_file__']))"""
        )
    )

    notebook["cells"].append(
        nbformat.v4.new_markdown_cell(  # type: ignore[no-untyped-call]
            """Open on of the source images, you can change it by uncommenting the corresponding line."""
        )
    )
    other_images_open = "\n".join(
        [
            f'# context.image = cv2.imread(os.path.join(base_folder, "{image}"))'
            for image in step["sources"][1:]
        ]
    )
    notebook["cells"].append(
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            f"""# Open Source image
context = process.Context({{"args": {{}}}}, {{}})

# Open one of the images
context.image = cv2.imread(os.path.join(base_folder, "{step["sources"][0]}"))
{other_images_open}

images_context = {{"original": context.image.clone()}}
"""
        )
    )

    notebook["cells"].append(
        nbformat.v4.new_markdown_cell(  # type: ignore[no-untyped-call]
            """Set the values that's used by more than one step."""
        )
    )
    notebook["cells"].append(
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            """context.config["args"] = {
    "dpi": {context.config["args"].get("dpi", schema.DPI_DEFAULT)},
    "background_color": {context.config["args"].get("background_color", schema.BACKGROUND_COLOR_DEFAULT)},
}"""
        )
    )

    notebook["cells"].append(
        nbformat.v4.new_markdown_cell(  # type: ignore[no-untyped-call]
            """Get the index that represent the part of the image we want to see."""
        )
    )
    notebook["cells"].append(
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            """
# Get a part of the image to display, by default, the top of the image
context.index = np.ix_(
    np.arange(0, 500),
    np.arange(0, context.image.shape[1]),
    np.arange(0, context.image.shape[2]),
)

display(Image.fromarray(cv2.cvtColor(images_context["original"][context.index], cv2.COLOR_BGR2RGB)))"""
        )
    )

    notebook["cells"].append(
        nbformat.v4.new_markdown_cell(  # type: ignore[no-untyped-call]
            """Calculate the image mask, the mask is used to hide some part of the image when we calculate the image skew and the image auto crop (based on the content).

The `lower_hsv_color`and the `upper_hsv_color` are used to define the color range to remove,
the `de_noise_size` is used to remove noise from the image,
the `buffer_size` is used to add a buffer around the image and
the `buffer_level` is used to define the level of the buffer (`0.0` to `1.0`).

To remove the gray background from the scanner, on document I use the following values:
```yaml
lower_hsv_color: [0, 0, 250]
upper_hsv_color: [255, 10, 255]
```
On leaflet I use the following values:
```yaml
lower_hsv_color: [0, 20, 0]
upper_hsv_color: [255, 255, 255]
```
"""
        )
    )
    notebook["cells"].append(
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            f"""context.image = images_context["original"]

context.config["args"]["auto_mask"] = {_pretty_repr(context.config["args"].get("auto_mask", {}), "    ")}

hsv = cv2.cvtColor(context.image, cv2.COLOR_BGR2HSV)
print("Hue (h)")
display(Image.fromarray(cv2.cvtColor(hsv[:, :, 0], cv2.COLOR_GRAY2RGB)[context.index]))
print("Saturation (s)")
display(Image.fromarray(cv2.cvtColor(hsv[:, :, 1], cv2.COLOR_GRAY2RGB)[context.index]))
print("Value (v)")
display(Image.fromarray(cv2.cvtColor(hsv[:, :, 2], cv2.COLOR_GRAY2RGB)[context.index]))

# Print the HSV value on some point of the image
points = [
    [10, 10],
    [100, 100],
]
image = context.image.copy()
for x, y in points:
    print(f"Pixel: {{x}}:{{y}}, with value: {{hsv[y, x, :]}}")
    cv2.drawMarker(image, [x, y], (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
display(Image.fromarray(cv2.cvtColor(image[context.index], cv2.COLOR_BGR2RGB)))

context.init_mask()
if context.mask is not None:
    display(Image.fromarray(cv2.cvtColor(context.mask, cv2.COLOR_GRAY2RGB)[context.index]))
display(Image.fromarray(cv2.cvtColor(context.get_masked()[context.index], cv2.COLOR_BGR2RGB)))

images_context["auto_mask"] = context.image.clone()
"""
        )
    )

    notebook["cells"].append(
        nbformat.v4.new_markdown_cell("Display the image histogram.")  # type: ignore[no-untyped-call]
    )
    notebook["cells"].append(
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            f"""context.image = images_context["auto_mask"]

context.config["args"]["level"] = {context.config["args"].get("level", schema.LEVEL_DEFAULT)},
context.config["args"]["cut_white"] = {context.config["args"].get("cut_white", schema.CUT_WHITE_DEFAULT)},
context.config["args"]["cut_black"] = {context.config["args"].get("cut_black", schema.CUT_BLACK_DEFAULT)},

process.histogram(context)"""
        )
    )

    notebook["cells"].append(
        nbformat.v4.new_markdown_cell(  # type: ignore[no-untyped-call]
            """Do the image level correction.

Some of the used values are displayed in the histogram chart."""
        )
    )
    notebook["cells"].append(
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            f"""context.image = images_context["auto_mask"]

context.config["args"]["auto_level"] = {context.config["args"].get("auto_level", schema.AUTO_LEVEL_DEFAULT)},
context.config["args"]["level"] = {context.config["args"].get("level", schema.LEVEL_DEFAULT)},

process.level(context)
display(Image.fromarray(cv2.cvtColor(context.image[context.index], cv2.COLOR_BGR2RGB)))

images_context["level"] = context.image.clone()"""
        )
    )

    notebook["cells"].append(
        nbformat.v4.new_markdown_cell(  # type: ignore[no-untyped-call]
            """Do the image level cut correction.

Some of the used values are displayed in the histogram chart."""
        )
    )
    notebook["cells"].append(
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            f"""context.image = images_context["level"]

print(f"Use cut_white: {context.config["args"]["cut_white"]}")
print(f"Use cut_black: {context.config["args"]["cut_black"]}")

process.color_cut(context)
display(Image.fromarray(cv2.cvtColor(context.image[context.index], cv2.COLOR_BGR2RGB)))

images_context["color_cut"] = context.image.clone()"""
        )
    )

    notebook["cells"].append(
        nbformat.v4.new_markdown_cell(  # type: ignore[no-untyped-call]
            """Cut some part of the image by auto removing a part of the image.

The needed of this step is to remove some part of the image that represent the part that is out of the page, witch is gray with some scanner.

The `lower_hsv_color`and the `upper_hsv_color` are used to define the color range to remove,
the `de_noise_size` is used to remove noise from the image,
the `buffer_size` is used to add a buffer around the image and
the `buffer_level` is used to define the level of the buffer (`0.0` to `1.0`)."""
        )
    )
    notebook["cells"].append(
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            f"""context.image = images_context["color_cut"]

context.config["args"]["auto_cut"] = {_pretty_repr(context.config["args"].get("auto_cut", {}), "    ")}"

# Print in HSV some point of the image
hsv = cv2.cvtColor(context.image, cv2.COLOR_BGR2HSV)
print("Pixel 10:10: ", hsv[10, 10])
print("Pixel 100:100: ", hsv[100, 100])

process.cut(context)
display(Image.fromarray(cv2.cvtColor(context.image[context.index], cv2.COLOR_BGR2RGB)))

images_context["cut"] = context.image.clone()"""
        )
    )

    notebook["cells"].append(
        nbformat.v4.new_markdown_cell("Do the image skew correction.")  # type: ignore[no-untyped-call]
    )
    notebook["cells"].append(
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            f"""context.image = images_context["cut"]

context.config["args"]["deskew"] = {_pretty_repr(context.config["args"].get("deskew", {}), "    ")}

# The angle can be forced in config.images_config.<image_name>.angle.
process.deskew(context)
display(Image.fromarray(cv2.cvtColor(context.image[context.index], cv2.COLOR_BGR2RGB)))"""
        )
    )

    notebook["cells"].append(
        nbformat.v4.new_markdown_cell(  # type: ignore[no-untyped-call]
            """Do the image auto crop base on the image content."""
        )
    )
    notebook["cells"].append(
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            f"""context.image = images_context["deskew"]

context.config["args"]["crop"] = {_pretty_repr(context.config["args"].get("crop", {}), "    ")}

process.docrop(context)
display(Image.fromarray(cv2.cvtColor(context.image[context.index], cv2.COLOR_BGR2RGB)))

images_context["crop"] = context.image.clone()"""
        )
    )

    notebook["cells"].append(
        nbformat.v4.new_markdown_cell("Do the image sharpen correction.")  # type: ignore[no-untyped-call]
    )
    notebook["cells"].append(
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            f"""context.image = images_context["crop"]

context.config["args"]["sharpen"] = {context.config["args"].get("sharpen", schema.SHARPEN_DEFAULT)}

process.sharpen(context)
display(Image.fromarray(cv2.cvtColor(context.image[context.index], cv2.COLOR_BGR2RGB)))

images_context["sharpen"] = context.image.clone()"""
        )
    )

    notebook["cells"].append(
        nbformat.v4.new_markdown_cell("Do the image dither correction.")  # type: ignore[no-untyped-call]
    )
    notebook["cells"].append(
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            f"""context.image = images_context["sharpen"]

context.config["args"]["dither"] = {context.config["args"].get("dither", schema.DITHER_DEFAULT)}

process.dither(context)
display(Image.fromarray(cv2.cvtColor(context.image[context.index], cv2.COLOR_BGR2RGB)))

images_context["dither"] = context.image.clone()"""
        )
    )

    notebook["cells"].append(
        nbformat.v4.new_markdown_cell(  # type: ignore[no-untyped-call]
            """Do the image auto rotate correction, based on the text orientation.

This require Tesseract to be installed."""
        )
    )
    notebook["cells"].append(
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            """context.image = images_context["dither"]

try:
    process.autorotate(context)
    display(Image.fromarray(cv2.cvtColor(context.image[context.index], cv2.COLOR_BGR2RGB)))
except FileNotFoundError as e:
    print("Tesseract not found, skipping autorotate: ", e)"""
        )
    )

    with open(os.path.join(dest_folder, "jupyter.ipynb"), "w", encoding="utf-8") as jupyter_file:
        nbformat.write(notebook, jupyter_file)  # type: ignore[no-untyped-call]


def transform(
    config: schema.Configuration,
    step: schema.Step,
    config_file_name: str,
    root_folder: str,
    status: Optional[scan_to_paperless.status.Status] = None,
) -> schema.Step:
    """Apply the transforms on a document."""

    if "intermediate_error" in config:
        del config["intermediate_error"]

    images = []
    process_count = 0

    if config["args"].setdefault("assisted_split", schema.ASSISTED_SPLIT_DEFAULT):
        config["assisted_split"] = []

    for index, image in enumerate(step["sources"]):
        if status is not None:
            status.set_status(config_file_name, -1, f"Transform ({os.path.basename(image)})", write=True)
        image_name = f"{os.path.basename(image).rsplit('.')[0]}.png"
        context = Context(config, step, config_file_name, root_folder, image_name)
        if context.image_name is None:
            raise ScanToPaperlessException("Image name is required")
        context.image = cv2.imread(os.path.join(root_folder, image))
        assert context.image is not None
        context.index = np.ix_(
            np.arange(0, context.image.shape[0]),
            np.arange(0, context.image.shape[1]),
            np.arange(0, context.image.shape[2]),
        )
        images_config = context.config.setdefault("images_config", {})
        image_config = images_config.setdefault(context.image_name, {})
        image_status = image_config.setdefault("status", {})
        assert context.image is not None
        image_status["size"] = list(context.image.shape[:2][::-1])
        context.init_mask()
        histogram(context)
        level(context)
        color_cut(context)
        cut(context)
        deskew(context)
        docrop(context)
        sharpen(context)
        dither(context)
        autorotate(context)

        # Is empty ?
        empty_config = config["args"].setdefault("empty", {})
        if empty_config.setdefault("enabled", schema.EMPTY_ENABLED_DEFAULT):
            contours = find_contours(
                context.get_masked(),
                context,
                "empty",
                empty_config.setdefault("contour", {}),
            )
            if not contours:
                print(f"Ignore image with no content: {image}")
                continue

        if config["args"].setdefault("assisted_split", schema.ASSISTED_SPLIT_DEFAULT):
            assisted_split: schema.AssistedSplit = {}
            name = os.path.join(root_folder, context.image_name)
            source = context.save_progress_images("assisted-split", context.image, force=True)
            assert source
            assisted_split["source"] = source

            config["assisted_split"].append(assisted_split)
            destinations = [len(step["sources"]) * 2 - index, index + 1]
            if index % 2 == 1:
                destinations.reverse()
            assisted_split["destinations"] = list(destinations)

            limits = []
            assert context.image is not None

            contours = find_contours(
                context.image,
                context,
                "limit",
                config["args"].setdefault("limit_detection", {}).setdefault("contour", {}),
            )
            vertical_limits_context = find_limits(context.image, True, context, contours)
            horizontal_limits_context = find_limits(context.image, False, context, contours)

            for contour_limit in contours:
                draw_rectangle(context.image, contour_limit, False)
            limits.extend(fill_limits(context.image, True, *vertical_limits_context))
            limits.extend(fill_limits(context.image, False, *horizontal_limits_context))
            assisted_split["limits"] = limits

            rule_config = config["args"].setdefault("rule", {})
            if rule_config.setdefault("enabled", schema.RULE_ENABLE_DEFAULT):
                minor_graduation_space = rule_config.setdefault(
                    "minor_graduation_space", schema.RULE_MINOR_GRADUATION_SPACE_DEFAULT
                )
                major_graduation_space = rule_config.setdefault(
                    "major_graduation_space", schema.RULE_MAJOR_GRADUATION_SPACE_DEFAULT
                )
                lines_space = rule_config.setdefault("lines_space", schema.RULE_LINES_SPACE_DEFAULT)
                minor_graduation_size = rule_config.setdefault(
                    "minor_graduation_size", schema.RULE_MINOR_GRADUATION_SIZE_DEFAULT
                )
                major_graduation_size = rule_config.setdefault(
                    "major_graduation_size", schema.RULE_MAJOR_GRADUATION_SIZE_DEFAULT
                )
                graduation_color = rule_config.setdefault(
                    "graduation_color", schema.RULE_GRADUATION_COLOR_DEFAULT
                )
                lines_color = rule_config.setdefault("lines_color", schema.RULE_LINES_COLOR_DEFAULT)
                lines_opacity = rule_config.setdefault("lines_opacity", schema.RULE_LINES_OPACITY_DEFAULT)
                graduation_text_font_filename = rule_config.setdefault(
                    "graduation_text_font_filename", schema.RULE_GRADUATION_TEXT_FONT_FILENAME_DEFAULT
                )
                graduation_text_font_size = rule_config.setdefault(
                    "graduation_text_font_size", schema.RULE_GRADUATION_TEXT_FONT_SIZE_DEFAULT
                )
                graduation_text_font_color = rule_config.setdefault(
                    "graduation_text_font_color", schema.RULE_GRADUATION_TEXT_FONT_COLOR_DEFAULT
                )
                graduation_text_margin = rule_config.setdefault(
                    "graduation_text_margin", schema.RULE_GRADUATION_TEXT_MARGIN_DEFAULT
                )

                x = minor_graduation_space
                while x < context.image.shape[1]:
                    if x % lines_space == 0:
                        sub_img = context.image[0 : context.image.shape[0], x : x + 1]
                        mask_image = np.zeros(sub_img.shape, dtype=np.uint8)
                        mask_image[:, :] = lines_color
                        opacity_result = cv2.addWeighted(
                            sub_img, 1 - lines_opacity, mask_image, lines_opacity, 1.0
                        )
                        if opacity_result is not None:
                            context.image[0 : context.image.shape[0], x : x + 1] = opacity_result

                    if x % major_graduation_space == 0:
                        cv2.rectangle(
                            context.image, (x, 0), (x + 1, major_graduation_size), graduation_color, -1
                        )
                    else:
                        cv2.rectangle(
                            context.image, (x, 0), (x + 1, minor_graduation_size), graduation_color, -1
                        )
                    x += minor_graduation_space

                y = minor_graduation_space
                while y < context.image.shape[0]:
                    if y % lines_space == 0:
                        sub_img = context.image[y : y + 1, 0 : context.image.shape[1]]
                        mask_image = np.zeros(sub_img.shape, dtype=np.uint8)
                        mask_image[:, :] = lines_color
                        opacity_result = cv2.addWeighted(
                            sub_img, 1 - lines_opacity, mask_image, lines_opacity, 1.0
                        )
                        if opacity_result is not None:
                            context.image[y : y + 1, 0 : context.image.shape[1]] = opacity_result
                    if y % major_graduation_space == 0:
                        cv2.rectangle(
                            context.image, (0, y), (major_graduation_size, y + 1), graduation_color, -1
                        )
                    else:
                        cv2.rectangle(
                            context.image, (0, y), (minor_graduation_size, y + 1), graduation_color, -1
                        )
                    y += minor_graduation_space

                pil_image = Image.fromarray(context.image)

                font = ImageFont.truetype(font=graduation_text_font_filename, size=graduation_text_font_size)
                draw = ImageDraw.Draw(pil_image)

                x = major_graduation_space
                print(graduation_text_font_color)
                while x < context.image.shape[1]:
                    draw.text(
                        (x + graduation_text_margin, major_graduation_size),
                        f"{x}",
                        fill=tuple(graduation_text_font_color),
                        anchor="lb",
                        font=font,
                    )
                    x += major_graduation_space

                pil_image = pil_image.rotate(-90, expand=True)
                draw = ImageDraw.Draw(pil_image)
                y = major_graduation_space
                while y < context.image.shape[0]:
                    draw.text(
                        (context.image.shape[0] - y + graduation_text_margin, major_graduation_size),
                        f"{y}",
                        fill=tuple(graduation_text_font_color),
                        anchor="lb",
                        font=font,
                    )
                    y += major_graduation_space
                pil_image = pil_image.rotate(90, expand=True)

                context.image = np.array(pil_image)

            cv2.imwrite(name, context.image)
            assisted_split["image"] = context.image_name
            images.append(name)
        else:
            img2 = os.path.join(root_folder, context.image_name)
            cv2.imwrite(img2, context.image)
            images.append(img2)
        process_count = context.process_count

    _create_jupyter_notebook(root_folder, context, step)

    progress = os.environ.get("PROGRESS", "FALSE") == "TRUE"

    count = context.get_process_count()
    for image in images:
        if progress:
            _save_progress(context.root_folder, count, "finalize", os.path.basename(image), image)

    if config["args"].setdefault("colors", schema.COLORS_DEFAULT):
        count = context.get_process_count()
        for image in images:
            call(CONVERT + ["-colors", str(config["args"]["colors"]), image, image])
            if progress:
                _save_progress(context.root_folder, count, "colors", os.path.basename(image), image)

    pngquant_config = config["args"].setdefault("pngquant", cast(schema.Pngquant, schema.PNGQUANT_DEFAULT))
    if not config["args"].setdefault("jpeg", cast(schema.Jpeg, schema.JPEG_DEFAULT)).setdefault(
        "enabled", schema.JPEG_ENABLED_DEFAULT
    ) and pngquant_config.setdefault("enabled", schema.PNGQUANT_ENABLED_DEFAULT):
        count = context.get_process_count()
        for image in images:
            with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
                call(
                    ["pngquant", f"--output={temp_file.name}"]
                    + pngquant_config.setdefault(
                        "options",
                        schema.PNGQUANT_OPTIONS_DEFAULT,
                    )
                    + ["--", image],
                    check=False,
                )
                if os.path.getsize(temp_file.name) > 0:
                    call(["cp", temp_file.name, image])
            if progress:
                _save_progress(context.root_folder, count, "pngquant", os.path.basename(image), image)

    if not config["args"].setdefault("jpeg", {}).setdefault(
        "enabled", schema.JPEG_ENABLED_DEFAULT
    ) and config["args"].setdefault("optipng", {}).setdefault(
        "enabled", not pngquant_config.setdefault("enabled", schema.PNGQUANT_ENABLED_DEFAULT)
    ):
        count = context.get_process_count()
        for image in images:
            call(["optipng", image], check=False)
            if progress:
                _save_progress(context.root_folder, count, "optipng", os.path.basename(image), image)

    if config["args"].setdefault("jpeg", {}).setdefault("enabled", schema.JPEG_ENABLED_DEFAULT):
        count = context.get_process_count()
        new_images = []
        for image in images:
            name = os.path.splitext(os.path.basename(image))[0]
            jpeg_img = f"{name}.jpeg"
            subprocess.run(  # nosec
                [
                    "gm",
                    "convert",
                    image,
                    "-quality",
                    str(
                        config["args"]
                        .setdefault("jpeg", {})
                        .setdefault("quality", schema.JPEG_QUALITY_DEFAULT)
                    ),
                    jpeg_img,
                ],
                check=True,
            )
            new_images.append(jpeg_img)
            if progress:
                _save_progress(context.root_folder, count, "to-jpeg", os.path.basename(image), image)

        images = new_images

    # Free matplotlib allocations
    plt.clf()
    plt.close("all")

    disable_remove_to_continue = config["args"].setdefault(
        "no_remove_to_continue", schema.NO_REMOVE_TO_CONTINUE_DEFAULT
    )
    if not disable_remove_to_continue or config["args"].setdefault(
        "assisted_split", schema.ASSISTED_SPLIT_DEFAULT
    ):
        with open(os.path.join(root_folder, "REMOVE_TO_CONTINUE"), "w", encoding="utf-8"):
            pass

    return {
        "sources": images,
        "name": scan_to_paperless.status.STATUS_ASSISTED_SPLIT
        if config["args"].setdefault("assisted_split", schema.ASSISTED_SPLIT_DEFAULT)
        else scan_to_paperless.status.STATUS_FINALIZE,
        "process_count": process_count,
    }


def _save_progress(root_folder: Optional[str], count: int, name: str, image_name: str, image: str) -> None:
    assert root_folder
    name = f"{count}-{name}"
    dest_folder = os.path.join(root_folder, name)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    dest_image = os.path.join(dest_folder, image_name)
    try:
        call(["cp", image, dest_image])
    except Exception as exception:
        print(exception)


def save(context: Context, root_folder: str, image: str, folder: str, force: bool = False) -> str:
    """Save the current image in a subfolder if progress mode in enabled."""

    if force or context.is_progress():
        dest_folder = os.path.join(root_folder, folder)
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        dest_file = os.path.join(dest_folder, os.path.basename(image))
        shutil.copyfile(image, dest_file)
        return dest_file
    return image


class Item(TypedDict, total=False):
    """
    Image content and position.

    Used to create the final document
    """

    pos: int
    file: IO[bytes]


def split(
    config: schema.Configuration,
    step: schema.Step,
    root_folder: str,
) -> schema.Step:
    """Split an image using the assisted split instructions."""

    process_count = 0
    for assisted_split in config["assisted_split"]:
        if assisted_split["limits"]:
            nb_horizontal = 1
            nb_vertical = 1

            for limit in assisted_split["limits"]:
                if limit["vertical"]:
                    nb_vertical += 1
                else:
                    nb_horizontal += 1

            if nb_vertical * nb_horizontal != len(assisted_split["destinations"]):
                raise ScanToPaperlessException(
                    f"Wrong number of destinations ({len(assisted_split['destinations'])}), "
                    f"vertical: {nb_horizontal}, height: {nb_vertical}, image: '{assisted_split['source']}'"
                )

    for assisted_split in config["assisted_split"]:
        if "image" in assisted_split:
            image_path = os.path.join(root_folder, assisted_split["image"])
            if os.path.exists(image_path):
                os.unlink(image_path)

    append: dict[Union[str, int], list[Item]] = {}
    transformed_images = []
    for assisted_split in config["assisted_split"]:
        image = assisted_split["source"]
        context = Context(config, step)
        width, height = (
            int(e) for e in output(CONVERT + [image, "-format", "%w %h", "info:-"]).strip().split(" ")
        )

        horizontal_limits = [limit for limit in assisted_split["limits"] if not limit["vertical"]]
        vertical_limits = [limit for limit in assisted_split["limits"] if limit["vertical"]]

        last_y = 0
        number = 0
        for horizontal_number in range(len(horizontal_limits) + 1):
            if horizontal_number < len(horizontal_limits):
                horizontal_limit = horizontal_limits[horizontal_number]
                horizontal_value = horizontal_limit["value"]
                horizontal_margin = horizontal_limit["margin"]
            else:
                horizontal_value = height
                horizontal_margin = 0
            last_x = 0
            for vertical_number in range(len(vertical_limits) + 1):
                destination = assisted_split["destinations"][number]
                if destination == "-" or destination is None:
                    if vertical_number < len(vertical_limits):
                        last_x = (
                            vertical_limits[vertical_number]["value"]
                            + vertical_limits[vertical_number]["margin"]
                        )
                else:
                    if vertical_number < len(vertical_limits):
                        vertical_limit = vertical_limits[vertical_number]
                        vertical_value = vertical_limit["value"]
                        vertical_margin = vertical_limit["margin"]
                    else:
                        vertical_value = width
                        vertical_margin = 0
                    process_file = tempfile.NamedTemporaryFile(  # pylint: disable=consider-using-with
                        suffix=".png"
                    )
                    call(
                        CONVERT
                        + [
                            "-crop",
                            f"{vertical_value - vertical_margin - last_x}x"
                            f"{horizontal_value - horizontal_margin - last_y}+{last_x}+{last_y}",
                            "+repage",
                            image,
                            process_file.name,
                        ]
                    )
                    last_x = vertical_value + vertical_margin

                    if re.match(r"[0-9]+\.[0-9]+", str(destination)):
                        page, page_pos = (int(e) for e in str(destination).split("."))
                    else:
                        page = int(destination)
                        page_pos = 0

                    save(context, root_folder, process_file.name, f"{context.get_process_count()}-split")
                    crop_config = context.config["args"].setdefault("crop", {})
                    margin_horizontal = context.get_px_value(
                        crop_config.setdefault("margin_horizontal", schema.MARGIN_HORIZONTAL_DEFAULT)
                    )
                    margin_vertical = context.get_px_value(
                        crop_config.setdefault("margin_vertical", schema.MARGIN_VERTICAL_DEFAULT)
                    )
                    context.image = cv2.imread(process_file.name)
                    if crop_config.setdefault("enabled", schema.CROP_ENABLED_DEFAULT):
                        crop(context, int(round(margin_horizontal)), int(round(margin_vertical)))
                        process_file = tempfile.NamedTemporaryFile(  # pylint: disable=consider-using-with
                            suffix=".png"
                        )
                        cv2.imwrite(process_file.name, context.image)
                        save(context, root_folder, process_file.name, f"{context.get_process_count()}-crop")
                    if page not in append:
                        append[page] = []
                    append[page].append({"file": process_file, "pos": page_pos})
                number += 1
            last_y = horizontal_value + horizontal_margin
        process_count = context.process_count

    for page_number in sorted(append.keys()):
        items: list[Item] = append[page_number]
        vertical = len(horizontal_limits) == 0
        if not vertical and len(vertical_limits) != 0 and len(items) > 1:
            raise ScanToPaperlessException(f"Mix of limit type for page '{page_number}'")

        with tempfile.NamedTemporaryFile(suffix=".png") as process_file:
            call(
                CONVERT
                + [e["file"].name for e in sorted(items, key=lambda e: e["pos"])]
                + [
                    "-background",
                    "#ffffff",
                    "-gravity",
                    "center",
                    "+append" if vertical else "-append",
                    process_file.name,
                ]
            )
            save(context, root_folder, process_file.name, f"{process_count}-split")
            img2 = os.path.join(root_folder, f"image-{page_number}.png")
            call(CONVERT + [process_file.name, img2])
            transformed_images.append(img2)
    process_count += 1

    return {
        "sources": transformed_images,
        "name": scan_to_paperless.status.STATUS_FINALIZE,
        "process_count": process_count,
    }


def finalize(
    config: schema.Configuration,
    step: schema.Step,
    root_folder: str,
    status: Optional[scan_to_paperless.status.Status] = None,
) -> None:
    """
    Do final step on document generation.

    convert in one pdf and copy with the right name in the consume folder
    """

    name = os.path.basename(root_folder)
    destination = os.path.join(os.environ.get("SCAN_CODES_FOLDER", "/scan-codes"), f"{name}.pdf")

    if os.path.exists(destination):
        return

    images = step["sources"]

    if config["args"].setdefault("append_credit_card", schema.APPEND_CREDIT_CARD_DEFAULT):
        if status is not None:
            status.set_status(name, -1, "Finalize (credit card append)", write=True)
        images2 = []
        for image in images:
            if os.path.exists(image):
                images2.append(image)

        file_name = os.path.join(root_folder, "append.png")
        call(CONVERT + images2 + ["-background", "#ffffff", "-gravity", "center", "-append", file_name])
        # To stack vertically (img1 over img2):
        # vis = np.concatenate((img1, img2), axis=0)
        # To stack horizontally (img1 to the left of img2):
        # vis = np.concatenate((img1, img2), axis=1)
        images = [file_name]

    pdf = []
    for image in images:
        if status is not None:
            status.set_status(name, -1, f"Finalize ({os.path.basename(image)})", write=True)
        if os.path.exists(image):
            name = os.path.splitext(os.path.basename(image))[0]
            file_name = os.path.join(root_folder, f"{name}.pdf")
            tesseract_configuration = config["args"].setdefault("tesseract", {})
            if tesseract_configuration.setdefault("enabled", schema.TESSERACT_ENABLED_DEFAULT):
                with open(file_name, "w", encoding="utf8") as output_file:
                    process = run(
                        [
                            "tesseract",
                            "--dpi",
                            str(config["args"].setdefault("dpi", schema.DPI_DEFAULT)),
                            "-l",
                            tesseract_configuration.setdefault("lang", schema.TESSERACT_LANG_DEFAULT),
                            image,
                            "stdout",
                            "pdf",
                        ],
                        stdout=output_file,
                    )
                    if process.stderr:
                        print(process.stderr)
            else:
                call(CONVERT + [image, "+repage", file_name])
            pdf.append(file_name)

    if status is not None:
        status.set_status(name, -1, "Finalize (optimize)", write=True)

    tesseract_producer = None
    if pdf:
        with pikepdf.open(pdf[0]) as pdf_:
            if tesseract_producer is None and pdf_.docinfo.get("/Producer") is not None:
                tesseract_producer = json.loads(pdf_.docinfo.get("/Producer").to_json())
                if "tesseract" not in tesseract_producer.lower():
                    tesseract_producer = None
                elif tesseract_producer.startswith("u:"):
                    tesseract_producer = tesseract_producer[2:]
            if tesseract_producer is None:
                with pdf_.open_metadata() as meta:
                    if "{http://purl.org/dc/elements/1.1/}producer" in meta:
                        tesseract_producer = meta["{http://purl.org/dc/elements/1.1/}producer"]
                        if "tesseract" not in tesseract_producer.lower():
                            tesseract_producer = None

    progress = os.environ.get("PROGRESS", "FALSE") == "TRUE"
    if progress:
        for pdf_file in pdf:
            basename = os.path.basename(pdf_file).split(".")
            call(
                [
                    "cp",
                    pdf_file,
                    os.path.join(root_folder, f"1-{'.'.join(basename[:-1])}-tesseract.{basename[-1]}"),
                ]
            )

    count = 1
    with tempfile.NamedTemporaryFile(suffix=".png") as temporary_pdf:
        call(["pdftk"] + pdf + ["output", temporary_pdf.name, "compress"])
        if progress:
            call(["cp", temporary_pdf.name, os.path.join(root_folder, f"{count}-pdftk.pdf")])
            count += 1

        if (
            config["args"]
            .setdefault("exiftool", cast(schema.Exiftool, schema.EXIFTOOL_DEFAULT))
            .setdefault("enabled", schema.EXIFTOOL_ENABLED_DEFAULT)
        ):
            call(["exiftool", "-overwrite_original_in_place", temporary_pdf.name])
            if progress:
                call(["cp", temporary_pdf.name, os.path.join(root_folder, f"{count}-exiftool.pdf")])
                count += 1

        if (
            config["args"]
            .setdefault("ps2pdf", cast(schema.Ps2Pdf, schema.PS2PDF_DEFAULT))
            .setdefault("enabled", schema.PS2PDF_ENABLED_DEFAULT)
        ):
            with tempfile.NamedTemporaryFile(suffix=".png") as temporary_ps2pdf:
                call(["ps2pdf", temporary_pdf.name, temporary_ps2pdf.name])
                if progress:
                    call(["cp", temporary_ps2pdf.name, f"{count}-ps2pdf.pdf"])
                    count += 1
                call(["cp", temporary_ps2pdf.name, temporary_pdf.name])

        with pikepdf.open(temporary_pdf.name, allow_overwriting_input=True) as pdf_:
            scan_to_paperless_meta = f"Scan to Paperless {os.environ.get('VERSION', 'undefined')}"
            with pdf_.open_metadata() as meta:
                meta["{http://purl.org/dc/elements/1.1/}creator"] = (
                    [scan_to_paperless_meta, tesseract_producer]
                    if tesseract_producer
                    else [scan_to_paperless_meta]
                )
            pdf_.save(temporary_pdf.name)
        if progress:
            call(["cp", temporary_pdf.name, os.path.join(root_folder, f"{count}-pikepdf.pdf")])
            count += 1
        call(["cp", temporary_pdf.name, destination])


def _process_code(name: str) -> None:
    """Detect ad add a page with the QR codes."""

    pdf_filename = os.path.join(os.environ.get("SCAN_CODES_FOLDER", "/scan-codes"), name)

    destination_filename = os.path.join(
        os.environ.get("SCAN_FINAL_FOLDER", "/destination"), os.path.basename(pdf_filename)
    )

    if os.path.exists(destination_filename):
        return

    try:
        _LOG.info("Processing codes for %s", pdf_filename)
        code.add_codes(
            pdf_filename,
            destination_filename,
            dpi=float(os.environ.get("SCAN_CODES_DPI", 200)),
            pdf_dpi=float(os.environ.get("SCAN_CODES_PDF_DPI", 72)),
            font_name=os.environ.get("SCAN_CODES_FONT_NAME", "Helvetica-Bold"),
            font_size=float(os.environ.get("SCAN_CODES_FONT_SIZE", 16)),
            margin_top=float(os.environ.get("SCAN_CODES_MARGIN_TOP", 0)),
            margin_left=float(os.environ.get("SCAN_CODES_MARGIN_LEFT", 2)),
        )
        if os.path.exists(destination_filename):
            # Remove the source file on success
            os.remove(pdf_filename)
        _LOG.info("Down processing codes for %s", pdf_filename)

    except Exception as exception:
        _LOG.exception("Error while processing %s: %s", pdf_filename, str(exception))


def is_sources_present(images: list[str], root_folder: str) -> bool:
    """Are sources present for the next step."""

    for image in images:
        if not os.path.exists(os.path.join(root_folder, image)):
            print(f"Missing {root_folder} - {image}")
            return False
    return True


def save_config(config: schema.Configuration, config_file_name: str) -> None:
    """Save the configuration."""

    yaml = YAML()
    yaml.default_flow_style = False
    with open(config_file_name + "_", "w", encoding="utf-8") as config_file:
        yaml.dump(config, config_file)
    os.rename(config_file_name + "_", config_file_name)


def _process(
    config_file_name: str,
    status: scan_to_paperless.status.Status,
    dirty: bool = False,
) -> bool:
    """Process one document."""

    if not os.path.exists(config_file_name):
        return dirty

    root_folder = os.path.dirname(config_file_name)

    if os.path.exists(os.path.join(root_folder, "error.yaml")):
        return dirty

    yaml = YAML()
    yaml.default_flow_style = False
    with open(config_file_name, encoding="utf-8") as config_file:
        config: schema.Configuration = yaml.load(config_file.read())
    if config is None:
        return dirty

    if not is_sources_present(config["images"], root_folder):
        return dirty

    try:
        rerun = False
        disable_remove_to_continue = config["args"].setdefault(
            "no_remove_to_continue", schema.NO_REMOVE_TO_CONTINUE_DEFAULT
        )
        if "steps" not in config:
            rerun = True
        while config.get("steps") and not is_sources_present(config["steps"][-1]["sources"], root_folder):
            config["steps"] = config["steps"][:-1]
            save_config(config, config_file_name)
            if os.path.exists(os.path.join(root_folder, "REMOVE_TO_CONTINUE")):
                os.remove(os.path.join(root_folder, "REMOVE_TO_CONTINUE"))
            rerun = True

        if "steps" not in config or not config["steps"]:
            step: schema.Step = {
                "sources": config["images"],
                "name": "transform",
            }
            config["steps"] = [step]
        step = config["steps"][-1]

        if is_sources_present(step["sources"], root_folder):
            if not disable_remove_to_continue:
                if os.path.exists(os.path.join(root_folder, "REMOVE_TO_CONTINUE")) and not rerun:
                    return dirty
            if os.path.exists(os.path.join(root_folder, "DONE")) and not rerun:
                return dirty

            status.set_global_status(f"Processing '{os.path.basename(os.path.dirname(config_file_name))}'...")
            status.set_current_folder(config_file_name)
            status.set_status(config_file_name, -1, "Processing")
            dirty = True

            done = False
            next_step = None
            if step["name"] == scan_to_paperless.status.STATUS_TRANSFORM:
                _update_config(config)
                next_step = transform(config, step, config_file_name, root_folder, status=status)
            elif step["name"] == scan_to_paperless.status.STATUS_ASSISTED_SPLIT:
                status.set_status(config_file_name, -1, "Split")
                next_step = split(config, step, root_folder)
            elif step["name"] == scan_to_paperless.status.STATUS_FINALIZE:
                finalize(config, step, root_folder, status=status)
                done = True

            if done and os.environ.get("PROGRESS", "FALSE") != "TRUE":
                shutil.rmtree(root_folder)
            else:
                if next_step is not None:
                    config["steps"].append(next_step)
                save_config(config, config_file_name)
                if done:
                    with open(os.path.join(root_folder, "DONE"), "w", encoding="utf-8"):
                        pass
                elif not disable_remove_to_continue:
                    with open(os.path.join(root_folder, "REMOVE_TO_CONTINUE"), "w", encoding="utf-8"):
                        pass
            status.set_current_folder(None)

    except Exception as exception:
        trace = traceback.format_exc()

        out = {"error": str(exception), "traceback": trace.split("\n")}
        for attribute in ("returncode", "cmd"):
            if hasattr(exception, attribute):
                out[attribute] = getattr(exception, attribute)
        for attribute in ("output", "stdout", "stderr"):
            if hasattr(exception, attribute):
                if getattr(exception, attribute):
                    out[attribute] = getattr(exception, attribute).decode()

        yaml = YAML(typ="safe")
        yaml.default_flow_style = False
        try:
            with open(os.path.join(root_folder, "error.yaml"), "w", encoding="utf-8") as error_file:
                yaml.dump(out, error_file)
            stream = ruamel.yaml.compat.StringIO()
            yaml.dump(out, stream)
        except Exception as exception2:
            print(exception2)
            print(traceback.format_exc())
            yaml = YAML()
            yaml.default_flow_style = False
            with open(os.path.join(root_folder, "error.yaml"), "w", encoding="utf-8") as error_file:
                yaml.dump(out, error_file)
            stream = ruamel.yaml.compat.StringIO()
            yaml.dump(out, stream)
    return dirty


def main() -> None:
    """Process the scanned documents."""

    parser = argparse.ArgumentParser("Process the scanned documents.")
    parser.add_argument("config", nargs="?", help="The config file to process.")
    args = parser.parse_args()

    if args.config:
        _process(args.config, scan_to_paperless.status.Status(no_write=True))
        sys.exit()

    print("Welcome to scanned images document to paperless.")
    print(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")

    status = scan_to_paperless.status.Status()
    status.write()

    while True:
        status.set_current_folder(None)
        name, job_type, step = status.get_next_job()

        if job_type in (
            scan_to_paperless.status.JobType.TRANSFORM,
            scan_to_paperless.status.JobType.ASSISTED_SPLIT,
            scan_to_paperless.status.JobType.FINALIZE,
        ):
            assert name is not None
            assert step is not None

            status.set_global_status(f"Processing '{name}'...")
            status.set_current_folder(name)

            root_folder = os.path.join(os.environ.get("SCAN_SOURCE_FOLDER", "/source"), name)
            config_file_name = os.path.join(root_folder, "config.yaml")
            yaml = YAML()
            yaml.default_flow_style = False
            with open(config_file_name, encoding="utf-8") as config_file:
                config: schema.Configuration = yaml.load(config_file.read())

            if "steps" not in config or not config["steps"]:
                config["steps"] = [step]
            else:
                used_index = -1
                for index, test_step in enumerate(config["steps"]):
                    if step["name"] == test_step["name"]:
                        used_index = index
                        break
                if used_index != -1:
                    config["steps"] = config["steps"][: used_index + 1]

            assert step is not None

            next_step = None
            if job_type == scan_to_paperless.status.JobType.TRANSFORM:
                _update_config(config)
                next_step = transform(config, step, config_file_name, root_folder, status=status)
            if job_type == scan_to_paperless.status.JobType.ASSISTED_SPLIT:
                status.set_status(name, -1, "Splitting in assisted-split mode", write=True)
                next_step = split(config, step, root_folder)
            if job_type == scan_to_paperless.status.JobType.FINALIZE:
                finalize(config, step, root_folder, status=status)
                with open(os.path.join(root_folder, "DONE"), "w", encoding="utf-8"):
                    pass
            if next_step is not None:
                config["steps"].append(next_step)

            save_config(config, config_file_name)

        elif job_type == scan_to_paperless.status.JobType.DOWN:
            assert name is not None
            root_folder = os.path.join(os.environ.get("SCAN_SOURCE_FOLDER", "/source"), name)
            shutil.rmtree(root_folder)
        elif job_type == scan_to_paperless.status.JobType.CODE:
            assert name is not None
            print(f"Process code '{name}'")
            status.set_global_status(f"Process code '{name}'...")
            status.set_current_folder(name)
            try:
                _process_code(name)
            except Exception as exception:
                print(exception)
                trace = traceback.format_exc()
                print(trace)
        elif job_type == scan_to_paperless.status.JobType.NONE:
            status.set_global_status("Waiting...")
            status.set_current_folder(None)
            time.sleep(30)
        else:
            raise ValueError(f"Unknown job type: {job_type}")


if __name__ == "__main__":
    main()
