#!/usr/bin/env python3

"""Process the scanned documents."""

import argparse
import asyncio
import datetime
import io
import json
import logging
import os
import re
import shutil
import subprocess  # nosec
import sys
import tempfile
import time
import traceback
from typing import IO, TYPE_CHECKING, Any, Protocol, TypedDict, cast

import aiohttp
import anyio

# read, write, rotate, crop, sharpen, draw_line, find_line, find_contour
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pikepdf
import sentry_sdk
from anyio import Path
from deskew import determine_skew_debug_images
from PIL import Image, ImageDraw, ImageFont
from ruamel.yaml.main import YAML
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from skimage.color import rgb2gray, rgba2rgb
from skimage.exposure import histogram as skimage_histogram
from skimage.metrics import structural_similarity
from skimage.util import img_as_ubyte

import scan_to_paperless
import scan_to_paperless.status
from scan_to_paperless import jupyter_utils, process_utils
from scan_to_paperless import process_schema as schema

if TYPE_CHECKING:
    NpNdarrayInt = np.ndarray[tuple[int, ...], np.dtype[np.integer[Any] | np.floating[Any]]]
    CompletedProcess = subprocess.CompletedProcess[str]
else:
    NpNdarrayInt = np.ndarray
    CompletedProcess = subprocess.CompletedProcess

# dither, crop, append, repage
CONVERT = ["gm", "convert"]
_DESKEW_LOCK = asyncio.Lock()
_LOG = logging.getLogger(__name__)
_ERROR_FILENAME = "error.yaml"


async def add_intermediate_error(
    config: schema.Configuration,
    config_file_name: Path | None,
    error: Exception,
    traceback_: list[str],
) -> None:
    """Add in the config non fatal error."""
    if config_file_name is None:
        msg = "The config file name is required"
        raise scan_to_paperless.ScanToPaperlessError(msg) from error
    if "intermediate_error" not in config:
        config["intermediate_error"] = []

    old_intermediate_error: list[schema.IntermediateError] = []
    old_intermediate_error.extend(config["intermediate_error"])
    yaml = YAML()
    yaml.default_flow_style = False
    temp_path = Path(str(config_file_name) + "_")
    try:
        config["intermediate_error"].append({"error": str(error), "traceback": traceback_})
        async with await temp_path.open("w", encoding="utf-8") as config_file:
            out = io.StringIO()
            yaml.dump(config, out)
            await config_file.write(out.getvalue())
    except Exception as exception:  # noqa: BLE001
        print(exception)
        config["intermediate_error"] = old_intermediate_error
        config["intermediate_error"].append({"error": str(error), "traceback": traceback_})
        async with await temp_path.open("w", encoding="utf-8") as config_file:
            out = io.StringIO()
            yaml.dump(config, out)
            await config_file.write(out.getvalue())
    await temp_path.rename(config_file_name)


async def call(cmd: str | list[str], check: bool = True, **kwargs: Any) -> None:
    """Verbose version of check_output with no returns."""
    if isinstance(cmd, list):
        cmd = [str(element) for element in cmd]
    print(" ".join(cmd) if isinstance(cmd, list) else cmd)
    sys.stdout.flush()
    proc = await asyncio.create_subprocess_exec(  # nosec # pylint: disable=subprocess-run-check
        *cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        **kwargs,
    )
    await proc.communicate()
    if check:
        assert proc.returncode == 0


async def run(cmd: str | list[str], **kwargs: Any) -> tuple[bytes, bytes, asyncio.subprocess.Process]:
    """Verbose version of check_output with no returns."""
    if isinstance(cmd, list):
        cmd = [str(element) for element in cmd]
    print(" ".join(cmd) if isinstance(cmd, list) else cmd)
    sys.stdout.flush()
    proc = await asyncio.create_subprocess_exec(*cmd, stderr=subprocess.PIPE, **kwargs)  # nosec
    stdout, stderr = await proc.communicate()
    assert proc.returncode == 0
    return stdout, stderr, proc


def output(cmd: str | list[str], **kwargs: Any) -> str:
    """Verbose version of check_output."""
    if isinstance(cmd, list):
        cmd = [str(element) for element in cmd]
    print(" ".join(cmd) if isinstance(cmd, list) else cmd)
    sys.stdout.flush()
    return cast("bytes", subprocess.check_output(cmd, stderr=subprocess.PIPE, **kwargs)).decode()  # noqa: S603


def image_diff(image1: NpNdarrayInt, image2: NpNdarrayInt) -> tuple[float, NpNdarrayInt]:
    """Do a diff between images."""
    width = max(image1.shape[1], image2.shape[1])
    height = max(image1.shape[0], image2.shape[0])
    image1 = cv2.resize(image1, (width, height))
    image2 = cv2.resize(image2, (width, height))

    image1 = image1 if len(image1.shape) == 2 else cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = image2 if len(image2.shape) == 2 else cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    score, diff = structural_similarity(image1, image2, full=True)  # type: ignore[no-untyped-call]
    diff = (255 - diff * 255).astype("uint8")
    return score, diff


class FunctionWithContextReturnsImage(Protocol):
    """Function with context and returns an image."""

    async def __call__(self, context: process_utils.Context) -> NpNdarrayInt | None:
        """Call the function."""


class FunctionWithContextReturnsNone(Protocol):
    """Function with context and no return."""

    async def __call__(self, context: process_utils.Context) -> None:
        """Call the function."""


class ExternalFunction(Protocol):
    """Function that call an external tool."""

    async def __call__(self, context: process_utils.Context, source: str, destination: str) -> None:
        """Call the function."""


# Decorate a step of the transform
class Process:
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

        async def wrapper(context: process_utils.Context) -> None:
            start_time = time.perf_counter()
            if self.ignore_error:
                try:
                    new_image = await func(context)
                    if new_image is not None and self.ignore_error:
                        context.image = new_image
                except Exception as exception:  # noqa: BLE001
                    print(exception)
                    if not jupyter_utils.is_ipython():
                        await add_intermediate_error(
                            context.config,
                            context.config_file_name,
                            exception,
                            traceback.format_exc().split("\n"),
                        )
            else:
                new_image = await func(context)
                if new_image is not None:
                    context.image = new_image
            elapsed_time = time.perf_counter() - start_time
            if os.environ.get("TIME", "FALSE") == "TRUE":
                print(f"Elapsed time in {self.name}: {round(elapsed_time)}s.")

            if self.progress:
                await context.save_progress_images(self.name)

        return wrapper


def external(func: ExternalFunction) -> FunctionWithContextReturnsImage:
    """Run an external tool."""

    async def wrapper(context: process_utils.Context) -> NpNdarrayInt | None:
        with tempfile.NamedTemporaryFile(suffix=".png") as source:
            assert context.image is not None
            cv2.imwrite(source.name, context.image)
            with tempfile.NamedTemporaryFile(suffix=".png") as destination:
                await func(context, source.name, destination.name)
                return cast("NpNdarrayInt", cv2.imread(destination.name))

    return wrapper


def get_contour_to_crop(
    contours: list[tuple[int, int, int, int]],
    margin_horizontal: int = 0,
    margin_vertical: int = 0,
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


async def crop(context: process_utils.Context, margin_horizontal: int, margin_vertical: int) -> None:
    """
    Do a crop on an image.

    Margin in px
    """
    image = context.get_masked()
    process_count = context.get_process_count()
    contours = await find_contours(
        image,
        context,
        "crop",
        context.config["args"].setdefault("crop", {}).setdefault("contour", {}),
    )

    if contours:
        for contour in contours:
            draw_rectangle(image, contour)
        await context.save_progress_images(
            "crop",
            image[context.get_index(image)] if jupyter_utils.is_ipython() else image,
            process_count=process_count,
            force=True,
        )

        x, y, width, height = get_contour_to_crop(contours, margin_horizontal, margin_vertical)
        context.crop(x, y, width, height)


def _get_level(context: process_utils.Context) -> tuple[bool, float, float]:
    level_config = context.config["args"].setdefault("level", {})
    level_ = level_config.setdefault("value", schema.LEVEL_VALUE_DEFAULT)
    min_p100 = 0.0
    max_p100 = 100.0
    if level_ is True:
        min_p100 = schema.MIN_LEVEL_DEFAULT
        max_p100 = schema.MAX_LEVEL_DEFAULT
    elif isinstance(level_, float | int):
        min_p100 = 0.0 + level_
        max_p100 = 100.0 - level_
    if level_ is not False:
        min_p100 = level_config.setdefault("min", min_p100)
        max_p100 = level_config.setdefault("max", max_p100)

    min_ = min_p100 / 100.0 * 255.0
    max_ = max_p100 / 100.0 * 255.0
    return level_ is not False, min_, max_


async def _histogram(
    context: process_utils.Context,
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
        if round(value) < len(histogram_data):
            hist_value = histogram_data[round(value)]
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
        if not jupyter_utils.is_ipython():
            plt.savefig(file.name)
            proc = await asyncio.create_subprocess_exec("gm", "convert", "-flatten", file.name, file.name)  # nosec
            await proc.communicate()
            assert proc.returncode == 0
            image = cv2.imread(file.name)
            await context.save_progress_images(
                "histogram",
                cast("NpNdarrayInt", image),
                image_prefix="log-" if log else "",
                process_count=process_count,
                force=True,
            )


@Process("histogram", progress=False)
async def histogram(context: process_utils.Context) -> None:
    """Create an image with the histogram of the current image."""
    noisy_image = img_as_ubyte(context.image)  # type: ignore[no-untyped-call]
    histogram_data, histogram_centers = skimage_histogram(noisy_image)
    histogram_max = max(histogram_data)
    process_count = context.get_process_count()

    await _histogram(context, histogram_data, histogram_centers, histogram_max, process_count, log=False)
    await _histogram(context, histogram_data, histogram_centers, histogram_max, process_count, log=True)


@Process("level")
async def level(context: process_utils.Context) -> NpNdarrayInt:
    """Do the level on an image."""
    assert context.image is not None
    img_yuv = cv2.cvtColor(context.image, cv2.COLOR_BGR2YUV)

    if context.config["args"].setdefault("level", {}).setdefault("auto", schema.AUTO_LEVEL_DEFAULT):
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cast("NpNdarrayInt", cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR))

    _, min_, max_ = _get_level(context)

    chanel_y = img_yuv[:, :, 0]
    mins = np.zeros(chanel_y.shape)
    maxs: NpNdarrayInt = np.zeros(chanel_y.shape) + 255

    values = (chanel_y - np.full_like(chanel_y, min_)) / (max_ - min_) * 255
    img_yuv[:, :, 0] = np.minimum(maxs, np.maximum(mins, values))
    return cast("NpNdarrayInt", cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR))


@Process("color-cut")
async def color_cut(context: process_utils.Context) -> None:
    """Set the near white to white and near black to black."""
    assert context.image is not None
    grayscale = cv2.cvtColor(context.image, cv2.COLOR_BGR2GRAY)

    white_mask = cv2.inRange(
        grayscale,
        cast(
            "np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]]",
            context.config["args"].setdefault("cut_white", schema.CUT_WHITE_DEFAULT),
        ),
        cast("np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]]", 255),
    )
    black_mask = cv2.inRange(
        grayscale,
        cast("np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]]", 0),
        cast(
            "np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]]",
            context.config["args"].setdefault("cut_black", schema.CUT_BLACK_DEFAULT),
        ),
    )
    context.image[white_mask == 255] = (255, 255, 255)
    context.image[black_mask == 255] = (0, 0, 0)


@Process("mask-cut")
async def cut(context: process_utils.Context) -> None:
    """Mask the image with the cut mask."""
    await context.do_initial_cut()


@Process("deskew")
async def deskew(context: process_utils.Context) -> None:
    """Deskew an image."""
    images_config = context.config.setdefault("images_config", {})
    image_config = images_config.setdefault(context.image_name, {}) if context.image_name else {}
    image_status = image_config.setdefault("status", {})
    angle = image_config.setdefault("angle", None)
    if angle is None:
        image = context.get_masked()
        image_rgb = rgba2rgb(image) if len(image.shape) == 3 and image.shape[2] == 4 else image  # type: ignore[no-untyped-call]
        grayscale = rgb2gray(image_rgb) if len(image_rgb.shape) == 3 else image_rgb

        deskew_configuration = context.config["args"].setdefault("deskew", {})
        async with _DESKEW_LOCK:
            skew_angle, debug_images = await asyncio.to_thread(
                determine_skew_debug_images,
                grayscale,
                min_angle=deskew_configuration.setdefault("min_angle", schema.DESKEW_MIN_ANGLE_DEFAULT),
                max_angle=deskew_configuration.setdefault("max_angle", schema.DESKEW_MAX_ANGLE_DEFAULT),
                min_deviation=deskew_configuration.setdefault(
                    "angle_derivation",
                    schema.DESKEW_ANGLE_DERIVATION_DEFAULT,
                ),
                sigma=deskew_configuration.setdefault("sigma", schema.DESKEW_SIGMA_DEFAULT),
                num_peaks=deskew_configuration.setdefault("num_peaks", schema.DESKEW_NUM_PEAKS_DEFAULT),
                angle_pm_90=deskew_configuration.setdefault("angle_pm_90", schema.DESKEW_ANGLE_PM_90_DEFAULT),
            )
        if skew_angle is not None:
            image_status["angle"] = float(skew_angle)
            angle = float(skew_angle)

        if not jupyter_utils.is_ipython():
            process_count = context.get_process_count()
            for name, debug_image in debug_images:
                await context.save_progress_images("skew", debug_image, name, process_count, force=True)

    if angle:
        context.rotate(angle)

        if context.image_name is not None:
            sources = [
                img for img in context.config.get("images", []) if f"{Path(context.image_name).stem}." in img
            ]
            if len(sources) == 1:
                assert context.root_folder
                image = process_utils.rotate_image(
                    cast("NpNdarrayInt", cv2.imread(str(context.root_folder / sources[0]))),
                    angle,
                    context.get_background_color(),
                )
                source_path = Path(sources[0])
                cv2.imwrite(
                    str(
                        context.root_folder
                        / "source"
                        / (source_path.stem + "-skew-corrected" + source_path.suffix),
                    ),
                    image,
                )


@Process("docrop")
async def docrop(context: process_utils.Context) -> None:
    """Crop an image."""
    # Margin in mm
    crop_config = context.config["args"].setdefault("crop", {})
    if not crop_config.setdefault("enabled", schema.CROP_ENABLED_DEFAULT):
        return
    margin_horizontal = context.get_px_value(
        crop_config.setdefault("margin_horizontal", schema.MARGIN_HORIZONTAL_DEFAULT),
    )
    margin_vertical = context.get_px_value(
        crop_config.setdefault("margin_vertical", schema.MARGIN_VERTICAL_DEFAULT),
    )
    await crop(context, round(margin_horizontal), round(margin_vertical))


@Process("sharpen")
async def sharpen(context: process_utils.Context) -> NpNdarrayInt | None:
    """Sharpen an image."""
    if (
        context.config["args"]
        .setdefault("sharpen", cast("schema.Sharpen", schema.SHARPEN_DEFAULT))
        .setdefault("enabled", schema.SHARPEN_ENABLED_DEFAULT)
        is False
    ):
        return None
    if context.image is None:
        msg = "The image is required"
        raise scan_to_paperless.ScanToPaperlessError(msg)
    image = cv2.GaussianBlur(context.image, (0, 0), 3)
    return cast("NpNdarrayInt", cv2.addWeighted(context.image, 1.5, image, -0.5, 0))


@Process("dither")
@external
async def dither(context: process_utils.Context, source: str, destination: str) -> None:
    """Dither an image."""
    if (
        context.config["args"]
        .setdefault("dither", cast("schema.Dither", schema.DITHER_DEFAULT))
        .setdefault("enabled", schema.DITHER_ENABLED_DEFAULT)
        is False
    ):
        return
    await call([*CONVERT, "+dither", source, destination])


@Process("autorotate", ignore_error=True)
async def autorotate(context: process_utils.Context) -> None:
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


def draw_line(
    image: NpNdarrayInt,
    vertical: bool,
    position: float | None,
    value: int | None,
    name: str,
    type_: str,
    color: tuple[int, int, int],
    line: tuple[int, int, int, int] | None = None,
) -> schema.Limit:
    """Draw a line on an image."""
    img_len = image.shape[0 if vertical else 1]
    if line is None:
        assert position is not None
        assert value is not None
        if vertical:
            cv2.rectangle(
                image,
                (int(position) - 1, img_len),
                (int(position) + 1, img_len - value),
                color,
                -1,
            )
            cv2.putText(
                image,
                name,
                (int(position), img_len - value),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                color,
                4,
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
    x = round(x)
    y = round(y)
    width = round(width)
    height = round(height)

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
    image: NpNdarrayInt,
    vertical: bool,
    config: schema.LineDetection,
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

    new_lines = [line for (line,) in lines if (line[0] == line[2] if vertical else line[1] == line[3])]

    def _key(line: tuple[int, int, int, int]) -> int:
        return line[1] - line[3] if vertical else line[2] - line[0]

    return cast("list[tuple[int, int, int, int]]", sorted(new_lines, key=_key)[:5])


def zero_ranges(values: NpNdarrayInt) -> NpNdarrayInt:
    """Create an array that is 1 where a is 0, and pad each end with an extra 0."""
    is_zero: NpNdarrayInt = np.concatenate([[0], np.equal(values, 0).view(np.int8), [0]])
    abs_diff = np.abs(np.diff(is_zero))
    # Runs start and end where abs_diff is 1.
    ranges = np.where(abs_diff == 1)[0].reshape(-1, 2)
    return cast("NpNdarrayInt", ranges)


def find_limit_contour(
    image: NpNdarrayInt,
    vertical: bool,
    contours: list[tuple[int, int, int, int]],
) -> list[int]:
    """Find the contour for assisted split."""
    image_size = image.shape[1 if vertical else 0]

    values = np.zeros(image_size)
    if vertical:
        for x, _, width, height in contours:
            x_int = round(x)
            for value in range(x_int, min(x_int + width, image_size)):
                values[value] += height
    else:
        for _, y, width, height in contours:
            y_int = round(y)
            for value in range(y_int, min(y_int + height, image_size)):
                values[value] += width

    ranges = zero_ranges(values)

    return [round(sum(ranges_) / 2) for ranges_ in ranges if ranges_[0] != 0 and ranges_[1] != image_size]


def find_limits(
    image: NpNdarrayInt,
    context: process_utils.Context,
    contours: list[tuple[int, int, int, int]],
    vertical: bool,
) -> tuple[list[int], list[tuple[int, int, int, int]]]:
    """Find the limit for assisted split."""
    contours_limits = find_limit_contour(image, vertical, contours)
    lines = find_lines(
        image,
        vertical,
        context.config["args"].setdefault("limit_detection", {}).setdefault("line", {}),
    )
    return contours_limits, lines


def fill_limits(
    image: NpNdarrayInt,
    contours_limits: list[int],
    lines: list[tuple[int, int, int, int]],
    vertical: bool,
) -> list[schema.Limit]:
    """Fill the limit for assisted split."""
    third_image_size = int(image.shape[0 if vertical else 1] / 3)
    limits: list[schema.Limit] = []
    prefix = "V" if vertical else "H"
    for index, line in enumerate(lines):
        limits.append(
            draw_line(image, vertical, None, None, f"{prefix}L{index}", "line detection", (255, 0, 0), line),
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
            ),
        )
    if not limits:
        half_image_size = image.shape[1 if vertical else 0] / 2
        limits.append(
            draw_line(
                image,
                vertical,
                half_image_size,
                third_image_size,
                f"{prefix}C",
                "image center",
                (0, 0, 255),
            ),
        )

    return limits


async def find_contours(
    image: NpNdarrayInt,
    context: process_utils.Context,
    name: str,
    config: schema.Contour,
) -> list[tuple[int, int, int, int]]:
    """Find the contours on an image."""
    block_size = context.get_px_value(
        config.setdefault("threshold_block_size", schema.THRESHOLD_BLOCK_SIZE_DEFAULT),
    )
    threshold_value_c = config.setdefault("threshold_value_c", schema.THRESHOLD_VALUE_C_DEFAULT)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    block_size = int(round(block_size / 2) * 2)

    # Clean the image using method with the inverted binarized image
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        block_size + 1,
        threshold_value_c,
    )
    if context.is_progress() or jupyter_utils.is_ipython():
        if jupyter_utils.is_ipython():
            print("Threshold")
        thresh_rgb = cast("NpNdarrayInt", cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB))
        await context.save_progress_images(
            "threshold",
            thresh_rgb[context.get_index(thresh_rgb)] if jupyter_utils.is_ipython() else thresh,
        )

    return _find_contours_thresh(image, cast("NpNdarrayInt", thresh), context, name, config)


def _find_contours_thresh(
    image: NpNdarrayInt,
    thresh: NpNdarrayInt,
    context: process_utils.Context,
    name: str,
    config: schema.Contour,
) -> list[tuple[int, int, int, int]]:
    min_size = context.get_px_value(config.setdefault("min_box_size", schema.MIN_BOX_SIZE_DEFAULT[name]))
    min_black = config.setdefault("min_box_black", schema.MIN_BOX_BLACK_DEFAULT)
    kernel_size = context.get_px_value(
        config.setdefault("contour_kernel_size", schema.CONTOUR_KERNEL_SIZE_DEFAULT),
    )

    kernel_size = round(kernel_size / 2)

    # Assign a rectangle kernel size
    kernel: NpNdarrayInt = np.ones((kernel_size, kernel_size), "uint8")
    par_img = cv2.dilate(thresh, kernel, iterations=5)

    contours, _ = cv2.findContours(par_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = []

    for cnt in contours:
        x, y, width, height = cv2.boundingRect(cnt)
        if width > min_size and height > min_size:
            contour_image = process_utils.crop_image(
                image,
                x,
                y,
                width,
                height,
                context.get_background_color(),
            )
            imagergb = (
                rgba2rgb(contour_image)  # type: ignore[no-untyped-call]
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
                    ),
                )

    return result


def _update_config(config: schema.Configuration) -> None:
    """Convert the old configuration to the new one."""
    old_config = cast("dict[str, Any]", config)
    # no_crop => crop.enabled (inverted)
    if "no_crop" in old_config["args"]:
        config["args"].setdefault("crop", {}).setdefault("enabled", not old_config["args"]["no_crop"])
        del old_config["args"]["no_crop"]
    # margin_horizontal => crop.margin_horizontal
    if "margin_horizontal" in old_config["args"]:
        config["args"].setdefault("crop", {}).setdefault(
            "margin_horizontal",
            old_config["args"]["margin_horizontal"],
        )
        del old_config["args"]["margin_horizontal"]
    # margin_vertical => crop.margin_vertical
    if "margin_vertical" in old_config["args"]:
        config["args"].setdefault("crop", {}).setdefault(
            "margin_vertical",
            old_config["args"]["margin_vertical"],
        )
        del old_config["args"]["margin_vertical"]
    # crop.min_box_size => crop.contour.min_box_size
    if "min_box_size" in old_config["args"].get("crop", {}):
        config["args"].setdefault("crop", {}).setdefault("contour", {}).setdefault(
            "min_box_size",
            old_config["args"]["crop"]["min_box_size"],
        )
        del old_config["args"]["crop"]["min_box_size"]
    # crop.min_box_black => crop.contour.min_box_black
    if "min_box_black" in old_config["args"].get("crop", {}):
        config["args"].setdefault("crop", {}).setdefault("contour", {}).setdefault(
            "min_box_black",
            old_config["args"]["crop"]["min_box_black"],
        )
        del old_config["args"]["crop"]["min_box_black"]
    # crop.contour_kernel_size => crop.contour.contour_kernel_size
    if "contour_kernel_size" in old_config["args"].get("crop", {}):
        config["args"].setdefault("crop", {}).setdefault("contour", {}).setdefault(
            "contour_kernel_size",
            old_config["args"]["crop"]["contour_kernel_size"],
        )
        del old_config["args"]["crop"]["contour_kernel_size"]
    # crop.threshold_block_size => crop.contour.threshold_block_size
    if "threshold_block_size" in old_config["args"].get("crop", {}):
        config["args"].setdefault("crop", {}).setdefault("contour", {}).setdefault(
            "threshold_block_size",
            old_config["args"]["crop"]["threshold_block_size"],
        )
        del old_config["args"]["crop"]["threshold_block_size"]
    # crop.threshold_value_c => crop.contour.threshold_value_c
    if "threshold_value_c" in old_config["args"].get("crop", {}):
        config["args"].setdefault("crop", {}).setdefault("contour", {}).setdefault(
            "threshold_value_c",
            old_config["args"]["crop"]["threshold_value_c"],
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
            "min_box_size",
            old_config["args"]["min_box_size_empty"],
        )
        del old_config["args"]["min_box_size_empty"]
    # min_box_black_empty => empty.contour.min_box_black
    if "min_box_black_empty" in old_config["args"]:
        config["args"].setdefault("empty", {}).setdefault("contour", {}).setdefault(
            "min_box_black",
            old_config["args"]["min_box_black_empty"],
        )
        del old_config["args"]["min_box_black_empty"]
    # contour_kernel_size_empty => empty.contour.contour_kernel_size
    if "contour_kernel_size_empty" in old_config["args"]:
        config["args"].setdefault("empty", {}).setdefault("contour", {}).setdefault(
            "contour_kernel_size",
            old_config["args"]["contour_kernel_size_empty"],
        )
        del old_config["args"]["contour_kernel_size_empty"]
    # threshold_block_size_empty => empty.contour.threshold_block_size
    if "threshold_block_size_empty" in old_config["args"]:
        config["args"].setdefault("empty", {}).setdefault("contour", {}).setdefault(
            "threshold_block_size",
            old_config["args"]["threshold_block_size_empty"],
        )
        del old_config["args"]["threshold_block_size_empty"]
    # threshold_value_c_empty => empty.contour.threshold_value_c
    if "threshold_value_c_empty" in old_config["args"]:
        config["args"].setdefault("empty", {}).setdefault("contour", {}).setdefault(
            "threshold_value_c",
            old_config["args"]["threshold_value_c_empty"],
        )
        del old_config["args"]["threshold_value_c_empty"]
    # min_box_size_limit => limit_detection.contour.min_box_size
    if "min_box_size_limit" in old_config["args"]:
        config["args"].setdefault("limit_detection", {}).setdefault("contour", {}).setdefault(
            "min_box_size",
            old_config["args"]["min_box_size_limit"],
        )
        del old_config["args"]["min_box_size_limit"]
    # min_box_black_limit => limit_detection.contour.min_box_black
    if "min_box_black_limit" in old_config["args"]:
        config["args"].setdefault("limit_detection", {}).setdefault("contour", {}).setdefault(
            "min_box_black",
            old_config["args"]["min_box_black_limit"],
        )
        del old_config["args"]["min_box_black_limit"]
    # contour_kernel_size_limit => limit_detection.contour.contour_kernel_size
    if "contour_kernel_size_limit" in old_config["args"]:
        config["args"].setdefault("limit_detection", {}).setdefault("contour", {}).setdefault(
            "contour_kernel_size",
            old_config["args"]["contour_kernel_size_limit"],
        )
        del old_config["args"]["contour_kernel_size_limit"]
    # threshold_block_size_limit => limit_detection.contour.threshold_block_size
    if "threshold_block_size_limit" in old_config["args"]:
        config["args"].setdefault("limit_detection", {}).setdefault("contour", {}).setdefault(
            "threshold_block_size",
            old_config["args"]["threshold_block_size_limit"],
        )
        del old_config["args"]["threshold_block_size_limit"]
    # threshold_value_c_limit => limit_detection.contour.threshold_value_c
    if "threshold_value_c_limit" in old_config["args"]:
        config["args"].setdefault("limit_detection", {}).setdefault("contour", {}).setdefault(
            "threshold_value_c",
            old_config["args"]["threshold_value_c_limit"],
        )
        del old_config["args"]["threshold_value_c_limit"]
    # auto_mask: null => auto_mask.enabled: false
    if "auto_mask" in old_config["args"] and (
        old_config["args"]["auto_mask"] is True or old_config["args"]["auto_mask"] is False
    ):
        config["args"]["mask"] = {"enabled": old_config["args"]["auto_mask"]}
    if "auto_mask" in old_config["args"] and old_config["args"]["auto_mask"] is None:
        config["args"]["mask"] = {"enabled": False}
    # auto_mask.lower_hsv_color => auto_mask.auto_mask.lower_hsv_color
    if "lower_hsv_color" in old_config["args"].get("auto_mask", {}):
        config["args"].setdefault("mask", {}).setdefault("auto_mask", {}).setdefault(
            "lower_hsv_color",
            old_config["args"]["auto_mask"]["lower_hsv_color"],
        )
        del old_config["args"]["auto_mask"]["lower_hsv_color"]
    # auto_mask.upper_hsv_color => auto_mask.auto_mask.upper_hsv_color
    if "upper_hsv_color" in old_config["args"].get("auto_mask", {}):
        config["args"].setdefault("mask", {}).setdefault("auto_mask", {}).setdefault(
            "upper_hsv_color",
            old_config["args"]["auto_mask"]["upper_hsv_color"],
        )
        del old_config["args"]["auto_mask"]["upper_hsv_color"]
    # auto_mask.de_noise_morphology => auto_mask.auto_mask.de_noise_morphology
    if "de_noise_morphology" in old_config["args"].get("auto_mask", {}):
        config["args"].setdefault("mask", {}).setdefault("auto_mask", {}).setdefault(
            "de_noise_morphology",
            old_config["args"]["auto_mask"]["de_noise_morphology"],
        )
        del old_config["args"]["auto_mask"]["de_noise_morphology"]
    # auto_mask.inverse_mask => auto_mask.auto_mask.inverse_mask
    if "inverse_mask" in old_config["args"].get("auto_mask", {}):
        config["args"].setdefault("mask", {}).setdefault("auto_mask", {}).setdefault(
            "inverse_mask",
            old_config["args"]["auto_mask"]["inverse_mask"],
        )
        del old_config["args"]["auto_mask"]["inverse_mask"]
    # auto_mask.de_noise_size => auto_mask.auto_mask.de_noise_size
    if "de_noise_size" in old_config["args"].get("auto_mask", {}):
        config["args"].setdefault("mask", {}).setdefault("auto_mask", {}).setdefault(
            "de_noise_size",
            old_config["args"]["auto_mask"]["de_noise_size"],
        )
        del old_config["args"]["auto_mask"]["de_noise_size"]
    # auto_mask.de_noise_level => auto_mask.auto_mask.de_noise_level
    if "de_noise_level" in old_config["args"].get("auto_mask", {}):
        config["args"].setdefault("mask", {}).setdefault("auto_mask", {}).setdefault(
            "de_noise_level",
            old_config["args"]["auto_mask"]["de_noise_level"],
        )
        del old_config["args"]["auto_mask"]["de_noise_level"]
    # auto_mask.buffer_size => auto_mask.auto_mask.buffer_size
    if "buffer_size" in old_config["args"].get("auto_mask", {}):
        config["args"].setdefault("mask", {}).setdefault("auto_mask", {}).setdefault(
            "buffer_size",
            old_config["args"]["auto_mask"]["buffer_size"],
        )
        del old_config["args"]["auto_mask"]["buffer_size"]
    # auto_mask.buffer_level => auto_mask.auto_mask.buffer_level
    if "buffer_level" in old_config["args"].get("auto_mask", {}):
        config["args"].setdefault("mask", {}).setdefault("auto_mask", {}).setdefault(
            "buffer_level",
            old_config["args"]["auto_mask"]["buffer_level"],
        )
        del old_config["args"]["auto_mask"]["buffer_level"]
    # auto_mask.additional_filename => auto_mask.auto_mask.additional_filename
    if "additional_filename" in old_config["args"].get("auto_mask", {}):
        config["args"].setdefault("mask", {}).setdefault(
            "additional_filename",
            old_config["args"]["auto_mask"]["additional_filename"],
        )
        del old_config["args"]["auto_mask"]["additional_filename"]
    # auto_cut: null => auto_cut.enabled: false
    if "auto_cut" in old_config["args"] and (
        old_config["args"]["auto_cut"] is True or old_config["args"]["auto_cut"] is False
    ):
        config["args"]["cut"] = {"enabled": old_config["args"]["auto_cut"]}
    if "auto_cut" in old_config["args"] and old_config["args"]["auto_cut"] is None:
        config["args"]["cut"] = {"enabled": False}
    # auto_cut.lower_hsv_color => auto_cut.auto_mask.lower_hsv_color
    if "lower_hsv_color" in old_config["args"].get("auto_cut", {}):
        config["args"].setdefault("cut", {}).setdefault("auto_mask", {}).setdefault(
            "lower_hsv_color",
            old_config["args"]["auto_cut"]["lower_hsv_color"],
        )
        del old_config["args"]["auto_cut"]["lower_hsv_color"]
    # auto_cut.upper_hsv_color => auto_cut.auto_mask.upper_hsv_color
    if "upper_hsv_color" in old_config["args"].get("auto_cut", {}):
        config["args"].setdefault("cut", {}).setdefault("auto_mask", {}).setdefault(
            "upper_hsv_color",
            old_config["args"]["auto_cut"]["upper_hsv_color"],
        )
        del old_config["args"]["auto_cut"]["upper_hsv_color"]
    # auto_cut.de_noise_morphology => auto_cut.auto_mask.de_noise_morphology
    if "de_noise_morphology" in old_config["args"].get("auto_cut", {}):
        config["args"].setdefault("cut", {}).setdefault("auto_mask", {}).setdefault(
            "de_noise_morphology",
            old_config["args"]["auto_cut"]["de_noise_morphology"],
        )
        del old_config["args"]["auto_cut"]["de_noise_morphology"]
    # auto_cut.inverse_mask => auto_cut.auto_mask.inverse_mask
    if "inverse_mask" in old_config["args"].get("auto_cut", {}):
        config["args"].setdefault("cut", {}).setdefault("auto_mask", {}).setdefault(
            "inverse_mask",
            old_config["args"]["auto_cut"]["inverse_mask"],
        )
        del old_config["args"]["auto_cut"]["inverse_mask"]
    # auto_cut.de_noise_size => auto_cut.auto_mask.de_noise_size
    if "de_noise_size" in old_config["args"].get("auto_cut", {}):
        config["args"].setdefault("cut", {}).setdefault("auto_mask", {}).setdefault(
            "de_noise_size",
            old_config["args"]["auto_cut"]["de_noise_size"],
        )
        del old_config["args"]["auto_cut"]["de_noise_size"]
    # auto_cut.de_noise_level => auto_cut.auto_mask.de_noise_level
    if "de_noise_level" in old_config["args"].get("auto_cut", {}):
        config["args"].setdefault("cut", {}).setdefault("auto_mask", {}).setdefault(
            "de_noise_level",
            old_config["args"]["auto_cut"]["de_noise_level"],
        )
        del old_config["args"]["auto_cut"]["de_noise_level"]
    # auto_cut.buffer_size => auto_cut.auto_mask.buffer_size
    if "buffer_size" in old_config["args"].get("auto_cut", {}):
        config["args"].setdefault("cut", {}).setdefault("auto_mask", {}).setdefault(
            "buffer_size",
            old_config["args"]["auto_cut"]["buffer_size"],
        )
        del old_config["args"]["auto_cut"]["buffer_size"]
    # auto_cut.buffer_level => auto_cut.auto_mask.buffer_level
    if "buffer_level" in old_config["args"].get("auto_cut", {}):
        config["args"].setdefault("cut", {}).setdefault("auto_mask", {}).setdefault(
            "buffer_level",
            old_config["args"]["auto_cut"]["buffer_level"],
        )
        del old_config["args"]["auto_cut"]["buffer_level"]
    # auto_cut.additional_filename => auto_cut.auto_mask.additional_filename
    if "additional_filename" in old_config["args"].get("auto_cut", {}):
        config["args"].setdefault("cut", {}).setdefault(
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
            "options",
            old_config["args"]["pngquant_options"],
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
            "enabled",
            not old_config["args"]["no_auto_rotate"],
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
    # level => level.value
    if "level" in old_config["args"] and not isinstance(old_config["args"]["level"], dict):
        config["args"]["level"] = {"value": old_config["args"]["level"]}
    # auto_level => level.auto
    if "auto_level" in old_config["args"]:
        config["args"].setdefault("level", {}).setdefault("auto", old_config["args"]["auto_level"])
        del old_config["args"]["auto_level"]
    # min_level => level.min
    if "min_level" in old_config["args"]:
        config["args"].setdefault("level", {}).setdefault("min", old_config["args"]["min_level"])
        del old_config["args"]["min_level"]
    # max_level => level.max
    if "max_level" in old_config["args"]:
        config["args"].setdefault("level", {}).setdefault("max", old_config["args"]["max_level"])
        del old_config["args"]["max_level"]


async def transform(
    config: schema.Configuration,
    step: schema.Step,
    config_file_name: Path,
    root_folder: Path,
    status: scan_to_paperless.status.Status | None = None,
) -> schema.Step:
    """Apply the transforms on a document."""
    if "intermediate_error" in config:
        del config["intermediate_error"]

    images_path = []
    process_count = 0

    if config["args"].setdefault("assisted_split", schema.ASSISTED_SPLIT_DEFAULT):
        config["assisted_split"] = []

    for index, image in enumerate(step["sources"]):
        image_path = Path(image)
        if status is not None:
            await status.set_status(config_file_name, -1, f"Transform ({image_path.name})", write=True)
        image_name = f"{image_path.name.rsplit('.')[0]}.png"
        context = process_utils.Context(config, step, config_file_name, root_folder, image_name)
        if context.image_name is None:
            msg = "Image name is required"
            raise scan_to_paperless.ScanToPaperlessError(msg)
        async with await anyio.open_file(root_folder / image, "rb") as f:
            img_array = np.asarray(bytearray(await f.read()), dtype=np.uint8)
            context.image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        assert context.image is not None
        images_config = context.config.setdefault("images_config", {})
        image_config = images_config.setdefault(context.image_name, {})
        image_status = image_config.setdefault("status", {})
        assert context.image is not None
        image_status["size"] = list(context.image.shape[:2][::-1])
        await histogram(context)
        await level(context)
        await color_cut(context)
        await context.init_mask()
        await cut(context)
        await deskew(context)
        await docrop(context)
        await sharpen(context)
        await dither(context)  # pylint: disable=no-value-for-parameter,unknown-option-value, added by decorator
        await autorotate(context)

        # Is empty ?
        empty_config = config["args"].setdefault("empty", {})
        if empty_config.setdefault("enabled", schema.EMPTY_ENABLED_DEFAULT):
            contours = await find_contours(
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
            name = root_folder / context.image_name
            source = await context.save_progress_images("assisted-split", context.image, force=True)
            assert source
            assisted_split["source"] = str(source)

            config["assisted_split"].append(assisted_split)
            destinations = [len(step["sources"]) * 2 - index, index + 1]
            if index % 2 == 1:
                destinations.reverse()
            assisted_split["destinations"] = list(destinations)

            limits = []
            assert context.image is not None

            contours = await find_contours(
                context.image,
                context,
                "limit",
                config["args"].setdefault("limit_detection", {}).setdefault("contour", {}),
            )
            vertical_contour_limits, vertical_lines = find_limits(
                context.image,
                context,
                contours,
                vertical=True,
            )
            horizontal_contour_limits, horizontal_lines = find_limits(
                context.image,
                context,
                contours,
                vertical=False,
            )

            for contour_limit in contours:
                draw_rectangle(context.image, contour_limit, border=False)
            limits.extend(fill_limits(context.image, vertical_contour_limits, vertical_lines, vertical=True))
            limits.extend(
                fill_limits(context.image, horizontal_contour_limits, horizontal_lines, vertical=False),
            )
            assisted_split["limits"] = limits

            rule_config = config["args"].setdefault("rule", {})
            if rule_config.setdefault("enabled", schema.RULE_ENABLE_DEFAULT):
                minor_graduation_space = rule_config.setdefault(
                    "minor_graduation_space",
                    schema.RULE_MINOR_GRADUATION_SPACE_DEFAULT,
                )
                major_graduation_space = rule_config.setdefault(
                    "major_graduation_space",
                    schema.RULE_MAJOR_GRADUATION_SPACE_DEFAULT,
                )
                lines_space = rule_config.setdefault("lines_space", schema.RULE_LINES_SPACE_DEFAULT)
                minor_graduation_size = rule_config.setdefault(
                    "minor_graduation_size",
                    schema.RULE_MINOR_GRADUATION_SIZE_DEFAULT,
                )
                major_graduation_size = rule_config.setdefault(
                    "major_graduation_size",
                    schema.RULE_MAJOR_GRADUATION_SIZE_DEFAULT,
                )
                graduation_color = rule_config.setdefault(
                    "graduation_color",
                    schema.RULE_GRADUATION_COLOR_DEFAULT,
                )
                lines_color = rule_config.setdefault("lines_color", schema.RULE_LINES_COLOR_DEFAULT)
                lines_opacity = rule_config.setdefault("lines_opacity", schema.RULE_LINES_OPACITY_DEFAULT)
                graduation_text_font_filename = rule_config.setdefault(
                    "graduation_text_font_filename",
                    schema.RULE_GRADUATION_TEXT_FONT_FILENAME_DEFAULT,
                )
                graduation_text_font_size = rule_config.setdefault(
                    "graduation_text_font_size",
                    schema.RULE_GRADUATION_TEXT_FONT_SIZE_DEFAULT,
                )
                graduation_text_font_color = rule_config.setdefault(
                    "graduation_text_font_color",
                    schema.RULE_GRADUATION_TEXT_FONT_COLOR_DEFAULT,
                )
                graduation_text_margin = rule_config.setdefault(
                    "graduation_text_margin",
                    schema.RULE_GRADUATION_TEXT_MARGIN_DEFAULT,
                )

                x = minor_graduation_space
                while x < context.image.shape[1]:
                    if x % lines_space == 0:
                        sub_img = context.image[0 : context.image.shape[0], x : x + 1]
                        mask_image = np.zeros(sub_img.shape, dtype=np.uint8)
                        mask_image[:, :] = lines_color
                        opacity_result = cv2.addWeighted(
                            sub_img,
                            1 - lines_opacity,
                            mask_image,
                            lines_opacity,
                            1.0,
                        )
                        if opacity_result is not None:
                            context.image[0 : context.image.shape[0], x : x + 1] = opacity_result

                    if x % major_graduation_space == 0:
                        cv2.rectangle(
                            context.image,
                            (x, 0),
                            (x + 1, major_graduation_size),
                            graduation_color,
                            -1,
                        )
                    else:
                        cv2.rectangle(
                            context.image,
                            (x, 0),
                            (x + 1, minor_graduation_size),
                            graduation_color,
                            -1,
                        )
                    x += minor_graduation_space

                y = minor_graduation_space
                while y < context.image.shape[0]:
                    if y % lines_space == 0:
                        sub_img = context.image[y : y + 1, 0 : context.image.shape[1]]
                        mask_image = np.zeros(sub_img.shape, dtype=np.uint8)
                        mask_image[:, :] = lines_color
                        opacity_result = cv2.addWeighted(
                            sub_img,
                            1 - lines_opacity,
                            mask_image,
                            lines_opacity,
                            1.0,
                        )
                        if opacity_result is not None:
                            context.image[y : y + 1, 0 : context.image.shape[1]] = opacity_result
                    if y % major_graduation_space == 0:
                        cv2.rectangle(
                            context.image,
                            (0, y),
                            (major_graduation_size, y + 1),
                            graduation_color,
                            -1,
                        )
                    else:
                        cv2.rectangle(
                            context.image,
                            (0, y),
                            (minor_graduation_size, y + 1),
                            graduation_color,
                            -1,
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

            cv2.imwrite(str(name), context.image)
            assisted_split["image"] = context.image_name
            images_path.append(name)
        else:
            img2 = root_folder / context.image_name
            cv2.imwrite(str(img2), context.image)
            images_path.append(img2)
        process_count = context.process_count

    from scan_to_paperless import jupyter  # noqa: PLC0415, RUF100

    await jupyter.create_transform_notebook(root_folder, context, step)

    progress = os.environ.get("PROGRESS", "FALSE") == "TRUE"

    count = context.get_process_count()
    for image_path in images_path:
        if progress:
            await _save_progress(context.root_folder, count, "finalize", image_path.name, image_path)

    if config["args"].setdefault("colors", schema.COLORS_DEFAULT):
        count = context.get_process_count()
        for image_path in images_path:
            await call([*CONVERT, "-colors", str(config["args"]["colors"]), str(image_path), str(image_path)])
            if progress:
                await _save_progress(context.root_folder, count, "colors", image_path.name, image_path)

    pngquant_config = config["args"].setdefault("pngquant", cast("schema.Pngquant", schema.PNGQUANT_DEFAULT))
    if not config["args"].setdefault("jpeg", cast("schema.Jpeg", schema.JPEG_DEFAULT)).setdefault(
        "enabled",
        schema.JPEG_ENABLED_DEFAULT,
    ) and pngquant_config.setdefault("enabled", schema.PNGQUANT_ENABLED_DEFAULT):
        count = context.get_process_count()
        for image_path in images_path:
            with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
                await call(
                    [
                        "pngquant",
                        f"--output={temp_file.name}",
                        *pngquant_config.setdefault(
                            "options",
                            schema.PNGQUANT_OPTIONS_DEFAULT,
                        ),
                        *["--", str(image_path)],
                    ],
                    check=False,
                )
                temp_path = Path(temp_file.name)
                if (await temp_path.stat()).st_size > 0:
                    await call(["cp", temp_file.name, str(image_path)])
            if progress:
                await _save_progress(context.root_folder, count, "pngquant", image_path.name, image_path)

    if not config["args"].setdefault("jpeg", {}).setdefault(
        "enabled",
        schema.JPEG_ENABLED_DEFAULT,
    ) and config["args"].setdefault("optipng", {}).setdefault(
        "enabled",
        not pngquant_config.setdefault("enabled", schema.PNGQUANT_ENABLED_DEFAULT),
    ):
        count = context.get_process_count()
        for image_path in images_path:
            await call(["optipng", str(image_path)], check=False)
            if progress:
                await _save_progress(context.root_folder, count, "optipng", image_path.name, image_path)

    if config["args"].setdefault("jpeg", {}).setdefault("enabled", schema.JPEG_ENABLED_DEFAULT):
        count = context.get_process_count()
        new_images = []
        for image_path in images_path:
            jpeg_img = Path(f"{image_path.stem}.jpeg")
            proc = await asyncio.create_subprocess_exec(  # nosec
                "gm",
                "convert",
                str(image_path),
                "-quality",
                str(config["args"].setdefault("jpeg", {}).setdefault("quality", schema.JPEG_QUALITY_DEFAULT)),
                str(jpeg_img),
            )
            await proc.communicate()
            assert proc.returncode == 0
            new_images.append(jpeg_img)
            if progress:
                await _save_progress(context.root_folder, count, "to-jpeg", image_path.name, image_path)

        images_path = new_images

    # Free matplotlib allocations
    plt.clf()
    plt.close("all")

    disable_remove_to_continue = config["args"].setdefault(
        "no_remove_to_continue",
        schema.NO_REMOVE_TO_CONTINUE_DEFAULT,
    )
    if not disable_remove_to_continue or config["args"].setdefault(
        "assisted_split",
        schema.ASSISTED_SPLIT_DEFAULT,
    ):
        async with await (root_folder / "REMOVE_TO_CONTINUE").open("w", encoding="utf-8"):
            pass

    return {
        "sources": [str(path) for path in images_path],
        "name": (
            scan_to_paperless.status.STATUS_ASSISTED_SPLIT
            if config["args"].setdefault("assisted_split", schema.ASSISTED_SPLIT_DEFAULT)
            else scan_to_paperless.status.STATUS_FINALIZE
        ),
        "process_count": process_count,
    }


async def _save_progress(
    root_folder: Path | None,
    count: int,
    name: str,
    image_name: str,
    image: Path,
) -> None:
    assert root_folder
    name = f"{count}-{name}"
    dest_folder = root_folder / name
    if not await dest_folder.exists():
        await dest_folder.mkdir(parents=True)
    dest_image = dest_folder / image_name
    try:
        await call(["cp", str(image), str(dest_image)])
    except Exception as exception:  # noqa: BLE001
        print(exception)


async def save(
    context: process_utils.Context,
    root_folder: Path,
    image: Path,
    folder: str,
    force: bool = False,
) -> Path:
    """Save the current image in a subfolder if progress mode in enabled."""
    if force or context.is_progress():
        dest_folder = root_folder / folder
        if not dest_folder.exists():
            await dest_folder.mkdir(parents=True)
        dest_file = dest_folder / image.name
        await anyio.to_thread.run_sync(shutil.copyfile, image, dest_file)
        return dest_file
    return Path(image)


class Item(TypedDict, total=False):
    """
    Image content and position.

    Used to create the final document
    """

    pos: int
    file: IO[bytes]


async def split(
    config: schema.Configuration,
    step: schema.Step,
    root_folder: Path,
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
                msg = (
                    f"Wrong number of destinations ({len(assisted_split['destinations'])}), "
                    f"vertical: {nb_horizontal}, height: {nb_vertical}, image: '{assisted_split['source']}'"
                )
                raise scan_to_paperless.ScanToPaperlessError(msg)

    for assisted_split in config["assisted_split"]:
        if "image" in assisted_split:
            image_path = root_folder / assisted_split["image"]
            if await image_path.exists():
                await image_path.unlink()

    append: dict[str | int, list[Item]] = {}
    transformed_images = []
    for assisted_split in config["assisted_split"]:
        image = assisted_split["source"]
        context = process_utils.Context(config, step)
        width, height = (
            int(e) for e in output([*CONVERT, image, "-format", "%w %h", "info:-"]).strip().split(" ")
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
                    process_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
                        suffix=".png",
                    )
                    await call(
                        [
                            *CONVERT,
                            "-crop",
                            f"{vertical_value - vertical_margin - last_x}x"
                            f"{horizontal_value - horizontal_margin - last_y}+{last_x}+{last_y}",
                            "+repage",
                            image,
                            process_file.name,
                        ],
                    )
                    last_x = vertical_value + vertical_margin

                    if re.match(r"[0-9]+\.[0-9]+", str(destination)):
                        page, page_pos = (int(e) for e in str(destination).split("."))
                    else:
                        page = int(destination)
                        page_pos = 0

                    await save(
                        context,
                        root_folder,
                        Path(process_file.name),
                        f"{context.get_process_count()}-split",
                    )
                    crop_config = context.config["args"].setdefault("crop", {})
                    margin_horizontal = context.get_px_value(
                        crop_config.setdefault("margin_horizontal", schema.MARGIN_HORIZONTAL_DEFAULT),
                    )
                    margin_vertical = context.get_px_value(
                        crop_config.setdefault("margin_vertical", schema.MARGIN_VERTICAL_DEFAULT),
                    )
                    context.image = cv2.imread(process_file.name)
                    if crop_config.setdefault("enabled", schema.CROP_ENABLED_DEFAULT):
                        await crop(context, round(margin_horizontal), round(margin_vertical))
                        process_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
                            suffix=".png",
                        )
                        cv2.imwrite(process_file.name, context.image)  # type: ignore[arg-type]
                        await save(
                            context,
                            root_folder,
                            Path(process_file.name),
                            f"{context.get_process_count()}-crop",
                        )
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
            msg = f"Mix of limit type for page '{page_number}'"
            raise scan_to_paperless.ScanToPaperlessError(msg)

        with tempfile.NamedTemporaryFile(suffix=".png") as process_file:
            await call(
                CONVERT
                + [e["file"].name for e in sorted(items, key=lambda e: e["pos"])]
                + [
                    "-background",
                    "#ffffff",
                    "-gravity",
                    "center",
                    "+append" if vertical else "-append",
                    process_file.name,
                ],
            )
            await save(context, root_folder, Path(process_file.name), f"{process_count}-split")
            img2 = root_folder / f"image-{page_number}.png"
            await call([*CONVERT, process_file.name, str(img2)])
            transformed_images.append(img2)
    process_count += 1

    return {
        "sources": [str(path) for path in transformed_images],
        "name": scan_to_paperless.status.STATUS_FINALIZE,
        "process_count": process_count,
    }


async def finalize(
    config: schema.Configuration,
    step: schema.Step,
    root_folder: Path,
    status: scan_to_paperless.status.Status | None = None,
) -> None:
    """
    Do final step on document generation.

    convert in one pdf and copy with the right name in the consume folder
    """
    name = root_folder.name
    destination = Path(os.environ.get("SCAN_CODES_FOLDER", "/scan-codes")) / f"{name}.pdf"

    if await destination.exists():
        return

    images_filenames = step["sources"]

    if config["args"].setdefault("append_credit_card", schema.APPEND_CREDIT_CARD_DEFAULT):
        if status is not None:
            await status.set_status(name, -1, "Finalize (credit card append)", write=True)
        images2 = [image for image in images_filenames if await Path(image).exists()]

        file_path = root_folder / "append.png"
        await call(
            [*CONVERT, *images2, "-background", "#ffffff", "-gravity", "center", "-append", str(file_path)],
        )
        # To stack vertically (img1 over img2):
        # vis = np.concatenate((img1, img2), axis=0)
        # To stack horizontally (img1 to the left of img2):
        # vis = np.concatenate((img1, img2), axis=1)
        images_filenames = [str(file_path)]

    pdf: list[Path] = []
    for image_filename in images_filenames:
        image_path = Path(image_filename)
        if status is not None:
            await status.set_status(name, -1, f"Finalize ({image_path.name})", write=True)
        if await image_path.exists():
            image_name = image_path.stem
            file_path = root_folder / f"{image_name}.pdf"
            tesseract_configuration = config["args"].setdefault("tesseract", {})
            if tesseract_configuration.setdefault("enabled", schema.TESSERACT_ENABLED_DEFAULT):
                async with await file_path.open("w", encoding="utf8") as output_file:
                    _, stderr, _ = await run(
                        [
                            "tesseract",
                            "--dpi",
                            str(config["args"].setdefault("dpi", schema.DPI_DEFAULT)),
                            "-l",
                            tesseract_configuration.setdefault("lang", schema.TESSERACT_LANG_DEFAULT),
                            str(image_filename),
                            "stdout",
                            "pdf",
                        ],
                        stdout=output_file,
                    )
                    if stderr:
                        print(stderr.decode())
            else:
                await call([*CONVERT, str(image_filename), "+repage", str(file_path)])
            pdf.append(file_path)

    if status is not None:
        await status.set_status(name, -1, "Finalize (optimize)", write=True)

    tesseract_producer = None
    if pdf:
        async with await pdf[0].open("rb") as pdf_file:
            pdf_content = await pdf_file.read()
            with pikepdf.open(io.BytesIO(pdf_content)) as pdf_:
                pdf_producer = pdf_.docinfo.get("/Producer")
                if tesseract_producer is None and pdf_producer is not None:
                    tesseract_producer = json.loads(pdf_producer.to_json())
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
        for pdf_path in pdf:
            basename = pdf_path.name.split(".")
            await call(
                [
                    "cp",
                    str(pdf_path),
                    str(root_folder / f"1-{'.'.join(basename[:-1])}-tesseract.{basename[-1]}"),
                ],
            )

    count = 1
    with tempfile.NamedTemporaryFile(suffix=".png") as temporary_pdf:
        await call(["pdftk", *[str(e) for e in pdf], "output", temporary_pdf.name, "compress"])
        if progress:
            await call(["cp", temporary_pdf.name, str(root_folder / f"{count}-pdftk.pdf")])
            count += 1

        if (
            config["args"]
            .setdefault("exiftool", cast("schema.Exiftool", schema.EXIFTOOL_DEFAULT))
            .setdefault("enabled", schema.EXIFTOOL_ENABLED_DEFAULT)
        ):
            await call(["exiftool", "-overwrite_original_in_place", temporary_pdf.name])
            if progress:
                await call(["cp", temporary_pdf.name, str(root_folder / f"{count}-exiftool.pdf")])
                count += 1

        if (
            config["args"]
            .setdefault("ps2pdf", cast("schema.Ps2Pdf", schema.PS2PDF_DEFAULT))
            .setdefault("enabled", schema.PS2PDF_ENABLED_DEFAULT)
        ):
            with tempfile.NamedTemporaryFile(suffix=".png") as temporary_ps2pdf:
                await call(["ps2pdf", temporary_pdf.name, temporary_ps2pdf.name])
                if progress:
                    await call(["cp", temporary_ps2pdf.name, f"{count}-ps2pdf.pdf"])
                    count += 1
                await call(["cp", temporary_ps2pdf.name, temporary_pdf.name])

        async with await anyio.open_file(temporary_pdf.name, "rb") as temp_file:
            pdf_content = await temp_file.read()

        pdf_buffer = io.BytesIO(pdf_content)
        with pikepdf.open(pdf_buffer) as pdf_:
            scan_to_paperless_meta = f"Scan to Paperless {os.environ.get('VERSION', 'undefined')}"
            with pdf_.open_metadata() as meta:
                meta["{http://purl.org/dc/elements/1.1/}creator"] = (
                    [scan_to_paperless_meta, tesseract_producer]
                    if tesseract_producer
                    else [scan_to_paperless_meta]
                )
            pdf_buffer = io.BytesIO()
            pdf_.save(pdf_buffer)

        async with await anyio.open_file(temporary_pdf.name, "wb") as temp_file:
            await temp_file.write(pdf_buffer.getvalue())

        if progress:
            await call(["cp", temporary_pdf.name, str(root_folder / f"{count}-pikepdf.pdf")])
            count += 1

        if (
            config["args"]
            .setdefault("consume_folder", {})
            .setdefault("enabled", schema.CONSUME_FOLDER_ENABLED_DEFAULT)
        ):
            await call(["cp", temporary_pdf.name, str(destination)])
        if (
            config["args"]
            .setdefault("rest_upload", cast("schema.RestUpload", {}))
            .setdefault("enabled", schema.REST_UPLOAD_ENABLED_DEFAULT)
        ):
            token = config["args"]["rest_upload"]["api_token"]
            url = config["args"]["rest_upload"]["api_url"]
            url = f"{url}/documents/post_document/"
            headers = {"authorization": f"Token {token}"}

            async with await anyio.open_file(temporary_pdf.name, "rb") as document_file:
                title = root_folder.name
                async with aiohttp.ClientSession() as session:
                    form = aiohttp.FormData()
                    form.add_field("document", await document_file.read())
                    form.add_field("title", title)
                    async with session.post(url, headers=headers, data=form, timeout=120) as response:  # type: ignore[arg-type]
                        if response.status != 200:
                            text = await response.text()
                            msg = f"Failed ({response.status}) upload to '{url}' with token '{token}'\n{text}"
                            raise scan_to_paperless.ScanToPaperlessError(msg)
                        print(f"Uploaded {temporary_pdf.name} with title {title}")


async def _process_code(name: str) -> bool:
    """Detect ad add a page with the QR codes."""
    pdf_filename = Path(os.environ.get("SCAN_CODES_FOLDER", "/scan-codes")) / name

    destination_filename = Path(os.environ.get("SCAN_FINAL_FOLDER", "/destination")) / pdf_filename.name

    if await destination_filename.exists():
        await asyncio.sleep(1)
        return False

    try:
        if await pdf_filename.exists():
            _LOG.info("Processing codes for %s", pdf_filename)
            from scan_to_paperless import add_code  # noqa: PLC0415, RUF100

            await add_code.add_codes(
                pdf_filename,
                str(destination_filename),
                dpi=float(os.environ.get("SCAN_CODES_DPI", "200")),
                pdf_dpi=float(os.environ.get("SCAN_CODES_PDF_DPI", "72")),
                font_name=os.environ.get("SCAN_CODES_FONT_NAME", "Helvetica-Bold"),
                font_size=float(os.environ.get("SCAN_CODES_FONT_SIZE", "16")),
                margin_top=float(os.environ.get("SCAN_CODES_MARGIN_TOP", "0")),
                margin_left=float(os.environ.get("SCAN_CODES_MARGIN_LEFT", "2")),
            )
            if await destination_filename.exists():
                # Remove the source file on success
                await pdf_filename.unlink()
            _LOG.info("Down processing codes for %s", pdf_filename)
            return True

    except Exception as exception:  # noqa: BLE001
        _LOG.exception("Error while processing %s: %s", pdf_filename, str(exception))

    await asyncio.sleep(1)
    return False


def is_sources_present(images: list[str], root_folder: Path) -> bool:
    """Are sources present for the next step."""
    for image in images:
        if not (root_folder / image).exists():
            print(f"Missing {root_folder} - {image}")
            return False
    return True


async def save_config(config: schema.Configuration, config_file_name: Path) -> None:
    """Save the configuration."""
    yaml = YAML()
    yaml.default_flow_style = False
    temp_path = Path(str(config_file_name) + "_")
    async with await temp_path.open("w", encoding="utf-8") as config_file:
        out = io.StringIO()
        yaml.dump(config, out)
        await config_file.write(out.getvalue())
    await temp_path.rename(config_file_name)


async def _process(
    config_file_name: Path,
    status: scan_to_paperless.status.Status,
    dirty: bool = False,
) -> bool:
    """Process one document."""
    config_file_name_anyio = Path(config_file_name)
    if not await config_file_name_anyio.exists():
        return dirty

    root_folder = config_file_name.parent

    if await (root_folder / _ERROR_FILENAME).exists():
        return dirty

    yaml = YAML()
    yaml.default_flow_style = False
    async with await config_file_name_anyio.open(encoding="utf-8") as config_file:
        config: schema.Configuration = yaml.load(await config_file.read())
    if config is None:
        return dirty

    if not is_sources_present(config["images"], root_folder):
        return dirty

    try:
        rerun = False
        disable_remove_to_continue = config["args"].setdefault(
            "no_remove_to_continue",
            schema.NO_REMOVE_TO_CONTINUE_DEFAULT,
        )
        if "steps" not in config:
            rerun = True
        while config.get("steps") and not is_sources_present(config["steps"][-1]["sources"], root_folder):
            config["steps"] = config["steps"][:-1]
            await save_config(config, config_file_name)
            if await (root_folder / "REMOVE_TO_CONTINUE").exists():
                await (root_folder / "REMOVE_TO_CONTINUE").unlink()
            rerun = True

        if "steps" not in config or not config["steps"]:
            step: schema.Step = {
                "sources": config["images"],
                "name": "transform",
            }
            config["steps"] = [step]
        step = config["steps"][-1]

        if is_sources_present(step["sources"], root_folder):
            if (
                not disable_remove_to_continue
                and await (root_folder / "REMOVE_TO_CONTINUE").exists()
                and not rerun
            ):
                return dirty
            if await (root_folder / "DONE").exists() and not rerun:
                return dirty

            await status.set_global_status(f"Processing '{config_file_name.parent.name}'...")
            await status.set_current_folder(config_file_name)
            await status.set_status(config_file_name, -1, "Processing")
            dirty = True

            done = False
            next_step = None
            if step["name"] == scan_to_paperless.status.STATUS_TRANSFORM:
                _update_config(config)
                next_step = await transform(config, step, config_file_name, root_folder, status=status)
            elif step["name"] == scan_to_paperless.status.STATUS_ASSISTED_SPLIT:
                await status.set_status(config_file_name, -1, "Split")
                next_step = await split(config, step, root_folder)
            elif step["name"] == scan_to_paperless.status.STATUS_FINALIZE:
                await finalize(config, step, root_folder, status=status)
                done = True

            if done and os.environ.get("PROGRESS", "FALSE") != "TRUE":
                shutil.rmtree(root_folder)
            else:
                if next_step is not None:
                    config["steps"].append(next_step)
                await save_config(config, config_file_name)
                if done:
                    async with await (root_folder / "DONE").open("w", encoding="utf-8"):
                        pass
                elif not disable_remove_to_continue:
                    async with await (root_folder / "REMOVE_TO_CONTINUE").open("w", encoding="utf-8"):
                        pass
            # Be sure that the status is up to date especially about the modifief files in the current folder
            await asyncio.sleep(0.1)
            await status.set_current_folder(None)

    except Exception as exception:  # noqa: BLE001
        trace = traceback.format_exc()

        out = {"error": str(exception), "traceback": trace.split("\n")}
        for attribute in ("returncode", "cmd", "description", "error_text"):
            if hasattr(exception, attribute):
                out[attribute] = getattr(exception, attribute)
        for attribute in ("output", "stdout", "stderr"):
            if hasattr(exception, attribute) and getattr(exception, attribute):
                out[attribute] = getattr(exception, attribute).decode()

        yaml = YAML(typ="safe")
        yaml.default_flow_style = False
        try:
            async with await anyio.open_file(
                root_folder / _ERROR_FILENAME, "w", encoding="utf-8"
            ) as error_file:
                yaml_out = io.StringIO()
                yaml.dump(out, yaml_out)
                await error_file.write(yaml_out.getvalue())
        except Exception as exception2:  # noqa: BLE001
            print(exception2)
            print(traceback.format_exc())
            yaml = YAML()
            yaml.default_flow_style = False
            async with await anyio.open_file(
                root_folder / _ERROR_FILENAME, "w", encoding="utf-8"
            ) as error_file:
                yaml_out = io.StringIO()
                yaml.dump(out, yaml_out)
                await error_file.write(yaml_out.getvalue())
    return dirty


async def _task(status: scan_to_paperless.status.Status) -> None:
    while True:
        await status.set_current_folder(None)
        # Be sure that the status is up to date
        await asyncio.sleep(0.1)
        name, job_type, step = status.get_next_job()
        if job_type == scan_to_paperless.status.JobType.NONE:
            await status.set_global_status("Waiting...")
            await status.set_current_folder(None)
            await asyncio.sleep(1)
            continue

        print(f"Processing '{name}' as {job_type}...")

        if job_type in (
            scan_to_paperless.status.JobType.TRANSFORM,
            scan_to_paperless.status.JobType.ASSISTED_SPLIT,
            scan_to_paperless.status.JobType.FINALIZE,
        ):
            assert isinstance(name, str)
            assert step is not None

            await status.set_global_status(f"Processing '{name}'...")
            await status.set_current_folder(name)
            try:
                root_folder = Path(os.environ.get("SCAN_SOURCE_FOLDER", "/source")) / name
                config_file_name = root_folder / "config.yaml"
                yaml = YAML()
                yaml.default_flow_style = False
                async with await config_file_name.open(encoding="utf-8") as config_file:
                    config: schema.Configuration = yaml.load(await config_file.read())
                    config.yaml_set_start_comment(  # type: ignore[attr-defined]
                        "# yaml-language-server: $schema=https://raw.githubusercontent.com/sbrunner/"
                        f"scan-to-paperless/{os.environ.get('SCHEMA_BRANCH', 'master')}/scan_to_paperless/"
                        "process_schema.json\n\n",
                    )

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

                for image in step["sources"]:
                    image_path = root_folder / image
                    if not await image_path.exists():
                        _LOG.warning("Missing image %s", image_path)
                        continue

                next_step = None
                if job_type == scan_to_paperless.status.JobType.TRANSFORM:
                    _update_config(config)
                    next_step = await transform(config, step, config_file_name, root_folder, status=status)
                if job_type == scan_to_paperless.status.JobType.ASSISTED_SPLIT:
                    await status.set_status(name, -1, "Splitting in assisted-split mode", write=True)
                    next_step = await split(config, step, root_folder)
                if job_type == scan_to_paperless.status.JobType.FINALIZE:
                    await finalize(config, step, root_folder, status=status)
                    async with await (root_folder / "DONE").open("w", encoding="utf-8"):
                        pass
                if next_step is not None:
                    config["steps"].append(next_step)

                await save_config(config, config_file_name)
            finally:
                await status.set_current_folder(None)

        elif job_type == scan_to_paperless.status.JobType.DOWN:
            assert name is not None
            await status.set_global_status(f"Removing '{name}'...")
            await status.set_current_folder(name)
            root_folder = Path(os.environ.get("SCAN_SOURCE_FOLDER", "/source")) / name
            if await root_folder.exists():
                shutil.rmtree(root_folder)
        elif job_type == scan_to_paperless.status.JobType.CODE:
            assert isinstance(name, str)
            print(f"Process code '{name}'")
            await status.set_global_status(f"Process code '{name}'...")
            await status.set_current_folder(name)
            try:
                if not await _process_code(name):
                    await status.update_scan_codes()
            except Exception as exception:  # noqa: BLE001
                print(exception)
                trace = traceback.format_exc()
                print(trace)
        else:
            msg = f"Unknown job type: {job_type}"
            raise ValueError(msg)

        await status.set_current_folder(None)

        print(f"End processing '{name}' as {job_type}...")


def main() -> None:
    """Process the scanned documents."""
    asyncio.run(async_main())


async def _watch_dog() -> None:
    while True:
        print("|===================")
        print("| Watch dog")
        for task in asyncio.all_tasks():
            print(f"| {task.get_name()}")
            for nb in range(1, 11):
                string_io = io.StringIO()
                task.print_stack(limit=2, file=string_io)
                value = string_io.getvalue()
                if nb == 10 or "/scan_to_paperless/" in value:
                    for line in value.split("\n"):
                        print(f"|   {line}")
                    break
        print("|===================")
        if os.environ.get("DEBUG_INOTIFY", "FALSE").lower() in ("true", "1", "yes"):
            await asyncio.sleep(10)
        else:
            await asyncio.sleep(60)


async def async_main() -> None:
    """Process the scanned documents."""
    if "SENTRY_DSN" in os.environ:
        sentry_sdk.init(
            dsn=os.environ["SENTRY_DSN"],
            integrations=[LoggingIntegration(), AsyncioIntegration()],
            traces_sample_rate=1.0,
            profiles_sample_rate=1.0,
        )

    parser = argparse.ArgumentParser("Process the scanned documents.")
    parser.add_argument("config", nargs="?", help="The config file to process.")
    args = parser.parse_args()

    if args.config:
        status = scan_to_paperless.status.Status(no_write=True)
        await status.init()
        await _process(args.config, status)
        sys.exit()

    print("Welcome to scanned images document to paperless.")
    print(f"Started at: {datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d %H:%M')}")

    status = scan_to_paperless.status.Status()
    await status.init()

    watch_dog_task = (
        asyncio.create_task(_watch_dog(), name="Watch dog")
        if os.environ.get("WATCH_DOG", "FALSE").lower() in ["true", "1", "yes"]
        else None
    )
    main_task = asyncio.create_task(_task(status), name="Main")
    status.start_watch()
    await main_task
    if watch_dog_task is not None:
        watch_dog_task.cancel()


if __name__ == "__main__":
    main()
