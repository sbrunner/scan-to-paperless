#!/usr/bin/env python3

"""Process the scanned documents."""

import argparse
import glob
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
from typing import IO, TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Set, Tuple, TypedDict, Union, cast

# read, write, rotate, crop, sharpen, draw_line, find_line, find_contour
import cv2
import numpy as np
import pikepdf
from deskew import determine_skew_dev
from ruamel.yaml.main import YAML
from scipy.signal import find_peaks
from skimage.color import rgb2gray, rgba2rgb
from skimage.metrics import structural_similarity

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


def rotate_image(
    image: NpNdarrayInt, angle: float, background: Union[int, Tuple[int, int, int]]
) -> NpNdarrayInt:
    """Rotate the image."""
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center: Tuple[Any, ...] = tuple(np.array(image.shape[1::-1]) / 2)
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
    background: Union[Tuple[int], Tuple[int, int, int]],
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
            if auto_mask_config.get("de_noise_morphology", True):
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

            inverse_mask = auto_mask_config.get("inverse_mask", False)
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
                mask_file: Optional[str] = os.path.join(self.root_folder, default_file_name)
                assert mask_file
                if not os.path.exists(mask_file):
                    base_folder = os.path.dirname(self.root_folder)
                    assert base_folder
                    mask_file = os.path.join(base_folder, default_file_name)
                    if not os.path.exists(mask_file):
                        mask_file = None
                mask_file = (
                    auto_mask_config.setdefault("additional_filename", mask_file)
                    if mask_file
                    else auto_mask_config.get("additional_filename")
                )
                if mask_file:
                    mask = cv2.add(
                        mask,
                        cv2.bitwise_not(
                            cv2.resize(
                                cv2.cvtColor(cv2.imread(mask_file), cv2.COLOR_BGR2GRAY),
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

            final_mask = cv2.imread(mask_file)

            maskbw = (
                final_mask if len(final_mask.shape) == 2 else cv2.cvtColor(final_mask, cv2.COLOR_BGR2GRAY)
            )
            if self.image is not None and final_mask is not None:
                return cast(NpNdarrayInt, cv2.resize(maskbw, (self.image.shape[1], self.image.shape[0])))
        return cast(NpNdarrayInt, final_mask)

    def init_mask(self) -> None:
        """Init the mask image used to mask the image on the crop and skew calculation."""
        self.mask = self._get_mask(self.config["args"].get("auto_mask"), "auto_mask", "mask.png")

    def get_background_color(self) -> Tuple[int, int, int]:
        """Get the background color."""
        return cast(
            Tuple[int, int, int],
            self.config["args"].setdefault("background_color", schema.BACKGROUND_COLOR_DEFAULT),
        )

    def do_initial_cut(self) -> None:
        """Definitively mask the original image."""
        if "auto_cut" in self.config["args"]:
            assert self.image is not None
            mask = self._get_mask(self.config["args"].get("auto_cut"), "auto_cut", "cut.png")
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
            raise Exception("The image is None")
        if self.mask is None:
            return self.image.copy()

        image = self.image.copy()
        image[self.mask == 0] = self.get_background_color()
        return image

    def crop(self, x: int, y: int, width: int, height: int) -> None:
        """Crop the image."""
        if self.image is None:
            raise Exception("The image is None")
        self.image = crop_image(self.image, x, y, width, height, self.get_background_color())
        if self.mask is not None:
            self.mask = crop_image(self.mask, x, y, width, height, (0,))

    def rotate(self, angle: float) -> None:
        """Rotate the image."""
        if self.image is None:
            raise Exception("The image is None")
        self.image = rotate_image(self.image, angle, self.get_background_color())
        if self.mask is not None:
            self.mask = rotate_image(self.mask, angle, 0)

    def get_px_value(self, name: str, default: Union[int, float]) -> float:
        """Get the value in px."""
        return (
            cast(float, cast(Dict[str, Any], self.config["args"]).setdefault(name, default))
            / 10
            / 2.51
            * self.config["args"].setdefault("dpi", schema.DPI_DEFAULT)
        )

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
    traceback_: List[str],
) -> None:
    """Add in the config non fatal error."""
    if config_file_name is None:
        raise Exception("The config file name is required")
    if "intermediate_error" not in config:
        config["intermediate_error"] = []

    old_intermediate_error: List[schema.IntermediateError] = []
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


def call(cmd: Union[str, List[str]], **kwargs: Any) -> None:
    """Verbose version of check_output with no returns."""
    if isinstance(cmd, list):
        cmd = [str(element) for element in cmd]
    print(" ".join(cmd) if isinstance(cmd, list) else cmd)
    sys.stdout.flush()
    kwargs.setdefault("check", True)
    subprocess.run(  # nosec # pylint: disable=subprocess-run-check
        cmd,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        **kwargs,
    )


def run(cmd: Union[str, List[str]], **kwargs: Any) -> CompletedProcess:
    """Verbose version of check_output with no returns."""
    if isinstance(cmd, list):
        cmd = [str(element) for element in cmd]
    print(" ".join(cmd) if isinstance(cmd, list) else cmd)
    sys.stdout.flush()
    return subprocess.run(cmd, stderr=subprocess.PIPE, check=True, **kwargs)  # nosec


def output(cmd: Union[str, List[str]], **kwargs: Any) -> str:
    """Verbose version of check_output."""
    if isinstance(cmd, list):
        cmd = [str(element) for element in cmd]
    print(" ".join(cmd) if isinstance(cmd, list) else cmd)
    sys.stdout.flush()
    return cast(bytes, subprocess.check_output(cmd, stderr=subprocess.PIPE, **kwargs)).decode()  # nosec


def image_diff(image1: NpNdarrayInt, image2: NpNdarrayInt) -> Tuple[float, NpNdarrayInt]:
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

    def __init__(self, name: str, ignore_error: bool = False) -> None:
        """Initialize."""
        self.name = name
        self.ignore_error = ignore_error

    def __call__(self, func: FunctionWithContextReturnsImage) -> FunctionWithContextReturnsNone:
        """Call the function."""

        def wrapper(context: Context) -> None:
            if context.image is None:
                raise Exception("The image is required")
            if context.root_folder is None:
                raise Exception("The root folder is required")
            if context.image_name is None:
                raise Exception("The image name is required")
            start_time = time.perf_counter()
            if self.ignore_error:
                try:
                    new_image = func(context)
                    if new_image is not None and self.ignore_error:
                        context.image = new_image
                except Exception as exception:
                    print(exception)
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
    contours: List[Tuple[int, int, int, int]], margin_horizontal: int = 0, margin_vertical: int = 0
) -> Tuple[int, int, int, int]:
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
    contours = find_contours(image, context, process_count, "crop", "crop", schema.MIN_BOX_SIZE_CROP_DEFAULT)

    if contours:
        for contour in contours:
            draw_rectangle(image, contour)
        context.save_progress_images("crop", image, process_count=process_count, force=True)

        x, y, width, height = get_contour_to_crop(contours, margin_horizontal, margin_vertical)
        context.crop(x, y, width, height)


@Process("level")
def level(context: Context) -> NpNdarrayInt:
    """Do the level on an image."""
    img_yuv = cv2.cvtColor(context.image, cv2.COLOR_BGR2YUV)

    if context.config["args"].setdefault("auto_level", schema.AUTO_LEVEL_DEFAULT):
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cast(NpNdarrayInt, cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR))
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
        grayscale, context.config["args"].setdefault("cut_black", schema.CUT_BLACK_DEFAULT), 0
    )
    context.image[white_mask == 1] = (255, 255, 255)
    context.image[black_mask == 1] = (0, 0, 0)


@Process("mask-cut")
def cut(context: Context) -> None:
    """Mask the image with the cut mask."""
    context.do_initial_cut()


def draw_angle(image: NpNdarrayInt, angle: np.float64, color: Tuple[int, int, int]) -> None:
    """Draw an angle on the image (as a line passed at the center of the image)."""
    angle = angle % 90
    height, width = image.shape[:2]
    center = (int(width / 2), int(height / 2))
    length = min(width, height) / 2

    angle_radian = math.radians(angle)
    sin_a = np.sin(angle_radian) * length
    cos_a = np.cos(angle_radian) * length
    for matrix in ([[0, -1], [-1, 0]], [[1, 0], [0, -1]], [[0, 1], [1, 0]], [[-1, 0], [0, 1]]):
        diff = np.dot(matrix, [sin_a, cos_a])
        x = diff[0] + width / 2
        y = diff[1] + height / 2

        cv2.line(image, center, (int(x), int(y)), color, 2)
        if matrix[0][0] == -1:
            cv2.putText(image, str(angle), (int(x), int(y + 50)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color)


def nice_angle(angle: np.float64) -> np.float64:
    """Fix the angle to be between -45° and 45°."""
    return ((angle + 45) % 90) - 45


@Process("deskew")
def deskew(context: Context) -> None:
    """Deskew an image."""
    images_config = context.config.setdefault("images_config", {})
    assert context.image_name
    image_config = images_config.setdefault(context.image_name, {})
    image_status = image_config.setdefault("status", {})
    angle = image_config.setdefault("angle", None)
    if angle is None:
        image = context.get_masked()
        imagergb = rgba2rgb(image) if len(image.shape) == 3 and image.shape[2] == 4 else image
        grayscale = rgb2gray(imagergb) if len(imagergb.shape) == 3 else imagergb
        image = cast(NpNdarrayInt, context.image).copy()

        skew_angle, angles, average_deviation, _ = determine_skew_dev(
            grayscale,
            min_angle=np.deg2rad(
                context.config["args"].setdefault("deskew_min_angle", schema.DESKEW_MIN_ANGLE_DEFAULT)
            ),
            max_angle=np.deg2rad(
                context.config["args"].setdefault("deskew_max_angle", schema.DESKEW_MAX_ANGLE_DEFAULT)
            ),
            min_deviation=np.deg2rad(
                context.config["args"].setdefault(
                    "deskew_angle_derivation", schema.DESKEW_ANGLE_DERIVATION_DEFAULT
                )
            ),
        )
        if skew_angle is not None:
            image_status["angle"] = float(nice_angle(skew_angle))
            draw_angle(image, skew_angle, (255, 0, 0))
            angle = float(skew_angle)

        float_angles: Set[np.float64] = set()
        average_deviation_float = average_deviation
        image_status["average_deviation"] = float(average_deviation_float)
        average_deviation2 = nice_angle(average_deviation_float - 45)
        image_status["average_deviation2"] = float(average_deviation2)
        if math.isfinite(average_deviation2):
            float_angles.add(average_deviation2)

        for current_angles in angles:
            for current_angle in current_angles:
                if current_angle is not None and math.isfinite(float(current_angle)):
                    float_angles.add(nice_angle(current_angle))
        for current_angle in float_angles:
            draw_angle(image, current_angle, (0, 255, 0))
        image_status["angles"] = [float(a) for a in float_angles]

        assert context.root_folder
        context.save_progress_images("skew-angles", image, force=True)

    if angle:
        context.rotate(angle)


@Process("docrop")
def docrop(context: Context) -> None:
    """Crop an image."""
    # Margin in mm
    if context.config["args"].setdefault("no_crop", schema.NO_CROP_DEFAULT):
        return
    margin_horizontal = context.get_px_value("margin_horizontal", schema.MARGIN_HORIZONTAL_DEFAULT)
    margin_vertical = context.get_px_value("margin_vertical", schema.MARGIN_VERTICAL_DEFAULT)
    crop(context, int(round(margin_horizontal)), int(round(margin_vertical)))


@Process("sharpen")
def sharpen(context: Context) -> Optional[NpNdarrayInt]:
    """Sharpen an image."""
    if context.config["args"].setdefault("sharpen", schema.SHARPEN_DEFAULT) is False:
        return None
    if context.image is None:
        raise Exception("The image is required")
    image = cv2.GaussianBlur(context.image, (0, 0), 3)
    return cast(NpNdarrayInt, cv2.addWeighted(context.image, 1.5, image, -0.5, 0))


@Process("dither")
@external
def dither(context: Context, source: str, destination: str) -> None:
    """Dither an image."""
    if context.config["args"].setdefault("dither", schema.DITHER_DEFAULT) is False:
        return
    call(CONVERT + ["+dither", source, destination])


@Process("autorotate", True)
def autorotate(context: Context) -> None:
    """
    Auto rotate an image.

    Put the text in the right position.
    """
    with tempfile.NamedTemporaryFile(suffix=".png") as source:
        cv2.imwrite(source.name, context.get_masked())
        orientation_lst = output(["tesseract", source.name, "-", "--psm", "0", "-l", "osd"]).splitlines()
        orientation_lst = [e for e in orientation_lst if "Orientation in degrees" in e]
        context.rotate(int(orientation_lst[0].split()[3]))


def draw_line(  # pylint: disable=too-many-arguments
    image: NpNdarrayInt, vertical: bool, position: float, value: int, name: str, type_: str
) -> schema.Limit:
    """Draw a line on an image."""
    img_len = image.shape[0 if vertical else 1]
    color = (255, 0, 0) if vertical else (0, 255, 0)
    if vertical:
        cv2.rectangle(image, (int(position) - 1, img_len), (int(position) + 0, img_len - value), color, -1)
        cv2.putText(image, name, (int(position), img_len - value), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)
    else:
        cv2.rectangle(image, (0, int(position) - 1), (value, int(position) + 0), color, -1)
        cv2.putText(image, name, (value, int(position)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)
    return {"name": name, "type": type_, "value": int(position), "vertical": vertical, "margin": 0}


def draw_rectangle(image: NpNdarrayInt, contour: Tuple[int, int, int, int]) -> None:
    """Draw a rectangle on an image."""
    color = (0, 255, 0)
    x, y, width, height = contour
    x = int(round(x))
    y = int(round(y))
    width = int(round(width))
    height = int(round(height))
    cv2.rectangle(image, (x, y), (x + 1, y + height), color, -1)
    cv2.rectangle(image, (x, y), (x + width, y + 1), color, -1)
    cv2.rectangle(image, (x, y + height - 1), (x + width, y + height), color, -1)
    cv2.rectangle(image, (x + width - 1, y), (x + width, y + height), color, -1)


def find_lines(image: NpNdarrayInt, vertical: bool) -> Tuple[NpNdarrayInt, Dict[str, NpNdarrayInt]]:
    """Find the lines on an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        image=edges,
        rho=0.02,
        theta=np.pi / 500,
        threshold=10,
        lines=np.array([]),
        minLineLength=100,
        maxLineGap=100,
    )

    values = np.zeros(image.shape[1 if vertical else 0])
    for index in range(lines.shape[0]):
        line = lines[index][0]
        if line[0 if vertical else 1] == line[2 if vertical else 3]:
            values[line[0 if vertical else 1]] += line[1 if vertical else 0] - line[3 if vertical else 2]
    correlated_values = np.correlate(values, [0.2, 0.6, 1, 0.6, 0.2])
    dist = 1.0
    peaks, properties = find_peaks(correlated_values, height=dist * 10, distance=dist)
    while len(peaks) > 5:
        dist *= 1.3
        peaks, properties = find_peaks(correlated_values, height=dist * 10, distance=dist)
    peaks += 2

    return peaks, properties


def zero_ranges(values: NpNdarrayInt) -> NpNdarrayInt:
    """Create an array that is 1 where a is 0, and pad each end with an extra 0."""
    iszero: NpNdarrayInt = np.concatenate([[0], np.equal(values, 0).view(np.int8), [0]])
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return cast(NpNdarrayInt, ranges)


def find_limit_contour(
    image: NpNdarrayInt, context: Context, vertical: bool
) -> Tuple[List[int], List[Tuple[int, int, int, int]]]:
    """Find the contour for assisted split."""
    contours = find_contours(image, context, context.get_process_count(), "limits", "limit")
    image_size = image.shape[1 if vertical else 0]

    values = np.zeros(image_size)
    for x, _, width, height in contours:
        x_int = int(round(x))
        for value in range(x_int, min(x_int + width, image_size)):
            values[value] += height

    ranges = zero_ranges(values)

    result: List[int] = []
    for ranges_ in ranges:
        if ranges_[0] != 0 and ranges_[1] != image_size:
            result.append(int(round(sum(ranges_) / 2)))

    return result, contours


def fill_limits(image: NpNdarrayInt, vertical: bool, context: Context) -> List[schema.Limit]:
    """Find the limit for assisted split."""
    contours_limits, contours = find_limit_contour(image, context, vertical)
    peaks, properties = find_lines(image, vertical)
    for contour_limit in contours:
        draw_rectangle(image, contour_limit)
    third_image_size = int(image.shape[0 if vertical else 1] / 3)
    limits: List[schema.Limit] = []
    prefix = "V" if vertical else "H"
    for index, peak in enumerate(peaks):
        value = int(round(properties["peak_heights"][index] / 3))
        limits.append(draw_line(image, vertical, peak, value, f"{prefix}L{index}", "line detection"))
    for index, contour in enumerate(contours_limits):
        limits.append(
            draw_line(image, vertical, contour, third_image_size, f"{prefix}C{index}", "contour detection")
        )
    if not limits:
        half_image_size = image.shape[1 if vertical else 0] / 2
        limits.append(
            draw_line(image, vertical, half_image_size, third_image_size, f"{prefix}C", "image center")
        )

    return limits


def find_contours(
    image: NpNdarrayInt,
    context: Context,
    progress_count: int,
    name: str,
    prefix: str,
    default_min_box_size: int = schema.MIN_BOX_SIZE_EMPTY_DEFAULT,
) -> List[Tuple[int, int, int, int]]:
    """Find the contours on an image."""
    block_size = context.get_px_value(
        f"threshold_block_size_{prefix}", schema.THRESHOLD_BLOCK_SIZE_CROP_DEFAULT
    )
    threshold_value_c = cast(Dict[str, int], context.config["args"]).setdefault(
        f"threshold_value_c_{prefix}", schema.THRESHOLD_VALUE_C_CROP_DEFAULT
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    block_size = int(round(block_size / 2) * 2)

    # Clean the image using otsu method with the inversed binarized image
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size + 1, threshold_value_c
    )
    if context.is_progress() and context.root_folder and context.image_name:
        context.save_progress_images("threshold", thresh)

        block_size_list = (block_size, 1.5, 5, 10, 15, 20, 50, 100, 200)
        threshold_value_c_list = (threshold_value_c, 20, 50, 100)

        for block_size2 in block_size_list:
            for threshold_value_c2 in threshold_value_c_list:
                block_size2 = int(round(block_size2 / 2) * 2)
                thresh2 = cv2.adaptiveThreshold(
                    gray,
                    255,
                    cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY_INV,
                    block_size2 + 1,
                    threshold_value_c2,
                )
                contours = _find_contours_thresh(image, thresh2, context, prefix, default_min_box_size)
                thresh2 = cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR)
                if contours:
                    for contour in contours:
                        draw_rectangle(thresh2, contour)

                context.save_progress_images(
                    f"{name}-threshold",
                    thresh2,
                    f"block_size_{prefix}-{block_size2}-value_c_{prefix}-{threshold_value_c2}-",
                    progress_count,
                )

    return _find_contours_thresh(image, thresh, context, prefix, default_min_box_size)


def _find_contours_thresh(
    image: NpNdarrayInt, thresh: NpNdarrayInt, context: Context, prefix: str, default_min_box_size: int = 10
) -> List[Tuple[int, int, int, int]]:
    min_size = context.get_px_value(f"min_box_size_{prefix}", default_min_box_size)
    min_black = cast(Dict[str, int], context.config["args"]).setdefault(
        f"min_box_black_{prefix}", schema.MIN_BOX_BLACK_CROP_DEFAULT
    )
    kernel_size = context.get_px_value(
        f"contour_kernel_size_{prefix}", schema.CONTOUR_KERNEL_SIZE_CROP_DEFAULT
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


def transform(
    config: schema.Configuration,
    step: schema.Step,
    config_file_name: str,
    root_folder: str,
) -> schema.Step:
    """Apply the transforms on a document."""
    if "intermediate_error" in config:
        del config["intermediate_error"]

    images = []
    process_count = 0

    if config["args"].setdefault("assisted_split", schema.ASSISTED_SPLIT_DEFAULT):
        config["assisted_split"] = []

    for index, img in enumerate(step["sources"]):
        image_name = f"{os.path.basename(img).rsplit('.')[0]}.png"
        context = Context(config, step, config_file_name, root_folder, image_name)
        if context.image_name is None:
            raise Exception("Image name is required")
        context.image = cv2.imread(os.path.join(root_folder, img))
        images_config = context.config.setdefault("images_config", {})
        image_config = images_config.setdefault(context.image_name, {})
        image_status = image_config.setdefault("status", {})
        assert context.image is not None
        image_status["size"] = list(context.image.shape[:2][::-1])
        context.init_mask()
        level(context)
        color_cut(context)
        cut(context)
        deskew(context)
        docrop(context)
        sharpen(context)
        dither(context)
        autorotate(context)

        # Is empty ?
        contours = find_contours(
            context.get_masked(), context, context.get_process_count(), "is-empty", "empty"
        )
        if not contours:
            print(f"Ignore image with no content: {img}")
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
            limits.extend(fill_limits(context.image, True, context))
            limits.extend(fill_limits(context.image, False, context))
            assisted_split["limits"] = limits

            cv2.imwrite(name, context.image)
            assisted_split["image"] = context.image_name
            images.append(name)
        else:
            img2 = os.path.join(root_folder, context.image_name)
            cv2.imwrite(img2, context.image)
            images.append(img2)
        process_count = context.process_count

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

    if not config["args"].setdefault("jpeg", False) and config["args"].setdefault(
        "run_pngquant", schema.RUN_PNGQUANT_DEFAULT
    ):
        count = context.get_process_count()
        for image in images:
            with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
                call(
                    ["pngquant", f"--output={temp_file.name}"]
                    + config["args"].setdefault(
                        "pngquant_options",
                        schema.PNGQUANT_OPTIONS_DEFAULT,
                    )
                    + ["--", image],
                    check=False,
                )
                if os.path.getsize(temp_file.name) > 0:
                    call(["cp", temp_file.name, image])
            if progress:
                _save_progress(context.root_folder, count, "pngquant", os.path.basename(image), image)

    if not config["args"].setdefault("jpeg", schema.JPEG_DEFAULT) and config["args"].setdefault(
        "run_optipng", not config["args"]["run_pngquant"]
    ):
        count = context.get_process_count()
        for image in images:
            call(["optipng", image], check=False)
            if progress:
                _save_progress(context.root_folder, count, "optipng", os.path.basename(image), image)

    if config["args"].setdefault("jpeg", schema.JPEG_DEFAULT):
        count = context.get_process_count()
        new_images = []
        for img in images:
            name = os.path.splitext(os.path.basename(img))[0]
            jpeg_img = f"{name}.jpeg"
            subprocess.run(  # nosec
                [
                    "gm",
                    "convert",
                    img,
                    "-quality",
                    str(config["args"].setdefault("jpeg_quality", schema.JPEG_QUALITY_DEFAULT)),
                    jpeg_img,
                ],
                check=True,
            )
            new_images.append(jpeg_img)
            if progress:
                _save_progress(context.root_folder, count, "to-jpeg", os.path.basename(image), image)

        images = new_images

    return {
        "sources": images,
        "name": "split"
        if config["args"].setdefault("assisted_split", schema.ASSISTED_SPLIT_DEFAULT)
        else "finalise",
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


def save(context: Context, root_folder: str, img: str, folder: str, force: bool = False) -> str:
    """Save the current image in a subfolder if progress mode in enabled."""
    if force or context.is_progress():
        dest_folder = os.path.join(root_folder, folder)
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        dest_file = os.path.join(dest_folder, os.path.basename(img))
        shutil.copyfile(img, dest_file)
        return dest_file
    return img


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
                raise Exception(
                    f"Wrong number of destinations ({len(assisted_split['destinations'])}), "
                    f"vertical: {nb_horizontal}, height: {nb_vertical}, img '{assisted_split['source']}'"
                )

    for assisted_split in config["assisted_split"]:
        if "image" in assisted_split:
            image_path = os.path.join(root_folder, assisted_split["image"])
            if os.path.exists(image_path):
                os.unlink(image_path)

    append: Dict[Union[str, int], List[Item]] = {}
    transformed_images = []
    for assisted_split in config["assisted_split"]:
        img = assisted_split["source"]
        context = Context(config, step)
        width, height = (
            int(e) for e in output(CONVERT + [img, "-format", "%w %h", "info:-"]).strip().split(" ")
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
                            img,
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
                    margin_horizontal = context.get_px_value(
                        "margin_horizontal", schema.MARGIN_HORIZONTAL_DEFAULT
                    )
                    margin_vertical = context.get_px_value("margin_vertical", schema.MARGIN_VERTICAL_DEFAULT)
                    context.image = cv2.imread(process_file.name)
                    if not context.config["args"].setdefault("no_crop", schema.NO_CROP_DEFAULT):
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
        items: List[Item] = append[page_number]
        vertical = len(horizontal_limits) == 0
        if not vertical and len(vertical_limits) != 0 and len(items) > 1:
            raise Exception(f"Mix of limit type for page '{page_number}'")

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

    return {"sources": transformed_images, "name": "finalise", "process_count": process_count}


def finalize(
    config: schema.Configuration,
    step: schema.Step,
    root_folder: str,
) -> None:
    """
    Do final step on document generation.

    convert in one pdf and copy with the right name in the consume folder
    """
    destination = os.path.join(
        os.environ.get("SCAN_CODES_FOLDER", "/scan-codes"), f"{os.path.basename(root_folder)}.pdf"
    )

    if os.path.exists(destination):
        return

    images = step["sources"]

    if config["args"].setdefault("append_credit_card", schema.APPEND_CREDIT_CARD_DEFAULT):
        images2 = []
        for img in images:
            if os.path.exists(img):
                images2.append(img)

        file_name = os.path.join(root_folder, "append.png")
        call(CONVERT + images2 + ["-background", "#ffffff", "-gravity", "center", "-append", file_name])
        # To stack vertically (img1 over img2):
        # vis = np.concatenate((img1, img2), axis=0)
        # To stack horizontally (img1 to the left of img2):
        # vis = np.concatenate((img1, img2), axis=1)
        images = [file_name]

    pdf = []
    for img in images:
        if os.path.exists(img):
            name = os.path.splitext(os.path.basename(img))[0]
            file_name = os.path.join(root_folder, f"{name}.pdf")
            if config["args"].setdefault("tesseract", schema.TESSERACT_DEFAULT):
                with open(file_name, "w", encoding="utf8") as output_file:
                    process = run(
                        [
                            "tesseract",
                            "--dpi",
                            str(config["args"].setdefault("dpi", schema.DPI_DEFAULT)),
                            "-l",
                            config["args"].setdefault("tesseract_lang", schema.TESSERACT_LANG_DEFAULT),
                            img,
                            "stdout",
                            "pdf",
                        ],
                        stdout=output_file,
                    )
                    if process.stderr:
                        print(process.stderr)
            else:
                call(CONVERT + [img, "+repage", file_name])
            pdf.append(file_name)

    tesseract_producer = None
    if pdf:
        with pikepdf.open(pdf[0]) as pdf_:
            if tesseract_producer is None and pdf_.docinfo.get("/Producer") is not None:
                tesseract_producer = json.loads(pdf_.docinfo.get("/Producer").to_json())  # type: ignore
                if "tesseract" not in tesseract_producer.lower():
                    tesseract_producer = None
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

        if config["args"].setdefault("run_exiftool", schema.RUN_EXIFTOOL_DEFAULT):
            call(["exiftool", "-overwrite_original_in_place", temporary_pdf.name])
            if progress:
                call(["cp", temporary_pdf.name, os.path.join(root_folder, f"{count}-exiftool.pdf")])
                count += 1

        if config["args"].setdefault("run_ps2pdf", schema.RUN_PS2PDF_DEFAULT):
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
                    f"{scan_to_paperless_meta}, {tesseract_producer}"
                    if tesseract_producer
                    else scan_to_paperless_meta
                )
            pdf_.save(temporary_pdf.name)
        if progress:
            call(["cp", temporary_pdf.name, os.path.join(root_folder, f"{count}-pikepdf.pdf")])
            count += 1
        call(["cp", temporary_pdf.name, destination])


def process_code() -> None:
    """Detect ad add a page with the QR codes."""
    for pdf_filename in glob.glob(os.path.join(os.environ.get("SCAN_CODES_FOLDER", "/scan-codes"), "*.pdf")):
        destination_filename = os.path.join(
            os.environ.get("SCAN_FINAL_FOLDER", "/destination"), os.path.basename(pdf_filename)
        )

        if os.path.exists(destination_filename):
            continue

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


def is_sources_present(images: List[str], root_folder: str) -> bool:
    """Are sources present for the next step."""
    for img in images:
        if not os.path.exists(os.path.join(root_folder, img)):
            print(f"Missing {root_folder} - {img}")
            return False
    return True


def save_config(config: schema.Configuration, config_file_name: str) -> None:
    """Save the configuration."""
    yaml = YAML()
    yaml.default_flow_style = False
    with open(config_file_name + "_", "w", encoding="utf-8") as config_file:
        yaml.dump(config, config_file)
    os.rename(config_file_name + "_", config_file_name)


def _process(config_file_name: str, dirty: bool = False, print_waiting: bool = True) -> Tuple[bool, bool]:
    """Propcess one document."""
    if not os.path.exists(config_file_name):
        return dirty, print_waiting

    root_folder = os.path.dirname(config_file_name)

    if os.path.exists(os.path.join(root_folder, "error.yaml")):
        return dirty, print_waiting

    yaml = YAML()
    yaml.default_flow_style = False
    with open(config_file_name, encoding="utf-8") as config_file:
        config: schema.Configuration = yaml.load(config_file.read())
    if config is None:
        print(config_file_name)
        print("Empty config")
        print_waiting = True
        return dirty, print_waiting

    if not is_sources_present(config["images"], root_folder):
        print(config_file_name)
        print("Missing images")
        print_waiting = True
        return dirty, print_waiting

    try:
        rerun = False
        if "steps" not in config:
            rerun = True
        while config.get("steps") and not is_sources_present(config["steps"][-1]["sources"], root_folder):
            config["steps"] = config["steps"][:-1]
            save_config(config, config_file_name)
            if os.path.exists(os.path.join(root_folder, "REMOVE_TO_CONTINUE")):
                os.remove(os.path.join(root_folder, "REMOVE_TO_CONTINUE"))
            print(config_file_name)
            print("Rerun step")
            print_waiting = True
            rerun = True

        if "steps" not in config or not config["steps"]:
            step: schema.Step = {
                "sources": config["images"],
                "name": "transform",
            }
            config["steps"] = [step]
        step = config["steps"][-1]

        if is_sources_present(step["sources"], root_folder):
            if os.path.exists(os.path.join(root_folder, "REMOVE_TO_CONTINUE")) and not rerun:
                return dirty, print_waiting
            if os.path.exists(os.path.join(root_folder, "DONE")) and not rerun:
                return dirty, print_waiting

            print(config_file_name)
            print_waiting = True
            dirty = True

            done = False
            next_step = None
            if step["name"] == "transform":
                print("Transform")
                next_step = transform(config, step, config_file_name, root_folder)
            elif step["name"] == "split":
                print("Split")
                next_step = split(config, step, root_folder)
            elif step["name"] == "finalise":
                print("Finalize")
                finalize(config, step, root_folder)
                done = True

            if done and os.environ.get("PROGRESS", "FALSE") != "TRUE":
                shutil.rmtree(root_folder)
            else:
                if next_step is not None:
                    config["steps"].append(next_step)
                save_config(config, config_file_name)
                with open(
                    os.path.join(root_folder, "DONE" if done else "REMOVE_TO_CONTINUE"),
                    "w",
                    encoding="utf-8",
                ):
                    pass
    except Exception as exception:
        print(exception)
        trace = traceback.format_exc()
        print(trace)
        print_waiting = True

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
        except Exception as exception2:
            print(exception2)
            print(traceback.format_exc())
            yaml = YAML()
            yaml.default_flow_style = False
            with open(os.path.join(root_folder, "error.yaml"), "w", encoding="utf-8") as error_file:
                yaml.dump(out, error_file)
    return dirty, print_waiting


def main() -> None:
    """Process the scanned documents."""
    parser = argparse.ArgumentParser("Process the scanned documents.")
    parser.add_argument("config", nargs="?", help="The config file to process.")
    args = parser.parse_args()

    if args.config:
        _process(args.config)
        sys.exit()

    print("Welcome to scanned images document to paperless.")
    print_waiting = True
    while True:
        dirty = False
        for config_file_name in glob.glob(
            os.path.join(os.environ.get("SCAN_SOURCE_FOLDER", "/source"), "*/config.yaml")
        ):
            dirty, print_waiting = _process(config_file_name, dirty, print_waiting)
        if not dirty:
            process_code()

        sys.stdout.flush()
        if not dirty:
            if print_waiting:
                print_waiting = False
                print("Waiting...")
            time.sleep(30)


if __name__ == "__main__":
    main()
