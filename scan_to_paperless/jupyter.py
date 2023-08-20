"""Functions used to generate a Jupyter notebook from the transform."""


import os
from typing import TYPE_CHECKING, Any

# read, write, rotate, crop, sharpen, draw_line, find_line, find_contour
import nbformat
import numpy as np

import scan_to_paperless
import scan_to_paperless.process_utils
from scan_to_paperless import process_schema as schema

if TYPE_CHECKING:
    NpNdarrayInt = np.ndarray[np.uint8, Any]
else:
    NpNdarrayInt = np.ndarray


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


def create_transform_notebook(
    root_folder: str, context: scan_to_paperless.process_utils.Context, step: schema.Step
) -> None:
    """Create a Jupyter notebook for the transform step."""

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

from scan_to_paperless import process, process_utils"""
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

jupyter_locals = IPython.extract_module_locals()[1]
base_folder = os.path.dirname(os.path.dirname(jupyter_locals['__vsc_ipynb_file__']) if '' in jupyter_locals else os.getcwd())"""
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
context = process_utils.Context({{"args": {{}}}}, {{}})

# Open one of the images
context.image = cv2.imread(os.path.join(base_folder, "{step["sources"][0]}"))
{other_images_open}

images_context = {{"original": context.image}}"""
        )
    )

    notebook["cells"].append(
        nbformat.v4.new_markdown_cell(  # type: ignore[no-untyped-call]
            """Set the values that's used by more than one step."""
        )
    )
    notebook["cells"].append(
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            f"""context.config["args"] = {{
    "dpi": {context.config["args"].get("dpi", schema.DPI_DEFAULT)},
    "background_color": {context.config["args"].get("background_color", schema.BACKGROUND_COLOR_DEFAULT)},
}}"""
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
context.get_index = lambda image: np.ix_(
    np.arange(0, min(image.shape[0], 500)),
    np.arange(0, image.shape[1]),
    np.arange(0, image.shape[2]),
)

context.display_image(images_context["original"])"""
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
            f"""context.image = images_context["original"].copy()

context.config["args"]["mask"] = {_pretty_repr(context.config["args"].get("mask", {}))}

hsv = cv2.cvtColor(context.image, cv2.COLOR_BGR2HSV)
print("Hue (h)")
context.display_image(cv2.cvtColor(hsv[:, :, 0], cv2.COLOR_GRAY2RGB))
print("Saturation (s)")
context.display_image(cv2.cvtColor(hsv[:, :, 1], cv2.COLOR_GRAY2RGB))
print("Value (v)")
context.display_image(cv2.cvtColor(hsv[:, :, 2], cv2.COLOR_GRAY2RGB))

# Print the HSV value on some point of the image
points = [
    [10, 10],
    [100, 100],
]
image = context.image.copy()
for x, y in points:
    print(f"Pixel: {{x}}:{{y}}, with value: {{hsv[y, x, :]}}")
    cv2.drawMarker(image, [x, y], (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
context.display_image(image)

context.init_mask()
if context.mask is not None:
    context.display_image(cv2.cvtColor(context.mask, cv2.COLOR_GRAY2RGB))
context.display_image(context.get_masked())

images_context["mask"] = context.image"""
        )
    )

    notebook["cells"].append(
        nbformat.v4.new_markdown_cell("Display the image histogram.")  # type: ignore[no-untyped-call]
    )
    notebook["cells"].append(
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            f"""context.image = images_context["mask"].copy()

context.config["args"]["level"] = {context.config["args"].get("level", schema.LEVEL_DEFAULT)}
context.config["args"]["cut_white"] = {context.config["args"].get("cut_white", schema.CUT_WHITE_DEFAULT)}
context.config["args"]["cut_black"] = {context.config["args"].get("cut_black", schema.CUT_BLACK_DEFAULT)}

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
            f"""context.image = images_context["mask"].copy()

context.config["args"]["auto_level"] = {context.config["args"].get("auto_level", schema.AUTO_LEVEL_DEFAULT)},
context.config["args"]["level"] = {context.config["args"].get("level", schema.LEVEL_DEFAULT)},

process.level(context)
context.display_image(context.image)

images_context["level"] = context.image"""
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
            f"""context.image = images_context["level"].copy()

print(f"Use cut_white: {context.config["args"]["cut_white"]}")
print(f"Use cut_black: {context.config["args"]["cut_black"]}")

process.color_cut(context)
context.display_image(context.image)

images_context["color_cut"] = context.image"""
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
            f"""context.image = images_context["color_cut"].copy()

context.config["args"]["cut"] = {_pretty_repr(context.config["args"].get("cut", {}))}

# Print in HSV some point of the image
hsv = cv2.cvtColor(context.image, cv2.COLOR_BGR2HSV)
print("Pixel 10:10: ", hsv[10, 10])
print("Pixel 100:100: ", hsv[100, 100])

process.cut(context)
context.display_image(context.image)

images_context["cut"] = context.image"""
        )
    )

    notebook["cells"].append(
        nbformat.v4.new_markdown_cell("Do the image skew correction.")  # type: ignore[no-untyped-call]
    )
    notebook["cells"].append(
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            f"""context.image = images_context["cut"].copy()

context.config["args"]["deskew"] = {_pretty_repr(context.config["args"].get("deskew", {}))}

# The angle can be forced in config.images_config.<image_name>.angle.
process.deskew(context)
context.display_image(context.image)

images_context["deskew"] = context.image"""
        )
    )

    notebook["cells"].append(
        nbformat.v4.new_markdown_cell(  # type: ignore[no-untyped-call]
            """Do the image auto crop base on the image content."""
        )
    )
    notebook["cells"].append(
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            f"""context.image = images_context["deskew"].copy()

context.config["args"]["crop"] = {_pretty_repr(context.config["args"].get("crop", {}))}

process.docrop(context)
context.display_image(context.image)

images_context["crop"] = context.image"""
        )
    )

    notebook["cells"].append(
        nbformat.v4.new_markdown_cell("Do the image sharpen correction.")  # type: ignore[no-untyped-call]
    )
    notebook["cells"].append(
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            f"""context.image = images_context["crop"].copy()

context.config["args"]["sharpen"] = {context.config["args"].get("sharpen", schema.SHARPEN_DEFAULT)}

process.sharpen(context)
context.display_image(context.image)

images_context["sharpen"] = context.image"""
        )
    )

    notebook["cells"].append(
        nbformat.v4.new_markdown_cell("Do the image dither correction.")  # type: ignore[no-untyped-call]
    )
    notebook["cells"].append(
        nbformat.v4.new_code_cell(  # type: ignore[no-untyped-call]
            f"""context.image = images_context["sharpen"].copy()

context.config["args"]["dither"] = {context.config["args"].get("dither", schema.DITHER_DEFAULT)}

process.dither(context)
context.display_image(context.image)

images_context["dither"] = context.image"""
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
            """context.image = images_context["dither"].copy()

try:
    process.autorotate(context)
    context.display_image(context.image)
except FileNotFoundError as e:
    print("Tesseract not found, skipping autorotate: ", e)"""
        )
    )

    with open(os.path.join(dest_folder, "jupyter.ipynb"), "w", encoding="utf-8") as jupyter_file:
        nbformat.write(notebook, jupyter_file)  # type: ignore[no-untyped-call]
