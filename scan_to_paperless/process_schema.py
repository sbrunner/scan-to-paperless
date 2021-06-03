from typing import Dict, List, TypedDict, Union

# Arguments
Arguments = TypedDict(
    "Arguments",
    {
        # true: => do level on 15% - 85% (under 15 % will be black above 85% will be white), false: => 0% - 100%, <number>: => (0 + <number>)% - (100 - number)%
        "level": Union[bool, int],
        # If no level specified, do auto level
        #
        # default: False
        "auto_level": bool,
        # Min level if no level end no autolovel
        #
        # default: 15
        "min_level": int,
        # Max level if no level end no autolovel
        #
        # default: 15
        "max_level": int,
        # Don't do any crop
        #
        # default: False
        "nocrop": bool,
        # The horizontal margin used on autodetect content [mm]
        #
        # default: 9
        "margin_horizontal": Union[int, float],
        # The vertical margin used on autodetect content [mm]
        #
        # default: 6
        "margin_vertical": Union[int, float],
        # The DPI used to convert the mm to pixel
        #
        # default: 300
        "dpi": Union[int, float],
        # Do the sharpen
        #
        # default: False
        "sharpen": bool,
        # Do the dither
        #
        # default: False
        "dither": bool,
        # Use tesseract to to an OCR on the document
        #
        # default: False
        "tesseract": bool,
        # The used language for tesseract
        #
        # default: fra+eng
        "tesseract_lang": str,
        # Do an assisted split
        #
        # default: False
        "append_credit_card": bool,
        # Do an assisted split
        #
        # default: False
        "assisted_split": bool,
    },
    total=False,
)


# Assisted split
#
# Assited split configuration
AssistedSplit = TypedDict(
    "AssistedSplit",
    {
        # The source image name
        "source": str,
        # The destination image positions
        "destinations": List[Union[int, str]],
        # The enhanced image name
        "image": str,
        # The (proposed) limits to do the assisted split, You should keep only the right one
        "limits": List["Limit"],
    },
    total=False,
)


# Configuration
Configuration = TypedDict(
    "Configuration",
    {
        # The images
        #
        # required
        "images": List[str],
        # The tile
        #
        # required
        "title": str,
        # The full name
        #
        # required
        "full_name": str,
        # The destination file name
        #
        # required
        "destination": str,
        # WARNING: The required are not correctly taken in account,
        # See: https://github.com/camptocamp/jsonschema-gentypes/issues/6
        #
        # required
        "args": "Arguments",
        # The carried out steps description
        "steps": List["Step"],
        "assisted_split": List["AssistedSplit"],
        # The transformed image, if removed the jobs will rag again from start
        "transformed_images": List[str],
        # The ignored errors
        "intermediate_error": List["IntermediateError"],
        "images_config": Dict[str, "_ConfigurationImagesConfigAdditionalproperties"],
        "images_status": Dict[str, "_ConfigurationImagesStatusAdditionalproperties"],
    },
    total=False,
)


# Intermediate error
IntermediateError = TypedDict(
    "IntermediateError",
    {
        "error": str,
        "traceback": List[str],
    },
    total=False,
)


# Limit
Limit = TypedDict(
    "Limit",
    {
        # The name visible on the generated image
        "name": str,
        # The kind of split
        "type": str,
        # The split position
        "value": int,
        # Is vertical?
        "vertical": bool,
        # The margin around the split, can be used to remove a fold
        "margin": int,
    },
    total=False,
)


# Step
Step = TypedDict(
    "Step",
    {
        # The step name
        "name": str,
        # The images obtain after the current step
        "sources": List[str],
        # The step number
        "process_count": int,
    },
    total=False,
)


_ConfigurationImagesConfigAdditionalproperties = TypedDict(
    "_ConfigurationImagesConfigAdditionalproperties",
    {
        # The used angle to deskex, can be change, restart by deleting one of the generated images
        "angle": Union[Union[int, float], None],
    },
    total=False,
)


_ConfigurationImagesStatusAdditionalproperties = TypedDict(
    "_ConfigurationImagesStatusAdditionalproperties",
    {
        # The measured deskew angle
        "angle": Union[int, float],
        # The measured deskew angle deviation
        "average_deviation": Union[int, float],
        # The measured possible deskew angles, visible on the generated image
        "angles": List[Union[int, float]],
        # The image dimensions
        "size": List[Union[int, float]],
    },
    total=False,
)
