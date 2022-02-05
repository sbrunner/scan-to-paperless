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
        # Min level if no level end no auto-level
        #
        # default: 15
        "min_level": Union[int, float],
        # Max level if no level end no auto-level
        #
        # default: 15
        "max_level": Union[int, float],
        # Don't do any crop
        #
        # default: False
        "no_crop": bool,
        # The horizontal margin used on auto-detect content [mm]
        #
        # default: 9
        "margin_horizontal": Union[int, float],
        # The vertical margin used on auto-detect content [mm]
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
        # The number of angle used to detect the image skew
        #
        # default: 1800
        "num_angles": Union[int, float],
        # The minimum box size to find the content on witch one we will crop [mm]
        #
        # default: 3
        "min_box_size_crop": Union[int, float],
        # The minimum black in a box on content find on witch one we will crop [%]
        #
        # default: 2
        "min_box_black_crop": Union[int, float],
        # The block size used in a box on content find on witch one we will crop [mm]
        #
        # default: 1.5
        "contour_kernel_size_crop": Union[int, float],
        # The block size used in a box on threshold for content find on witch one we will crop [mm]
        #
        # default: 1.5
        "threshold_block_size_crop": Union[int, float],
        # A variable used on threshold, should be low on low contrast image, used in a box on content find on witch one we will crop
        #
        # default: 70
        "threshold_value_c_crop": Union[int, float],
        # The minimum box size to find the content to determine if the page is empty [mm]
        #
        # default: 10
        "min_box_size_empty": Union[int, float],
        # The minimum black in a box on content find if the page is empty [%]
        #
        # default: 2
        "min_box_black_empty": Union[int, float],
        # The block size used in a box on content find if the page is empty [mm]
        #
        # default: 1.5
        "contour_kernel_size_empty": Union[int, float],
        # The block size used in a box on threshold for content find if the page is empty [mm]
        #
        # default: 1.5
        "threshold_block_size_empty": Union[int, float],
        # A variable used on threshold, should be low on low contrast image, used in a box on content find if the page is empty
        #
        # default: 70
        "threshold_value_c_empty": Union[int, float],
        # The minimum box size to find the limits based on content [mm]
        #
        # default: 3
        "min_box_size_limit": Union[int, float],
        # The minimum black in a box on content find the limits based on content [%]
        #
        # default: 2
        "min_box_black_limit": Union[int, float],
        # The block size used in a box on content find the limits based on content [mm]
        #
        # default: 1.5
        "contour_kernel_size_limit": Union[int, float],
        # The block size used in a box on threshold for content find the limits based on content [mm]
        #
        # default: 1.5
        "threshold_block_size_limit": Union[int, float],
        # A variable used on threshold, should be low on low contrast image, used in a box on content find the limits based on content
        #
        # default: 70
        "threshold_value_c_limit": Union[int, float],
    },
    total=False,
)


# Assisted split
#
# Assisted split configuration
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
        # The destination file name
        #
        # required
        "destination": str,
        # WARNING: The required are not correctly taken in account,
        # See: https://github.com/camptocamp/jsonschema-gentypes/issues/6
        #
        # required
        "args": "Arguments",
        # Run in progress mode
        #
        # default: False
        "progress": bool,
        # Run the experimental features
        #
        # default: False
        "experimental": bool,
        # The carried out steps description
        "steps": List["Step"],
        "assisted_split": List["AssistedSplit"],
        # The transformed image, if removed the jobs will rag again from start
        "transformed_images": List[str],
        # The ignored errors
        "intermediate_error": List["IntermediateError"],
        "images_config": Dict[str, "_ConfigurationImagesConfigAdditionalproperties"],
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
        # The used angle to deskew, can be change, restart by deleting one of the generated images
        "angle": Union[Union[int, float], None],
        # WARNING: The required are not correctly taken in account,
        # See: https://github.com/camptocamp/jsonschema-gentypes/issues/6
        "status": "_ConfigurationImagesConfigAdditionalpropertiesStatus",
    },
    total=False,
)


_ConfigurationImagesConfigAdditionalpropertiesStatus = TypedDict(
    "_ConfigurationImagesConfigAdditionalpropertiesStatus",
    {
        # The measured deskew angle
        "angle": Union[int, float],
        # The measured deskew angle deviation
        "average_deviation": Union[int, float],
        # The measured deskew angle deviation, corrected
        "average_deviation2": Union[int, float],
        # The measured possible deskew angles, visible on the generated image
        "angles": List[Union[int, float]],
        # The image dimensions
        "size": List[Union[int, float]],
    },
    total=False,
)
