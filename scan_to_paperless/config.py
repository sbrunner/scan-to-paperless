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
        # Set the near white pixels on the image to white
        #
        # default: 250
        "cut_white": Union[int, float],
        # Set the near black pixels on the image to black
        #
        # default: 0
        "cut_black": Union[int, float],
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
        # The number of colors in the png
        #
        # default: 0
        "colors": int,
        # Run the optipng optimizer
        #
        # default: True
        "run_optipng": bool,
        # Run the pngquant optimizer
        #
        # default: False
        "run_pngquant": bool,
        # The pngquant options
        #
        # default:
        #   - --force
        #   - --speed=1
        #   - --strip
        #   - --quality=0-32
        "pngquant_options": List[str],
        # Run the exiftool optimizer
        #
        # default: False
        "run_exiftool": bool,
        # Run the ps2pdf optimizer (=> JPEG)
        #
        # default: False
        "run_ps2pdf": bool,
        # Convert images to JPEG
        #
        # default: False
        "jpeg": bool,
        # The JPEG quality
        #
        # default: 90
        "jpeg_quality": int,
        # The background color
        #
        # default:
        #   - 255
        #   - 255
        #   - 255
        "background_color": List[int],
        # WARNING: The required are not correctly taken in account,
        # See: https://github.com/camptocamp/jsonschema-gentypes/issues/6
        "auto_mask": "AutoMask",
        # WARNING: The required are not correctly taken in account,
        # See: https://github.com/camptocamp/jsonschema-gentypes/issues/6
        "auto_cut": "AutoMask",
        # The minimum angle to detect the image skew [degree]
        #
        # default: -10
        "deskew_min_angle": Union[int, float],
        # The maximum angle to detect the image skew [degree]
        #
        # default: 10
        "deskew_max_angle": Union[int, float],
        # The step of angle to detect the image skew [degree]
        #
        # default: 0.1
        "deskew_angle_derivation": Union[int, float],
    },
    total=False,
)


# Auto mask
#
# The auto mask configuration, the mask is used to mask the image on crop and deskew calculation
AutoMask = TypedDict(
    "AutoMask",
    {
        # The lower color in HSV representation
        #
        # default:
        #   - 0
        #   - 0
        #   - 250
        "lower_hsv_color": List[int],
        # The upper color in HSV representation
        #
        # default:
        #   - 255
        #   - 10
        #   - 255
        "upper_hsv_color": List[int],
        # Apply a morphology operation to remove noise
        #
        # default: True
        "de_noise_morphology": bool,
        # Inverse the mask
        #
        # default: False
        "inverse_mask": bool,
        # The size of the artifact that will be de noise
        #
        # default: 20
        "de_noise_size": int,
        # The threshold level used in de noise on the blurry image
        #
        # default: 220
        "de_noise_level": int,
        # The size of the buffer add on the mask
        #
        # default: 50 an case of mask, 20 in case of cut
        "buffer_size": int,
        # The threshold level used in buffer on the blurry image
        #
        # default: 20
        "buffer_level": int,
        # An image file used to add on the mask
        "additional_filename": str,
    },
    total=False,
)


# Configuration
Configuration = TypedDict(
    "Configuration",
    {
        # The configuration to extends
        "extends": str,
        # WARNING: The required are not correctly taken in account,
        # See: https://github.com/camptocamp/jsonschema-gentypes/issues/6
        "merge_strategies": "MergeStrategies",
        # This should be shared with the process container in 'source'.
        "scan_folder": str,
        # The scanimage command
        #
        # default: scanimage
        "scanimage": str,
        # The scanimage arguments
        #
        # default:
        #   - --format=png
        #   - --mode=color
        #   - --resolution=300
        "scanimage_arguments": List[str],
        # The extension of generate image (png or tiff)
        #
        # default: png
        "extension": str,
        # WARNING: The required are not correctly taken in account,
        # See: https://github.com/camptocamp/jsonschema-gentypes/issues/6
        "default_args": "Arguments",
        # The command used to start the viewer
        #
        # default: eog
        "viewer": str,
        # Customize the modes
        "modes": Dict[str, "_ConfigurationModesAdditionalproperties"],
    },
    total=False,
)


# Merge strategies
#
# The merge strategy to use, see https://deepmerge.readthedocs.io/en/latest/strategies.html#builtin-strategies
MergeStrategies = TypedDict(
    "MergeStrategies",
    {
        # The merge strategy to use on list
        #
        # default:
        #   - override
        "list": List[str],
        # The merge strategy to use on dict
        #
        # default:
        #   - merge
        "dict": List[str],
        # The fallback merge strategy
        #
        # default:
        #   - override
        "fallback": List[str],
        # The type_conflict merge strategy
        #
        # default:
        #   - override
        "type_conflict": List[str],
    },
    total=False,
)


# Default value of the field path 'Arguments append_credit_card'
_ARGUMENTS_APPEND_CREDIT_CARD_DEFAULT = False


# Default value of the field path 'Arguments assisted_split'
_ARGUMENTS_ASSISTED_SPLIT_DEFAULT = False


# Default value of the field path 'Arguments auto_level'
_ARGUMENTS_AUTO_LEVEL_DEFAULT = False


# Default value of the field path 'Arguments background_color'
_ARGUMENTS_BACKGROUND_COLOR_DEFAULT = [255, 255, 255]


# Default value of the field path 'Arguments colors'
_ARGUMENTS_COLORS_DEFAULT = 0


# Default value of the field path 'Arguments contour_kernel_size_crop'
_ARGUMENTS_CONTOUR_KERNEL_SIZE_CROP_DEFAULT = 1.5


# Default value of the field path 'Arguments contour_kernel_size_empty'
_ARGUMENTS_CONTOUR_KERNEL_SIZE_EMPTY_DEFAULT = 1.5


# Default value of the field path 'Arguments contour_kernel_size_limit'
_ARGUMENTS_CONTOUR_KERNEL_SIZE_LIMIT_DEFAULT = 1.5


# Default value of the field path 'Arguments cut_black'
_ARGUMENTS_CUT_BLACK_DEFAULT = 0


# Default value of the field path 'Arguments cut_white'
_ARGUMENTS_CUT_WHITE_DEFAULT = 250


# Default value of the field path 'Arguments deskew_angle_derivation'
_ARGUMENTS_DESKEW_ANGLE_DERIVATION_DEFAULT = 0.1


# Default value of the field path 'Arguments deskew_max_angle'
_ARGUMENTS_DESKEW_MAX_ANGLE_DEFAULT = 10


# Default value of the field path 'Arguments deskew_min_angle'
_ARGUMENTS_DESKEW_MIN_ANGLE_DEFAULT = -10


# Default value of the field path 'Arguments dither'
_ARGUMENTS_DITHER_DEFAULT = False


# Default value of the field path 'Arguments dpi'
_ARGUMENTS_DPI_DEFAULT = 300


# Default value of the field path 'Arguments jpeg'
_ARGUMENTS_JPEG_DEFAULT = False


# Default value of the field path 'Arguments jpeg_quality'
_ARGUMENTS_JPEG_QUALITY_DEFAULT = 90


# Default value of the field path 'Arguments margin_horizontal'
_ARGUMENTS_MARGIN_HORIZONTAL_DEFAULT = 9


# Default value of the field path 'Arguments margin_vertical'
_ARGUMENTS_MARGIN_VERTICAL_DEFAULT = 6


# Default value of the field path 'Arguments max_level'
_ARGUMENTS_MAX_LEVEL_DEFAULT = 15


# Default value of the field path 'Arguments min_box_black_crop'
_ARGUMENTS_MIN_BOX_BLACK_CROP_DEFAULT = 2


# Default value of the field path 'Arguments min_box_black_empty'
_ARGUMENTS_MIN_BOX_BLACK_EMPTY_DEFAULT = 2


# Default value of the field path 'Arguments min_box_black_limit'
_ARGUMENTS_MIN_BOX_BLACK_LIMIT_DEFAULT = 2


# Default value of the field path 'Arguments min_box_size_crop'
_ARGUMENTS_MIN_BOX_SIZE_CROP_DEFAULT = 3


# Default value of the field path 'Arguments min_box_size_empty'
_ARGUMENTS_MIN_BOX_SIZE_EMPTY_DEFAULT = 10


# Default value of the field path 'Arguments min_box_size_limit'
_ARGUMENTS_MIN_BOX_SIZE_LIMIT_DEFAULT = 3


# Default value of the field path 'Arguments min_level'
_ARGUMENTS_MIN_LEVEL_DEFAULT = 15


# Default value of the field path 'Arguments no_crop'
_ARGUMENTS_NO_CROP_DEFAULT = False


# Default value of the field path 'Arguments pngquant_options'
_ARGUMENTS_PNGQUANT_OPTIONS_DEFAULT = ["--force", "--speed=1", "--strip", "--quality=0-32"]


# Default value of the field path 'Arguments run_exiftool'
_ARGUMENTS_RUN_EXIFTOOL_DEFAULT = False


# Default value of the field path 'Arguments run_optipng'
_ARGUMENTS_RUN_OPTIPNG_DEFAULT = True


# Default value of the field path 'Arguments run_pngquant'
_ARGUMENTS_RUN_PNGQUANT_DEFAULT = False


# Default value of the field path 'Arguments run_ps2pdf'
_ARGUMENTS_RUN_PS2PDF_DEFAULT = False


# Default value of the field path 'Arguments sharpen'
_ARGUMENTS_SHARPEN_DEFAULT = False


# Default value of the field path 'Arguments tesseract'
_ARGUMENTS_TESSERACT_DEFAULT = False


# Default value of the field path 'Arguments tesseract_lang'
_ARGUMENTS_TESSERACT_LANG_DEFAULT = "fra+eng"


# Default value of the field path 'Arguments threshold_block_size_crop'
_ARGUMENTS_THRESHOLD_BLOCK_SIZE_CROP_DEFAULT = 1.5


# Default value of the field path 'Arguments threshold_block_size_empty'
_ARGUMENTS_THRESHOLD_BLOCK_SIZE_EMPTY_DEFAULT = 1.5


# Default value of the field path 'Arguments threshold_block_size_limit'
_ARGUMENTS_THRESHOLD_BLOCK_SIZE_LIMIT_DEFAULT = 1.5


# Default value of the field path 'Arguments threshold_value_c_crop'
_ARGUMENTS_THRESHOLD_VALUE_C_CROP_DEFAULT = 70


# Default value of the field path 'Arguments threshold_value_c_empty'
_ARGUMENTS_THRESHOLD_VALUE_C_EMPTY_DEFAULT = 70


# Default value of the field path 'Arguments threshold_value_c_limit'
_ARGUMENTS_THRESHOLD_VALUE_C_LIMIT_DEFAULT = 70


# Default value of the field path 'Auto mask buffer_level'
_AUTO_MASK_BUFFER_LEVEL_DEFAULT = 20


# Default value of the field path 'Auto mask buffer_size'
_AUTO_MASK_BUFFER_SIZE_DEFAULT = "50 an case of mask, 20 in case of cut"


# Default value of the field path 'Auto mask de_noise_level'
_AUTO_MASK_DE_NOISE_LEVEL_DEFAULT = 220


# Default value of the field path 'Auto mask de_noise_morphology'
_AUTO_MASK_DE_NOISE_MORPHOLOGY_DEFAULT = True


# Default value of the field path 'Auto mask de_noise_size'
_AUTO_MASK_DE_NOISE_SIZE_DEFAULT = 20


# Default value of the field path 'Auto mask inverse_mask'
_AUTO_MASK_INVERSE_MASK_DEFAULT = False


# Default value of the field path 'Auto mask lower_hsv_color'
_AUTO_MASK_LOWER_HSV_COLOR_DEFAULT = [0, 0, 250]


# Default value of the field path 'Auto mask upper_hsv_color'
_AUTO_MASK_UPPER_HSV_COLOR_DEFAULT = [255, 10, 255]


# Default value of the field path 'Configuration extension'
_CONFIGURATION_EXTENSION_DEFAULT = "png"


# Default value of the field path 'Configuration scanimage_arguments'
_CONFIGURATION_SCANIMAGE_ARGUMENTS_DEFAULT = ["--format=png", "--mode=color", "--resolution=300"]


# Default value of the field path 'Configuration scanimage'
_CONFIGURATION_SCANIMAGE_DEFAULT = "scanimage"


# Default value of the field path 'Configuration viewer'
_CONFIGURATION_VIEWER_DEFAULT = "eog"


_ConfigurationModesAdditionalproperties = TypedDict(
    "_ConfigurationModesAdditionalproperties",
    {
        # Additional scanimage arguments
        "scanimage_arguments": List[str],
        # Run the ADF in tow step odd and even, needed for scanner that don't support double face
        "auto_bash": bool,
        # Rotate the even pages, to use in conjunction with auto_bash
        "rotate_even": bool,
    },
    total=False,
)


# Default value of the field path 'Merge strategies dict'
_MERGE_STRATEGIES_DICT_DEFAULT = ["merge"]


# Default value of the field path 'Merge strategies fallback'
_MERGE_STRATEGIES_FALLBACK_DEFAULT = ["override"]


# Default value of the field path 'Merge strategies list'
_MERGE_STRATEGIES_LIST_DEFAULT = ["override"]


# Default value of the field path 'Merge strategies type_conflict'
_MERGE_STRATEGIES_TYPE_CONFLICT_DEFAULT = ["override"]
