from typing import Dict, List, TypedDict, Union

# Default value of the field path 'Arguments append_credit_card'
APPEND_CREDIT_CARD_DEFAULT = False


# Default value of the field path 'Arguments assisted_split'
ASSISTED_SPLIT_DEFAULT = False


# Default value of the field path 'Arguments auto_level'
AUTO_LEVEL_DEFAULT = False


# Arguments
#
# Editor note: The properties of this object should be modified in the config_schema.json file
Arguments = TypedDict(
    "Arguments",
    {
        # Level
        #
        # true: => do level on 15% - 85% (under 15 % will be black above 85% will be white), false: => 0% - 100%, <number>: => (0 + <number>)% - (100 - number)%
        #
        # default: False
        "level": Union[bool, int],
        # Auto level
        #
        # If no level specified, do auto level
        #
        # default: False
        "auto_level": bool,
        # Min level
        #
        # Min level if no level end no auto-level
        #
        # default: 15
        "min_level": Union[int, float],
        # Max level
        #
        # Max level if no level end no auto-level
        #
        # default: 85
        "max_level": Union[int, float],
        # Cut white
        #
        # Set the near white pixels on the image to white
        #
        # default: 200
        "cut_white": Union[int, float],
        # Cut black
        #
        # Set the near black pixels on the image to black
        #
        # default: 0
        "cut_black": Union[int, float],
        # No crop
        #
        # Don't do any crop
        #
        # default: False
        "no_crop": bool,
        # Margin horizontal
        #
        # The horizontal margin used on auto-detect content [mm]
        #
        # default: 9
        "margin_horizontal": Union[int, float],
        # Margin vertical
        #
        # The vertical margin used on auto-detect content [mm]
        #
        # default: 6
        "margin_vertical": Union[int, float],
        # Dpi
        #
        # The DPI used to convert the mm to pixel
        #
        # default: 300
        "dpi": Union[int, float],
        # Sharpen
        #
        # Do the sharpen
        #
        # default: False
        "sharpen": bool,
        # Dither
        #
        # Do the dither
        #
        # default: False
        "dither": bool,
        # Tesseract
        #
        # Use tesseract to to an OCR on the document
        #
        # default: True
        "tesseract": bool,
        # Tesseract lang
        #
        # The used language for tesseract
        #
        # default: fra+eng
        "tesseract_lang": str,
        # Append credit card
        #
        # Do an assisted split
        #
        # default: False
        "append_credit_card": bool,
        # Assisted split
        #
        # Do an assisted split
        #
        # default: False
        "assisted_split": bool,
        # Min box size crop
        #
        # The minimum box size to find the content on witch one we will crop [mm]
        #
        # default: 3
        "min_box_size_crop": Union[int, float],
        # Min box black crop
        #
        # The minimum black in a box on content find on witch one we will crop [%]
        #
        # default: 2
        "min_box_black_crop": Union[int, float],
        # Contour kernel size crop
        #
        # The block size used in a box on content find on witch one we will crop [mm]
        #
        # default: 1.5
        "contour_kernel_size_crop": Union[int, float],
        # Threshold block size crop
        #
        # The block size used in a box on threshold for content find on witch one we will crop [mm]
        #
        # default: 1.5
        "threshold_block_size_crop": Union[int, float],
        # Threshold value c crop
        #
        # A variable used on threshold, should be low on low contrast image, used in a box on content find on witch one we will crop
        #
        # default: 70
        "threshold_value_c_crop": Union[int, float],
        # Min box size empty
        #
        # The minimum box size to find the content to determine if the page is empty [mm]
        #
        # default: 10
        "min_box_size_empty": Union[int, float],
        # Min box black empty
        #
        # The minimum black in a box on content find if the page is empty [%]
        #
        # default: 2
        "min_box_black_empty": Union[int, float],
        # Contour kernel size empty
        #
        # The block size used in a box on content find if the page is empty [mm]
        #
        # default: 1.5
        "contour_kernel_size_empty": Union[int, float],
        # Threshold block size empty
        #
        # The block size used in a box on threshold for content find if the page is empty [mm]
        #
        # default: 1.5
        "threshold_block_size_empty": Union[int, float],
        # Threshold value c empty
        #
        # A variable used on threshold, should be low on low contrast image, used in a box on content find if the page is empty
        #
        # default: 70
        "threshold_value_c_empty": Union[int, float],
        # Min box size limit
        #
        # The minimum box size to find the limits based on content [mm]
        #
        # default: 10
        "min_box_size_limit": Union[int, float],
        # Min box black limit
        #
        # The minimum black in a box on content find the limits based on content [%]
        #
        # default: 2
        "min_box_black_limit": Union[int, float],
        # Contour kernel size limit
        #
        # The block size used in a box on content find the limits based on content [mm]
        #
        # default: 1.5
        "contour_kernel_size_limit": Union[int, float],
        # Threshold block size limit
        #
        # The block size used in a box on threshold for content find the limits based on content [mm]
        #
        # default: 1.5
        "threshold_block_size_limit": Union[int, float],
        # Threshold value c limit
        #
        # A variable used on threshold, should be low on low contrast image, used in a box on content find the limits based on content
        #
        # default: 70
        "threshold_value_c_limit": Union[int, float],
        # Colors
        #
        # The number of colors in the png
        #
        # default: 0
        "colors": int,
        # Run optipng
        #
        # Run the optipng optimizer
        #
        # default: True
        "run_optipng": bool,
        # Run pngquant
        #
        # Run the pngquant optimizer
        #
        # default: False
        "run_pngquant": bool,
        # Pngquant options
        #
        # The pngquant options
        #
        # default:
        #   - --force
        #   - --speed=1
        #   - --strip
        #   - --quality=0-32
        "pngquant_options": List[str],
        # Run exiftool
        #
        # Run the exiftool optimizer
        #
        # default: False
        "run_exiftool": bool,
        # Run ps2pdf
        #
        # Run the ps2pdf optimizer (=> JPEG)
        #
        # default: False
        "run_ps2pdf": bool,
        # Jpeg
        #
        # Convert images to JPEG
        #
        # default: False
        "jpeg": bool,
        # Jpeg quality
        #
        # The JPEG quality
        #
        # default: 90
        "jpeg_quality": int,
        # Background color
        #
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
        # WARNING: The required are not correctly taken in account,
        # See: https://github.com/camptocamp/jsonschema-gentypes/issues/6
        "deskew": "_ArgumentsDeskew",
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


# Auto mask
#
# The auto mask configuration, the mask is used to mask the image on crop and skew calculation
#
# Editor note: The properties of this object should be modified in the config_schema.json file
AutoMask = TypedDict(
    "AutoMask",
    {
        # Lower hsv color
        #
        # The lower color in HSV representation
        #
        # default:
        #   - 0
        #   - 0
        #   - 250
        "lower_hsv_color": List[int],
        # Upper hsv color
        #
        # The upper color in HSV representation
        #
        # default:
        #   - 255
        #   - 10
        #   - 255
        "upper_hsv_color": List[int],
        # De noise morphology
        #
        # Apply a morphology operation to remove noise
        #
        # default: True
        "de_noise_morphology": bool,
        # Inverse mask
        #
        # Inverse the mask
        #
        # default: False
        "inverse_mask": bool,
        # De noise size
        #
        # The size of the artifact that will be de noise
        #
        # default: 1000
        "de_noise_size": int,
        # De noise level
        #
        # The threshold level used in de noise on the blurry image
        #
        # default: 220
        "de_noise_level": int,
        # Buffer size
        #
        # The size of the buffer add on the mask
        #
        # default: 20
        "buffer_size": int,
        # Buffer level
        #
        # The threshold level used in buffer on the blurry image
        #
        # default: 20
        "buffer_level": int,
        # An image file used to add on the mask
        "additional_filename": str,
    },
    total=False,
)


# Default value of the field path 'Arguments background_color'
BACKGROUND_COLOR_DEFAULT = [255, 255, 255]


# Default value of the field path 'Auto mask buffer_level'
BUFFER_LEVEL_DEFAULT = 20


# Default value of the field path 'Auto mask buffer_size'
BUFFER_SIZE_DEFAULT = 20


# Default value of the field path 'Arguments colors'
COLORS_DEFAULT = 0


# Default value of the field path 'Arguments contour_kernel_size_crop'
CONTOUR_KERNEL_SIZE_CROP_DEFAULT = 1.5


# Default value of the field path 'Arguments contour_kernel_size_empty'
CONTOUR_KERNEL_SIZE_EMPTY_DEFAULT = 1.5


# Default value of the field path 'Arguments contour_kernel_size_limit'
CONTOUR_KERNEL_SIZE_LIMIT_DEFAULT = 1.5


# Default value of the field path 'Arguments cut_black'
CUT_BLACK_DEFAULT = 0


# Default value of the field path 'Arguments cut_white'
CUT_WHITE_DEFAULT = 200


# Configuration
Configuration = TypedDict(
    "Configuration",
    {
        # The images
        #
        # required
        "images": List[str],
        # WARNING: The required are not correctly taken in account,
        # See: https://github.com/camptocamp/jsonschema-gentypes/issues/6
        #
        # required
        "args": "Arguments",
        # Progress
        #
        # Run in progress mode
        #
        # default: False
        "progress": bool,
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


# Default value of the field path 'Arguments deskew angle_derivation'
DESKEW_ANGLE_DERIVATION_DEFAULT = 0.1


# Default value of the field path 'Arguments deskew angle_pm_90'
DESKEW_ANGLE_PM_90_DEFAULT = False


# Default value of the field path 'Arguments deskew max_angle'
DESKEW_MAX_ANGLE_DEFAULT = 10


# Default value of the field path 'Arguments deskew min_angle'
DESKEW_MIN_ANGLE_DEFAULT = -10


# Default value of the field path 'Arguments deskew num_peaks'
DESKEW_NUM_PEAKS_DEFAULT = 20


# Default value of the field path 'Arguments deskew sigma'
DESKEW_SIGMA_DEFAULT = 3.0


# Default value of the field path 'Auto mask de_noise_level'
DE_NOISE_LEVEL_DEFAULT = 220


# Default value of the field path 'Auto mask de_noise_morphology'
DE_NOISE_MORPHOLOGY_DEFAULT = True


# Default value of the field path 'Auto mask de_noise_size'
DE_NOISE_SIZE_DEFAULT = 1000


# Default value of the field path 'Arguments dither'
DITHER_DEFAULT = False


# Default value of the field path 'Arguments dpi'
DPI_DEFAULT = 300


# Default value of the field path 'Auto mask inverse_mask'
INVERSE_MASK_DEFAULT = False


# Intermediate error
IntermediateError = TypedDict(
    "IntermediateError",
    {
        "error": str,
        "traceback": List[str],
    },
    total=False,
)


# Default value of the field path 'Arguments jpeg'
JPEG_DEFAULT = False


# Default value of the field path 'Arguments jpeg_quality'
JPEG_QUALITY_DEFAULT = 90


# Default value of the field path 'Arguments level'
LEVEL_DEFAULT = False


# Default value of the field path 'Auto mask lower_hsv_color'
LOWER_HSV_COLOR_DEFAULT = [0, 0, 250]


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


# Default value of the field path 'Arguments margin_horizontal'
MARGIN_HORIZONTAL_DEFAULT = 9


# Default value of the field path 'Arguments margin_vertical'
MARGIN_VERTICAL_DEFAULT = 6


# Default value of the field path 'Arguments max_level'
MAX_LEVEL_DEFAULT = 85


# Default value of the field path 'Arguments min_box_black_crop'
MIN_BOX_BLACK_CROP_DEFAULT = 2


# Default value of the field path 'Arguments min_box_black_empty'
MIN_BOX_BLACK_EMPTY_DEFAULT = 2


# Default value of the field path 'Arguments min_box_black_limit'
MIN_BOX_BLACK_LIMIT_DEFAULT = 2


# Default value of the field path 'Arguments min_box_size_crop'
MIN_BOX_SIZE_CROP_DEFAULT = 3


# Default value of the field path 'Arguments min_box_size_empty'
MIN_BOX_SIZE_EMPTY_DEFAULT = 10


# Default value of the field path 'Arguments min_box_size_limit'
MIN_BOX_SIZE_LIMIT_DEFAULT = 10


# Default value of the field path 'Arguments min_level'
MIN_LEVEL_DEFAULT = 15


# Default value of the field path 'Arguments no_crop'
NO_CROP_DEFAULT = False


# Default value of the field path 'Arguments pngquant_options'
PNGQUANT_OPTIONS_DEFAULT = ["--force", "--speed=1", "--strip", "--quality=0-32"]


# Default value of the field path 'Configuration progress'
PROGRESS_DEFAULT = False


# Default value of the field path 'Arguments run_exiftool'
RUN_EXIFTOOL_DEFAULT = False


# Default value of the field path 'Arguments run_optipng'
RUN_OPTIPNG_DEFAULT = True


# Default value of the field path 'Arguments run_pngquant'
RUN_PNGQUANT_DEFAULT = False


# Default value of the field path 'Arguments run_ps2pdf'
RUN_PS2PDF_DEFAULT = False


# Default value of the field path 'Arguments sharpen'
SHARPEN_DEFAULT = False


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


# Default value of the field path 'Arguments tesseract'
TESSERACT_DEFAULT = True


# Default value of the field path 'Arguments tesseract_lang'
TESSERACT_LANG_DEFAULT = "fra+eng"


# Default value of the field path 'Arguments threshold_block_size_crop'
THRESHOLD_BLOCK_SIZE_CROP_DEFAULT = 1.5


# Default value of the field path 'Arguments threshold_block_size_empty'
THRESHOLD_BLOCK_SIZE_EMPTY_DEFAULT = 1.5


# Default value of the field path 'Arguments threshold_block_size_limit'
THRESHOLD_BLOCK_SIZE_LIMIT_DEFAULT = 1.5


# Default value of the field path 'Arguments threshold_value_c_crop'
THRESHOLD_VALUE_C_CROP_DEFAULT = 70


# Default value of the field path 'Arguments threshold_value_c_empty'
THRESHOLD_VALUE_C_EMPTY_DEFAULT = 70


# Default value of the field path 'Arguments threshold_value_c_limit'
THRESHOLD_VALUE_C_LIMIT_DEFAULT = 70


# Default value of the field path 'Auto mask upper_hsv_color'
UPPER_HSV_COLOR_DEFAULT = [255, 10, 255]


# The deskew configuration
_ArgumentsDeskew = TypedDict(
    "_ArgumentsDeskew",
    {
        # Deskew min angle
        #
        # The minimum angle to detect the image skew [degree]
        #
        # default: -10
        "min_angle": Union[int, float],
        # Deskew max angle
        #
        # The maximum angle to detect the image skew [degree]
        #
        # default: 10
        "max_angle": Union[int, float],
        # Deskew angle derivation
        #
        # The step of angle to detect the image skew [degree]
        #
        # default: 0.1
        "angle_derivation": Union[int, float],
        # Deskew sigma
        #
        # Used in the `canny` function
        #
        # default: 3.0
        "sigma": Union[int, float],
        # Deskew num peaks
        #
        # number of peaks we ask for
        #
        # default: 20
        "num_peaks": int,
        # Deskew angle pm 90
        #
        # Detect an angle of +/- 90 degree, also +/- 45 degree
        #
        # default: False
        "angle_pm_90": bool,
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
        # The image dimensions
        "size": List[Union[int, float]],
    },
    total=False,
)
