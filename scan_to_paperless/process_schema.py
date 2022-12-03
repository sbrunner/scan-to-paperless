from typing import Dict, List, TypedDict, Union

APPEND_CREDIT_CARD_DEFAULT = False
"""Default value of the field path 'Arguments append_credit_card'"""


ASSISTED_SPLIT_DEFAULT = False
"""Default value of the field path 'Arguments assisted_split'"""


AUTO_LEVEL_DEFAULT = False
"""Default value of the field path 'Arguments auto_level'"""


class Arguments(TypedDict, total=False):
    """
    Arguments.

    Editor note: The properties of this object should be modified in the config_schema.json file
    """

    level: Union[bool, int]
    """
    Level.

    true: => do level on 15% - 85% (under 15 % will be black above 85% will be white), false: => 0% - 100%, <number>: => (0 + <number>)% - (100 - number)%

    default: False
    """

    auto_level: bool
    """
    Auto level.

    If no level specified, do auto level

    default: False
    """

    min_level: Union[int, float]
    """
    Min level.

    Min level if no level end no auto-level

    default: 0
    """

    max_level: Union[int, float]
    """
    Max level.

    Max level if no level end no auto-level

    default: 100
    """

    cut_white: Union[int, float]
    """
    Cut white.

    Set the near white pixels on the image to white

    default: 255
    """

    cut_black: Union[int, float]
    """
    Cut black.

    Set the near black pixels on the image to black

    default: 0
    """

    no_crop: bool
    """
    No crop.

    Don't do any crop

    default: False
    """

    margin_horizontal: Union[int, float]
    """
    Margin horizontal.

    The horizontal margin used on auto-detect content [mm]

    default: 9
    """

    margin_vertical: Union[int, float]
    """
    Margin vertical.

    The vertical margin used on auto-detect content [mm]

    default: 6
    """

    dpi: Union[int, float]
    """
    Dpi.

    The DPI used to convert the mm to pixel

    default: 300
    """

    sharpen: bool
    """
    Sharpen.

    Do the sharpen

    default: False
    """

    dither: bool
    """
    Dither.

    Do the dither

    default: False
    """

    tesseract: bool
    """
    Tesseract.

    Use tesseract to to an OCR on the document

    default: True
    """

    tesseract_lang: str
    """
    Tesseract lang.

    The used language for tesseract

    default: fra+eng
    """

    append_credit_card: bool
    """
    Append credit card.

    Do an assisted split

    default: False
    """

    assisted_split: bool
    """
    Assisted split.

    Do an assisted split

    default: False
    """

    min_box_size_crop: Union[int, float]
    """
    Min box size crop.

    The minimum box size to find the content on witch one we will crop [mm]

    default: 3
    """

    min_box_black_crop: Union[int, float]
    """
    Min box black crop.

    The minimum black in a box on content find on witch one we will crop [%]

    default: 2
    """

    contour_kernel_size_crop: Union[int, float]
    """
    Contour kernel size crop.

    The block size used in a box on content find on witch one we will crop [mm]

    default: 1.5
    """

    threshold_block_size_crop: Union[int, float]
    """
    Threshold block size crop.

    The block size used in a box on threshold for content find on witch one we will crop [mm]

    default: 1.5
    """

    threshold_value_c_crop: Union[int, float]
    """
    Threshold value c crop.

    A variable used on threshold, should be low on low contrast image, used in a box on content find on witch one we will crop

    default: 70
    """

    min_box_size_empty: Union[int, float]
    """
    Min box size empty.

    The minimum box size to find the content to determine if the page is empty [mm]

    default: 10
    """

    min_box_black_empty: Union[int, float]
    """
    Min box black empty.

    The minimum black in a box on content find if the page is empty [%]

    default: 2
    """

    contour_kernel_size_empty: Union[int, float]
    """
    Contour kernel size empty.

    The block size used in a box on content find if the page is empty [mm]

    default: 1.5
    """

    threshold_block_size_empty: Union[int, float]
    """
    Threshold block size empty.

    The block size used in a box on threshold for content find if the page is empty [mm]

    default: 1.5
    """

    threshold_value_c_empty: Union[int, float]
    """
    Threshold value c empty.

    A variable used on threshold, should be low on low contrast image, used in a box on content find if the page is empty

    default: 70
    """

    min_box_size_limit: Union[int, float]
    """
    Min box size limit.

    The minimum box size to find the limits based on content [mm]

    default: 10
    """

    min_box_black_limit: Union[int, float]
    """
    Min box black limit.

    The minimum black in a box on content find the limits based on content [%]

    default: 2
    """

    contour_kernel_size_limit: Union[int, float]
    """
    Contour kernel size limit.

    The block size used in a box on content find the limits based on content [mm]

    default: 1.5
    """

    threshold_block_size_limit: Union[int, float]
    """
    Threshold block size limit.

    The block size used in a box on threshold for content find the limits based on content [mm]

    default: 1.5
    """

    threshold_value_c_limit: Union[int, float]
    """
    Threshold value c limit.

    A variable used on threshold, should be low on low contrast image, used in a box on content find the limits based on content

    default: 70
    """

    colors: int
    """
    Colors.

    The number of colors in the png

    default: 0
    """

    run_optipng: bool
    """
    Run optipng.

    Run the optipng optimizer

    default: True
    """

    run_pngquant: bool
    """
    Run pngquant.

    Run the pngquant optimizer

    default: False
    """

    pngquant_options: List[str]
    """
    Pngquant options.

    The pngquant options

    default:
      - --force
      - --speed=1
      - --strip
      - --quality=0-32
    """

    run_exiftool: bool
    """
    Run exiftool.

    Run the exiftool optimizer

    default: False
    """

    run_ps2pdf: bool
    """
    Run ps2pdf.

    Run the ps2pdf optimizer (=> JPEG)

    default: False
    """

    no_auto_rotate: bool
    """
    No auto rotate.

    Run the auto rotate detected by Tesseract

    default: False
    """

    jpeg: bool
    """
    Jpeg.

    Convert images to JPEG

    default: False
    """

    jpeg_quality: int
    """
    Jpeg quality.

    The JPEG quality

    default: 90
    """

    background_color: List[int]
    """
    Background color.

    The background color

    default:
      - 255
      - 255
      - 255
    """

    auto_mask: "AutoMask"
    """
    WARNING: The required are not correctly taken in account,
    See: https://github.com/camptocamp/jsonschema-gentypes/issues/6
    """

    auto_cut: "AutoMask"
    """
    WARNING: The required are not correctly taken in account,
    See: https://github.com/camptocamp/jsonschema-gentypes/issues/6
    """

    deskew: "_ArgumentsDeskew"
    """
    WARNING: The required are not correctly taken in account,
    See: https://github.com/camptocamp/jsonschema-gentypes/issues/6
    """

    line_detection: "LineDetection"
    """
    WARNING: The required are not correctly taken in account,
    See: https://github.com/camptocamp/jsonschema-gentypes/issues/6
    """

    rule: "Rule"
    """
    WARNING: The required are not correctly taken in account,
    See: https://github.com/camptocamp/jsonschema-gentypes/issues/6
    """


class AssistedSplit(TypedDict, total=False):
    """
    Assisted split.

    Assisted split configuration
    """

    source: str
    """The source image name."""

    destinations: List[Union[int, str]]
    """The destination image positions."""

    image: str
    """The enhanced image name."""

    limits: List["Limit"]
    """The (proposed) limits to do the assisted split, You should keep only the right one"""


class AutoMask(TypedDict, total=False):
    """
    Auto mask.

    The auto mask configuration, the mask is used to mask the image on crop and skew calculation

    Editor note: The properties of this object should be modified in the config_schema.json file
    """

    lower_hsv_color: List[int]
    """
    Lower hsv color.

    The lower color in HSV representation

    default:
      - 0
      - 0
      - 250
    """

    upper_hsv_color: List[int]
    """
    Upper hsv color.

    The upper color in HSV representation

    default:
      - 255
      - 10
      - 255
    """

    de_noise_morphology: bool
    """
    De noise morphology.

    Apply a morphology operation to remove noise

    default: True
    """

    inverse_mask: bool
    """
    Inverse mask.

    Inverse the mask

    default: False
    """

    de_noise_size: int
    """
    De noise size.

    The size of the artifact that will be de noise

    default: 1000
    """

    de_noise_level: int
    """
    De noise level.

    The threshold level used in de noise on the blurry image

    default: 220
    """

    buffer_size: int
    """
    Buffer size.

    The size of the buffer add on the mask

    default: 20
    """

    buffer_level: int
    """
    Buffer level.

    The threshold level used in buffer on the blurry image

    default: 20
    """

    additional_filename: str
    """An image file used to add on the mask"""


BACKGROUND_COLOR_DEFAULT = [255, 255, 255]
"""Default value of the field path 'Arguments background_color'"""


BUFFER_LEVEL_DEFAULT = 20
"""Default value of the field path 'Auto mask buffer_level'"""


BUFFER_SIZE_DEFAULT = 20
"""Default value of the field path 'Auto mask buffer_size'"""


COLORS_DEFAULT = 0
"""Default value of the field path 'Arguments colors'"""


CONTOUR_KERNEL_SIZE_CROP_DEFAULT = 1.5
"""Default value of the field path 'Arguments contour_kernel_size_crop'"""


CONTOUR_KERNEL_SIZE_EMPTY_DEFAULT = 1.5
"""Default value of the field path 'Arguments contour_kernel_size_empty'"""


CONTOUR_KERNEL_SIZE_LIMIT_DEFAULT = 1.5
"""Default value of the field path 'Arguments contour_kernel_size_limit'"""


CUT_BLACK_DEFAULT = 0
"""Default value of the field path 'Arguments cut_black'"""


CUT_WHITE_DEFAULT = 255
"""Default value of the field path 'Arguments cut_white'"""


class Configuration(TypedDict, total=False):
    """Configuration."""

    images: List[str]
    """
    The images

    required
    """

    args: "Arguments"
    """
    WARNING: The required are not correctly taken in account,
    See: https://github.com/camptocamp/jsonschema-gentypes/issues/6

    required
    """

    progress: bool
    """
    Progress.

    Run in progress mode

    default: False
    """

    steps: List["Step"]
    """The carried out steps description"""

    assisted_split: List["AssistedSplit"]
    transformed_images: List[str]
    """The transformed image, if removed the jobs will rag again from start"""

    intermediate_error: List["IntermediateError"]
    """The ignored errors"""

    images_config: Dict[str, "_ConfigurationImagesConfigAdditionalproperties"]


DESKEW_ANGLE_DERIVATION_DEFAULT = 0.1
"""Default value of the field path 'Arguments deskew angle_derivation'"""


DESKEW_ANGLE_PM_90_DEFAULT = False
"""Default value of the field path 'Arguments deskew angle_pm_90'"""


DESKEW_MAX_ANGLE_DEFAULT = 45
"""Default value of the field path 'Arguments deskew max_angle'"""


DESKEW_MIN_ANGLE_DEFAULT = -45
"""Default value of the field path 'Arguments deskew min_angle'"""


DESKEW_NUM_PEAKS_DEFAULT = 20
"""Default value of the field path 'Arguments deskew num_peaks'"""


DESKEW_SIGMA_DEFAULT = 3.0
"""Default value of the field path 'Arguments deskew sigma'"""


DE_NOISE_LEVEL_DEFAULT = 220
"""Default value of the field path 'Auto mask de_noise_level'"""


DE_NOISE_MORPHOLOGY_DEFAULT = True
"""Default value of the field path 'Auto mask de_noise_morphology'"""


DE_NOISE_SIZE_DEFAULT = 1000
"""Default value of the field path 'Auto mask de_noise_size'"""


DITHER_DEFAULT = False
"""Default value of the field path 'Arguments dither'"""


DPI_DEFAULT = 300
"""Default value of the field path 'Arguments dpi'"""


INVERSE_MASK_DEFAULT = False
"""Default value of the field path 'Auto mask inverse_mask'"""


class IntermediateError(TypedDict, total=False):
    """Intermediate error."""

    error: str
    traceback: List[str]


JPEG_DEFAULT = False
"""Default value of the field path 'Arguments jpeg'"""


JPEG_QUALITY_DEFAULT = 90
"""Default value of the field path 'Arguments jpeg_quality'"""


LEVEL_DEFAULT = False
"""Default value of the field path 'Arguments level'"""


LINE_DETECTION_APERTURE_SIZE_DEFAULT = 3
"""Default value of the field path 'Line detection aperture_size'"""


LINE_DETECTION_HIGH_THRESHOLD_DEFAULT = 1000
"""Default value of the field path 'Line detection high_threshold'"""


LINE_DETECTION_LOW_THRESHOLD_DEFAULT = 0
"""Default value of the field path 'Line detection low_threshold'"""


LINE_DETECTION_MAX_LINE_GAP_DEFAULT = 100
"""Default value of the field path 'Line detection max_line_gap'"""


LINE_DETECTION_MIN_LINE_LENGTH_DEFAULT = 50
"""Default value of the field path 'Line detection min_line_length'"""


LINE_DETECTION_RHO_DEFAULT = 1
"""Default value of the field path 'Line detection rho'"""


LINE_DETECTION_THRESHOLD_DEFAULT = 100
"""Default value of the field path 'Line detection threshold'"""


LOWER_HSV_COLOR_DEFAULT = [0, 0, 250]
"""Default value of the field path 'Auto mask lower_hsv_color'"""


class Limit(TypedDict, total=False):
    """Limit."""

    name: str
    """The name visible on the generated image"""

    type: str
    """The kind of split"""

    value: int
    """The split position"""

    vertical: bool
    """Is vertical?"""

    margin: int
    """The margin around the split, can be used to remove a fold"""


class LineDetection(TypedDict, total=False):
    """
    Line detection.

    The line detection used in assisted split
    """

    low_threshold: int
    """
    Line detection low threshold.

    The low threshold used in the Canny edge detector

    default: 0
    """

    high_threshold: int
    """
    Line detection high threshold.

    The high threshold used in the Canny edge detector

    default: 1000
    """

    aperture_size: int
    """
    Line detection aperture size.

    The aperture size used in the Canny edge detector

    default: 3
    """

    rho: int
    """
    Line detection rho.

    The rho used in the Hough transform

    default: 1
    """

    threshold: int
    """
    Line detection threshold.

    The threshold used in the Hough transform

    default: 100
    """

    min_line_length: int
    """
    Line detection min line length.

    The minimum line length in percentage of the image size used in the Hough transform

    default: 50
    """

    max_line_gap: int
    """
    Line detection max line gap.

    The maximum line gap in percentage of the image size used in the Hough transform

    default: 100
    """


MARGIN_HORIZONTAL_DEFAULT = 9
"""Default value of the field path 'Arguments margin_horizontal'"""


MARGIN_VERTICAL_DEFAULT = 6
"""Default value of the field path 'Arguments margin_vertical'"""


MAX_LEVEL_DEFAULT = 100
"""Default value of the field path 'Arguments max_level'"""


MIN_BOX_BLACK_CROP_DEFAULT = 2
"""Default value of the field path 'Arguments min_box_black_crop'"""


MIN_BOX_BLACK_EMPTY_DEFAULT = 2
"""Default value of the field path 'Arguments min_box_black_empty'"""


MIN_BOX_BLACK_LIMIT_DEFAULT = 2
"""Default value of the field path 'Arguments min_box_black_limit'"""


MIN_BOX_SIZE_CROP_DEFAULT = 3
"""Default value of the field path 'Arguments min_box_size_crop'"""


MIN_BOX_SIZE_EMPTY_DEFAULT = 10
"""Default value of the field path 'Arguments min_box_size_empty'"""


MIN_BOX_SIZE_LIMIT_DEFAULT = 10
"""Default value of the field path 'Arguments min_box_size_limit'"""


MIN_LEVEL_DEFAULT = 0
"""Default value of the field path 'Arguments min_level'"""


NO_AUTO_ROTATE_DEFAULT = False
"""Default value of the field path 'Arguments no_auto_rotate'"""


NO_CROP_DEFAULT = False
"""Default value of the field path 'Arguments no_crop'"""


PNGQUANT_OPTIONS_DEFAULT = ["--force", "--speed=1", "--strip", "--quality=0-32"]
"""Default value of the field path 'Arguments pngquant_options'"""


PROGRESS_DEFAULT = False
"""Default value of the field path 'Configuration progress'"""


RULE_ENABLE_DEFAULT = True
"""Default value of the field path 'Rule enable'"""


RULE_GRADUATION_COLOR_DEFAULT = [0, 0, 0]
"""Default value of the field path 'Rule graduation_color'"""


RULE_GRADUATION_TEXT_FONT_COLOR_DEFAULT = [0, 0, 0]
"""Default value of the field path 'Rule graduation_text_font_color'"""


RULE_GRADUATION_TEXT_FONT_FILENAME_DEFAULT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
"""Default value of the field path 'Rule graduation_text_font_filename'"""


RULE_GRADUATION_TEXT_FONT_SIZE_DEFAULT = 17
"""Default value of the field path 'Rule graduation_text_font_size'"""


RULE_GRADUATION_TEXT_MARGIN_DEFAULT = 6
"""Default value of the field path 'Rule graduation_text_margin'"""


RULE_LINES_COLOR_DEFAULT = [0, 0, 0]
"""Default value of the field path 'Rule lines_color'"""


RULE_LINES_OPACITY_DEFAULT = 0.2
"""Default value of the field path 'Rule lines_opacity'"""


RULE_LINES_SPACE_DEFAULT = 100
"""Default value of the field path 'Rule lines_space'"""


RULE_MAJOR_GRADUATION_SIZE_DEFAULT = 30
"""Default value of the field path 'Rule major_graduation_size'"""


RULE_MAJOR_GRADUATION_SPACE_DEFAULT = 100
"""Default value of the field path 'Rule major_graduation_space'"""


RULE_MINOR_GRADUATION_SIZE_DEFAULT = 10
"""Default value of the field path 'Rule minor_graduation_size'"""


RULE_MINOR_GRADUATION_SPACE_DEFAULT = 10
"""Default value of the field path 'Rule minor_graduation_space'"""


RUN_EXIFTOOL_DEFAULT = False
"""Default value of the field path 'Arguments run_exiftool'"""


RUN_OPTIPNG_DEFAULT = True
"""Default value of the field path 'Arguments run_optipng'"""


RUN_PNGQUANT_DEFAULT = False
"""Default value of the field path 'Arguments run_pngquant'"""


RUN_PS2PDF_DEFAULT = False
"""Default value of the field path 'Arguments run_ps2pdf'"""


class Rule(TypedDict, total=False):
    """
    Rule.

    Configuration of rule displayed in assisted split images
    """

    enable: bool
    """
    Rule enable.

    default: True
    """

    minor_graduation_space: int
    """
    Rule minor graduation space.

    default: 10
    """

    major_graduation_space: int
    """
    Rule major graduation space.

    default: 100
    """

    lines_space: int
    """
    Rule lines space.

    default: 100
    """

    minor_graduation_size: int
    """
    Rule minor graduation size.

    default: 10
    """

    major_graduation_size: int
    """
    Rule major graduation size.

    default: 30
    """

    graduation_color: List[int]
    """
    Rule graduation color.

    default:
      - 0
      - 0
      - 0
    """

    lines_color: List[int]
    """
    Rule lines color.

    default:
      - 0
      - 0
      - 0
    """

    lines_opacity: Union[int, float]
    """
    Rule lines opacity.

    default: 0.2
    """

    graduation_text_font_filename: str
    """
    Rule graduation text font filename.

    default: /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
    """

    graduation_text_font_size: Union[int, float]
    """
    Rule graduation text font size.

    default: 17
    """

    graduation_text_font_color: List[int]
    """
    Rule graduation text font color.

    default:
      - 0
      - 0
      - 0
    """

    graduation_text_margin: int
    """
    Rule graduation text margin.

    default: 6
    """


SHARPEN_DEFAULT = False
"""Default value of the field path 'Arguments sharpen'"""


class Step(TypedDict, total=False):
    """Step."""

    name: str
    """The step name"""

    sources: List[str]
    """The images obtain after the current step"""

    process_count: int
    """The step number"""


TESSERACT_DEFAULT = True
"""Default value of the field path 'Arguments tesseract'"""


TESSERACT_LANG_DEFAULT = "fra+eng"
"""Default value of the field path 'Arguments tesseract_lang'"""


THRESHOLD_BLOCK_SIZE_CROP_DEFAULT = 1.5
"""Default value of the field path 'Arguments threshold_block_size_crop'"""


THRESHOLD_BLOCK_SIZE_EMPTY_DEFAULT = 1.5
"""Default value of the field path 'Arguments threshold_block_size_empty'"""


THRESHOLD_BLOCK_SIZE_LIMIT_DEFAULT = 1.5
"""Default value of the field path 'Arguments threshold_block_size_limit'"""


THRESHOLD_VALUE_C_CROP_DEFAULT = 70
"""Default value of the field path 'Arguments threshold_value_c_crop'"""


THRESHOLD_VALUE_C_EMPTY_DEFAULT = 70
"""Default value of the field path 'Arguments threshold_value_c_empty'"""


THRESHOLD_VALUE_C_LIMIT_DEFAULT = 70
"""Default value of the field path 'Arguments threshold_value_c_limit'"""


UPPER_HSV_COLOR_DEFAULT = [255, 10, 255]
"""Default value of the field path 'Auto mask upper_hsv_color'"""


class _ArgumentsDeskew(TypedDict, total=False):
    """The deskew configuration"""

    min_angle: Union[int, float]
    """
    Deskew min angle.

    The minimum angle to detect the image skew [degree]

    default: -45
    """

    max_angle: Union[int, float]
    """
    Deskew max angle.

    The maximum angle to detect the image skew [degree]

    default: 45
    """

    angle_derivation: Union[int, float]
    """
    Deskew angle derivation.

    The step of angle to detect the image skew [degree]

    default: 0.1
    """

    sigma: Union[int, float]
    """
    Deskew sigma.

    Used in the `canny` function

    default: 3.0
    """

    num_peaks: int
    """
    Deskew num peaks.

    number of peaks we ask for

    default: 20
    """

    angle_pm_90: bool
    """
    Deskew angle pm 90.

    Detect an angle of +/- 90 degree, also +/- 45 degree

    default: False
    """


class _ConfigurationImagesConfigAdditionalproperties(TypedDict, total=False):
    angle: Union[Union[int, float], None]
    """The used angle to deskew, can be change, restart by deleting one of the generated images"""

    status: "_ConfigurationImagesConfigAdditionalpropertiesStatus"
    """
    WARNING: The required are not correctly taken in account,
    See: https://github.com/camptocamp/jsonschema-gentypes/issues/6
    """


class _ConfigurationImagesConfigAdditionalpropertiesStatus(TypedDict, total=False):
    angle: Union[int, float]
    """The measured deskew angle"""

    size: List[Union[int, float]]
    """The image dimensions"""
