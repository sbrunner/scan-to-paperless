# Automatically generated file from a JSON schema
# Used to correctly format the generated file


from typing import Dict, List, TypedDict, Union

from typing_extensions import Required

APPEND_CREDIT_CARD_DEFAULT = False
""" Default value of the field path 'Arguments append_credit_card' """


ASSISTED_SPLIT_DEFAULT = False
""" Default value of the field path 'Arguments assisted_split' """


AUTO_BASH_DEFAULT = False
""" Default value of the field path 'Mode auto_bash' """


AUTO_CUT_ENABLED_DEFAULT = True
""" Default value of the field path 'Cut operation enabled' """


AUTO_DETECTION_ENABLED_DEFAULT = True
""" Default value of the field path 'Auto mask enabled' """


AUTO_LEVEL_DEFAULT = False
""" Default value of the field path 'Level auto' """


AUTO_ROTATE_ENABLED_DEFAULT = True
""" Default value of the field path 'Auto rotate enabled' """


class Arguments(TypedDict, total=False):
    """Arguments."""

    level: "Level"
    """
    Level.

    The level configuration
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

    crop: "Crop"
    """
    Crop.

    The crop configuration
    """

    dpi: Union[int, float]
    """
    Dpi.

    The DPI used to convert the mm to pixel

    default: 300
    """

    sharpen: "Sharpen"
    """
    Sharpen.

    Sharpen configuration

    default:
      enabled: false
    """

    dither: "Dither"
    """
    Dither.

    The dither configuration

    default:
      enabled: false
    """

    tesseract: "Tesseract"
    """
    Tesseract.

    The Tesseract configuration
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

    empty: "Empty"
    """
    Empty.

    The empty page detection configuration
    """

    limit_detection: "LimitDetection"
    """
    Limit detection.

    The limit page detection configuration
    """

    colors: int
    """
    Colors.

    The number of colors in the png

    default: 0
    """

    optipng: "Optipng"
    """
    Optipng.

    The optipng optimization tool configuration
    """

    pngquant: "Pngquant"
    """
    Pngquant.

    The pngquant optimization tool configuration

    default:
      enabled: false
    """

    exiftool: "Exiftool"
    """
    Exiftool.

    The exiftool optimization tool configuration

    default:
      enabled: false
    """

    ps2pdf: "Ps2Pdf"
    """
    Ps2pdf.

    The ps2pdf optimization tool configuration

    default:
      enabled: false
    """

    auto_rotate: "AutoRotate"
    """
    Auto rotate.

    The auto rotate configuration
    """

    jpeg: "Jpeg"
    """
    Jpeg.

    Convert images to JPEG configuration

    default:
      enabled: false
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

    mask: "MaskOperation"
    """
    Mask operation.

    The mask configuration, the a is used to mask the image on crop and skew calculation

    default:
      enabled: false
    """

    cut: "CutOperation"
    """
    Cut operation.

    The cut configuration, a mask is used to definitively mask the source image

    default:
      enabled: false
    """

    no_remove_to_continue: bool
    """
    No REMOVE_TO_CONTINUE.

    Don't wait for the deletion of the REMOVE_TO_CONTINUE file before exporting the PDF.

    default: False
    """

    deskew: "_ArgumentsDeskew"
    """ The deskew configuration """

    rule: "Rule"
    """
    Rule.

    Configuration of rule displayed in assisted split images
    """

    rest_upload: "RestUpload"
    """
    REST upload.

    Upload the final PDF via Paperless REST API
    """

    consume_folder: "ConsumeFolder"
    """
    Consume folder.

    Send the final PDF to Paperless using the consume folder
    """


class AutoMask(TypedDict, total=False):
    """Auto mask."""

    enabled: bool
    """
    Auto detection enabled.

    Enable the auto detection of the mask

    default: True
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


class AutoRotate(TypedDict, total=False):
    """
    Auto rotate.

    The auto rotate configuration
    """

    enabled: bool
    """
    Auto rotate enabled.

    Enable the auto rotate detected by Tesseract

    default: True
    """


BACKGROUND_COLOR_DEFAULT = [255, 255, 255]
""" Default value of the field path 'Arguments background_color' """


BUFFER_LEVEL_DEFAULT = 20
""" Default value of the field path 'Auto mask buffer_level' """


BUFFER_SIZE_DEFAULT = 20
""" Default value of the field path 'Auto mask buffer_size' """


COLORS_DEFAULT = 0
""" Default value of the field path 'Arguments colors' """


CONSUME_FOLDER_ENABLED_DEFAULT = True
""" Default value of the field path 'Consume folder enabled' """


CONTOUR_KERNEL_SIZE_DEFAULT = 1.5
""" Default value of the field path 'Contour contour_kernel_size' """


CROP_ENABLED_DEFAULT = True
""" Default value of the field path 'Crop enabled' """


CUT_BLACK_DEFAULT = 0
""" Default value of the field path 'Arguments cut_black' """


CUT_OPERATION_DEFAULT = {"enabled": False}
""" Default value of the field path 'Arguments cut' """


CUT_WHITE_DEFAULT = 255
""" Default value of the field path 'Arguments cut_white' """


class Configuration(TypedDict, total=False):
    """Configuration."""

    extends: str
    """ The configuration to extends """

    merge_strategies: "MergeStrategies"
    """
    Merge strategies.

    The merge strategy to use, see https://deepmerge.readthedocs.io/en/latest/strategies.html#builtin-strategies
    """

    scan_folder: str
    """ This should be shared with the process container in 'source'. """

    scanimage: str
    """
    Scanimage.

    The scanimage command

    default: scanimage
    """

    scanimage_arguments: List[str]
    """
    Scanimage arguments.

    The scanimage arguments

    default:
      - --format=png
      - --mode=color
      - --resolution=300
    """

    extension: str
    """
    Extension.

    The extension of generate image (png or tiff)

    default: png
    """

    default_args: "Arguments"
    """ Arguments. """

    viewer: str
    """
    Viewer.

    The command used to start the viewer

    default: eog
    """

    modes: Dict[str, "Mode"]
    """
    Modes.

    Customize the modes

    default:
      adf:
        scanimage_arguments:
        - --source=ADF
      double:
        auto_bash: true
        rotate_even: true
        scanimage_arguments:
        - --source=ADF
      multi:
        scanimage_arguments:
        - --batch-prompt
      one:
        scanimage_arguments:
        - --batch-count=1
    """


class ConsumeFolder(TypedDict, total=False):
    """
    Consume folder.

    Send the final PDF to Paperless using the consume folder
    """

    enabled: bool
    """
    Consume folder enabled.

    Enable using the consume folder

    default: True
    """


class Contour(TypedDict, total=False):
    """
    Contour.

    The configuration used to find the contour
    """

    min_box_size: Union[int, float]
    """
    Min box size.

    The minimum box size to find the content [mm]

    default:
      crop: 3
      empty: 10
      limit: 10
    """

    min_box_black: Union[int, float]
    """
    Min box black.

    The minimum black in a box on content find [%]

    default: 2
    """

    contour_kernel_size: Union[int, float]
    """
    Contour kernel size.

    The block size used in a box on content find [mm]

    default: 1.5
    """

    threshold_block_size: Union[int, float]
    """
    Threshold block size.

    The block size used in a box on threshold for content find [mm]

    default: 1.5
    """

    threshold_value_c: Union[int, float]
    """
    Threshold value c.

    A variable used on threshold, should be low on low contrast image, used in a box on content find on witch one we will crop

    default: 70
    """


class Crop(TypedDict, total=False):
    """
    Crop.

    The crop configuration
    """

    enabled: bool
    """
    Crop enabled.

    Enable the crop

    default: True
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

    contour: "Contour"
    """
    Contour.

    The configuration used to find the contour
    """


class CutOperation(TypedDict, total=False):
    """
    Cut operation.

    The cut configuration, a mask is used to definitively mask the source image

    default:
      enabled: false
    """

    enabled: bool
    """
    Auto cut enabled.

    Enable the cut

    default: True
    """

    auto_mask: "AutoMask"
    """ Auto mask. """

    additional_filename: str
    """ An image file used to add on the mask """


DESKEW_ANGLE_DERIVATION_DEFAULT = 0.1
""" Default value of the field path 'Arguments deskew angle_derivation' """


DESKEW_ANGLE_PM_90_DEFAULT = False
""" Default value of the field path 'Arguments deskew angle_pm_90' """


DESKEW_MAX_ANGLE_DEFAULT = 45
""" Default value of the field path 'Arguments deskew max_angle' """


DESKEW_MIN_ANGLE_DEFAULT = -45
""" Default value of the field path 'Arguments deskew min_angle' """


DESKEW_NUM_PEAKS_DEFAULT = 20
""" Default value of the field path 'Arguments deskew num_peaks' """


DESKEW_SIGMA_DEFAULT = 3.0
""" Default value of the field path 'Arguments deskew sigma' """


DE_NOISE_LEVEL_DEFAULT = 220
""" Default value of the field path 'Auto mask de_noise_level' """


DE_NOISE_MORPHOLOGY_DEFAULT = True
""" Default value of the field path 'Auto mask de_noise_morphology' """


DE_NOISE_SIZE_DEFAULT = 1000
""" Default value of the field path 'Auto mask de_noise_size' """


DICT_DEFAULT = ["merge"]
""" Default value of the field path 'Merge strategies dict' """


DITHER_DEFAULT = {"enabled": False}
""" Default value of the field path 'Arguments dither' """


DITHER_ENABLED_DEFAULT = True
""" Default value of the field path 'Dither enabled' """


DPI_DEFAULT = 300
""" Default value of the field path 'Arguments dpi' """


class Dither(TypedDict, total=False):
    """
    Dither.

    The dither configuration

    default:
      enabled: false
    """

    enabled: bool
    """
    Dither enabled.

    Enable the dither

    default: True
    """


EMPTY_ENABLED_DEFAULT = True
""" Default value of the field path 'Empty enabled' """


EXIFTOOL_DEFAULT = {"enabled": False}
""" Default value of the field path 'Arguments exiftool' """


EXIFTOOL_ENABLED_DEFAULT = True
""" Default value of the field path 'Exiftool enabled' """


EXTENSION_DEFAULT = "png"
""" Default value of the field path 'Configuration extension' """


class Empty(TypedDict, total=False):
    """
    Empty.

    The empty page detection configuration
    """

    enabled: bool
    """
    Empty enabled.

    Enable the empty page detection

    default: True
    """

    contour: "Contour"
    """
    Contour.

    The configuration used to find the contour
    """


class Exiftool(TypedDict, total=False):
    """
    Exiftool.

    The exiftool optimization tool configuration

    default:
      enabled: false
    """

    enabled: bool
    """
    Exiftool enabled.

    Use the exiftool optimizer

    default: True
    """


FALLBACK_DEFAULT = ["override"]
""" Default value of the field path 'Merge strategies fallback' """


INVERSE_MASK_DEFAULT = False
""" Default value of the field path 'Auto mask inverse_mask' """


JPEG_DEFAULT = {"enabled": False}
""" Default value of the field path 'Arguments jpeg' """


JPEG_ENABLED_DEFAULT = True
""" Default value of the field path 'Jpeg enabled' """


JPEG_QUALITY_DEFAULT = 90
""" Default value of the field path 'Jpeg quality' """


class Jpeg(TypedDict, total=False):
    """
    Jpeg.

    Convert images to JPEG configuration

    default:
      enabled: false
    """

    enabled: bool
    """
    Jpeg enabled.

    Convert images to JPEG

    default: True
    """

    quality: int
    """
    Jpeg quality.

    The JPEG quality

    default: 90
    """


LEVEL_VALUE_DEFAULT = False
""" Default value of the field path 'Level value' """


LINE_DETECTION_APERTURE_SIZE_DEFAULT = 3
""" Default value of the field path 'Line detection aperture_size' """


LINE_DETECTION_HIGH_THRESHOLD_DEFAULT = 1000
""" Default value of the field path 'Line detection high_threshold' """


LINE_DETECTION_LOW_THRESHOLD_DEFAULT = 0
""" Default value of the field path 'Line detection low_threshold' """


LINE_DETECTION_MAX_LINE_GAP_DEFAULT = 100
""" Default value of the field path 'Line detection max_line_gap' """


LINE_DETECTION_MIN_LINE_LENGTH_DEFAULT = 50
""" Default value of the field path 'Line detection min_line_length' """


LINE_DETECTION_RHO_DEFAULT = 1
""" Default value of the field path 'Line detection rho' """


LINE_DETECTION_THRESHOLD_DEFAULT = 100
""" Default value of the field path 'Line detection threshold' """


LIST_DEFAULT = ["override"]
""" Default value of the field path 'Merge strategies list' """


LOWER_HSV_COLOR_DEFAULT = [0, 0, 250]
""" Default value of the field path 'Auto mask lower_hsv_color' """


class Level(TypedDict, total=False):
    """
    Level.

    The level configuration
    """

    value: "LevelValue"
    """
    Level value.

    true: => do level on 15% - 85% (under 15 % will be black above 85% will be white), false: => 0% - 100%, <number>: => (0 + <number>)% - (100 - number)%

    default: False
    """

    auto: bool
    """
    Auto level.

    If no level specified, do auto level

    default: False
    """

    min: Union[int, float]
    """
    Min level.

    Min level if no level end no auto-level

    default: 0
    """

    max: Union[int, float]
    """
    Max level.

    Max level if no level end no auto-level

    default: 100
    """


LevelValue = Union[bool, int]
"""
Level value.

true: => do level on 15% - 85% (under 15 % will be black above 85% will be white), false: => 0% - 100%, <number>: => (0 + <number>)% - (100 - number)%

default: False
"""


class LimitDetection(TypedDict, total=False):
    """
    Limit detection.

    The limit page detection configuration
    """

    contour: "Contour"
    """
    Contour.

    The configuration used to find the contour
    """

    line: "LineDetection"
    """
    Line detection.

    The line detection used in assisted split
    """


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
""" Default value of the field path 'Crop margin_horizontal' """


MARGIN_VERTICAL_DEFAULT = 6
""" Default value of the field path 'Crop margin_vertical' """


MASK_ENABLED_DEFAULT = True
""" Default value of the field path 'Mask operation enabled' """


MASK_OPERATION_DEFAULT = {"enabled": False}
""" Default value of the field path 'Arguments mask' """


MAX_LEVEL_DEFAULT = 100
""" Default value of the field path 'Level max' """


MIN_BOX_BLACK_DEFAULT = 2
""" Default value of the field path 'Contour min_box_black' """


MIN_BOX_SIZE_DEFAULT = {"crop": 3, "empty": 10, "limit": 10}
""" Default value of the field path 'Contour min_box_size' """


MIN_LEVEL_DEFAULT = 0
""" Default value of the field path 'Level min' """


MODES_DEFAULT = {
    "adf": {"scanimage_arguments": ["--source=ADF"]},
    "double": {"scanimage_arguments": ["--source=ADF"], "auto_bash": True, "rotate_even": True},
    "multi": {"scanimage_arguments": ["--batch-prompt"]},
    "one": {"scanimage_arguments": ["--batch-count=1"]},
}
""" Default value of the field path 'Configuration modes' """


class MaskOperation(TypedDict, total=False):
    """
    Mask operation.

    The mask configuration, the a is used to mask the image on crop and skew calculation

    default:
      enabled: false
    """

    enabled: bool
    """
    mask enabled.

    Enable the mask

    default: True
    """

    auto_mask: "AutoMask"
    """ Auto mask. """

    additional_filename: str
    """ An image file used to add on the mask """


class MergeStrategies(TypedDict, total=False):
    """
    Merge strategies.

    The merge strategy to use, see https://deepmerge.readthedocs.io/en/latest/strategies.html#builtin-strategies
    """

    list: List[str]
    """
    List.

    The merge strategy to use on list

    default:
      - override
    """

    dict: List[str]
    """
    Dict.

    The merge strategy to use on dict

    default:
      - merge
    """

    fallback: List[str]
    """
    Fallback.

    The fallback merge strategy

    default:
      - override
    """

    type_conflict: List[str]
    """
    Type conflict.

    The type_conflict merge strategy

    default:
      - override
    """


class Mode(TypedDict, total=False):
    """Mode."""

    scanimage_arguments: List[str]
    """ Additional scanimage arguments """

    auto_bash: bool
    """
    Auto bash.

    Run the ADF in tow step odd and even, needed for scanner that don't support double face

    default: False
    """

    rotate_even: bool
    """
    Rotate even.

    Rotate the even pages, to use in conjunction with auto_bash

    default: False
    """


NO_REMOVE_TO_CONTINUE_DEFAULT = False
""" Default value of the field path 'Arguments no_remove_to_continue' """


OPTIPNG_ENABLED_DEFAULT = True
""" Default value of the field path 'Optipng enabled' """


class Optipng(TypedDict, total=False):
    """
    Optipng.

    The optipng optimization tool configuration
    """

    enabled: bool
    """
    Optipng enabled.

    Use the optipng optimizer

    default: True
    """


PNGQUANT_DEFAULT = {"enabled": False}
""" Default value of the field path 'Arguments pngquant' """


PNGQUANT_ENABLED_DEFAULT = True
""" Default value of the field path 'Pngquant enabled' """


PNGQUANT_OPTIONS_DEFAULT = ["--force", "--speed=1", "--strip", "--quality=0-32"]
""" Default value of the field path 'Pngquant options' """


PS2PDF_DEFAULT = {"enabled": False}
""" Default value of the field path 'Arguments ps2pdf' """


PS2PDF_ENABLED_DEFAULT = True
""" Default value of the field path 'Ps2pdf enabled' """


class Pngquant(TypedDict, total=False):
    """
    Pngquant.

    The pngquant optimization tool configuration

    default:
      enabled: false
    """

    enabled: bool
    """
    Pngquant enabled.

    Use the pngquant optimizer

    default: True
    """

    options: List[str]
    """
    Pngquant options.

    The pngquant options

    default:
      - --force
      - --speed=1
      - --strip
      - --quality=0-32
    """


class Ps2Pdf(TypedDict, total=False):
    """
    Ps2pdf.

    The ps2pdf optimization tool configuration

    default:
      enabled: false
    """

    enabled: bool
    """
    Ps2pdf enabled.

    Use the ps2pdf optimizer (=> JPEG)

    default: True
    """


REST_UPLOAD_ENABLED_DEFAULT = False
""" Default value of the field path 'REST upload enabled' """


ROTATE_EVEN_DEFAULT = False
""" Default value of the field path 'Mode rotate_even' """


RULE_ENABLE_DEFAULT = True
""" Default value of the field path 'Rule enabled' """


RULE_GRADUATION_COLOR_DEFAULT = [0, 0, 0]
""" Default value of the field path 'Rule graduation_color' """


RULE_GRADUATION_TEXT_FONT_COLOR_DEFAULT = [0, 0, 0]
""" Default value of the field path 'Rule graduation_text_font_color' """


RULE_GRADUATION_TEXT_FONT_FILENAME_DEFAULT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
""" Default value of the field path 'Rule graduation_text_font_filename' """


RULE_GRADUATION_TEXT_FONT_SIZE_DEFAULT = 17
""" Default value of the field path 'Rule graduation_text_font_size' """


RULE_GRADUATION_TEXT_MARGIN_DEFAULT = 6
""" Default value of the field path 'Rule graduation_text_margin' """


RULE_LINES_COLOR_DEFAULT = [0, 0, 0]
""" Default value of the field path 'Rule lines_color' """


RULE_LINES_OPACITY_DEFAULT = 0.2
""" Default value of the field path 'Rule lines_opacity' """


RULE_LINES_SPACE_DEFAULT = 100
""" Default value of the field path 'Rule lines_space' """


RULE_MAJOR_GRADUATION_SIZE_DEFAULT = 30
""" Default value of the field path 'Rule major_graduation_size' """


RULE_MAJOR_GRADUATION_SPACE_DEFAULT = 100
""" Default value of the field path 'Rule major_graduation_space' """


RULE_MINOR_GRADUATION_SIZE_DEFAULT = 10
""" Default value of the field path 'Rule minor_graduation_size' """


RULE_MINOR_GRADUATION_SPACE_DEFAULT = 10
""" Default value of the field path 'Rule minor_graduation_space' """


class RestUpload(TypedDict, total=False):
    """
    REST upload.

    Upload the final PDF via Paperless REST API
    """

    enabled: bool
    """
    REST upload enabled.

    Enable the upload of the PDF via REST API

    default: False
    """

    api_url: Required[str]
    """
    REST upload API url.

    The URL address of the REST API, usually http://server.name/api

    Required property
    """

    api_token: Required[str]
    """
    REST upload API token.

    The API token

    Required property
    """


class Rule(TypedDict, total=False):
    """
    Rule.

    Configuration of rule displayed in assisted split images
    """

    enabled: bool
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


SCANIMAGE_ARGUMENTS_DEFAULT = ["--format=png", "--mode=color", "--resolution=300"]
""" Default value of the field path 'Configuration scanimage_arguments' """


SCANIMAGE_DEFAULT = "scanimage"
""" Default value of the field path 'Configuration scanimage' """


SHARPEN_DEFAULT = {"enabled": False}
""" Default value of the field path 'Arguments sharpen' """


SHARPEN_ENABLED_DEFAULT = True
""" Default value of the field path 'Sharpen enabled' """


class Sharpen(TypedDict, total=False):
    """
    Sharpen.

    Sharpen configuration

    default:
      enabled: false
    """

    enabled: bool
    """
    Sharpen enabled.

    Enable the sharpen

    default: True
    """


TESSERACT_ENABLED_DEFAULT = True
""" Default value of the field path 'Tesseract enabled' """


TESSERACT_LANG_DEFAULT = "fra+eng"
""" Default value of the field path 'Tesseract lang' """


THRESHOLD_BLOCK_SIZE_DEFAULT = 1.5
""" Default value of the field path 'Contour threshold_block_size' """


THRESHOLD_VALUE_C_DEFAULT = 70
""" Default value of the field path 'Contour threshold_value_c' """


TYPE_CONFLICT_DEFAULT = ["override"]
""" Default value of the field path 'Merge strategies type_conflict' """


class Tesseract(TypedDict, total=False):
    """
    Tesseract.

    The Tesseract configuration
    """

    enabled: bool
    """
    Tesseract enabled.

    Use Tesseract to to an OCR on the document

    default: True
    """

    lang: str
    """
    Tesseract lang.

    The used language for tesseract

    default: fra+eng
    """


UPPER_HSV_COLOR_DEFAULT = [255, 10, 255]
""" Default value of the field path 'Auto mask upper_hsv_color' """


VIEWER_DEFAULT = "eog"
""" Default value of the field path 'Configuration viewer' """


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
