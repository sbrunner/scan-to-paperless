# Automatically generated file from a JSON schema
# Used to correctly format the generated file


from typing import Required, TypedDict


APPEND_CREDIT_CARD_DEFAULT = False
r""" Default value of the field path 'Arguments append_credit_card' """



ASSISTED_SPLIT_DEFAULT = False
r""" Default value of the field path 'Arguments assisted_split' """



AUTO_CUT_ENABLED_DEFAULT = True
r""" Default value of the field path 'Cut operation enabled' """



AUTO_DETECTION_ENABLED_DEFAULT = True
r""" Default value of the field path 'Auto mask enabled' """



AUTO_LEVEL_DEFAULT = False
r""" Default value of the field path 'Level auto' """



AUTO_ROTATE_ENABLED_DEFAULT = True
r""" Default value of the field path 'Auto rotate enabled' """



class Arguments(TypedDict, total=False):
    r"""
    Arguments.

    Editor note: The properties of this object should be modified in the config_schema.json file
    """

    level: "Level"
    r"""
    Level.

    The level configuration
    """

    cut_white: int | float
    r"""
    Cut white.

    Set the near white pixels on the image to white

    default: 255
    """

    cut_black: int | float
    r"""
    Cut black.

    Set the near black pixels on the image to black

    default: 0
    """

    crop: "Crop"
    r"""
    Crop.

    The crop configuration
    """

    dpi: int | float
    r"""
    Dpi.

    The DPI used to convert the mm to pixel

    default: 300
    """

    sharpen: "Sharpen"
    r"""
    Sharpen.

    Sharpen configuration

    default:
      enabled: false
    """

    dither: "Dither"
    r"""
    Dither.

    The dither configuration

    default:
      enabled: false
    """

    tesseract: "Tesseract"
    r"""
    Tesseract.

    The Tesseract configuration
    """

    append_credit_card: bool
    r"""
    Append credit card.

    Do an assisted split

    default: False
    """

    assisted_split: bool
    r"""
    Assisted split.

    Do an assisted split

    default: False
    """

    empty: "Empty"
    r"""
    Empty.

    The empty page detection configuration
    """

    limit_detection: "LimitDetection"
    r"""
    Limit detection.

    The limit page detection configuration
    """

    colors: int
    r"""
    Colors.

    The number of colors in the png

    default: 0
    """

    optipng: "Optipng"
    r"""
    Optipng.

    The optipng optimization tool configuration
    """

    pngquant: "Pngquant"
    r"""
    Pngquant.

    The pngquant optimization tool configuration

    default:
      enabled: false
    """

    exiftool: "Exiftool"
    r"""
    Exiftool.

    The exiftool optimization tool configuration

    default:
      enabled: false
    """

    ps2pdf: "Ps2Pdf"
    r"""
    Ps2pdf.

    The ps2pdf optimization tool configuration

    default:
      enabled: false
    """

    auto_rotate: "AutoRotate"
    r"""
    Auto rotate.

    The auto rotate configuration
    """

    jpeg: "Jpeg"
    r"""
    Jpeg.

    Convert images to JPEG configuration

    default:
      enabled: false
    """

    background_color: list[int]
    r"""
    Background color.

    The background color

    default:
      - 255
      - 255
      - 255
    """

    mask: "MaskOperation"
    r"""
    Mask operation.

    The mask configuration, the a is used to mask the image on crop and skew calculation

    default:
      enabled: false
    """

    cut: "CutOperation"
    r"""
    Cut operation.

    The cut configuration, a mask is used to definitively mask the source image

    default:
      enabled: false
    """

    no_remove_to_continue: bool
    r"""
    No REMOVE_TO_CONTINUE.

    Don't wait for the deletion of the REMOVE_TO_CONTINUE file before exporting the PDF.

    default: False
    """

    deskew: "_ArgumentsDeskew"
    r""" The deskew configuration """

    rule: "Rule"
    r"""
    Rule.

    Configuration of rule displayed in assisted split images
    """

    rest_upload: "RestUpload"
    r"""
    REST upload.

    Upload the final PDF via Paperless REST API
    """

    consume_folder: "ConsumeFolder"
    r"""
    Consume folder.

    Send the final PDF to Paperless using the consume folder
    """



class AssistedSplit(TypedDict, total=False):
    r"""
    Assisted split.

    Assisted split configuration
    """

    source: str
    r""" The source image name. """

    destinations: list[int | str]
    r""" The destination image positions. """

    image: str
    r""" The enhanced image name. """

    limits: list["Limit"]
    r"""
    Limits.

    The (proposed) limits to do the assisted split, You should keep only the right one
    """



class AutoMask(TypedDict, total=False):
    r"""
    Auto mask.

    Editor note: The properties of this object should be modified in the config_schema.json file
    """

    enabled: bool
    r"""
    Auto detection enabled.

    Enable the auto detection of the mask

    default: True
    """

    lower_hsv_color: list[int]
    r"""
    Lower hsv color.

    The lower color in HSV representation

    default:
      - 0
      - 0
      - 250
    """

    upper_hsv_color: list[int]
    r"""
    Upper hsv color.

    The upper color in HSV representation

    default:
      - 255
      - 10
      - 255
    """

    de_noise_morphology: bool
    r"""
    De noise morphology.

    Apply a morphology operation to remove noise

    default: True
    """

    inverse_mask: bool
    r"""
    Inverse mask.

    Inverse the mask

    default: False
    """

    de_noise_size: int
    r"""
    De noise size.

    The size of the artifact that will be de noise

    default: 1000
    """

    de_noise_level: int
    r"""
    De noise level.

    The threshold level used in de noise on the blurry image

    default: 220
    """

    buffer_size: int
    r"""
    Buffer size.

    The size of the buffer add on the mask

    default: 20
    """

    buffer_level: int
    r"""
    Buffer level.

    The threshold level used in buffer on the blurry image

    default: 20
    """



class AutoRotate(TypedDict, total=False):
    r"""
    Auto rotate.

    The auto rotate configuration
    """

    enabled: bool
    r"""
    Auto rotate enabled.

    Enable the auto rotate detected by Tesseract

    default: True
    """



BACKGROUND_COLOR_DEFAULT = [255, 255, 255]
r""" Default value of the field path 'Arguments background_color' """



BUFFER_LEVEL_DEFAULT = 20
r""" Default value of the field path 'Auto mask buffer_level' """



BUFFER_SIZE_DEFAULT = 20
r""" Default value of the field path 'Auto mask buffer_size' """



COLORS_DEFAULT = 0
r""" Default value of the field path 'Arguments colors' """



CONSUME_FOLDER_ENABLED_DEFAULT = True
r""" Default value of the field path 'Consume folder enabled' """



CONTOUR_KERNEL_SIZE_DEFAULT = 1.5
r""" Default value of the field path 'Contour contour_kernel_size' """



CROP_ENABLED_DEFAULT = True
r""" Default value of the field path 'Crop enabled' """



CUT_BLACK_DEFAULT = 0
r""" Default value of the field path 'Arguments cut_black' """



CUT_OPERATION_DEFAULT = {'enabled': False}
r""" Default value of the field path 'Arguments cut' """



CUT_WHITE_DEFAULT = 255
r""" Default value of the field path 'Arguments cut_white' """



class Configuration(TypedDict, total=False):
    r""" Configuration. """

    images: Required[list[str]]
    r"""
    The images

    Required property
    """

    args: Required["Arguments"]
    r"""
    Arguments.

    Editor note: The properties of this object should be modified in the config_schema.json file

    Required property
    """

    progress: bool
    r"""
    Progress.

    Run in progress mode

    default: False
    """

    steps: list["Step"]
    r""" The carried out steps description """

    assisted_split: list["AssistedSplit"]
    transformed_images: list[str]
    r""" The transformed image, if removed the jobs will rag again from start """

    intermediate_error: list["IntermediateError"]
    r""" The ignored errors """

    images_config: dict[str, "_ConfigurationImagesConfigAdditionalproperties"]


class ConsumeFolder(TypedDict, total=False):
    r"""
    Consume folder.

    Send the final PDF to Paperless using the consume folder
    """

    enabled: bool
    r"""
    Consume folder enabled.

    Enable using the consume folder

    default: True
    """



class Contour(TypedDict, total=False):
    r"""
    Contour.

    Editor note: The properties of this object should be modified in the config_schema.json file
    """

    min_box_size: int | float
    r"""
    Min box size.

    The minimum box size to find the content [mm]

    default:
      crop: 3
      empty: 10
      limit: 10
    """

    min_box_black: int | float
    r"""
    Min box black.

    The minimum black in a box on content find [%]

    default: 2
    """

    contour_kernel_size: int | float
    r"""
    Contour kernel size.

    The block size used in a box on content find [mm]

    default: 1.5
    """

    threshold_block_size: int | float
    r"""
    Threshold block size.

    The block size used in a box on threshold for content find [mm]

    default: 1.5
    """

    threshold_value_c: int | float
    r"""
    Threshold value c.

    A variable used on threshold, should be low on low contrast image, used in a box on content find on witch one we will crop

    default: 70
    """



class Crop(TypedDict, total=False):
    r"""
    Crop.

    The crop configuration
    """

    enabled: bool
    r"""
    Crop enabled.

    Enable the crop

    default: True
    """

    margin_horizontal: int | float
    r"""
    Margin horizontal.

    The horizontal margin used on auto-detect content [mm]

    default: 9
    """

    margin_vertical: int | float
    r"""
    Margin vertical.

    The vertical margin used on auto-detect content [mm]

    default: 6
    """

    contour: "Contour"
    r"""
    Contour.

    Editor note: The properties of this object should be modified in the config_schema.json file
    """



class CutOperation(TypedDict, total=False):
    r"""
    Cut operation.

    The cut configuration, a mask is used to definitively mask the source image

    default:
      enabled: false
    """

    enabled: bool
    r"""
    Auto cut enabled.

    Enable the cut

    default: True
    """

    auto_mask: "AutoMask"
    r"""
    Auto mask.

    Editor note: The properties of this object should be modified in the config_schema.json file
    """

    additional_filename: str | None
    r""" An image file used to add on the mask """



DESKEW_ANGLE_DERIVATION_DEFAULT = 0.1
r""" Default value of the field path 'Arguments deskew angle_derivation' """



DESKEW_ANGLE_PM_90_DEFAULT = False
r""" Default value of the field path 'Arguments deskew angle_pm_90' """



DESKEW_MAX_ANGLE_DEFAULT = 45
r""" Default value of the field path 'Arguments deskew max_angle' """



DESKEW_MIN_ANGLE_DEFAULT = -45
r""" Default value of the field path 'Arguments deskew min_angle' """



DESKEW_NUM_PEAKS_DEFAULT = 20
r""" Default value of the field path 'Arguments deskew num_peaks' """



DESKEW_SIGMA_DEFAULT = 3.0
r""" Default value of the field path 'Arguments deskew sigma' """



DE_NOISE_LEVEL_DEFAULT = 220
r""" Default value of the field path 'Auto mask de_noise_level' """



DE_NOISE_MORPHOLOGY_DEFAULT = True
r""" Default value of the field path 'Auto mask de_noise_morphology' """



DE_NOISE_SIZE_DEFAULT = 1000
r""" Default value of the field path 'Auto mask de_noise_size' """



DITHER_DEFAULT = {'enabled': False}
r""" Default value of the field path 'Arguments dither' """



DITHER_ENABLED_DEFAULT = True
r""" Default value of the field path 'Dither enabled' """



DPI_DEFAULT = 300
r""" Default value of the field path 'Arguments dpi' """



class Dither(TypedDict, total=False):
    r"""
    Dither.

    The dither configuration

    default:
      enabled: false
    """

    enabled: bool
    r"""
    Dither enabled.

    Enable the dither

    default: True
    """



EMPTY_ENABLED_DEFAULT = True
r""" Default value of the field path 'Empty enabled' """



EXIFTOOL_DEFAULT = {'enabled': False}
r""" Default value of the field path 'Arguments exiftool' """



EXIFTOOL_ENABLED_DEFAULT = True
r""" Default value of the field path 'Exiftool enabled' """



class Empty(TypedDict, total=False):
    r"""
    Empty.

    The empty page detection configuration
    """

    enabled: bool
    r"""
    Empty enabled.

    Enable the empty page detection

    default: True
    """

    contour: "Contour"
    r"""
    Contour.

    Editor note: The properties of this object should be modified in the config_schema.json file
    """



class Exiftool(TypedDict, total=False):
    r"""
    Exiftool.

    The exiftool optimization tool configuration

    default:
      enabled: false
    """

    enabled: bool
    r"""
    Exiftool enabled.

    Use the exiftool optimizer

    default: True
    """



INVERSE_MASK_DEFAULT = False
r""" Default value of the field path 'Auto mask inverse_mask' """



class IntermediateError(TypedDict, total=False):
    r""" Intermediate error. """

    error: str
    traceback: list[str]


JPEG_DEFAULT = {'enabled': False}
r""" Default value of the field path 'Arguments jpeg' """



JPEG_ENABLED_DEFAULT = True
r""" Default value of the field path 'Jpeg enabled' """



JPEG_QUALITY_DEFAULT = 90
r""" Default value of the field path 'Jpeg quality' """



class Jpeg(TypedDict, total=False):
    r"""
    Jpeg.

    Convert images to JPEG configuration

    default:
      enabled: false
    """

    enabled: bool
    r"""
    Jpeg enabled.

    Convert images to JPEG

    default: True
    """

    quality: int
    r"""
    Jpeg quality.

    The JPEG quality

    default: 90
    """



LEVEL_VALUE_DEFAULT = False
r""" Default value of the field path 'Level value' """



LINE_DETECTION_APERTURE_SIZE_DEFAULT = 3
r""" Default value of the field path 'Line detection aperture_size' """



LINE_DETECTION_HIGH_THRESHOLD_DEFAULT = 1000
r""" Default value of the field path 'Line detection high_threshold' """



LINE_DETECTION_LOW_THRESHOLD_DEFAULT = 0
r""" Default value of the field path 'Line detection low_threshold' """



LINE_DETECTION_MAX_LINE_GAP_DEFAULT = 100
r""" Default value of the field path 'Line detection max_line_gap' """



LINE_DETECTION_MIN_LINE_LENGTH_DEFAULT = 50
r""" Default value of the field path 'Line detection min_line_length' """



LINE_DETECTION_RHO_DEFAULT = 1
r""" Default value of the field path 'Line detection rho' """



LINE_DETECTION_THRESHOLD_DEFAULT = 100
r""" Default value of the field path 'Line detection threshold' """



LOWER_HSV_COLOR_DEFAULT = [0, 0, 250]
r""" Default value of the field path 'Auto mask lower_hsv_color' """



class Level(TypedDict, total=False):
    r"""
    Level.

    The level configuration
    """

    value: "LevelValue"
    r"""
    Level value.

    true: => do level on 15% - 85% (under 15 % will be black above 85% will be white), false: => 0% - 100%, <number>: => (0 + <number>)% - (100 - number)%

    default: False
    """

    auto: bool
    r"""
    Auto level.

    If no level specified, do auto level

    default: False
    """

    min: int | float
    r"""
    Min level.

    Min level if no level end no auto-level

    default: 0
    """

    max: int | float
    r"""
    Max level.

    Max level if no level end no auto-level

    default: 100
    """



LevelValue = bool | int
r"""
Level value.

true: => do level on 15% - 85% (under 15 % will be black above 85% will be white), false: => 0% - 100%, <number>: => (0 + <number>)% - (100 - number)%

default: False
"""



class Limit(TypedDict, total=False):
    r""" Limit. """

    name: str
    r""" The name visible on the generated image """

    type: str
    r""" The kind of split """

    value: int
    r""" The split position """

    vertical: bool
    r""" Is vertical? """

    margin: int
    r""" The margin around the split, can be used to remove a fold """



class LimitDetection(TypedDict, total=False):
    r"""
    Limit detection.

    The limit page detection configuration
    """

    contour: "Contour"
    r"""
    Contour.

    Editor note: The properties of this object should be modified in the config_schema.json file
    """

    line: "LineDetection"
    r"""
    Line detection.

    The line detection used in assisted split
    """



class LineDetection(TypedDict, total=False):
    r"""
    Line detection.

    The line detection used in assisted split
    """

    low_threshold: int
    r"""
    Line detection low threshold.

    The low threshold used in the Canny edge detector

    default: 0
    """

    high_threshold: int
    r"""
    Line detection high threshold.

    The high threshold used in the Canny edge detector

    default: 1000
    """

    aperture_size: int
    r"""
    Line detection aperture size.

    The aperture size used in the Canny edge detector

    default: 3
    """

    rho: int
    r"""
    Line detection rho.

    The rho used in the Hough transform

    default: 1
    """

    threshold: int
    r"""
    Line detection threshold.

    The threshold used in the Hough transform

    default: 100
    """

    min_line_length: int
    r"""
    Line detection min line length.

    The minimum line length in percentage of the image size used in the Hough transform

    default: 50
    """

    max_line_gap: int
    r"""
    Line detection max line gap.

    The maximum line gap in percentage of the image size used in the Hough transform

    default: 100
    """



MARGIN_HORIZONTAL_DEFAULT = 9
r""" Default value of the field path 'Crop margin_horizontal' """



MARGIN_VERTICAL_DEFAULT = 6
r""" Default value of the field path 'Crop margin_vertical' """



MASK_ENABLED_DEFAULT = True
r""" Default value of the field path 'Mask operation enabled' """



MASK_OPERATION_DEFAULT = {'enabled': False}
r""" Default value of the field path 'Arguments mask' """



MAX_LEVEL_DEFAULT = 100
r""" Default value of the field path 'Level max' """



MIN_BOX_BLACK_DEFAULT = 2
r""" Default value of the field path 'Contour min_box_black' """



MIN_BOX_SIZE_DEFAULT = {'crop': 3, 'empty': 10, 'limit': 10}
r""" Default value of the field path 'Contour min_box_size' """



MIN_LEVEL_DEFAULT = 0
r""" Default value of the field path 'Level min' """



class MaskOperation(TypedDict, total=False):
    r"""
    Mask operation.

    The mask configuration, the a is used to mask the image on crop and skew calculation

    default:
      enabled: false
    """

    enabled: bool
    r"""
    mask enabled.

    Enable the mask

    default: True
    """

    auto_mask: "AutoMask"
    r"""
    Auto mask.

    Editor note: The properties of this object should be modified in the config_schema.json file
    """

    additional_filename: str | None
    r""" An image file used to add on the mask """



NO_REMOVE_TO_CONTINUE_DEFAULT = False
r""" Default value of the field path 'Arguments no_remove_to_continue' """



OPTIPNG_ENABLED_DEFAULT = True
r""" Default value of the field path 'Optipng enabled' """



class Optipng(TypedDict, total=False):
    r"""
    Optipng.

    The optipng optimization tool configuration
    """

    enabled: bool
    r"""
    Optipng enabled.

    Use the optipng optimizer

    default: True
    """



PNGQUANT_DEFAULT = {'enabled': False}
r""" Default value of the field path 'Arguments pngquant' """



PNGQUANT_ENABLED_DEFAULT = True
r""" Default value of the field path 'Pngquant enabled' """



PNGQUANT_OPTIONS_DEFAULT = ['--force', '--speed=1', '--strip', '--quality=0-32']
r""" Default value of the field path 'Pngquant options' """



PROGRESS_DEFAULT = False
r""" Default value of the field path 'Configuration progress' """



PS2PDF_DEFAULT = {'enabled': False}
r""" Default value of the field path 'Arguments ps2pdf' """



PS2PDF_ENABLED_DEFAULT = True
r""" Default value of the field path 'Ps2pdf enabled' """



class Pngquant(TypedDict, total=False):
    r"""
    Pngquant.

    The pngquant optimization tool configuration

    default:
      enabled: false
    """

    enabled: bool
    r"""
    Pngquant enabled.

    Use the pngquant optimizer

    default: True
    """

    options: list[str]
    r"""
    Pngquant options.

    The pngquant options

    default:
      - --force
      - --speed=1
      - --strip
      - --quality=0-32
    """



class Ps2Pdf(TypedDict, total=False):
    r"""
    Ps2pdf.

    The ps2pdf optimization tool configuration

    default:
      enabled: false
    """

    enabled: bool
    r"""
    Ps2pdf enabled.

    Use the ps2pdf optimizer (=> JPEG)

    default: True
    """



REST_UPLOAD_ENABLED_DEFAULT = False
r""" Default value of the field path 'REST upload enabled' """



RULE_ENABLE_DEFAULT = True
r""" Default value of the field path 'Rule enabled' """



RULE_GRADUATION_COLOR_DEFAULT = [0, 0, 0]
r""" Default value of the field path 'Rule graduation_color' """



RULE_GRADUATION_TEXT_FONT_COLOR_DEFAULT = [0, 0, 0]
r""" Default value of the field path 'Rule graduation_text_font_color' """



RULE_GRADUATION_TEXT_FONT_FILENAME_DEFAULT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
r""" Default value of the field path 'Rule graduation_text_font_filename' """



RULE_GRADUATION_TEXT_FONT_SIZE_DEFAULT = 17
r""" Default value of the field path 'Rule graduation_text_font_size' """



RULE_GRADUATION_TEXT_MARGIN_DEFAULT = 6
r""" Default value of the field path 'Rule graduation_text_margin' """



RULE_LINES_COLOR_DEFAULT = [0, 0, 0]
r""" Default value of the field path 'Rule lines_color' """



RULE_LINES_OPACITY_DEFAULT = 0.2
r""" Default value of the field path 'Rule lines_opacity' """



RULE_LINES_SPACE_DEFAULT = 100
r""" Default value of the field path 'Rule lines_space' """



RULE_MAJOR_GRADUATION_SIZE_DEFAULT = 30
r""" Default value of the field path 'Rule major_graduation_size' """



RULE_MAJOR_GRADUATION_SPACE_DEFAULT = 100
r""" Default value of the field path 'Rule major_graduation_space' """



RULE_MINOR_GRADUATION_SIZE_DEFAULT = 10
r""" Default value of the field path 'Rule minor_graduation_size' """



RULE_MINOR_GRADUATION_SPACE_DEFAULT = 10
r""" Default value of the field path 'Rule minor_graduation_space' """



class RestUpload(TypedDict, total=False):
    r"""
    REST upload.

    Upload the final PDF via Paperless REST API
    """

    enabled: bool
    r"""
    REST upload enabled.

    Enable the upload of the PDF via REST API

    default: False
    """

    api_url: Required[str]
    r"""
    REST upload API url.

    The URL address of the REST API, usually http://server.name/api

    Required property
    """

    api_token: Required[str]
    r"""
    REST upload API token.

    The API token

    Required property
    """



class Rule(TypedDict, total=False):
    r"""
    Rule.

    Configuration of rule displayed in assisted split images
    """

    enabled: bool
    r"""
    Rule enable.

    default: True
    """

    minor_graduation_space: int
    r"""
    Rule minor graduation space.

    default: 10
    """

    major_graduation_space: int
    r"""
    Rule major graduation space.

    default: 100
    """

    lines_space: int
    r"""
    Rule lines space.

    default: 100
    """

    minor_graduation_size: int
    r"""
    Rule minor graduation size.

    default: 10
    """

    major_graduation_size: int
    r"""
    Rule major graduation size.

    default: 30
    """

    graduation_color: list[int]
    r"""
    Rule graduation color.

    default:
      - 0
      - 0
      - 0
    """

    lines_color: list[int]
    r"""
    Rule lines color.

    default:
      - 0
      - 0
      - 0
    """

    lines_opacity: int | float
    r"""
    Rule lines opacity.

    default: 0.2
    """

    graduation_text_font_filename: str
    r"""
    Rule graduation text font filename.

    default: /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
    """

    graduation_text_font_size: int | float
    r"""
    Rule graduation text font size.

    default: 17
    """

    graduation_text_font_color: list[int]
    r"""
    Rule graduation text font color.

    default:
      - 0
      - 0
      - 0
    """

    graduation_text_margin: int
    r"""
    Rule graduation text margin.

    default: 6
    """



SHARPEN_DEFAULT = {'enabled': False}
r""" Default value of the field path 'Arguments sharpen' """



SHARPEN_ENABLED_DEFAULT = True
r""" Default value of the field path 'Sharpen enabled' """



class Sharpen(TypedDict, total=False):
    r"""
    Sharpen.

    Sharpen configuration

    default:
      enabled: false
    """

    enabled: bool
    r"""
    Sharpen enabled.

    Enable the sharpen

    default: True
    """



class Step(TypedDict, total=False):
    r""" Step. """

    name: str
    r""" The step name """

    sources: list[str]
    r""" The images obtain after the current step """

    process_count: int
    r""" The step number """



TESSERACT_ENABLED_DEFAULT = True
r""" Default value of the field path 'Tesseract enabled' """



TESSERACT_LANG_DEFAULT = 'fra+eng'
r""" Default value of the field path 'Tesseract lang' """



THRESHOLD_BLOCK_SIZE_DEFAULT = 1.5
r""" Default value of the field path 'Contour threshold_block_size' """



THRESHOLD_VALUE_C_DEFAULT = 70
r""" Default value of the field path 'Contour threshold_value_c' """



class Tesseract(TypedDict, total=False):
    r"""
    Tesseract.

    The Tesseract configuration
    """

    enabled: bool
    r"""
    Tesseract enabled.

    Use Tesseract to to an OCR on the document

    default: True
    """

    lang: str
    r"""
    Tesseract lang.

    The used language for tesseract

    default: fra+eng
    """



UPPER_HSV_COLOR_DEFAULT = [255, 10, 255]
r""" Default value of the field path 'Auto mask upper_hsv_color' """



class _ArgumentsDeskew(TypedDict, total=False):
    r""" The deskew configuration """

    min_angle: int | float
    r"""
    Deskew min angle.

    The minimum angle to detect the image skew [degree]

    default: -45
    """

    max_angle: int | float
    r"""
    Deskew max angle.

    The maximum angle to detect the image skew [degree]

    default: 45
    """

    angle_derivation: int | float
    r"""
    Deskew angle derivation.

    The step of angle to detect the image skew [degree]

    default: 0.1
    """

    sigma: int | float
    r"""
    Deskew sigma.

    Used in the `canny` function

    default: 3.0
    """

    num_peaks: int
    r"""
    Deskew num peaks.

    number of peaks we ask for

    default: 20
    """

    angle_pm_90: bool
    r"""
    Deskew angle pm 90.

    Detect an angle of +/- 90 degree, also +/- 45 degree

    default: False
    """



class _ConfigurationImagesConfigAdditionalproperties(TypedDict, total=False):
    angle: int | float | None
    r""" The used angle to deskew, can be change, restart by deleting one of the generated images """

    status: "_ConfigurationImagesConfigAdditionalpropertiesStatus"


class _ConfigurationImagesConfigAdditionalpropertiesStatus(TypedDict, total=False):
    angle: int | float
    r""" The measured deskew angle """

    size: list[int | float]
    r""" The image dimensions """

