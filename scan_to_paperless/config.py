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
        "num_angles": int,
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
        #   - --skip-if-larger
        #   - --speed=1
        #   - --strip
        #   - --quality=0-64
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
        # WARNING: The required are not correctly taken in account,
        # See: https://github.com/camptocamp/jsonschema-gentypes/issues/6
        "auto_mask": "_ArgumentsAutoMask",
    },
    total=False,
)


# Configuration
Configuration = TypedDict(
    "Configuration",
    {
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


# The auto mask configuration
_ArgumentsAutoMask = TypedDict(
    "_ArgumentsAutoMask",
    {
        # The lower color in HSV representation
        #
        # default:
        #   - 0
        #   - 0
        #   - 108
        "lower_hsv_color": List[int],
        # The upper color in HSV representation
        #
        # default:
        #   - 255
        #   - 10
        #   - 148
        "upper_hsv_color": List[int],
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
        # default: 100
        "buffer_size": int,
        # The threshold level used in buffer on the blurry image
        #
        # default: 20
        "buffer_level": int,
    },
    total=False,
)


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
