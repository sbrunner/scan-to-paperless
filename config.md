# Configuration

## Properties

- **`scan_folder`** _(string)_: This should be shared with the process container in 'source'.
- **`scanimage`** _(string)_: The scanimage command. Default: `scanimage`.
- **`scanimage_arguments`** _(array)_: The scanimage arguments. Default: `['--format=png', '--mode=color', '--resolution=300']`.
  - **Items** _(string)_
- **`extension`** _(string)_: The extension of generate image (png or tiff). Default: `png`.
- **`default_args`**: Refer to _#/definitions/args_.
- **`viewer`** _(string)_: The command used to start the viewer. Default: `eog`.
- **`modes`** _(object)_: Customize the modes. Can contain additional properties.
  - **Additional Properties** _(object)_
    - **`scanimage_arguments`** _(array)_: Additional scanimage arguments.
      - **Items** _(string)_
    - **`auto_bash`** _(boolean)_: Run the ADF in tow step odd and even, needed for scanner that don't support double face.
    - **`rotate_even`** _(boolean)_: Rotate the even pages, to use in conjunction with auto_bash.

## Definitions

- **`args`** _(object)_: Cannot contain additional properties.
  - **`level`** _(['boolean', 'integer'])_: true: => do level on 15% - 85% (under 15 % will be black above 85% will be white), false: => 0% - 100%, <number>: => (0 + <number>)% - (100 - number)%.
  - **`auto_level`** _(boolean)_: If no level specified, do auto level. Default: `False`.
  - **`min_level`** _(number)_: Min level if no level end no auto-level. Default: `15`.
  - **`max_level`** _(number)_: Max level if no level end no auto-level. Default: `15`.
  - **`no_crop`** _(boolean)_: Don't do any crop. Default: `False`.
  - **`margin_horizontal`** _(number)_: The horizontal margin used on auto-detect content [mm]. Default: `9`.
  - **`margin_vertical`** _(number)_: The vertical margin used on auto-detect content [mm]. Default: `6`.
  - **`dpi`** _(number)_: The DPI used to convert the mm to pixel. Default: `300`.
  - **`sharpen`** _(boolean)_: Do the sharpen. Default: `False`.
  - **`dither`** _(boolean)_: Do the dither. Default: `False`.
  - **`tesseract`** _(boolean)_: Use tesseract to to an OCR on the document. Default: `False`.
  - **`tesseract_lang`** _(string)_: The used language for tesseract. Default: `fra+eng`.
  - **`append_credit_card`** _(boolean)_: Do an assisted split. Default: `False`.
  - **`assisted_split`** _(boolean)_: Do an assisted split. Default: `False`.
  - **`num_angles`** _(integer)_: The number of angle used to detect the image skew. Default: `1800`.
  - **`min_box_size_crop`** _(number)_: The minimum box size to find the content on witch one we will crop [mm]. Default: `3`.
  - **`min_box_black_crop`** _(number)_: The minimum black in a box on content find on witch one we will crop [%]. Default: `2`.
  - **`contour_kernel_size_crop`** _(number)_: The block size used in a box on content find on witch one we will crop [mm]. Default: `1.5`.
  - **`threshold_block_size_crop`** _(number)_: The block size used in a box on threshold for content find on witch one we will crop [mm]. Default: `1.5`.
  - **`threshold_value_c_crop`** _(number)_: A variable used on threshold, should be low on low contrast image, used in a box on content find on witch one we will crop. Default: `70`.
  - **`min_box_size_empty`** _(number)_: The minimum box size to find the content to determine if the page is empty [mm]. Default: `10`.
  - **`min_box_black_empty`** _(number)_: The minimum black in a box on content find if the page is empty [%]. Default: `2`.
  - **`contour_kernel_size_empty`** _(number)_: The block size used in a box on content find if the page is empty [mm]. Default: `1.5`.
  - **`threshold_block_size_empty`** _(number)_: The block size used in a box on threshold for content find if the page is empty [mm]. Default: `1.5`.
  - **`threshold_value_c_empty`** _(number)_: A variable used on threshold, should be low on low contrast image, used in a box on content find if the page is empty. Default: `70`.
  - **`min_box_size_limit`** _(number)_: The minimum box size to find the limits based on content [mm]. Default: `3`.
  - **`min_box_black_limit`** _(number)_: The minimum black in a box on content find the limits based on content [%]. Default: `2`.
  - **`contour_kernel_size_limit`** _(number)_: The block size used in a box on content find the limits based on content [mm]. Default: `1.5`.
  - **`threshold_block_size_limit`** _(number)_: The block size used in a box on threshold for content find the limits based on content [mm]. Default: `1.5`.
  - **`threshold_value_c_limit`** _(number)_: A variable used on threshold, should be low on low contrast image, used in a box on content find the limits based on content. Default: `70`.
  - **`colors`** _(integer)_: The number of colors in the png. Default: `0`.
  - **`run_optipng`** _(boolean)_: Run the optipng optimizer. Default: `True`.
  - **`run_pngquant`** _(boolean)_: Run the pngquant optimizer. Default: `False`.
  - **`pngquant_options`** _(array)_: The pngquant options.
    - **Items** _(string)_
  - **`run_exiftool`** _(boolean)_: Run the exiftool optimizer. Default: `False`.
  - **`run_ps2pdf`** _(boolean)_: Run the ps2pdf optimizer (=> JPEG). Default: `False`.
  - **`jpeg`** _(boolean)_: Convert images to JPEG. Default: `False`.
  - **`jpeg_quality`** _(integer)_: The JPEG quality. Default: `90`.
