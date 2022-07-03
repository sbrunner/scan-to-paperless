# Configuration

## Properties

- **`scan_folder`** _(string)_: This should be shared with the process container in 'source'.
- **`scanimage`** _(string)_: The scanimage command. Default: `scanimage`.
- **`scanimage_arguments`** _(array)_: Default: `['--format=png', '--mode=color', '--resolution=300']`.
  - **Items** _(string)_
- **`default_args`**: Refer to _#/definitions/args_.
- **`viewer`** _(string)_: The command used to start the viewer. Default: `eog`.

## Definitions

- **`args`** _(object)_: Cannot contain additional properties.
  - **`level`** _(['boolean', 'integer'])_: true: => do level on 15% - 85% (under 15 % will be black above 85% will be white), false: => 0% - 100%, <number>: => (0 + <number>)% - (100 - number)%.
  - **`auto_level`** _(boolean)_: If no level specified, do auto level. Default: `False`.
  - **`min_level`** _(integer)_: Min level if no level end no auto-level. Default: `15`.
  - **`max_level`** _(integer)_: Max level if no level end no auto-level. Default: `15`.
  - **`no_crop`** _(boolean)_: Don't do any crop. Default: `False`.
  - **`margin_horizontal`** _(number)_: The horizontal margin used on auto-detect content [mm]. Default: `9`.
  - **`margin_vertical`** _(number)_: The vertical margin used on auto-detect content [mm]. Default: `6`.
  - **`dpi`** _(number)_: The DPI used to convert the mm to pixel. Default: `300`.
  - **`sharpen`** _(boolean)_: Do the sharpen. Default: `False`.
  - **`dither`** _(boolean)_: Do the dither. Default: `False`.
  - **`tesseract`** _(boolean)_: Use tesseract to to an OCR on the document. Default: `False`.
  - **`tesseract_lang`** _(string)_: The used language for tesseract. Default: `fra+eng`.
  - **`assisted_split`** _(boolean)_: Do en assisted split. Default: `False`.
  - **`min_box_size_crop`** _(number)_: The minimum box size to find the content on witch one we will crop [mm]. Default: `3`.
  - **`min_box_size_limit`** _(number)_: The minimum box size to find the limits based on content [mm]. Default: `10`.
  - **`min_box_size_empty`** _(number)_: The minimum box size to find the content to determine if the page is empty [mm]. Default: `10`.
  - **`min_box_black_crop`** _(number)_: The minimum black in a box on content find on witch one we will crop [%]. Default: `2`.
  - **`min_box_black_limit`** _(number)_: The minimum black in a box on content find the limits based on content [%]. Default: `2`.
  - **`min_box_black_empty`** _(number)_: The minimum black in a box on content find to determine if the page is empty [%]. Default: `2`.
  - **`box_kernel_size`** _(number)_: The block size used in a box on content find [mm]. Default: `1.5`.
  - **`box_block_size`** _(number)_: The block size used in a box on threshold for content find [mm]. Default: `1.5`.
  - **`box_threshold_value_c`** _(number)_: A variable used on threshold, should be low on low contrast image, used in a box on content find. Default: `70`.
  - **`colors`** _(integer)_: The number of colors in the png. Default: `0`.
  - **`run_optipng`** _(boolean)_: Run the optipng optimizer. Default: `True`.
  - **`run_exiftool`** _(boolean)_: Run the exiftool optimizer. Default: `True`.
  - **`run_ps2pdf`** _(boolean)_: Run the ps2pdf optimizer (=> JPEG). Default: `False`.
