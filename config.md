# Configuration

## Properties

- **`scan_folder`** *(string)*: This should be shared with the process container in 'source'.
- **`scanimage`** *(string)*: The scanimage command. Default: `scanimage`.
- **`scanimage_arguments`** *(array)*: Default: `['--format=png', '--mode=color', '--resolution=300']`.
  - **Items** *(string)*
- **`default_args`**: Refer to *#/definitions/args*.
- **`viewer`** *(string)*: The command used to start the viewer. Default: `eog`.
## Definitions

- **`args`** *(object)*: Cannot contain additional properties.
  - **`level`** *(['boolean', 'integer'])*: true: => do level on 15% - 85% (under 15 % will be black above 85% will be white), false: => 0% - 100%, <number>: => (0 + <number>)% - (100 - number)%.
  - **`auto_level`** *(boolean)*: If no level specified, do auto level. Default: `False`.
  - **`min_level`** *(integer)*: Min level if no level end no autolovel. Default: `15`.
  - **`max_level`** *(integer)*: Max level if no level end no autolovel. Default: `15`.
  - **`nocrop`** *(boolean)*: Don't do any crop. Default: `False`.
  - **`margin_horizontal`** *(number)*: The horizontal margin used on autodetect content [mm]. Default: `9`.
  - **`margin_vertical`** *(number)*: The vertical margin used on autodetect content [mm]. Default: `6`.
  - **`dpi`** *(number)*: The DPI used to convert the mm to pixel. Default: `300`.
  - **`sharpen`** *(boolean)*: Do the sharpen. Default: `False`.
  - **`dither`** *(boolean)*: Do the dither. Default: `False`.
  - **`tesseract`** *(boolean)*: Use tesseract to to an OCR on the document. Default: `False`.
  - **`tesseract_lang`** *(string)*: The used language for tesseract. Default: `fra+eng`.
  - **`assisted_split`** *(boolean)*: Do en assited split. Default: `False`.
