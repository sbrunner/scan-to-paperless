# Scan and prepare your document for [Paperless](https://github.com/paperless-ngx/paperless-ngx)

The main goal of this project is to have some productive process from the document scanning to
[Paperless](https://github.com/paperless-ngx/paperless-ngx).
For that we need to prepare the documents some tools that need many resources, then the idea to do it
in the background and ideally on another host like a NAS.
A consequence of that it's a not easy to put it in place, but then you will be really productive.
The interface between the user and the process is the `scan` command to do the initial scan, and the file system
to verify that the result is OK (and do some advance operations describe below) and validate it.

## Features

- Scan the images optionally by using the Automatic Document Feeder
- Easily scan double-sided images using the Automatic Document Feeder
- Extract the DPI from the TIFF images
- Change the images levels
- Remove the area out of the image
- Deskew the images
- Crop the images
- Sharpen the images (disable by default)
- Dither the images (disable by default)
- Auto rotate the images by using tesseract (To have the text on the right side)
- Optimize the images using `pngquant`, `optipng`, `ps2pdf` or `jpeg` (using quality from GraphicsMagick convert)
- Assisted split, used to split a prospectus page in more pages (Requires to modify the YAML...)
- Append credit cart, used to have the two faces of a credit cart on the same page
- Be able to copy the OCR result from the PDF
- Scan the QR code and Bar code and add a new page with the values (separate process)
- Manage the empty lines in the QR code (replace by a pipe (`|`) in the PDF,
  and run `scan --convert-clipboard` to scan your clipboard to do the inverse transform)

## Requirements

On the desktop:

- [Python](https://www.python.org/) >= 3.6
- The [scanimage](http://www.sane-project.org/) command, on Windows it should be able to use another command,
  but it's never be tested.
  This command would be an adapter that interpret the following arguments:
  `--batch`, `--batch-start`, `--batch-increment`, `--batch-count`,
  `--batch` for the destination file name template (`%d` is replaced by the page number),
  and the others for the `auto_bash`.

On the NAS:

- [Docker](https://www.docker.com/)

## Install

Scan-to-paperless requires a desktop and a server part, the two parts communicate through the scan folder.

The server part is where the document were processed, and the desktop part is
from where we want we will scan the document, on witch one the scanner is connected.

The scan folder should be synchronized between the desktop and the server,
I use [Syncthing](https://syncthing.net/) for that.

### On the desktop

```bash
$ python3 -m pip install scan-to-paperless
$ echo PATH=$PATH:~/venv/bin >> ~/.bashrc
$ echo source <(register-python-argcomplete scan) >> ~/.bashrc
$ echo source <(register-python-argcomplete scan-progress-status) >> ~/.bashrc
```

Create the configuration file on `<home_config>/scan-to-paperless.yaml` (on Linux it's `~/.config/scan-to-paperless.yaml`), with:

```yaml
# yaml-language-server: $schema=https://raw.githubusercontent.com/sbrunner/scan-to-paperless/master/scan_to_paperless/config_schema.json

scan_folder: /home/sbrunner/Paperless/scan/
scanimage_arguments: # Additional argument passed to the scanimage command
  - --device=... # Use `scanimage --list` to get the possible values
  - --format=png
  - --mode=color
  - --resolution=300
default_args:
  auto_mask: {}
  auto_cut: {}
  run_pngquant: true
  cut_white: 200 # cut the near white color to have a uniform background
  dpi: 300 # Not necessary if the scanner generate a tiff file
  tesseract_lang: fra+eng # The used languages for the OCR
```

[Full config documentation](./config.md)

### On the NAS

The Docker support is required, Personally I use a [Synology DiskStation DS918+](https://www.synology.com/products/DS918+),
and you can get the \*.syno.json files to configure your Docker services.

Otherwise, use:

```bash
SCAN_FOLDER=<scan_folder>
CONSUME_FOLDER=<consume_folder>
docker run --name=scan-to-paperless --restart=unless-stopped --detach \
  --volume=${SCAN_FOLDER}:/source \
  --volume=${CONSUME_FOLDER}:/destination \
  sbrunner/scan-to-paperless
```

You can set the environment variable `PROGRESS` to `TRUE` to get all the intermediate images.

To stop run:

```bash
docker stop scan-to-paperless
docker rm scan-to-paperless
```

### Repertory link

You should find a way to synchronize or using sharing to link the scan folder on your desktop and on your NAS.

You should also link to consume folder to `paperless-ngx` probably just by using the same folder.

## Usage

1. Use the `scan` command to import your document, to scan your documents.

2. The document is transferred to your NAS (I use [Syncthing](https://syncthing.net/)).

3. The documents will be processed on the NAS.

4. Use `scan-process-status` to know the status of your documents.

5. Validate your documents.

6. If your happy with that remove the `REMOVE_TO_CONTINUE` file.
   (To restart the process remove one of the generated images, to cancel the job just remove the folder).

7. The process will continue his job and import the document in `paperless-ngx`.

## Job config file

In the `config.yaml` file present in the document folder, you can find some information generated during
the processing and some can be modified.

E.g. you can modify an image angle to fix the skew, then remove a generated image for force to regenerate
the images.

[Full job config documentation](./process.md)

## Advance feature

### Add a mask

If your scanner add some margin around the scanned image it will relay case some issue the skew and the
content detection.

To solve that you can add a black and white image named `mask.png` in the root folder and draw in black the
part that should not be taken in account.

Scan to Paperless is also able to create a mask automatically, to enable is with the default configuration,
just add `args` name `auto_mask` with an empty dictionary (`{}`).

See also: [The documentation](./config.md#definitions)

Configuration note:

By default, the options `lower_hsv_color` and `upper_hsv_color` select the page (white).
Yon can also select the scanner background, for that you also should set the option `inverse_mask` to `true`
and the option `de_noise_morphology` to `false`.

### Mask the image

If your scanner add some margin around the scanned image you can definitively mask them.

To solve that you can add a black and white image named `cut.png` in the root folder and draw in black the
part that should not be taken in account.

Scan to Paperless is also able to create a mask automatically, to enable is with the default configuration,
just add `args` name `auto_cut` with an empty dictionary (`{}`).

See also: [The documentation](./config.md#definitions)

### Double sized scanning

1. Pour your sheets on the Automatic Document Feeder.

2. Run `scan` with the option `--mode=double`.

3. Press enter to start scanning the first side of all sheets.

4. Put again all your sheets on the Automatic Document Feeder without turning them.

The scan utils will rotate and reorder all the sheets to get a good document.

### Credit card scanning

The options `--append-credit-card` will append all the sheets vertically to have the booth face of the credit card on the same page.

### Assisted split

1. Do your scan as usual with the extra option `--assisted-split`.

2. After the process do his first pass you will have images with lines and numbers.
   The lines represent the detected potential split of the image, the length indicate the strength of the detection.
   In your config you will have something like:

```yaml
assisted_split:
-   destinations:
    -   4 # Page number of the left part of the image
    -   1 # Same for the right page of the image
        image: image-1.png # name of the image
        limits:
    -   margin: 0 # Margin around the split
        name: 0 # Number visible on the generated image
        value: 375 # The position of the split (can be manually edited)
        vertical: true # Will split the image vertically
    -   ...
        source: /source/975468/7-assisted-split/image-1.png
-   ...

```

Edit your config file, you should have one more destination than the limits.
If you put destination like that: 2.1, it means that it will be the first part of the page 2 and the 2.2 will be the second part.

3. Delete the file `REMOVE_TO_CONTINUE`.

4. After the process do his first pass you will have the final generated images.

5. If it's OK delete the file `REMOVE_TO_CONTINUE`.

## The scan modes configuration

First of all the `scanimage` command and arguments can be configured with the `scanimage` and
`scanimage_argumentss` options in the configuration file (`~/.config/scan-to-paperless.yaml`).

In this file there is also a `modes` section that can configure each modes.

See also: [The documentation](./config.md)

## Extends an existing configuration

To create the `preset` configuration file it can be useful to extends an existing configuration.
For that you can use the `extends` (and `merge_strategies`) option in the configuration file.

See also: [The documentation](./config.md)

## Server configuration

Environment variable:

- `SCAN_SOURCE_FOLDER`: The main input folder for the scan process.
- `SCAN_CODES_FOLDER`: The input folder for the codes (QR code ad Barcode) detection and add a new page.
- `SCAN_FINAL_FOLDER`: The final folder for the scan process.
- `SCAN_CODES_DPI`: The used DPI to decode the codes.
- `SCAN_CODES_PDF_DPI`: The used PDF DPI to create the codes document.
- `SCAN_CODES_FONT_NAME`: The used font of code number.
- `SCAN_CODES_FONT_SIZE`: The used font size of code number.
- `SCAN_CODES_MARGIN_TOP`: The top margin of code number.
- `SCAN_CODES_MARGIN_LEFT`:The left margin of code number.
- `TIME`: Print the elapsed time.
- `PROGRESS`: Save some intermediate files, don't clean the folder at the end.
