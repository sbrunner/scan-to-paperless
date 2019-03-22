# Scan and prepare your document for paperless

## Usage

1. Use the scan command, with auto-completion on your correspondents and your tags to import your document, to scan your documents.

2. The document is transferred to your NAS (I use [Syncthing](https://syncthing.net/)).

3. The documents will be processed on the NAS.

4. Use `scan-process-status` to know the status of your documents.

5. Validate your documents.

4. If your happy with that remove the REMOVE_TO_CONTINUE file.
   (To restart the process remove one of the generated images, to cancel the job just remove the folder).

5. The process will continue his job and import the document in paperless.

## Nice feature

### Double sized scanning

1. Pour your sheets on the Automatic Document Feeder.

2. Run `scan` with the option `--double-sided`.

3. Press enter to start scanning the first side of all sheets.

4. Put again all your sheets on the Automatic Document Feeder without turning them.

The scan utils will rotate and reorder all the sheets to get a good document.

### Credit card scanning

The options `--append-credit-card` will append all the sheets vertically to have the booth face of the credit card on the same page.

### Assisted split

1. Do your scan as usual with the extra option `--assisted-split`.

2. After the process do his first pass you will have images with lines and numbers.
   The lines represent the detected potential split of the image, the length indicate the strength of the detection.
   In your config you will have somthing like:

```
assisted_split:
- destinations:
  - 4  # Page number of the left part of the image
  - 1  # Same for the right page of the image
  image: image-1.png  # name of the image
  limits:
  - margin: 0  # Margin around the split
    name: 0  # Number visible on the generated image
    value: 375  # The position of the split (can be manually edited)
    vertical: true  # Will split the image vertically
  - ...
  source: /source/975468/7-assisted-split/image-1.png
- ...
```

   Edit your config file, you should have one more destination then the limits.
   If you put destinatination like that: 2.1, it mean that it will be the first part of the page 2 and the 2.2 will be the secound part.

3. Delete the file `REMOVE_TO_CONTINUE`.

4. After the process do his first pass you will have the final generated images.

5. If it's OK delete the file `REMOVE_TO_CONTINUE`.


## Install

Install in a venv in the home directory:

```
$ cd
$ python3 -m venv venv
$ ~/venv/bin/pip install scan-to-paperless
$ sudo activate-global-python-argcomplete
$ echo PATH=$PATH:~/venv/bin >> ~/.bashrc
```

## NAS

The Docker support is required, Personally I use a [Synology DiskStation DS918+](https://www.synology.com/products/DS918+),
and you can get the *.syno.json files to configure your Docker services.
