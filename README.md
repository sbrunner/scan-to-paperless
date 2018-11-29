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
