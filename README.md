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


## Install (with clone)

* Clone the repository.
* Create a simlink to from the `scan`, `scan-complete` and `scan-process-status` to a repository of the `PATH` or add the Git repository to the `PATH`.
* Add `. scan-complete` in your `.bashrc` to have the autocompletion.

## Install (direct)

* Get the files [scan](https://raw.githubusercontent.com/sbrunner/scan-to-paperless/master/scan),
  [scan-complete](https://raw.githubusercontent.com/sbrunner/scan-to-paperless/master/scan-complete),
  [scan-process-status](https://raw.githubusercontent.com/sbrunner/scan-to-paperless/master/scan-process-status)
  and put them in a folder that is in your PATH.
* Add `. scan-complete` in your `.bashrc` to have the autocompletion.

## NAS

The Docker support is required, pPersonally I use a [Synology DiskStation DS918+](https://www.synology.com/products/DS918+),
and you can get the *.syno.json files to configure your Docker services.
