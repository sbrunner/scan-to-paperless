# AI Scan Optimization Guide

This file defines how an AI agent should improve a scan by editing `config.yaml`.

## Goal

Improve the final document quality with minimal clipping and robust masking.

## Structure

In this folder there is one folder per job. In each job there is:

- a `config.yaml` file with the process configuration and some result values.
- if there is a `REMOVE_TO_CONTINUE` file it means that the process is waiting for user intervention.
- if there is a `DOWN` file it means that the process is performing its final step.
- if there is no `REMOVE_TO_CONTINUE` or `DOWN` file it means that the job is waiting to be processed.
- there is also a `jupyter` folder with a Jupyter notebook.
- the `source` folder contains the source images and also deskew-related images.
- the `crop` folder contains images where the mask is applied.
- the `histogram` folder contains histograms of the images.
- the `skew` folder contains images with skew information.
- and in the base folder there are the generated images.

Finally, in the root folder there is a `status.html` file that contains the global status.

## Primary data source

For each image, read `images_config.<image>.status` in `config.yaml`:

- `histogram.text`: textual grayscale histogram (`000-007 | #### | 3.10%` format).
- `histogram.current`: current cut settings and clipping percentages.
- `histogram.suggested`: conservative/balanced/aggressive proposals for `cut_black` and `cut_white`.
- `auto_mask_hsv`: HSV suggestions for `auto_mask` values.
- `deskew`: detected/applied angle and deskew search configuration.

## Note on generated metadata

The metadata blocks (`histogram`, `auto_mask_hsv`, `deskew`) are generated hints.
Treat them as guidance, not absolute truth:

- always verify on the rendered image,
- prefer small iterative changes,
- when visual result and metadata disagree, trust the visual result.

## Tuning strategy

1. Start with `histogram.suggested.balanced` for `cut_black` and `cut_white`.
2. If text strokes are being lost, reduce `cut_black`.
3. If background is still gray/noisy, increase `cut_white` gradually.
4. Prefer clipping percentages from `histogram.current` as objective signals.
5. For masking, start with `auto_mask_hsv.suggestions.page_white`.
6. If scanner background should be removed instead, use `scanner_background` and keep
   `inverse_mask: true` with `de_noise_morphology: false`.
7. If `deskew.near_search_limit` is true, broaden `args.deskew.min_angle`/`max_angle`.

## Where to write changes

- Global defaults: `args.cut_black`, `args.cut_white`, `args.mask.auto_mask`, `args.cut.auto_mask`.
- Per image angle override (manual deskew): `images_config.<image>.angle`.

## Validation loop

1. Edit `config.yaml`.
2. Delete generated transformed images for the page(s) being tuned.
3. Re-run processing.
4. Re-check `images_config.<image>.status` and final images.
5. Repeat until acceptable.

## Safety rules

- Apply small parameter changes between iterations.
- Do not change unrelated keys.
- Keep a short changelog in the folder containing the config.yaml file.
