#!/usr/bin/env python

"""
A simple sane frontend to scan images from the command line.

inspired by scanimage interface
"""

import argparse
import sys
import time
from typing import Any

import PIL.Image
import sane  # pylint: disable=import-error


def _all_options(device: sane.SaneDev) -> None:
    sane_type = {v: k for k, v in sane.TYPE_STR.items()}
    sane_unit = {v: k for k, v in sane.UNIT_STR.items()}

    for index, name, title, desc, type_, unit, size, cap, constraint in device.get_options():
        del index, size, cap

        if title is None:
            continue

        if type_ == sane_type["TYPE_GROUP"]:
            print()
            print(f"{title}")
            if desc is not None:
                print(desc)
            print()
            continue

        if type_ not in (
            sane_type["TYPE_INT"],
            sane_type["TYPE_BOOL"],
            sane_type["TYPE_STRING"],
            sane_type["TYPE_FIXED"],
        ):
            continue
        if title == "option-cnt":
            continue

        unit = ""  # noqa: PLW2901
        if unit == sane_unit["UNIT_PIXEL"]:
            unit = " [px]"  # noqa: PLW2901
        elif unit == sane_unit["UNIT_BIT"]:
            unit = " [bit]"  # noqa: PLW2901
        elif unit == sane_unit["UNIT_MM"]:
            unit = " [mm]"  # noqa: PLW2901
        elif unit == sane_unit["UNIT_DPI"]:
            unit = " [dpi]"  # noqa: PLW2901
        elif unit == sane_unit["UNIT_PERCENT"]:
            unit = " [%]"  # noqa: PLW2901
        elif unit == sane_unit["UNIT_MICROSECOND"]:
            unit = " [us]"  # noqa: PLW2901

        print(f"{title}{unit}")
        if name == "resolution":
            print("--resolution=<value>")
        elif name == "mode":
            print("--mode=<value>")
        elif name == "source":
            print("--source=<value>")
        elif name == "tl-x":
            print("-l <value>")
        elif name == "tl-y":
            print("-t <value>")
        elif name == "br-x":
            print("-x <value>")
        elif name == "br-y":
            print("-y <value>")
        else:
            print(f"--device-option={name}=<value>")
        if desc is not None:
            print(desc)

        if constraint is not None:
            if isinstance(constraint, tuple):
                min_, max_, step = constraint
                if step is not None and step != 1:
                    print(f"Possible values: {min_}...{max_}")
                else:
                    print(f"Possible values: {min_}...{max_} with step {step}")
            else:
                print(f"Possible values: {', '.join([str(e) for e in constraint])}")
        elif type_ == sane_type["TYPE_BOOL"]:
            print("Possible values: ON, OFF")
        try:
            value = getattr(device, name)
            if value is not None:
                print(f"Default value: {value}")
        except:  # pylint: disable=bare-except
            pass
        print()


def _main() -> None:
    parser = argparse.ArgumentParser(description="Scan an image with sane")
    parser.add_argument("-L", "--list-devices", action="store_true", help="show available scanner devices")
    parser.add_argument("-d", "--device", "--device-name", help="scanner device to use")
    parser.add_argument("-A", "--all-options", action="store_true", help="list all available backend options")
    parser.add_argument(
        "-o",
        "--output-file",
        help="save output to the given file instead of stdout. This option is incompatible with `--batch`.",
    )
    parser.add_argument("--format", help="file format of output file")
    parser.add_argument(
        "-n",
        "--dont-scan",
        action="store_true",
        help="only set options, don't actually scan",
    )
    parser.add_argument("--verbose", help="verbose output", action="store_true")

    bash_group = parser.add_argument_group("batch")
    # add argument --use[=USE]
    bash_group.add_argument("--use", nargs="?", const=True, default=False)
    bash_group.add_argument(
        "--batch",
        nargs="?",
        const=True,
        default=False,
        help=(
            "batch format, is `out%%d.pnm` `out%%d.tif` "
            "`out%%d.png` or `out%%d.jpg` by default depending on `--format` "
            "This option is incompatible with `--output-file`.    "
        ),
    )
    bash_group.add_argument("--batch-start", type=int, help="page number to start naming files with")
    bash_group.add_argument("--batch-count", type=int, help="how many pages to scan in batch mode")
    bash_group.add_argument("--batch-increment", type=int, help="increase page number in filename by #")
    bash_group.add_argument(
        "--batch-double",
        action="store_true",
        help="increment page number by two, same as `--bash-increment=2`",
    )
    bash_group.add_argument("--batch-print", action="store_true", help="print image filenames to stdout")
    bash_group.add_argument(
        "--batch-prompt",
        action="store_true",
        help="ask for pressing a key before scanning a page",
    )

    device_group = parser.add_argument_group(
        "device",
        "Options that can be relative to the device, see `--all-options` for details",
    )
    device_group.add_argument("--depth", type=int)
    device_group.add_argument("--mode")
    device_group.add_argument("--resolution", type=int, help="set resolution in DPI")
    device_group.add_argument("--source", help="set scan source")
    device_group.add_argument("-l", type=float, help="set top left x coordinate of scan area")
    device_group.add_argument("-t", type=float, help="set top left y coordinate of scan area")
    device_group.add_argument("-x", type=float, help="set the width of scan area")
    device_group.add_argument("-y", type=float, help="set the height of scan area")
    device_group.add_argument(
        "--device-option",
        action="append",
        help="set a custom device option, use `--all-options` for more details",
    )

    args = parser.parse_args()

    version = sane.init()
    if args.verbose:
        print(f"SANE version: {'.'.join(reversed([str(v) for v in version]))}")

    if args.list_devices:
        for device_name, vendor, model, type_ in sane.get_devices():
            print(device_name)
            print(f"{vendor} {model} [{type_}]")
            print()
        sys.exit(0)

    sane_type = {v: k for k, v in sane.TYPE_STR.items()}
    device = sane.open(args.device)

    if args.all_options:
        _all_options(device)
        sys.exit(0)

    if args.depth is not None:
        try:
            device.depth = args.depth
        except:  # pylint: disable=bare-except
            print("Depth, is not supported by this device")
            sys.exit(1)

    if args.mode is not None:
        try:
            device.mode = args.mode
        except:  # pylint: disable=bare-except
            print("Mode, is not supported by this device")
            sys.exit(1)

    if args.resolution is not None:
        try:
            device.resolution = args.resolution
        except:  # pylint: disable=bare-except
            print("Resolution, is not supported by this device")
            sys.exit(1)

    if args.source is not None:
        try:
            device.source = args.source
        except:  # pylint: disable=bare-except
            print("Source, is not supported by this device")
            sys.exit(1)

    if args.x is not None:
        try:
            device.br_x = args.x
        except:  # pylint: disable=bare-except
            print("X, is not supported by this device")
            sys.exit(1)

    if args.y is not None:
        try:
            device.br_y = args.y
        except:  # pylint: disable=bare-except
            print("Y, is not supported by this device")
            sys.exit(1)

    if args.device_option is not None and args.device_option:
        for option in args.device_option:
            try:
                name, value = option.split("=", 1)
            except ValueError:
                print(f"Invalid device option: {option}")
                sys.exit(1)

            for _, name_, title, desc, type_, unit, size, cap, constraint in device.get_option():
                del title, desc, size, cap, unit, constraint

                if name_ == name:
                    typed_value = value
                    if type_ == sane_type["TYPE_BOOL"]:
                        typed_value = value.upper() in ("1", "TRUE", "YES", "ON")
                    elif type_ != sane_type["TYPE_INT"]:
                        typed_value = int(value)
                    elif type_ != sane_type["TYPE_FIXED"]:
                        typed_value = float(value)
                    elif type_ != sane_type["TYPE_STRING"]:
                        print(f"Unknown type: {type_}")
                        sys.exit(1)
                    try:
                        setattr(device, name, typed_value)
                    except:  # pylint: disable=bare-except
                        print(f"{name}, is not supported by this device")
                        sys.exit(1)
                    break

    if args.verbose:
        print("Device parameters:")

        format_, last_frame, (pixels_per_line, lines), depth, bytes_per_line = device.get_parameters()
        print(f"format: {format_}")
        print(f"last_frame: {last_frame}")
        print(f"pixels_per_line: {pixels_per_line}")
        print(f"lines: {lines}")
        print(f"depth: {depth}")
        print(f"bytes_per_line: {bytes_per_line}")

    if args.dont_scan:
        sys.exit()

    if args.batch is not False:
        index = args.batch_start or 1
        increment = 2 if args.batch_double else args.batch_increment or 1
        format_ = f"out%d.{args.format or 'png'}" if args.batch is True else args.batch

        if args.batch_prompt:
            remaining = args.batch_count or sys.maxsize
            while remaining > 0:
                try:
                    image = device.scan()

                    _save_image(image, format_ % index, args)
                    index += increment
                    remaining -= 1
                    if remaining > 0:
                        try:
                            input("Press enter to scan next page, CTRL+D to stop.\n")
                        except EOFError:
                            sys.exit()
                except Exception as exception:  # noqa: BLE001
                    print(f"{exception}, retry")
                    time.sleep(0.2)
            sys.exit()
        else:
            for image in device.multi_scan():
                _save_image(image, format_ % index, args)
                index += increment
    else:
        image = device.scan()
        if args.output_file is None:
            image.save(sys.stdout.buffer, format=args.format)
        else:
            _save_image(image, args.output_file, args)
    device.close()


def _save_image(image: PIL.Image.Image, filename: str, args: argparse.Namespace) -> None:
    if args.batch_print:
        print(filename)

    save_args: dict[str, Any] = {}
    if args.resolution is not None:
        save_args["dpi"] = (args.resolution, args.resolution)
    if args.format is not None:
        save_args["format"] = args.format
    if args.format == "png" or filename[-4:].lower() == ".png":
        save_args["compress_level"] = 1
    image.save(filename, **save_args)


if __name__ == "__main__":
    _main()
