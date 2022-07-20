"""Add the QRCode and the BarCodes to a PDF in an additional page."""

import argparse
import io
import logging
import math
import os
import random
import subprocess  # nosec
import tempfile
from typing import List, Optional, Set, Tuple, TypedDict

import cv2
from PIL import Image
from PyPDF2 import PdfFileReader, PdfFileWriter
from pyzbar import pyzbar
from reportlab.lib.colors import Color
from reportlab.pdfgen import canvas
from weasyprint import CSS, HTML

_LOG = logging.getLogger(__name__)
Image.MAX_IMAGE_PIXELS = 500000000


class _Code(TypedDict):
    pos: int
    type: str
    data: str


class _PageCode(TypedDict):
    pos: int
    rect: List[Tuple[float, float]]


def _point(point: Tuple[int, int], deg_angle: float, width: int, height: int) -> Tuple[float, float]:
    assert -90 <= deg_angle <= 90
    angle = math.radians(deg_angle)
    diff_x = 0.0
    diff_y = 0.0
    if deg_angle < 0:
        diff_y = width * math.sin(angle)
    else:
        diff_x = height * math.sin(angle)
    x = point[0] - diff_x
    y = point[1] - diff_y
    return (
        x * math.cos(angle) + y * math.sin(angle),
        -x * math.sin(angle) + y * math.cos(angle),
    )


def _get_codes_with_open_cv(
    image: str,
    alpha: float,
    width: int,
    height: int,
    page_index: int,
    all_codes: Optional[List[_Code]] = None,
    added_codes: Optional[Set[str]] = None,
) -> List[_PageCode]:

    if added_codes is None:
        added_codes = set()
    if all_codes is None:
        all_codes = []
    codes: List[_PageCode] = []

    decoded_image = cv2.imread(image, flags=cv2.IMREAD_COLOR)
    if decoded_image is not None:
        detector = cv2.QRCodeDetector()
        retval, decoded_info, points, straight_qr_code = detector.detectAndDecodeMulti(decoded_image)
        if retval:
            if os.environ.get("PROGRESS", "FALSE") == "TRUE":
                base_path = os.path.dirname(image)
                filename = ".".join(os.path.basename(image).split(".")[:-1])
                suffix = random.randint(0, 1000)  # nosec
                for img_index, img in enumerate(straight_qr_code):
                    dest_filename = os.path.join(
                        base_path,
                        f"{filename}-qrcode-{page_index}-{suffix}-{img_index}.png",
                    )
                    cv2.imwrite(dest_filename, img)

            for index, data in enumerate(decoded_info):
                bbox = points[index]
                if bbox is not None and len(data) > 0 and data not in added_codes:
                    added_codes.add(data)
                    pos = len(all_codes)
                    all_codes.append(
                        {
                            "type": "QRCode",
                            "pos": pos,
                            "data": data,
                        }
                    )
                    codes.append(
                        {
                            "pos": pos,
                            "rect": [_point(p, alpha, width, height) for p in bbox],
                        }
                    )
        try:
            detector = cv2.barcode.BarcodeDetector()
            retval, decoded_info, decoded_type, points = detector.detectAndDecode(decoded_image)
            if retval:
                for index, data in enumerate(decoded_info):
                    bbox = points[index]
                    type_ = decoded_type[index]
                    if bbox is not None and len(data) > 0 and data not in added_codes:
                        added_codes.add(data)
                        pos = len(all_codes)
                        all_codes.append(
                            {
                                "type": type_[0] + type_[1:].lower(),
                                "pos": pos,
                                "data": data,
                            }
                        )
                        codes.append(
                            {
                                "pos": pos,
                                "rect": [_point(p, alpha, width, height) for p in bbox],
                            }
                        )
        except Exception:
            _LOG.warning("Open CV barcode decoder not available")

    return codes


def _get_codes_with_z_bar(
    image: str,
    alpha: float,
    width: int,
    height: int,
    all_codes: Optional[List[_Code]] = None,
    added_codes: Optional[Set[str]] = None,
) -> List[_PageCode]:

    if added_codes is None:
        added_codes = set()
    if all_codes is None:
        all_codes = []
    codes: List[_PageCode] = []

    img = Image.open(image)
    for output in pyzbar.decode(img):
        if output.data.decode().replace("\\n", "\n") not in added_codes:
            added_codes.add(output.data.decode().replace("\\n", "\n"))
            pos = len(all_codes)
            all_codes.append(
                {
                    "type": "QR code"
                    if output.type == "QRCODE"
                    else output.type[0] + output.type[1:].lower(),
                    "pos": pos,
                    "data": output.data.decode().replace("\\n", "\n"),
                }
            )
            codes.append(
                {
                    "pos": pos,
                    "rect": [_point((p.x, p.y), alpha, width, height) for p in output.polygon],
                }
            )

    return codes


def add_codes(
    input_filename: str,
    output_filename: str,
    dpi: float = 200,
    pdf_dpi: float = 72,
    font_size: float = 16,
    font_name: str = "Helvetica-Bold",
    margin_left: float = 2,
    margin_top: float = 0,
) -> None:
    """Add the QRCode and the BarCodes to a PDF in an additional page."""
    all_codes: List[_Code] = []
    added_codes: Set[str] = set()

    with open(input_filename, "rb") as input_file:
        existing_pdf = PdfFileReader(input_file)
        output_pdf = PdfFileWriter()
        for index, page in enumerate(existing_pdf.pages):
            # Get the QR code from the page
            image = f"img-{index}.png"
            subprocess.run(  # nosec
                [
                    "gm",
                    "convert",
                    "-density",
                    str(dpi),
                    f"{input_filename}[{index}]",
                    image,
                ],
                check=True,
            )
            img0 = Image.open(image)

            codes: List[_PageCode] = []
            codes += _get_codes_with_z_bar(image, 0, img0.width, img0.height, all_codes, added_codes)
            for angle in range(-10, 11, 2):
                subprocess.run(  # nosec
                    [
                        "gm",
                        "convert",
                        "-density",
                        str(dpi),
                        "-rotate",
                        str(angle),
                        f"{input_filename}[{index}]",
                        image,
                    ],
                    check=True,
                )
                codes += _get_codes_with_z_bar(image, angle, img0.width, img0.height, all_codes, added_codes)
            # codes += _get_codes_with_open_cv(image, 0, img0.width,  img0.height, all_codes, added_codes)

            if codes:
                packet = io.BytesIO()
                can = canvas.Canvas(
                    packet, pagesize=(page.mediabox.width, page.mediabox.height), bottomup=False
                )
                for code in codes:
                    can.setFillColor(Color(1, 1, 1, alpha=0.7))
                    path = can.beginPath()
                    path.moveTo(code["rect"][0][0] / dpi * pdf_dpi, code["rect"][0][1] / dpi * pdf_dpi)
                    for point in code["rect"][1:]:
                        path.lineTo(point[0] / dpi * pdf_dpi, point[1] / dpi * pdf_dpi)
                    path.close()
                    can.drawPath(path, stroke=0, fill=1)

                    can.setFillColorRGB(0, 0, 0)
                    can.setFont(font_name, font_size)
                    can.drawString(
                        min((p[0] for p in code["rect"])) / dpi * pdf_dpi + margin_left,
                        min((p[1] for p in code["rect"])) / dpi * pdf_dpi + font_size + 0 + margin_top,
                        str(code["pos"]),
                    )

                can.save()
                # move to the beginning of the StringIO buffer
                packet.seek(0)

                # Create a new PDF with Reportlab
                new_pdf = PdfFileReader(packet)

                page.mergePage(new_pdf.getPage(0))
            output_pdf.addPage(page)

        with tempfile.NamedTemporaryFile(suffix=".pdf") as dest_1, tempfile.NamedTemporaryFile(
            suffix=".pdf"
        ) as dest_2:
            # Finally, write "output" to a real file

            with open(dest_1.name, "wb") as output_stream:
                output_pdf.write(output_stream)

            for code_ in all_codes:
                code_["data"] = code_["data"].replace("\n", "<br />")
            sections = [
                f"<h2>{code_['type']} [{code_['pos']}]</h2><p>{code_['data']}</p>" for code_ in all_codes
            ]

            html = HTML(
                string=f"""<html>
            <head>
                <meta charset="utf-8">
            </head>
            <body>
                <section id="heading">
                    <h4>QRCode and Barcode</h4>
                </section>
                {'<hr />'.join(sections)}
            </body>
            </html>"""
            )

            css = CSS(string="@page { size: A4; margin: 2cm } P { font-size: 5pt; font-family: 'sans'; }")

            html.write_pdf(dest_2.name, stylesheets=[css])

            subprocess.run(  # nosec
                ["pdftk", dest_1.name, dest_2.name, "output", output_filename, "compress"], check=True
            )


def main() -> None:
    """Add the QRCode and the BarCodes to a PDF in an additional page."""
    arg_parser = argparse.ArgumentParser("Add the QRCode and the BarCodes to a PDF in an additional page")
    arg_parser.add_argument(
        "--dpi",
        help="The DPI used in the intermediate image to detect the QR code and the BarCode",
        type=int,
        default=300,
    )
    arg_parser.add_argument("--pdf-dpi", help="The DPI used in the PDF", type=int, default=72)
    arg_parser.add_argument(
        "--font-size", help="The font size used in the PDF to add the number", type=int, default=10
    )
    arg_parser.add_argument(
        "--margin-left", help="The margin left used in the PDF to add the number", type=int, default=2
    )
    arg_parser.add_argument(
        "--margin-top", help="The margin top used in the PDF to add the number", type=int, default=0
    )
    arg_parser.add_argument("input_filename", help="The input PDF filename")
    arg_parser.add_argument("output_filename", help="The output PDF filename")
    args = arg_parser.parse_args()

    add_codes(
        args.input_filename,
        args.output_filename,
        args.dpi,
        args.pdf_dpi,
        args.font_size,
        args.margin_left,
        args.margin_top,
    )
