"""Add the QRCode and the BarCodes to a PDF in an additional page."""

import argparse
import io
import logging
import os
import random
import subprocess  # nosec
import tempfile
from typing import List, Set, TypedDict

import cv2
from PIL import Image
from PyPDF2 import PdfFileReader, PdfFileWriter
from pyzbar import pyzbar
from reportlab.lib.colors import Color
from reportlab.pdfgen import canvas
from weasyprint import CSS, HTML

_LOG = logging.getLogger(__name__)


class _Code(TypedDict):
    pos: int
    type: str
    data: str


class _PageCode(TypedDict):
    pos: int
    top: int
    left: int
    width: int
    height: int


def add_codes(
    input_filename: str,
    output_filename: str,
    dpi: int = 300,
    pdf_dpi: int = 72,
    font_size: int = 16,
    margin_left: int = 2,
    margin_top: int = 0,
) -> None:
    """Add the QRCode and the BarCodes to a PDF in an additional page."""
    all_codes: List[_Code] = []
    added_codes: Set[str] = set()

    with open(input_filename, "rb") as input_file:
        existing_pdf = PdfFileReader(input_file)
        output_pdf = PdfFileWriter()
        for index, page in enumerate(existing_pdf.pages):
            codes: List[_PageCode] = []

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
                            "left": output.rect.left,
                            "top": output.rect.top,
                            "width": output.rect.width,
                            "height": output.rect.height,
                        }
                    )

            decoded_image = cv2.imread(image, flags=cv2.IMREAD_COLOR)
            if decoded_image is not None:
                detector = cv2.QRCodeDetector()
                retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(decoded_image)
                if retval:
                    if os.environ.get("PROGRESS", "FALSE") == "TRUE":
                        basepath = os.path.dirname(input_filename)
                        filename = ".".join(os.path.basename(input_filename).split(".")[:-1])
                        suffix = random.randint(0, 1000)  # nosec
                        for img_index, img in enumerate(straight_qrcode):
                            dest_filename = os.path.join(
                                basepath,
                                f"{filename}-qrcode-{index}-{suffix}-{img_index}.png",
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
                            bbox = bbox[0]
                            codes.append(
                                {
                                    "pos": pos,
                                    "left": bbox[0][0],
                                    "top": bbox[0][1],
                                    "width": bbox[3][0] - bbox[0][0],
                                    "height": bbox[3][1] - bbox[0][1],
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
                                bbox = bbox[0]
                                codes.append(
                                    {
                                        "pos": pos,
                                        "left": bbox[0][0],
                                        "top": bbox[0][1],
                                        "width": bbox[3][0] - bbox[0][0],
                                        "height": bbox[3][1] - bbox[0][1],
                                    }
                                )
                except Exception:
                    _LOG.warning("Open CV barcode decoder not available")

            if codes:
                packet = io.BytesIO()
                can = canvas.Canvas(
                    packet, pagesize=(page.mediabox.width, page.mediabox.height), bottomup=False
                )
                for code in codes:
                    min_size = 10
                    width = max(code["width"] / dpi * pdf_dpi, min_size)
                    height = max(code["height"] / dpi * pdf_dpi, min_size)

                    can.setFillColor(Color(1, 1, 1, alpha=0.7))
                    can.rect(
                        code["left"] / dpi * pdf_dpi,
                        code["top"] / dpi * pdf_dpi,
                        width + 1,
                        height + 1,
                        fill=1,
                        stroke=0,
                    )
                    can.setFillColorRGB(0, 0, 0)

                    can.setFont("Courier", font_size)
                    can.drawString(
                        code["left"] / dpi * pdf_dpi + margin_left,
                        code["top"] / dpi * pdf_dpi + font_size + 0 + margin_top,
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

            sections = [f"<h2>{code['type']} [{code['pos']}]</h2><p>{code['data']}</p>" for code in all_codes]

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
