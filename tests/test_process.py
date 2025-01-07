import glob
import os.path
import re
import shutil
import subprocess
from typing import Any

import cv2
import nbformat
import pikepdf
import pytest
import skimage.color
import skimage.io
from c2cwsgiutils.acceptance.image import check_image, check_image_file
from nbconvert.preprocessors import ExecutePreprocessor

from scan_to_paperless import code, process, process_utils

REGENERATE = False


def load_image(image_name) -> None:
    return cv2.imread(os.path.join(os.path.dirname(__file__), image_name))


def test_should_not_commit() -> None:
    assert REGENERATE is False


# @pytest.mark.skip(reason="for test")
def test_find_lines() -> None:
    lines = process.find_lines(load_image("limit-lines-1.png"), True, {})
    assert 1821 in [l[0] for l in lines]


# @pytest.mark.skip(reason="for test")
def test_find_limit_contour() -> None:
    context = process_utils.Context({"args": {}}, {})
    context.image = load_image("limit-contour-1.png")
    contours = process.find_contours(context.image, context, "limit", {})
    limits = process.find_limit_contour(context.image, True, contours)
    assert limits == [1589]


# @pytest.mark.skip(reason="for test")
def test_crop() -> None:
    image = load_image("image-1.png")
    root_folder = "/results/crop"
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    check_image(
        root_folder,
        cv2.cvtColor(process_utils.crop_image(image, 100, 0, 100, 300, (255, 255, 255)), cv2.COLOR_BGR2RGB),
        os.path.join(os.path.dirname(__file__), "crop-1.expected.png"),
        generate_expected_image=REGENERATE,
    )
    check_image(
        root_folder,
        cv2.cvtColor(process_utils.crop_image(image, 0, 100, 300, 100, (255, 255, 255)), cv2.COLOR_BGR2RGB),
        os.path.join(os.path.dirname(__file__), "crop-2.expected.png"),
        generate_expected_image=REGENERATE,
    )
    check_image(
        root_folder,
        cv2.cvtColor(
            process_utils.crop_image(image, 100, -100, 100, 200, (255, 255, 255)), cv2.COLOR_BGR2RGB
        ),
        os.path.join(os.path.dirname(__file__), "crop-3.expected.png"),
        generate_expected_image=REGENERATE,
    )
    check_image(
        root_folder,
        cv2.cvtColor(
            process_utils.crop_image(image, -100, 100, 200, 100, (255, 255, 255)), cv2.COLOR_BGR2RGB
        ),
        os.path.join(os.path.dirname(__file__), "crop-4.expected.png"),
        generate_expected_image=REGENERATE,
    )
    check_image(
        root_folder,
        cv2.cvtColor(process_utils.crop_image(image, 100, 200, 100, 200, (255, 255, 255)), cv2.COLOR_BGR2RGB),
        os.path.join(os.path.dirname(__file__), "crop-5.expected.png"),
        generate_expected_image=REGENERATE,
    )
    check_image(
        root_folder,
        cv2.cvtColor(process_utils.crop_image(image, 200, 100, 200, 100, (255, 255, 255)), cv2.COLOR_BGR2RGB),
        os.path.join(os.path.dirname(__file__), "crop-6.expected.png"),
        generate_expected_image=REGENERATE,
    )
    shutil.rmtree(root_folder)


# @pytest.mark.skip(reason="for test")
def test_rotate() -> None:
    image = load_image("image-1.png")
    root_folder = "/results/rotate"
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    image = process_utils.crop_image(image, 0, 50, 300, 200, (255, 255, 255))
    check_image(
        root_folder,
        cv2.cvtColor(process_utils.rotate_image(image, 10, (255, 255, 255)), cv2.COLOR_BGR2RGB),
        os.path.join(os.path.dirname(__file__), "rotate-1.expected.png"),
        generate_expected_image=REGENERATE,
    )
    check_image(
        root_folder,
        cv2.cvtColor(process_utils.rotate_image(image, -10, (255, 255, 255)), cv2.COLOR_BGR2RGB),
        os.path.join(os.path.dirname(__file__), "rotate-2.expected.png"),
        generate_expected_image=REGENERATE,
    )
    check_image(
        root_folder,
        cv2.cvtColor(process_utils.rotate_image(image, 90, (255, 255, 255)), cv2.COLOR_BGR2RGB),
        os.path.join(os.path.dirname(__file__), "rotate-3.expected.png"),
        generate_expected_image=REGENERATE,
    )
    check_image(
        root_folder,
        cv2.cvtColor(process_utils.rotate_image(image, -90, (255, 255, 255)), cv2.COLOR_BGR2RGB),
        os.path.join(os.path.dirname(__file__), "rotate-4.expected.png"),
        generate_expected_image=REGENERATE,
    )
    check_image(
        root_folder,
        cv2.cvtColor(process_utils.rotate_image(image, 270, (255, 255, 255)), cv2.COLOR_BGR2RGB),
        os.path.join(os.path.dirname(__file__), "rotate-4.expected.png"),
        generate_expected_image=REGENERATE,
    )
    check_image(
        root_folder,
        cv2.cvtColor(process_utils.rotate_image(image, 180, (255, 255, 255)), cv2.COLOR_BGR2RGB),
        os.path.join(os.path.dirname(__file__), "rotate-5.expected.png"),
        generate_expected_image=REGENERATE,
    )
    shutil.rmtree(root_folder)


def init_test() -> None:
    os.environ["PROGRESS"] = "FALSE"
    os.environ["TIME"] = "TRUE"
    os.environ["SCAN_CODES_FOLDER"] = "/results"
    shutil.copyfile(os.path.join(os.path.dirname(__file__), "mask.png"), "/results/mask.png")


# @pytest.mark.skip(reason="for test")
@pytest.mark.parametrize(
    "type_,limit,better_value,cut_white",
    [
        pytest.param(
            "lines",
            {"name": "VL0", "type": "line detection", "value": 979, "vertical": True, "margin": 0},
            979,
            240,
            id="lines",
        ),
        pytest.param(
            "contour",
            {"name": "VC0", "type": "contour detection", "value": 864, "vertical": True, "margin": 0},
            864,
            200,
            id="contour",
        ),
    ],
)
async def test_assisted_split_full(type_, limit, better_value, cut_white) -> None:
    init_test()
    #    os.environ['PROGRESS'] = 'TRUE'
    root_folder = f"/results/assisted-split-full-{type_}"
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), f"limit-{type_}-all-1.png"),
        os.path.join(root_folder, "image-1.png"),
    )
    config = {
        "args": {
            "assisted_split": True,
            "tesseract": {
                "enabled": False,
            },
            "deskew": {
                "num_angles": 179,
            },
            "crop": {"contour": {"threshold_block_size": 20, "threshold_value_c": 20}},
            "cut_white": cut_white,
            "auto_rotate": {"enabled": False},
        },
    }
    step = {
        "sources": ["image-1.png"],
    }
    config_file_name = os.path.join(root_folder, "config.yaml")
    step = await process.transform(config, step, config_file_name, root_folder)
    assert step["name"] == "split"
    images = step["sources"]
    assert len(images) == 1
    assert os.path.basename(images[0]) == config["assisted_split"][0]["image"]
    print(f"Compare '{images[0]}' with expected image 'assisted-split-{type_}-1.expected.png'.")
    check_image_file(
        root_folder,
        images[0],
        os.path.join(os.path.dirname(__file__), f"assisted-split-{type_}-1.expected.png"),
        generate_expected_image=REGENERATE,
    )
    limits = [item for item in config["assisted_split"][0]["limits"] if item["vertical"]]
    assert not [item for item in limits if item["name"] == "C"], "We shouldn't have center limit"
    limits = [item for item in limits if item["name"] == limit["name"]]
    assert limits == [limit], limits
    config["assisted_split"][0]["limits"] = limits
    config["assisted_split"][0]["limits"][0]["value"] = better_value
    step = await process.split(config, step, root_folder)
    assert len(step["sources"]) == 2
    assert step["name"] == "finalize"
    print(f"Compare '{step['sources'][0]}' with expected image 'assisted-split-{type_}-3.expected.png'.")
    check_image_file(
        root_folder,
        step["sources"][0],
        os.path.join(os.path.dirname(__file__), f"assisted-split-{type_}-3.expected.png"),
        generate_expected_image=REGENERATE,
    )
    print(f"Compare '{step['sources'][1]}' with expected image 'assisted-split-{type_}-4.expected.png'.")
    check_image_file(
        root_folder,
        step["sources"][1],
        os.path.join(os.path.dirname(__file__), f"assisted-split-{type_}-4.expected.png"),
        generate_expected_image=REGENERATE,
    )
    process.finalize(config, step, root_folder)
    pdfinfo = process.output(
        ["pdfinfo", os.path.join("/results", f"{os.path.basename(root_folder)}.pdf")]
    ).split("\n")
    regex = re.compile(r"([a-zA-Z ]+): +(.*)")
    pdfinfo = [regex.match(e) for e in pdfinfo]
    pdfinfo = dict([e.groups() for e in pdfinfo if e is not None])
    assert pdfinfo["Pages"] == "2"
    process.call(
        [
            "gm",
            "convert",
            os.path.join("/results", f"{os.path.basename(root_folder)}.pdf"),
            "+adjoin",
            os.path.join("/results", f"{os.path.basename(root_folder)}.png"),
        ]
    )
    print(
        f"Compare '{os.path.join('/results', f'{os.path.basename(root_folder)}.png')}' with expected image 'assisted-split-{type_}-5.expected.png'."
    )
    check_image_file(
        root_folder,
        os.path.join("/results", f"{os.path.basename(root_folder)}.png"),
        os.path.join(os.path.dirname(__file__), f"assisted-split-{type_}-5.expected.png"),
        generate_expected_image=REGENERATE,
    )
    shutil.rmtree(root_folder)


# @pytest.mark.skip(reason="for test")
async def test_assisted_split_join_full() -> None:
    init_test()
    #    os.environ['PROGRESS'] = 'TRUE'
    root_folder = "/results/assisted-split-join-full"
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    for number in (1, 2):
        shutil.copyfile(
            os.path.join(os.path.dirname(__file__), f"split-join-{number}.png"),
            os.path.join(root_folder, f"image-{number}.png"),
        )

    config = {
        "args": {
            "assisted_split": True,
            "level": {"value": True},
            "tesseract": {"enabled": False},
            "deskew": {"num_angles": 179},
        },
        "destination": os.path.join(root_folder, "final.pdf"),
    }
    step = {
        "sources": ["image-1.png", "image-2.png"],
    }
    config_file_name = os.path.join(root_folder, "config.yaml")
    step = await process.transform(config, step, config_file_name, root_folder)
    assert step["name"] == "split"
    images = step["sources"]
    assert os.path.basename(images[0]) == config["assisted_split"][0]["image"]
    assert len(images) == 2
    for number, elements in enumerate(
        [
            ({"value": 738, "vertical": True, "margin": 0}, ["-", "1.2"]),
            ({"value": 3300, "vertical": True, "margin": 0}, ["1.1", "-"]),
        ]
    ):
        limit, destinations = elements
        config["assisted_split"][number]["limits"] = [limit]
        config["assisted_split"][number]["destinations"] = destinations
    step = await process.split(config, step, root_folder)
    assert step["name"] == "finalize"
    assert len(step["sources"]) == 1
    print(f"Compare '{step['sources'][0]}' with expected image 'assisted-split-join-1.expected.png'.")
    check_image_file(
        root_folder,
        step["sources"][0],
        os.path.join(os.path.dirname(__file__), "assisted-split-join-1.expected.png"),
        generate_expected_image=REGENERATE,
    )

    process.finalize(config, step, root_folder)
    pdfinfo = process.output(
        ["pdfinfo", os.path.join("/results", f"{os.path.basename(root_folder)}.pdf")]
    ).split("\n")
    regex = re.compile(r"([a-zA-Z ]+): +(.*)")
    pdfinfo = [regex.match(e) for e in pdfinfo]
    pdfinfo = dict([e.groups() for e in pdfinfo if e is not None])
    assert pdfinfo["Pages"] == "1"
    process.call(
        [
            "gm",
            "convert",
            os.path.join("/results", f"{os.path.basename(root_folder)}.pdf"),
            "+adjoin",
            os.path.join("/results", f"{os.path.basename(root_folder)}.png"),
        ]
    )
    print(
        f"Compare '{os.path.join('/results', f'{os.path.basename(root_folder)}.png')}' with expected image 'assisted-split-join-2.expected.png'."
    )
    check_image_file(
        root_folder,
        os.path.join("/results", f"{os.path.basename(root_folder)}.png"),
        os.path.join(os.path.dirname(__file__), "assisted-split-join-2.expected.png"),
        generate_expected_image=REGENERATE,
    )
    shutil.rmtree(root_folder)


# @pytest.mark.skip(reason="for test")
async def test_assisted_split_booth() -> None:
    init_test()
    #    os.environ['PROGRESS'] = 'TRUE'
    root_folder = "/results/assisted-split-booth"
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), "image-1.png"),
        os.path.join(root_folder, "image-1.png"),
    )

    config = {
        "args": {
            "assisted_split": True,
            "level": {"value": False},
            "crop": {
                "enabled": False,
            },
            "tesseract": {
                "enabled": False,
            },
        },
        "destination": os.path.join(root_folder, "final.pdf"),
        "assisted_split": [
            {
                "image": os.path.join(root_folder, "image-1.png"),
                "source": os.path.join(os.path.dirname(__file__), "image-1.png"),
                "limits": [
                    {"value": 150, "vertical": True, "margin": 0},
                    {"value": 150, "vertical": False, "margin": 0},
                ],
                "destinations": ["1", "2", "3", "4"],
            },
        ],
    }
    step = {
        "name": "split",
        "sources": ["image-1.png"],
    }
    step = await process.split(config, step, root_folder)
    assert step["name"] == "finalize"
    assert len(step["sources"]) == 4
    print(f"Compare '{step['sources'][0]}' with expected image 'assisted-split-booth-1.expected.png'.")
    check_image_file(
        root_folder,
        step["sources"][0],
        os.path.join(os.path.dirname(__file__), "assisted-split-booth-1.expected.png"),
        generate_expected_image=REGENERATE,
    )
    print(f"Compare '{step['sources'][1]}' with expected image 'assisted-split-booth-2.expected.png'.")
    check_image_file(
        root_folder,
        step["sources"][1],
        os.path.join(os.path.dirname(__file__), "assisted-split-booth-2.expected.png"),
        generate_expected_image=REGENERATE,
    )
    print(f"Compare '{step['sources'][2]}' with expected image 'assisted-split-booth-3.expected.png'.")
    check_image_file(
        root_folder,
        step["sources"][2],
        os.path.join(os.path.dirname(__file__), "assisted-split-booth-3.expected.png"),
        generate_expected_image=REGENERATE,
    )
    print(f"Compare '{step['sources'][3]}' with expected image 'assisted-split-booth-4.expected.png'.")
    check_image_file(
        root_folder,
        step["sources"][3],
        os.path.join(os.path.dirname(__file__), "assisted-split-booth-4.expected.png"),
        generate_expected_image=REGENERATE,
    )
    shutil.rmtree(root_folder)


# @pytest.mark.skip(reason="for test")
@pytest.mark.parametrize("progress", ["FALSE", "TRUE"])
async def test_full(progress) -> None:
    init_test()
    os.environ["PROGRESS"] = progress
    root_folder = f"/results/full-{progress}"
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    config = {
        "args": {"level": {"value": 15}, "cut_white": 200},
        "images": [os.path.join(os.path.dirname(__file__), "all-1.png")],
    }
    step = {"sources": [os.path.join(os.path.dirname(__file__), "all-1.png")]}
    step = await process.transform(config, step, "/tmp/test-config.yaml", root_folder)
    assert len(step["sources"]) == 1
    print(f"Compare '{step['sources'][0]}' with expected image 'all-1.expected.png'.")
    check_image_file(
        root_folder,
        step["sources"][0],
        os.path.join(os.path.dirname(__file__), "all-1.expected.png"),
        generate_expected_image=REGENERATE,
    )
    assert os.path.exists(os.path.join(os.path.dirname(__file__), "all-1-skew-corrected.png")), os.listdir(
        os.path.dirname(__file__)
    )

    if progress == "TRUE":
        assert os.path.exists(os.path.join(root_folder, "1-level/all-1.png"))
    else:
        assert not os.path.exists(os.path.join(root_folder, "1-level"))

    assert step["name"] == "finalize"
    process.finalize(config, step, root_folder)

    pdf_filename = os.path.join("/results", f"{os.path.basename(root_folder)}.pdf")

    creator_scan_tp_paperless_re = re.compile(r"^Scan to Paperless 1.[0-9]+.[0-9]+\+[0-9]+$")
    creator_tesseract_re = re.compile(r"^Tesseract [0-9]+.[0-9]+.[0-9]+$")
    with pikepdf.open(pdf_filename) as pdf_, pdf_.open_metadata() as meta:
        creator = meta["{http://purl.org/dc/elements/1.1/}creator"]
        assert len(creator) == 2, creator
        assert creator_scan_tp_paperless_re.match(creator[0]), creator
        assert creator_tesseract_re.match(creator[1]), creator

    pdfinfo = process.output(["pdfinfo", pdf_filename]).split("\n")
    regex = re.compile(r"([a-zA-Z ]+): +(.*)")
    pdfinfo = [regex.match(e) for e in pdfinfo]
    pdfinfo = dict([e.groups() for e in pdfinfo if e is not None])
    assert pdfinfo["Pages"] == "1"
    process.call(
        [
            "gm",
            "convert",
            os.path.join("/results", f"{os.path.basename(root_folder)}.pdf"),
            "+adjoin",
            os.path.join("/results", f"{os.path.basename(root_folder)}.png"),
        ]
    )
    print(
        f"Compare '{os.path.join('/results', f'{os.path.basename(root_folder)}.png')}' with expected image 'all-2.expected.png'."
    )
    check_image_file(
        root_folder,
        os.path.join("/results", f"{os.path.basename(root_folder)}.png"),
        os.path.join(os.path.dirname(__file__), "all-2.expected.png"),
        generate_expected_image=REGENERATE,
    )
    shutil.rmtree(root_folder)


# @pytest.mark.skip(reason="for test")
async def test_credit_card_full() -> None:
    init_test()
    #    os.environ['PROGRESS'] = 'TRUE'
    root_folder = "/results/credit-card"
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    config = {
        "args": {
            "append_credit_card": True,
            "deskew": {"num_angles": 179},
            "cut_white": 200,
            "mask": {},
            "auto_rotate": {"enabled": False},
        },
    }
    step = {
        "sources": [
            os.path.join(os.path.dirname(__file__), "credit-card-1.png"),
            os.path.join(os.path.dirname(__file__), "credit-card-2.png"),
        ]
    }
    step = await process.transform(config, step, "/tmp/test-config.yaml", root_folder)
    assert len(step["sources"]) == 2
    assert step["name"] == "finalize"
    process.finalize(config, step, root_folder)
    pdfinfo = process.output(
        ["pdfinfo", os.path.join("/results", f"{os.path.basename(root_folder)}.pdf")]
    ).split("\n")
    regex = re.compile(r"([a-zA-Z ]+): +(.*)")
    pdfinfo = [regex.match(e) for e in pdfinfo]
    pdfinfo = dict([e.groups() for e in pdfinfo if e is not None])
    assert pdfinfo["Pages"] == "1"
    process.call(
        [
            "gm",
            "convert",
            os.path.join("/results", f"{os.path.basename(root_folder)}.pdf"),
            "+adjoin",
            os.path.join("/results", f"{os.path.basename(root_folder)}.png"),
        ]
    )
    print(
        f"Compare '{os.path.join('/results', f'{os.path.basename(root_folder)}.png')}' with expected image 'credit-card-1.expected.png'."
    )
    check_image_file(
        root_folder,
        os.path.join("/results", f"{os.path.basename(root_folder)}.png"),
        os.path.join(os.path.dirname(__file__), "credit-card-1.expected.png"),
        generate_expected_image=REGENERATE,
    )
    shutil.rmtree(root_folder)


# @pytest.mark.skip(reason="for test")
async def test_empty() -> None:
    init_test()
    #    os.environ['PROGRESS'] = 'TRUE'
    root_folder = "/results/empty"
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    config = {
        "args": {
            "level": {"value": True},
            "mask": {},
        }
    }
    step = {
        "sources": [
            os.path.join(os.path.dirname(__file__), "empty.png"),
        ]
    }
    step = await process.transform(config, step, "/tmp/test-config.yaml", root_folder)
    assert len(step["sources"]) == 0
    assert step["name"] == "finalize"
    shutil.rmtree(root_folder)


# @pytest.mark.skip(reason="for test")
@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize(
    "test,args", [pytest.param("600", {"dpi": 600, "deskew": {"num_angles": 179}}, id="600")]
)
async def test_custom_process(test: str, args: dict[str, Any]) -> None:
    init_test()
    root_folder = "/results/600"
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    config = {"args": args}
    step = {"sources": [os.path.join(os.path.dirname(__file__), f"{test}.png")]}
    step = await process.transform(config, step, "/tmp/test-config.yaml", root_folder)
    assert len(step["sources"]) == 1
    try:
        print(f"Compare '{step['sources'][0]}' with expected image '{test}.expected.png'.")
        check_image_file(
            root_folder,
            step["sources"][0],
            os.path.join(os.path.dirname(__file__), f"{test}.expected.png"),
            generate_expected_image=REGENERATE,
        )
    except ValueError:
        print(f"Compare '{step['sources'][0]}' with expected image '{test}-bis.expected.png'.")
        check_image_file(
            root_folder,
            step["sources"][0],
            os.path.join(os.path.dirname(__file__), f"{test}-bis.expected.png"),
            generate_expected_image=REGENERATE,
        )

    shutil.rmtree(root_folder)


# @pytest.mark.skip(reason="for test")
@pytest.mark.parametrize("name", ["qrcode", "qrbill", "qrbill2"])
def test_qr_code(name) -> None:
    init_test()
    code.add_codes(os.path.join(os.path.dirname(__file__), f"{name}.pdf"), f"/results/{name}.pdf")
    root_folder = "/results/qrcode"
    for page in range(2):
        subprocess.run(
            [
                "gm",
                "convert",
                "-density",
                "150",
                f"/results/{name}.pdf[{page}]",
                f"/results/{name}-{page}.png",
            ],
            check=True,
        )

    for page in range(2):
        image = skimage.io.imread(f"/results/{name}-{page}.png")
        if len(image.shape) == 2:
            image = skimage.color.gray2rgb(image)

        check_image(
            root_folder,
            image,
            os.path.join(os.path.dirname(__file__), f"{name}-{page}.expected.png"),
            generate_expected_image=REGENERATE,
        )


# @pytest.mark.skip(reason="for test")
def test_qr_code_metadata() -> None:
    init_test()
    code.add_codes(os.path.join(os.path.dirname(__file__), "qrbill.pdf"), "/results/qrbill.pdf")

    with pikepdf.open("/results/qrbill.pdf") as pdf:
        for k, v in {
            "/Title": "qrbill",
            "/CreationDate": "D:20220720213803",
            "/ModDate": "D:20220720213803",
            "/Producer": "GraphicsMagick 1.3.38 2022-03-26 Q16 http://www.GraphicsMagick.org/",
        }.items():
            assert pdf.docinfo[k] == pikepdf.objects.String(v)
        with pdf.open_metadata() as meta:
            assert (
                meta.get("{http://purl.org/dc/elements/1.1/}description")
                == """QR code [0]
SPC
0200
1
CH3908704016075473007
K
Robert Schneider AG
Rue du Lac 1268
2501 Biel


CH







5923.50
CHF
K
Pia-Maria Rutschmann-Schnyder
Grosse Marktgasse 28
9400 Rorschach


CH
SCOR
RF9720200227JS
20200227JS- - """
                + """
EPD

"""
            )


# @pytest.mark.skip(reason="for test")
def test_multi_code() -> None:
    init_test()
    code.add_codes(os.path.join(os.path.dirname(__file__), "qrbill-multi.pdf"), "/results/qrbill-multi.pdf")
    root_folder = "/results/qrcode"
    for page in range(3):
        subprocess.run(
            [
                "gm",
                "convert",
                "-density",
                "150",
                f"/results/qrbill-multi.pdf[{page}]",
                f"/results/qrbill-multi-{page}.png",
            ],
            check=True,
        )
    for page in range(3):
        image = skimage.io.imread(f"/results/qrbill-multi-{page}.png")
        if len(image.shape) == 2:
            image = skimage.color.gray2rgb(image)

        check_image(
            root_folder,
            image,
            os.path.join(os.path.dirname(__file__), f"qrbill-multi-{page}.expected.png"),
            generate_expected_image=REGENERATE,
        )


# @pytest.mark.skip(reason="for test")
async def test_tiff_jupyter() -> None:
    init_test()
    root_folder = "/results/tiff"
    source_folder = os.path.join(root_folder, "source")
    if not os.path.exists(source_folder):
        os.makedirs(source_folder)

    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), "image-1.tiff"),
        os.path.join(source_folder, "image-1.tiff"),
    )

    config = {
        "args": {},
        "destination": os.path.join(root_folder, "final.pdf"),
    }
    step = {
        "sources": ["source/image-1.tiff"],
    }
    config_file_name = os.path.join(root_folder, "config.yaml")
    step = await process.transform(config, step, config_file_name, root_folder)
    assert step["sources"] == ["/results/tiff/image-1.png"]
    assert list(glob.glob(f"{root_folder}/**/*.tiff")) == [os.path.join(root_folder, "source/image-1.tiff")]

    with open("/results/tiff/jupyter/jupyter.ipynb") as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": "/results/tiff/jupyter/"}})


# @pytest.mark.skip(reason="for test")
@pytest.mark.flaky(reruns=3, only_rerun="ValueError")
@pytest.mark.parametrize(
    "name,config",
    [
        pytest.param("default", {}, id="default"),
        pytest.param(
            "inverse",
            {
                "lower_hsv_color": [0, 0, 108],
                "upper_hsv_color": [255, 10, 148],
                "inverse_mask": True,
                "de_noise_size": 20,
            },
            id="inverse",
        ),
        pytest.param(
            "no-morphology", {"de_noise_morphology": False, "de_noise_size": 20}, id="no-morphology"
        ),
        pytest.param(
            "inverse-no-morphology",
            {
                "lower_hsv_color": [0, 0, 108],
                "upper_hsv_color": [255, 10, 148],
                "inverse_mask": True,
                "de_noise_morphology": False,
                "de_noise_size": 20,
            },
            id="inverse-no-morphology",
        ),
    ],
)
def test_auto_mask(config, name) -> None:
    init_test()
    context = process_utils.Context({"args": {"mask": {"auto_mask": config}}}, {})
    context.image = cv2.imread(os.path.join(os.path.dirname(__file__), "auto-mask-source.png"))
    context.init_mask()
    check_image(
        "/results/",
        cv2.cvtColor(context.mask, cv2.COLOR_BGR2RGB),
        os.path.join(os.path.dirname(__file__), f"auto_mask-{name}.expected.png"),
        generate_expected_image=REGENERATE,
    )


# @pytest.mark.skip(reason="for test")
def test_auto_mask_combine() -> None:
    init_test()
    context = process_utils.Context({"args": {"mask": {}}}, {})
    context.image = cv2.imread(os.path.join(os.path.dirname(__file__), "auto-mask-source.png"))
    context.root_folder = os.path.join(os.path.join(os.path.dirname(__file__), "auto-mask-other"))
    context.image_name = "image.png"
    context.init_mask()
    check_image(
        "/results/",
        cv2.cvtColor(context.mask, cv2.COLOR_BGR2RGB),
        os.path.join(os.path.dirname(__file__), "auto_mask_combine.expected.png"),
        generate_expected_image=REGENERATE,
    )


# @pytest.mark.skip(reason="for test")
def test_auto_cut() -> None:
    init_test()
    context = process_utils.Context({"args": {"cut": {}, "background_color": [255, 0, 0]}}, {})
    context.image = cv2.imread(os.path.join(os.path.dirname(__file__), "auto-mask-source.png"))
    context.do_initial_cut()
    check_image(
        "/results/",
        cv2.cvtColor(context.image, cv2.COLOR_BGR2RGB),
        os.path.join(os.path.dirname(__file__), "auto_cut.expected.png"),
        generate_expected_image=REGENERATE,
    )


# @pytest.mark.skip(reason="for test")
def test_color_cut() -> None:
    init_test()
    context = process_utils.Context({"args": {"cut_white": 200}}, {})
    context.image = cv2.imread(os.path.join(os.path.dirname(__file__), "white-cut.png"))
    process.color_cut(context)
    check_image(
        "/results/",
        cv2.cvtColor(context.image, cv2.COLOR_BGR2RGB),
        os.path.join(os.path.dirname(__file__), "white-cut.expected.png"),
        generate_expected_image=REGENERATE,
    )


# @pytest.mark.skip(reason="for test")
async def test_histogram() -> None:
    init_test()
    context = process_utils.Context(
        {
            "args": {
                "level": {"value": True, "min": 10, "max": 90},
                "cut_black": 20,
                "cut_white": 200,
            }
        },
        {},
    )
    context.image = cv2.imread(os.path.join(os.path.dirname(__file__), "limit-contour-all-1.png"))
    context.image_name = "histogram.png"
    context.root_folder = "/tmp"
    await process.histogram(context)
    print("Compare '/results/histogram/histogram.png' with expected image 'histogram.expected.png'.")
    check_image_file(
        "/results/histogram/",
        "/tmp/histogram/histogram.png",
        os.path.join(os.path.dirname(__file__), "histogram.expected.png"),
        generate_expected_image=REGENERATE,
    )
    print("Compare '/results/histogram/log-histogram.png' with expected image 'histogram-log.expected.png'.")
    check_image_file(
        "/results/histogram/",
        "/tmp/histogram/log-histogram.png",
        os.path.join(os.path.dirname(__file__), "histogram-log.expected.png"),
        generate_expected_image=REGENERATE,
    )
