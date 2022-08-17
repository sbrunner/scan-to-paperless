import glob
import json
import os.path
import re
import shutil
import subprocess

import cv2
import pikepdf
import pytest

from scan_to_paperless import code, process


def load_image(image_name):
    return cv2.imread(os.path.join(os.path.dirname(__file__), image_name))


# @pytest.mark.skip(reason="for test")
def test_find_lines():
    peaks, _ = process.find_lines(load_image("limit-lines-1.png"), True)
    assert 2844 in peaks


# @pytest.mark.skip(reason="for test")
def test_find_limit_contour():
    limits, _ = process.find_limit_contour(
        load_image("limit-contour-1.png"),
        process.Context({"args": {"min_box_size_empty": 40}}, {}),
        True,
    )
    assert limits == [1589]


def check_image_file(root_folder, image, name, level=0.9):
    result = cv2.imread(image)
    subprocess.run(["ls", "-l", image])
    assert result is not None, "Wrong image: " + image
    check_image(root_folder, result, name, level)


def check_image(root_folder, image, name, level=0.9):
    assert image is not None, "Image required"
    expected_name = os.path.join(os.path.dirname(__file__), f"{name}.expected.png")
    result_name = os.path.join(root_folder, f"{name}.result.png")
    diff_name = os.path.join(root_folder, f"{name}.diff.png")
    # Set to True to regenerate images
    if False:
        cv2.imwrite(expected_name, image)
        return
    if not os.path.exists(expected_name):
        cv2.imwrite(expected_name, image)
        assert not os.path.isfile(expected_name), "Expected image not found: " + expected_name
    expected = cv2.imread(expected_name)
    assert expected is not None, "Wrong image: " + expected_name
    score, diff = process.image_diff(expected, image)
    if diff is None:
        cv2.imwrite(result_name, image)
        assert diff is not None, "No diff generated"
    if score > level:
        cv2.imwrite(result_name, image)
        cv2.imwrite(diff_name, diff)
        assert score > level, f"{result_name} != {expected_name} => {diff_name} ({score} > {level})"


# @pytest.mark.skip(reason="for test")
def test_crop():
    image = load_image("image-1.png")
    root_folder = "/results/crop"
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    check_image(root_folder, process.crop_image(image, 100, 0, 100, 300, (255, 255, 255)), "crop-1")
    check_image(root_folder, process.crop_image(image, 0, 100, 300, 100, (255, 255, 255)), "crop-2")
    check_image(root_folder, process.crop_image(image, 100, -100, 100, 200, (255, 255, 255)), "crop-3")
    check_image(root_folder, process.crop_image(image, -100, 100, 200, 100, (255, 255, 255)), "crop-4")
    check_image(root_folder, process.crop_image(image, 100, 200, 100, 200, (255, 255, 255)), "crop-5")
    check_image(root_folder, process.crop_image(image, 200, 100, 200, 100, (255, 255, 255)), "crop-6")
    shutil.rmtree(root_folder)


# @pytest.mark.skip(reason="for test")
def test_rotate():
    image = load_image("image-1.png")
    root_folder = "/results/rotate"
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    image = process.crop_image(image, 0, 50, 300, 200, (255, 255, 255))
    check_image(root_folder, process.rotate_image(image, 10, (255, 255, 255)), "rotate-1")
    check_image(root_folder, process.rotate_image(image, -10, (255, 255, 255)), "rotate-2")
    check_image(root_folder, process.rotate_image(image, 90, (255, 255, 255)), "rotate-3")
    check_image(root_folder, process.rotate_image(image, -90, (255, 255, 255)), "rotate-4")
    check_image(root_folder, process.rotate_image(image, 270, (255, 255, 255)), "rotate-4")
    check_image(root_folder, process.rotate_image(image, 180, (255, 255, 255)), "rotate-5")
    shutil.rmtree(root_folder)


def init_test():
    os.environ["PROGRESS"] = "FALSE"
    os.environ["TIME"] = "TRUE"
    os.environ["SCAN_CODES_FOLDER"] = "/results"
    shutil.copyfile(os.path.join(os.path.dirname(__file__), "mask.png"), "/results/mask.png")


# @pytest.mark.skip(reason="for test")
@pytest.mark.parametrize(
    "type_,limit",
    [
        ("lines", {"name": "VL0", "type": "line detection", "value": 1031, "vertical": True, "margin": 0}),
        (
            "contour",
            {"name": "VC0", "type": "contour detection", "value": 1584, "vertical": True, "margin": 0},
        ),
    ],
)
def test_assisted_split_full(type_, limit):
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
            "level": True,
            "tesseract": False,
            "sharpen": True,
            "num_angles": 179,
            "threshold_block_size_crop": 20,
            "threshold_value_c_crop": 20,
        },
    }
    step = {
        "sources": ["image-1.png"],
    }
    config_file_name = os.path.join(root_folder, "config.yaml")
    step = process.transform(config, step, config_file_name, root_folder)
    assert step["name"] == "split"
    images = step["sources"]
    assert len(images) == 1
    assert os.path.basename(images[0]) == config["assisted_split"][0]["image"]
    check_image_file(root_folder, images[0], f"assisted-split-{type_}-1", 0.998)
    # check_image_file(root_folder, config['assisted_split'][0]['source'], 'assisted-split-{}-2'.format(type_))
    limits = [item for item in config["assisted_split"][0]["limits"] if item["vertical"]]
    print(json.dumps(limits))
    assert not [item for item in limits if item["name"] == "C"], "We shouldn't have center limit"
    limits = [item for item in limits if item["name"] == limit["name"]]
    assert limits == [limit], limits
    config["assisted_split"][0]["limits"] = limits
    check_image_file(root_folder, images[0], f"assisted-split-{type_}-1", 0.998)
    step = process.split(config, step, root_folder)
    assert len(step["sources"]) == 2
    assert step["name"] == "finalise"
    check_image_file(root_folder, step["sources"][0], f"assisted-split-{type_}-3")
    check_image_file(root_folder, step["sources"][1], f"assisted-split-{type_}-4")
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
    check_image_file(
        root_folder,
        os.path.join("/results", f"{os.path.basename(root_folder)}.png"),
        f"assisted-split-{type_}-5",
    )
    shutil.rmtree(root_folder)


# @pytest.mark.skip(reason="for test")
def test_assisted_split_join_full():
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
        "args": {"assisted_split": True, "level": True, "tesseract": False, "num_angles": 179},
        "destination": os.path.join(root_folder, "final.pdf"),
    }
    step = {
        "sources": ["image-1.png", "image-2.png"],
    }
    config_file_name = os.path.join(root_folder, "config.yaml")
    step = process.transform(config, step, config_file_name, root_folder)
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
    step = process.split(config, step, root_folder)
    assert step["name"] == "finalise"
    assert len(step["sources"]) == 1
    check_image_file(root_folder, step["sources"][0], "assisted-split-join-1")

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
    check_image_file(
        root_folder, os.path.join("/results", f"{os.path.basename(root_folder)}.png"), "assisted-split-join-2"
    )
    shutil.rmtree(root_folder)


# @pytest.mark.skip(reason="for test")
def test_assisted_split_booth():
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
            "level": False,
            "no_crop": True,
            "tesseract": False,
            "margin_horizontal": 0,
            "margin_vertical": 0,
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
    step = process.split(config, step, root_folder)
    assert step["name"] == "finalise"
    assert len(step["sources"]) == 4
    check_image_file(root_folder, step["sources"][0], "assisted-split-booth-1")
    check_image_file(root_folder, step["sources"][1], "assisted-split-booth-2")
    check_image_file(root_folder, step["sources"][2], "assisted-split-booth-3")
    check_image_file(root_folder, step["sources"][3], "assisted-split-booth-4")
    shutil.rmtree(root_folder)


# @pytest.mark.skip(reason="for test")
@pytest.mark.parametrize("progress", ["FALSE", "TRUE"])
def test_full(progress):
    init_test()
    os.environ["PROGRESS"] = progress
    root_folder = f"/results/full-{progress}"
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    config = {
        "args": {"level": True},
    }
    step = {"sources": [os.path.join(os.path.dirname(__file__), "all-1.png")]}
    step = process.transform(config, step, "/tmp/test-config.yaml", root_folder)
    assert len(step["sources"]) == 1
    check_image_file(root_folder, step["sources"][0], "all-1")

    if progress == "TRUE":
        assert os.path.exists(os.path.join(root_folder, "0-level/all-1.png"))
    else:
        assert not os.path.exists(os.path.join(root_folder, "0-level"))

    assert step["name"] == "finalise"
    process.finalize(config, step, root_folder)

    pdf_filename = os.path.join("/results", f"{os.path.basename(root_folder)}.pdf")

    creator_re = re.compile(
        r"^Scan to Paperless [0-9]+.[0-9]+.[0-9]+\+[0-9]+, Tesseract [0-9]+.[0-9]+.[0-9]+$"
    )
    with pikepdf.open(pdf_filename) as pdf_:
        with pdf_.open_metadata() as meta:
            assert creator_re.match(meta["{http://purl.org/dc/elements/1.1/}creator"])

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
    check_image_file(root_folder, os.path.join("/results", f"{os.path.basename(root_folder)}.png"), "all-2")
    shutil.rmtree(root_folder)


# @pytest.mark.skip(reason="for test")
def test_credit_card_full():
    init_test()
    #    os.environ['PROGRESS'] = 'TRUE'
    root_folder = "/results/credit-card"
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    config = {
        "args": {"level": True, "append_credit_card": True, "num_angles": 179},
    }
    step = {
        "sources": [
            os.path.join(os.path.dirname(__file__), "credit-card-1.png"),
            os.path.join(os.path.dirname(__file__), "credit-card-2.png"),
        ]
    }
    step = process.transform(config, step, "/tmp/test-config.yaml", root_folder)
    assert len(step["sources"]) == 2
    assert step["name"] == "finalise"
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
    check_image_file(
        root_folder, os.path.join("/results", f"{os.path.basename(root_folder)}.png"), "credit-card-1"
    )
    shutil.rmtree(root_folder)


# @pytest.mark.skip(reason="for test")
def test_empty():
    init_test()
    #    os.environ['PROGRESS'] = 'TRUE'
    root_folder = "/results/empty"
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    config = {
        "args": {"level": True},
    }
    step = {
        "sources": [
            os.path.join(os.path.dirname(__file__), "empty.png"),
        ]
    }
    step = process.transform(config, step, "/tmp/test-config.yaml", root_folder)
    assert len(step["sources"]) == 0
    assert step["name"] == "finalise"
    shutil.rmtree(root_folder)


# @pytest.mark.skip(reason="for test")
@pytest.mark.parametrize("test,args", [("600", {"dpi": 600, "num_angles": 179})])
def test_custom_process(test, args):
    init_test()
    root_folder = f"/results/600"
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    config = {
        "args": args,
    }
    step = {"sources": [os.path.join(os.path.dirname(__file__), f"{test}.png")]}
    step = process.transform(config, step, "/tmp/test-config.yaml", root_folder)
    assert len(step["sources"]) == 1
    check_image_file(root_folder, step["sources"][0], test)
    shutil.rmtree(root_folder)


# @pytest.mark.skip(reason="for test")
@pytest.mark.parametrize("name", ["qrcode", "qrbill", "qrbill2"])
def test_qr_code(name):
    code.add_codes(os.path.join(os.path.dirname(__file__), f"{name}.pdf"), f"/results/{name}.pdf")
    root_folder = f"/results/qrcode"
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
        check_image_file(root_folder, f"/results/{name}-{page}.png", f"{name}-{page}", level=0.99999)


# @pytest.mark.skip(reason="for test")
def test_qr_code_metadata():
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
def test_multi_code():
    code.add_codes(os.path.join(os.path.dirname(__file__), "qrbill-multi.pdf"), "/results/qrbill-multi.pdf")
    root_folder = f"/results/qrcode"
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
        check_image_file(
            root_folder, f"/results/qrbill-multi-{page}.png", f"qrbill-multi-{page}", level=0.99999
        )


# @pytest.mark.skip(reason="for test")
def test_tiff():
    init_test()
    os.environ["PROGRESS"] = "TRUE"
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
    step = process.transform(config, step, config_file_name, root_folder)
    assert step["sources"] == ["/results/tiff/image-1.png"]
    assert list(glob.glob(f"{root_folder}/**/*.tiff")) == ["/results/tiff/source/image-1.tiff"]


# @pytest.mark.skip(reason="for test")
@pytest.mark.parametrize(
    "name,config",
    [
        ("default", {}),
        (
            "inverse",
            {"lower_hsv_color": [0, 0, 108], "upper_hsv_color": [255, 10, 148], "inverse_mask": True},
        ),
        ("no-morphology", {"de_noise_morphology": False}),
        (
            "inverse-no-morphology",
            {
                "lower_hsv_color": [0, 0, 108],
                "upper_hsv_color": [255, 10, 148],
                "inverse_mask": True,
                "de_noise_morphology": False,
            },
        ),
    ],
)
def test_auto_mask(config, name):
    context = process.Context({"args": {"auto_mask": config}}, {})
    context.image = cv2.imread(os.path.join(os.path.dirname(__file__), "auto-mask-source.png"))
    context.init_mask()
    check_image("/results/auto_mask", context.mask, f"auto_mask-{name}")


# @pytest.mark.skip(reason="for test")
def test_auto_mask_combine():
    context = process.Context({"args": {"auto_mask": {}}}, {})
    context.image = cv2.imread(os.path.join(os.path.dirname(__file__), "auto-mask-source.png"))
    context.root_folder = os.path.join(os.path.join(os.path.dirname(__file__), "auto-mask-other"))
    context.image_name = "image.png"
    context.init_mask()
    check_image("/results/auto_mask_combine", context.mask, "auto_mask_combine")


# @pytest.mark.skip(reason="for test")
def test_auto_cut():
    context = process.Context({"args": {"auto_cut": {}, "background_color": [255, 0, 0]}}, {})
    context.image = cv2.imread(os.path.join(os.path.dirname(__file__), "auto-mask-source.png"))
    context.do_initial_cut()
    check_image("/results/auto_cut", context.image, "auto_cut")
