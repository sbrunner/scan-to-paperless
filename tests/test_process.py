import os.path
import process
import cv2
import pytest
import shutil
import json


def load_image(image_name):
    return cv2.imread(os.path.join(os.path.dirname(__file__), image_name))


#def test_crop():
#    process.crop(...)


#def test_no_content():
#    process.no_content(...)


def test_find_lines():
    peaks, _ = process.find_lines(load_image('limit-lines-1.png'), True)
    assert 2844 in peaks


def test_find_limit_contour():
    contour = process.find_limit_contour(load_image('limit-contour-1.png'), True)
    assert contour == [1588]


def check_image(root_folder, image, name):
    result = cv2.imread(image)
    assert result is not None, 'Wrong image: ' + image
    cv2.imwrite(os.path.join(root_folder, '{}.result.png'.format(name)), result)
    expected_name = os.path.join(
        os.path.dirname(__file__), '{}.expected.png'.format(name)
    )
    expected = cv2.imread(expected_name)
    assert expected is not None, 'Wrong image: ' + expected_name
    score, diff = process.image_diff(expected, result)
    if diff is not None:
        cv2.imwrite(os.path.join(root_folder, '{}.diff.png'.format(name)), diff)
    assert score < 2, '{} ({}) != {} ({})'.format(expected, result, expected_name, image)


@pytest.mark.parametrize('type_,limit', [
    ('lines', {
        'name': 'L0',
        'type': 'line detection',
        'value': 1821,
        'vertical': True,
        'margin': 0
    }),
    ('contour', {
        'name': 'C0',
        'type': 'contour detection',
        'value': 1588,
        'vertical': True,
        'margin': 0
    })
])
def test_assisted_split_full(type_, limit):
    os.environ['PROCESS'] = 'TRUE'
    os.environ['EXPERIMENTAL'] = 'FALSE'
    root_folder = '/results/assisted-split-full-{}/'.format(type_)
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), 'limit-{}-all-1.png'.format(type_)),
        os.path.join(root_folder, 'image-1.png'),
    )
    config = {
        'args': {
            'assisted_split': True,
            'level': True,
        },
        'images': ['image-1.png']
    }
    config_file_name = os.path.join(root_folder, 'config.yaml')
    images = process.transform(config, config_file_name, root_folder)
    config['transformed_images'] = images
    assert os.path.basename(images[0]) == config['assisted_split'][0]['image']
    assert len(images) == 1
    check_image(root_folder, images[0], 'assisted-split-{}-1'.format(type_))
    check_image(root_folder, config['assisted_split'][0]['source'], 'assisted-split-{}-2'.format(type_))
    limits = [item for item in config['assisted_split'][0]['limits'] if item['vertical']]
    print(json.dumps(limits))
    assert not [item for item in limits if item['name'] == 'C'], "We shouldn't have center limit"
    limits = [item for item in limits if item['name'] == limit['name']]
    assert limits == [limit]
    config['assisted_split'][0]['limits'] = limits
    process.split(config, root_folder, config_file_name)
    assert config['splitted'] is True
    assert len(config['transformed_images']) == 2
    check_image(root_folder, config['transformed_images'][0], 'assisted-split-{}-3'.format(type_))
    check_image(root_folder, config['transformed_images'][1], 'assisted-split-{}-4'.format(type_))


@pytest.mark.skip(reason='for test')
def test_assisted_split_join_full():
    os.environ['PROCESS'] = 'FALSE'
    os.environ['EXPERIMENTAL'] = 'FALSE'
    root_folder = '/results/assisted-split-join-full/'
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    for i in (1, 2):
        shutil.copyfile(
            os.path.join(os.path.dirname(__file__), 'split-join-{}.png'.format(i)),
            os.path.join(root_folder, 'image-{}.png'.format(i)),
        )

    # TODO


# @pytest.mark.skip(reason='for test')
@pytest.mark.parametrize('progress', ['FALSE', 'TRUE'])
@pytest.mark.parametrize('experimental', ['FALSE', 'TRUE'])
def test_full(progress, experimental):
    os.environ['PROGRESS'] = progress
    os.environ['EXPERIMENTAL'] = experimental
    root_folder = '/results/full-{}-{}/'.format(progress, experimental)
    config = {
        'args': {
            'assisted_split': False,
            'level': True,
        },
        'images': [os.path.join(os.path.dirname(__file__), 'all-1.png')]
    }
    images = process.transform(config, '/tmp/test-config.yaml', root_folder)
    check_image(root_folder, images[0], 'all-1')
    assert len(images) == 1
