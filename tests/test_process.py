import os.path
import process
import cv2
import pytest
import shutil
import json
import re


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
    assert score > 0.999, '{} ({}) != {} ({})'.format(expected, result, expected_name, image)


@pytest.mark.parametrize('type_,limit,size', [
    ('lines', {
        'name': 'VL0',
        'type': 'line detection',
        'value': 1821,
        'vertical': True,
        'margin': 0
    }, '543 x 792 pts'),
    ('contour', {
        'name': 'VC0',
        'type': 'contour detection',
        'value': 1588,
        'vertical': True,
        'margin': 0
    }, '550 x 792 pts')
])
def test_assisted_split_full(type_, limit, size):
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
            'append_credit_card': False
        },
        'full_name': 'Test title 1',
        'destination': os.path.join(root_folder, 'final.pdf'),
    }
    step = {
        'sources': ['image-1.png'],
    }
    config_file_name = os.path.join(root_folder, 'config.yaml')
    step = process.transform(config, step, config_file_name, root_folder)
    assert step['name'] == 'split'
    images = step['sources']
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
    step = process.split(config, step, root_folder)
    assert len(step['sources']) == 2
    check_image(root_folder, step['sources'][0], 'assisted-split-{}-3'.format(type_))
    check_image(root_folder, step['sources'][1], 'assisted-split-{}-4'.format(type_))
    assert step['name'] == 'finalise'
    process.finalise(config, step, root_folder)
    pdfinfo = process.output(['pdfinfo', os.path.join(root_folder, 'final.pdf')]).split('\n')
    regex = re.compile(r'([a-zA-Z ]+): +(.*)')
    pdfinfo = [regex.match(e) for e in pdfinfo]
    pdfinfo = dict([e.groups() for e in pdfinfo if e is not None])
    assert pdfinfo['Title'] == 'Test title 1'
    assert pdfinfo['Pages'] == '2'
    assert pdfinfo['Page size'] == size
    process.call([
        'gm', 'convert', os.path.join(root_folder, 'final.pdf'),
        '+adjoin', os.path.join(root_folder, 'final.png')
    ])
    check_image(root_folder, step['sources'][1], 'assisted-split-{}-4'.format(type_))


# @pytest.mark.skip(reason='for test')
def test_assisted_split_join_full():
    os.environ['PROCESS'] = 'FALSE'
    os.environ['EXPERIMENTAL'] = 'FALSE'
    root_folder = '/results/assisted-split-join-full/'
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    for number in (1, 2):
        shutil.copyfile(
            os.path.join(os.path.dirname(__file__), 'split-join-{}.png'.format(number)),
            os.path.join(root_folder, 'image-{}.png'.format(number)),
        )

    config = {
        'args': {
            'assisted_split': True,
            'level': True,
            'append_credit_card': False,
        },
        'full_name': 'Test title 2',
        'destination': os.path.join(root_folder, 'final.pdf'),
    }
    step = {
        'sources': ['image-1.png', 'image-2.png'],
    }
    config_file_name = os.path.join(root_folder, 'config.yaml')
    step = process.transform(config, step, config_file_name, root_folder)
    assert step['name'] == 'split'
    images = step['sources']
    assert os.path.basename(images[0]) == config['assisted_split'][0]['image']
    assert len(images) == 2
    for number, elements in enumerate([({
        'value': 700,
        'vertical': True,
        'margin': 0
    }, ['-', '1.2']), ({
        'value': 3310,
        'vertical': True,
        'margin': 0
    }, ['1.1', '-'])]):
        limit, destinations = elements
        config['assisted_split'][number]['limits'] = [limit]
        config['assisted_split'][number]['destinations'] = destinations
    step = process.split(config, step, root_folder)
    assert step['name'] == 'finalise'
    assert len(step['sources']) == 1
    check_image(root_folder, step['sources'][0], 'assisted-split-join-1')

    process.finalise(config, step, root_folder)
    pdfinfo = process.output(['pdfinfo', os.path.join(root_folder, 'final.pdf')]).split('\n')
    regex = re.compile(r'([a-zA-Z ]+): +(.*)')
    pdfinfo = [regex.match(e) for e in pdfinfo]
    pdfinfo = dict([e.groups() for e in pdfinfo if e is not None])
    assert pdfinfo['Title'] == 'Test title 2'
    assert pdfinfo['Pages'] == '1'
    assert pdfinfo['Page size'] == '612 x 229 pts'
    process.call([
        'gm', 'convert', os.path.join(root_folder, 'final.pdf'),
        '+adjoin', os.path.join(root_folder, 'final.png')
    ])
    check_image(root_folder, step['sources'][0], 'assisted-split-join-1')


# @pytest.mark.skip(reason='for test')
@pytest.mark.parametrize('progress', ['FALSE', 'TRUE'])
@pytest.mark.parametrize('experimental', ['FALSE', 'TRUE'])
def test_full(progress, experimental):
    os.environ['PROGRESS'] = progress
    os.environ['EXPERIMENTAL'] = experimental
    root_folder = '/results/full-{}-{}/'.format(progress, experimental)
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    config = {
        'args': {
            'assisted_split': False,
            'level': True,
            'append_credit_card': False,
        },
        'full_name': 'Test title 3',
        'destination': os.path.join(root_folder, 'final.pdf'),
    }
    step = {
        'sources': [os.path.join(os.path.dirname(__file__), 'all-1.png')]
    }
    step = process.transform(config, step, '/tmp/test-config.yaml', root_folder)
    check_image(root_folder, step['sources'][0], 'all-1')
    assert len(step['sources']) == 1

    if progress == 'TRUE':
        assert os.path.exists(os.path.join(root_folder, '0-force-cleanup/all-1.png'))
    else:
        assert not os.path.exists(os.path.join(root_folder, '0-force-cleanup'))
    if experimental == 'TRUE':
        assert os.path.exists(os.path.join(root_folder, 'scantailor/all-1.png'))
    else:
        assert not os.path.exists(os.path.join(root_folder, 'scantailor'))

    assert step['name'] == 'finalise'
    process.finalise(config, step, root_folder)
    pdfinfo = process.output(['pdfinfo', os.path.join(root_folder, 'final.pdf')]).split('\n')
    regex = re.compile(r'([a-zA-Z ]+): +(.*)')
    pdfinfo = [regex.match(e) for e in pdfinfo]
    pdfinfo = dict([e.groups() for e in pdfinfo if e is not None])
    assert pdfinfo['Title'] == 'Test title 3'
    assert pdfinfo['Pages'] == '1'
    assert pdfinfo['Page size'] == '309 x 792 pts'
    process.call([
        'gm', 'convert', os.path.join(root_folder, 'final.pdf'),
        '+adjoin', os.path.join(root_folder, 'final.png')
    ])
    check_image(root_folder, os.path.join(root_folder, 'final.png'), 'all-2')
