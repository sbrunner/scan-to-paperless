#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# TODO
# - Do blur only one times
# - other layouts
# - fix content detection for romande energie
# - add http://www.graphicsmagick.org/GraphicsMagick.html#details-sharpen


import argparse
import datetime
import json
import os
import random
import re
import sqlite3
import subprocess
import time

import argcomplete
from argcomplete.completers import ChoicesCompleter
import yaml


CONFIG_FILENAME = "scan-to-paperless.yaml"
CACHE_FILENAME = "scan-to-paperless-cache.json"

if 'APPDATA' in os.environ:
    CONFIG_PATH = os.path.join(os.environ['APPDATA'], CONFIG_FILENAME)
    CACHE_PATH = os.path.join(os.environ['APPDATA'], CACHE_FILENAME)
elif 'XDG_CONFIG_HOME' in os.environ:
    CONFIG_PATH = os.path.join(os.environ['XDG_CONFIG_HOME'], CONFIG_FILENAME)
    CACHE_PATH = os.path.join(os.environ['XDG_CONFIG_HOME'], CACHE_FILENAME)
else:
    CONFIG_PATH = os.path.join(os.environ['HOME'], '.config', CONFIG_FILENAME)
    CACHE_PATH = os.path.join(os.environ['HOME'], '.config', CACHE_FILENAME)


def call(cmd, cmd2=None, **kwargs):
    print(' '.join(cmd) if isinstance(cmd, list) else cmd)
    try:
        subprocess.check_call(cmd, **kwargs)
    except subprocess.CalledProcessError as e:
        print(e)
        exit(1)


def output(cmd, cmd2=None, **kwargs):
    print(' '.join(cmd) if isinstance(cmd, list) else cmd)
    try:
        return subprocess.check_output(cmd, **kwargs)
    except subprocess.CalledProcessError as e:
        print(e)
        exit(1)


def main():

    config = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding='utf-8') as f:
            config = yaml.safe_load(f.read())

    cache = {
        'correspondents': [],
        'tags': [],
        'time': 0
    }
    update_cache = True
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding='utf-8') as f:
            cache = json.loads(f.read())
            if cache['time'] > time.monotonic() - 3600:
                update_cache = False

    if update_cache:

        if 'paperless_db' in config:
            connection = sqlite3.connect(config['paperless_db'])
            cursor = connection.cursor()

            cache = {
                'correspondents': [t[0] for t in cursor.execute('select name from documents_correspondent')],
                'tags': [t[0] for t in cursor.execute('select name from documents_tag')],
                'time': time.monotonic()
            }

            connection.close()

        elif 'paperless_dump' in config:
            cache['time'] = time.monotonic()
            with open(config['paperless_dump']) as dumpdata:
                dump = json.loads(dumpdata.read())
                {
                    "model": "contenttypes.contenttype",
                    "pk": 1,
                    "fields": {
                        "app_label": "auth",
                        "model": "permission"
                    }
                }

            for element in dump:
                if element['model'] == 'documents.correspondent':
                    cache['correspondents'].append(element['fields']['name'])
                elif element['model'] == 'documents.tag':
                    cache['tags'].append(element['fields']['name'])

        else:
            print('''The PaperLess source path isn\'t set, use:
                scan --set-settings paperless_db <the_path>, or
                scan --set-settings paperless_dump <the_path>.''')

        with open(CACHE_PATH, 'w', encoding='utf-8') as f:
            f.write(json.dumps(cache))

    parser = argparse.ArgumentParser()

    def add_argument(name, choices=None, **kwargs):
        arg = parser.add_argument(name, **kwargs)
        if choices is not None:
            arg.completer = ChoicesCompleter(choices)

    add_argument(
        '--no-adf',
        dest='adf',
        action='store_false',
        help="Don't use ADF"
    )
    add_argument(
        '--no-level',
        dest='level',
        action='store_false',
        help="Don't use level correction"
    )
    add_argument(
        '--correspondent',
        choices=cache['correspondents'],
        help='The correspondent'
    )
    add_argument(
        'title',
        nargs='*',
        choices=['No title'],
        help='The document title'
    )
    add_argument(
        '--date',
        choices=[datetime.date.today().strftime('%Y%m%d')],
        help='The document date'
    )
    add_argument(
        '--tag',
        action='append',
        dest="tags",
        default=[],
        choices=cache['tags'],
        help='The document tags'
    )
    add_argument(
        '--double-sided',
        action='store_true',
        help='Number of pages in double sided mode'
    )
    add_argument(
        '--append-credit-card',
        action='store_true',
        help='Append vertically the credit card'
    )
    add_argument(
        '--assisted-split',
        action='store_true',
        help='Split operation, se help'
    )
    add_argument(
        '--set-config',
        nargs=2,
        action='append',
        default=[],
        help='Set a configuration option'
    )

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    dirty = False
    for conf in args.set_config:
        config[conf[0]] = conf[1]
        dirty = True
    if dirty:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            f.write(yaml.safe_dump(config, default_flow_style=False))

    if len(args.title) == 0:
        exit(0)

    if 'scan_folder' not in config:
        print('''The scan folder isn\'t set, use:
    scan --set-settings scan_folder <a_folder>
    This should be shared with the process container in '\source'.''')
        exit(1)

    full_name = ' '.join(args.title)
    if args.correspondent is not None:
        full_name = '{} - {}'.format(args.correspondent, full_name)
    if args.date is not None:
        full_name = '{}Z - {}'.format(args.date, full_name)
    full_name = '{} - {}'.format(full_name, ','.join(args.tags))
    destination = '/destination/{}.pdf'.format(
        full_name
    )
    if '/' in full_name:
        print("The name can't contans some '/' (from correspondent, tags or title).")
        exit(1)

    root_folder = os.path.join(config['scan_folder'], str(random.randint(0, 999999)), 'source')
    os.makedirs(root_folder)

    mogrify = ['gm', 'mogrify']

    try:
        scanimage = [
            'scanimage',
            # TODO: put in config
            '--device=hpaio:/usb/HP_LaserJet_MFP_M129-M134?serial=VNC8K00063',
            '--format=png',
            '--mode=color',
            '--resolution=300',
            '--batch={}/image-%d.png'.format(root_folder),
            '--source=ADF' if args.adf else '--batch-prompt'
        ]

        if args.double_sided:
            call(scanimage + ['--batch-start=1', '--batch-increment=2'])
            odd = os.listdir(root_folder)
            input('Put your document in the automatic document feeder for the other side, and press enter.')
            call(scanimage + [
                '--batch-start={}'.format(len(odd) * 2),
                '--batch-increment=-2',
                '--batch-count={}'.format(len(odd))
            ])
            for img in os.listdir(root_folder):
                if img not in odd:
                    call(mogrify + ['-rotate', '180', os.path.join(root_folder, img)])
        else:
            call(scanimage)

        images = []
        for img in os.listdir(root_folder):
            if not img.startswith('image-'):
                continue
            images.append(os.path.join('source', img))

        regex = re.compile(r'^source\/image\-([0-9]+)\.png$')
        images = sorted(images, key=lambda e: int(regex.match(e).group(1)))
        config = {
            'images': images,
            'full_name': full_name,
            'destination': destination,
            'args': dict(args._get_kwargs()),
        }
        with open(os.path.join(os.path.dirname(root_folder), 'config.yaml'), 'w') as config_file:
            config_file.write(yaml.safe_dump(config, default_flow_style=False))

    except subprocess.CalledProcessError as e:
        print(e)
        exit(1)

    print(root_folder)
    subprocess.call(['eog', root_folder])
