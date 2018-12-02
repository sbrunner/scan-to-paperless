#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# TODO
# - Do blure only one times
# - other layouts
# - fix content detection for romande energie
# - add http://www.graphicsmagick.org/GraphicsMagick.html#details-sharpen

import argcomplete
import argparse
import sqlite3
import subprocess
import os
import datetime
import random
import re
import yaml


CONFIG_FILENAME = "scan-to-paperless.yaml"

if 'APPDATA' in os.environ:
    CONFIG_PATH = os.path.join(os.environ['APPDATA'], CONFIG_FILENAME)
elif 'XDG_CONFIG_HOME' in os.environ:
    CONFIG_PATH = os.path.join(os.environ['XDG_CONFIG_HOME'], CONFIG_FILENAME)
else:
    CONFIG_PATH = os.path.join(os.environ['HOME'], '.config', CONFIG_FILENAME)


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

    connection = None
    cursor = None
    if 'paperless_db' not in config:
        print('''The PaperLess db path isn\'t set, use:
    scan --set-settings paperless_db <the_path>.''')
    else:
        connection = sqlite3.connect(config['paperless_db'])
        cursor = connection.cursor()

    def execute(sql):
        return cursor.execute(sql) if cursor is not None else []

    auto_complete_parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()

    def add_argument(name, **kwargs):
        auto_complete_parser.add_argument(name, **kwargs)
        if 'choices' in kwargs:
            del kwargs['choices']
        parser.add_argument(name, **kwargs)

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
        choices=[t[0] for t in execute('select name from documents_correspondent')],
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
        choices=[t[0] for t in execute('select name from documents_tag')],
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
        '--set-config',
        nargs=2,
        action='append',
        default=[],
        help='Set a configuration option'
    )

    if connection is not None:
        connection.close()
    argcomplete.autocomplete(auto_complete_parser)
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
    if len(args.tags) > 0:
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
            'scanimage', '--device=hpaio:/usb/HP_LaserJet_MFP_M129-M134?serial=VNC8K00063', '--format=png', '--mode=color', '--resolution=300', '--batch={}/image-%d.png'.format(root_folder),
            '--source=ADF' if args.adf else '--batch-prompt'
        ]

        if args.double_sided:
            call(scanimage + ['--batch-start=1', '--batch-increment=2'])
            odd = os.listdir(root_folder)
            input('Put your document in the automatic document feeder for the other side, and press enter.')
            call(scanimage + ['--batch-start={}'.format(len(odd) * 2), '--batch-increment=-2', '--batch-count={}'.format(len(odd))])
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
