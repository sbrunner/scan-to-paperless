#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from setuptools import setup, find_packages

HERE = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(HERE, 'README.md')) as f:
    README = f.read()

setup(
    name='scan-to-paperless',
    version='0.5.0',
    description='Tool to scan and process documents to palerless',
    long_description=README,
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
    ],
    author='Stéphane Brunner',
    author_email='stephane.brunner@gmail.com',
    url='https://hub.docker.com/r/sbrunner/scan-to-paperless/',
    packages=find_packages(exclude=['tests.*']),
    install_requires=['argcomplete', 'pyyaml'],
    entry_points={
        'console_scripts': [
            'scan = scan_to_paperless.scan:main',
            'scan-process-status = scan_to_paperless.scan_process_status:main',
        ],
    },
)
