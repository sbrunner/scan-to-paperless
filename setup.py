#!/usr/bin/env python3

import os
import site
import sys

from setuptools import find_packages, setup

site.ENABLE_USER_SITE = "--user" in sys.argv

HERE = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(HERE, "README.md")) as f:
    README = f.read()

setup(
    name="scan-to-paperless",
    version="1.3.0",
    description="Tool to scan and process documents to palerless",
    long_description=README,
    long_description_content_type="text/markdown",
    keywords=["scan", "paperless"],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: BSD License",
    ],
    author="Stéphane Brunner",
    author_email="stephane.brunner@gmail.com",
    url="https://hub.docker.com/r/sbrunner/scan-to-paperless/",
    packages=find_packages(exclude=["tests.*"]),
    install_requires=["argcomplete", "ruamel.yaml", "scikit-image"],
    entry_points={
        "console_scripts": [
            "scan = scan_to_paperless.scan:main",
            "scan-process-status = scan_to_paperless.scan_process_status:main",
            "scan-process = scan_to_paperless.process:main",
        ],
    },
    data={"scan_to_paperless": ["*.json"]},
)
