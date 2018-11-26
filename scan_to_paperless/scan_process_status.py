#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import re
import subprocess
import yaml


def main():
    for folder in glob.glob('/home/sbrunner/dsl/paperless/scan/*/'):
        print(re.sub(r'.', '-', folder))
        print(folder)

        if not os.path.exists(os.path.join(folder, 'config.yaml')):
            print("No config")
        else:
            with open(os.path.join(folder, 'config.yaml')) as config_file:
                config = yaml.load(config_file.read())

            if os.path.exists(os.path.join(folder, 'error.yaml')):
                with open(os.path.join(folder, 'error.yaml')) as error_file:
                    error = yaml.load(error_file.read())
                    if 'error' in error:
                        print(error['error'])
                        if isinstance(error, subprocess.CalledProcessError):
                            print(error.output.decode())
                            if error.stderr:
                                print(error.stderr)
                        if 'traceback' in error:
                            print('\n'.join(error['traceback']))
                    else:
                        print('Unknown error')
                        print(error)
            else:
                if config.get('waiting', False):
                    if os.path.exists(os.path.join(folder, 'waiting')):
                        print('To be validated')
                    else:
                        print('Waiting to be imported')
                else:
                    print('Not ready')
