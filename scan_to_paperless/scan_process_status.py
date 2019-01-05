#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import re
import subprocess
import yaml


def main():
    for folder in glob.glob(os.path.expanduser('~/Paperless/scan/*/')):
        print(re.sub(r'.', '-', folder))
        print(folder)

        if not os.path.exists(os.path.join(folder, 'config.yaml')):
            print("No config")
        else:
            with open(os.path.join(folder, 'config.yaml')) as config_file:
                config = yaml.safe_load(config_file.read())

            if os.path.exists(os.path.join(folder, 'error.yaml')):
                with open(os.path.join(folder, 'error.yaml')) as error_file:
                    error = yaml.load(error_file.read())
                    if error is not None and 'error' in error:
                        print(error['error'])
                        if isinstance(error['error'], subprocess.CalledProcessError):
                            print(error['error'].output.decode())
                            if error['error'].stderr:
                                print(error['error'].stderr)
                        if 'traceback' in error:
                            print('\n'.join(error['traceback']))
                    else:
                        print('Unknown error')
                        print(error)
            else:
                allready_proceed = True
                if 'transformed_images' not in config:
                    allready_proceed = False
                else:
                    for img in config['transformed_images']:
                        img = os.path.join(folder, os.path.basename(img))
                        if not os.path.exists(img):
                            allready_proceed = False
                if allready_proceed:
                    if os.path.exists(os.path.join(folder, 'REMOVE_TO_CONTINUE')):
                        print('To be validated')
                    else:
                        print('Waiting to be imported')
                else:
                    print('Not ready')
