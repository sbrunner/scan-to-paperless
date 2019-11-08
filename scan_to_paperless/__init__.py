import os.path
import yaml

CONFIG_FILENAME = 'scan-to-paperless.yaml'

if 'APPDATA' in os.environ:
    CONFIG_FOLDER = os.environ['APPDATA']
elif 'XDG_CONFIG_HOME' in os.environ:
    CONFIG_FOLDER = os.environ['XDG_CONFIG_HOME']
else:
    CONFIG_FOLDER = os.path.expanduser('~/.config')

CONFIG_PATH = os.path.join(CONFIG_FOLDER, CONFIG_FILENAME)

def get_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding='utf-8') as f:
            return yaml.safe_load(f.read())
    return {}
