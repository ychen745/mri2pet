import os
import json
import platform
import yaml
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
DEFAULT_CFG_PATH = ROOT / 'cfg' / 'default.yaml'
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])

def print_cfg(cfg_dict):
    """
        Pretty print cfg dict.
    :param cfg_dict:
    :return: 
    """
    print(yaml.dump(cfg_dict, sort_keys=False, allow_unicode=True, width=-1, Dumper=yaml.SafeDumper))

with open(DEFAULT_CFG_PATH) as f:
    DEFAULT_CFG = yaml.load(f, Loader=yaml.SafeLoader)
    DEFAULT_CFG_KEYS = DEFAULT_CFG.keys()

# print_cfg(DEFAULT_CFG)