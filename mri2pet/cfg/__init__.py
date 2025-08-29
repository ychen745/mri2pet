from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


def get_cfg(cfg: Union[Path, str, Dict]) -> Dict[str, Any]:
    """
    Load and merge configuration from given path.

    Args:
        cfg (Union[Path, str, Dict]): Configuration path.

    Returns:
        Dict[str, Any]: Configuration.
    """
    if isinstance(cfg, (Path, str)):
        with open(cfg) as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
            return cfg
    elif isinstance(cfg, Dict):
        return cfg
    else:
        raise TypeError(f"cfg must be either a Path or a dict, not {type(cfg)}.")

