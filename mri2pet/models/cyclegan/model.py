# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from mri2pet.engine.model import Model
from mri2pet.models import cyclegan
from mri2pet.utils import utils


class CycleGAN(Model):
    """
    CycleGAN model.

    This class provides a unified interface for CycleGAN models.

    Attributes:
        model: The loaded CyceGAN model instance.

    Methods:
        __init__: Initialize a CycleGAN model.
        task_map: Map tasks to their corresponding model, trainer, validator, and predictor classes.

    Examples:
        Load a pretrained CycleGAN detection model
        >>> model = CyeleGAN("cyclegan.pt")
    """

    def __init__(self, model: Union[str, Path] = "cyclegan.pt"):
        """
        Initialize a CycleGAN model.

        This constructor initializes a CycleGAN model.

        Args:
            model (str | Path): Model name or path to model file, i.e. 'cyclegan.pt'.

        Examples:
            >>> from mri2pet import CycleGAN
            >>> model = CycleGAN("cyclegan.pt")  # load a pretrained CycleGAN model
        """
        path = Path(model if isinstance(model, (str, Path)) else "")
        super().__init__(model=model)





