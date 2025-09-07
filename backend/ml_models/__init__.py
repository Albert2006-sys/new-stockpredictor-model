"""

Code to shorten imports: instead of writing the longer:
from ml_models.predict import ModelPredictor

write:
from ml_models import ModelPredictor
"""

from .predict import ModelPredictor

# The __all__ variable defines the public API of the package.
# When `from ml_models import *` is used, only the names in this list will be imported.
__all__ = [
    "ModelPredictor"
]