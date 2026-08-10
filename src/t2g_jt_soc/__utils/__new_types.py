# File to store new types definition for typing
import numpy as np
from typing import TypeVar

Scalar = int | float | complex
RealScalar = int | float
DecimalScalar = float | complex
ArrayKey = int | slice | tuple[int | slice]
NDArray1D = np.ndarray[tuple[np.number], np.dtype[TypeVar("DType", bound=np.generic)]]