# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Type stub for the compiled SYCL extension ``egg._cpp.cpp_core``.

The extension has no inline type info, so this stub declares the surface the
Python layer calls. Arguments are the opaque SoA wire structures assembled in
Python (typed ``object`` here); the return shapes are what callers rely on.
"""

import numpy as np

class CppStructuredSweepSession:
    def __init__(
        self,
        wire: object,
        structured: object,
        X_flat: np.ndarray,
        *,
        device: str,
        dim: int,
        control: object | None = ...,
    ) -> None: ...
    def run(
        self,
        n_sweeps: int,
        *,
        phase: str = ...,
        delta: float = ...,
        omega: float = ...,
        report_every: int = ...,
    ) -> tuple[np.ndarray, np.ndarray]: ...
    def run_fas(
        self, n_cycles: int, **kwargs: object
    ) -> tuple[np.ndarray, np.ndarray]: ...
    def run_control(self, n_outer: int, **kwargs: object) -> dict: ...
    def get_X(self) -> np.ndarray: ...
    def get_C(self) -> np.ndarray: ...
    def set_C(self, C: np.ndarray) -> None: ...
    def set_control_b(self, *args: object) -> None: ...
    def set_control_penalty(self, *args: object) -> None: ...
    def set_control_reduction(self, *args: object) -> None: ...
    def mg_levels(self) -> list: ...

def cpp_structured_sweep(
    wire: object,
    structured: object,
    X_flat: np.ndarray,
    n_sweeps: int,
    *,
    device: str,
    phase: str = ...,
    delta: float = ...,
    dim: int = ...,
    omega: float = ...,
    report_every: int = ...,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def geometry_project_batch(
    Q: np.ndarray,
    tag: int,
    records: np.ndarray,
    seg_dicts: list[dict],
) -> tuple[np.ndarray, np.ndarray]: ...
