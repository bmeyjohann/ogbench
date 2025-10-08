"""OGBench: Benchmarking Offline Goal-Conditioned RL"""

# Ensure headless MuJoCo rendering works reliably (esp. on clusters using EGL).
# We monkey-patch the EGL context teardown to ignore sporadic destroy errors
# that arise when the renderer is garbage-collected after the driver unloads.
try:
    import os

    os.environ.setdefault('MUJOCO_GL', os.environ.get('MUJOCO_GL', 'egl'))

    import mujoco
    from mujoco import egl as _mj_egl  # type: ignore
    from OpenGL import error as _gl_error  # type: ignore

    _original_free = _mj_egl.GLContext.free

    def _safe_free(self):  # type: ignore
        try:
            _original_free(self)
        except (_gl_error.GLError, Exception):
            # Ignore teardown failures; the context is already gone.
            pass

    _mj_egl.GLContext.free = _safe_free  # type: ignore
    del _original_free  # cleanup
except Exception:
    # If MuJoCo/OpenGL are unavailable (e.g., docs builds), fall through.
    pass

import ogbench.locomaze
import ogbench.manipspace
import ogbench.powderworld
import ogbench.ui
import ogbench.wrappers
from ogbench.utils import download_datasets, load_dataset, make_env_and_datasets

__all__ = (
    'locomaze',
    'manipspace',
    'powderworld',
    'ui',
    'wrappers',
    'download_datasets',
    'load_dataset',
    'make_env_and_datasets',
)
