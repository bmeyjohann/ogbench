"""OGBench: Benchmarking Offline Goal-Conditioned RL"""

# Ensure MuJoCo picks a sensible backend before any env modules import it.
# Linux headless runs default to EGL; Windows defaults to GLFW because MuJoCo
# rejects MUJOCO_GL=egl there. Only patch EGL teardown when EGL is active.
try:
    import os
    import sys

    _configured_gl = str(os.environ.get('MUJOCO_GL', '')).strip().lower()
    if not _configured_gl:
        _configured_gl = 'glfw' if sys.platform.startswith('win') else 'egl'
        os.environ['MUJOCO_GL'] = _configured_gl

    if _configured_gl == 'egl':
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
