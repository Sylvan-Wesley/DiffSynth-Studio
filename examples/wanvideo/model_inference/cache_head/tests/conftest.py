"""Make the cache_head modules importable no matter where pytest is invoked from.

The cache_head scripts import each other flatly (``from cache_head_model import ...``)
because they are run as standalone entry points, not as a package.  Without this
the suite only collects when ``examples/wanvideo/model_inference/cache_head`` is
already on ``PYTHONPATH``.
"""

import os
import sys

CACHE_HEAD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.abspath(os.path.join(CACHE_HEAD_DIR, "..", "..", "..", ".."))

for path in (CACHE_HEAD_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)
