# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

import json
import sys
import time

from wu.models import apply_memory_policy, load, setting_dir
from wu.ppl import window_nll

apply_memory_policy()
model_name = sys.argv[1]
for setting in sys.argv[2].split(","):
    t = time.time()
    m, tok = load(model_name, setting)
    rows = window_nll(m, tok, windows=32)
    (setting_dir(model_name, setting) / "ppl-windows.json").write_text(json.dumps(rows))
    print(setting, f"{time.time() - t:.0f}s", flush=True)
    del m
