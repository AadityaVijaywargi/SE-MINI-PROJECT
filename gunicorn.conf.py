"""Gunicorn settings (SM-66, CON-2, PERF-2).

One worker because the job store lives in process memory; four threads so status polls
are answered while a training thread is busy.
"""

# Gunicorn reads these exact lower-case names, so pylint's constant naming rule
# does not apply to this file.
# pylint: disable=invalid-name

import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = 1
threads = 4
timeout = 600
wsgi_app = "app:create_app()"
