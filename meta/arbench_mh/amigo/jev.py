"""The Jev client the environment answers through, kept in the package so a run directory needs
nothing from a scratch directory. Pinned to jev-1.14.0: images are a danger-zone feature and the
format may change under `latest`.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

API = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.14.0"


class Jev:
    def __init__(self, token=None, model=MODEL):
        token = token or os.environ.get("TYPESAFE_API_KEY") or \
            open(os.path.expanduser("~/.typesafe_token")).read().strip()
        self.h = {"Authorization": "Bearer " + token, "Content-Type": "application/json",
                  "TypeSafe-Danger-Zone-Images": "true"}
        self.model = model
        self.calls = self.in_tok = 0

    def ask(self, body, tries=4):
        body["model"] = self.model
        data = json.dumps(body).encode()
        last = None
        for k in range(tries):
            try:
                req = urllib.request.Request(API, data=data, headers=self.h)
                with urllib.request.urlopen(req, timeout=300) as r:
                    o = json.loads(r.read().decode())
                self.in_tok += (o.get("usage") or {}).get("input_tokens", 0)
                self.calls += 1
                return o["answers"]
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
                last = e
                time.sleep(1.5 * (k + 1))
        raise RuntimeError("jev failed after %d tries: %s" % (tries, last))
