"""Where ACE's episodes are played: `play(key, harness_path, cell, split, instance_ids, visible_only)`.

Returns one dict per instance, with only what the method may read:
    {"instance": id, "trace": trace.md text, "score": float | None, "success": bool | None}
Before deployment (`visible_only=False`) score and success are the search split's scores, which every
method may use. At test time (`visible_only=True`) they are None, and the trace is what the model saw
and did and what the environment said, with the broker's score line removed.

MailboxBackend plays through the cluster's long-lived worker (mb/evolve/worker.py) with
mb/evolve/remote.play. It never reads play_results/<key>/private.json; `private_scores` is for our
reports only and is not called by the method's code.
"""
from __future__ import annotations

import json
from pathlib import Path


class MailboxBackend:
    def __init__(self, repo: Path, poll=10):
        from mb.evolve.remote import Mailbox
        self.box, self.poll = Mailbox(Path(repo).expanduser()), poll

    def play(self, key, harness_path, cell, split, instance_ids, visible_only):
        from mb.evolve import remote
        res = self.box.repo / "play_results" / key
        self.box.pull()
        if (res / "private.json").exists():            # finished before an interruption: reuse it
            vis = res / "visible"
        else:
            vis = remote.play(self.box, key, harness_path, cell, split, instance_ids,
                              visible_only=visible_only, poll=self.poll)
        if (vis / "rejected.json").exists():
            raise RuntimeError(f"{key}: harness rejected: {(vis / 'rejected.json').read_text()[:400]}")
        return read_visible(vis, instance_ids, visible_only)

    def private_scores(self, key):
        """For reports only (never called by the method)."""
        p = self.box.repo / "play_results" / key / "private.json"
        return json.loads(p.read_text()) if p.exists() else None


def read_visible(vis: Path, instance_ids, visible_only):
    from mb.screen import _slug
    per = {}
    if not visible_only and (vis / "scores.json").exists():
        per = json.loads((vis / "scores.json").read_text()).get("per_episode", {})
    out = []
    for inst in instance_ids:
        d = vis / "episodes" / _slug(inst)
        trace = (d / "trace.md").read_text() if (d / "trace.md").exists() else ""
        if not trace:
            raise RuntimeError(f"no trace for {inst} in {vis}")
        r = per.get(inst, {}) if not visible_only else {}
        if not visible_only and (d / "result.json").exists():
            r = json.loads((d / "result.json").read_text())
        out.append({"instance": inst, "trace": trace,
                    "score": None if visible_only else r.get("score"),
                    "success": None if visible_only else bool(r.get("success"))})
    return out
