"""Helpers a harness may import (`from harness_lib import ...`). Copied from the suite's seed harness."""
import re

WINDOW = 16          # exchanges of history the seed keeps
MAX_ADMISSIBLE = 60
SYSTEM = ("You are playing a text game. Read the instructions, then on every turn reply with "
          "exactly one action in the format the game asks for. You may think briefly first, but "
          "end your reply with a line of the form `Action: <action>`.")


def with_admissible(text, env):
    """Append the admissible actions, when the game provides them and there are not too many."""
    adm = env.admissible()
    if adm and len(adm) <= MAX_ADMISSIBLE:
        text += "\n\nAdmissible actions: " + ", ".join(adm)
    return text


def extract_action(reply):
    """The text after the last `Action:` line, else the last non-empty line."""
    m = re.findall(r"^\s*Action\s*:\s*(.+?)\s*$", reply or "", re.M | re.I)
    if m:
        a = m[-1]
    else:
        lines = [l.strip() for l in (reply or "").splitlines() if l.strip()]
        a = lines[-1] if lines else ""
    return a.strip().strip("`").rstrip(".!").strip()
