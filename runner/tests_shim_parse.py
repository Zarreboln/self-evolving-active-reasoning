"""Cases for the referee-prompt parser, including the one that produced 50 meaningless verdicts."""
import json
import sys

sys.path.insert(0, "/home/tmp_l/code/AR-Bench")
sys.path.insert(0, "/home/tmp_l/code")
from arbench.reasoner.sp.prompt import system_prompt_with_2shots as TPL
import jev_referee_shim as shim

fails = 0


def check(name, got, want):
    global fails
    ok = got == want
    print(("  ok   " if ok else "  FAIL ") + name + ("" if ok else f"\n         got {got!r}\n        want {want!r}"))
    if not ok:
        fails += 1


puzzles = json.load(open("/home/tmp_l/code/AR-Bench/data/sp/test.json"))

print("AR-Bench's real 2-shot referee prompt, every test puzzle:")
bad = 0
for p in puzzles:
    msg = TPL.format(question=p["surface"], answer=p["bottom"])
    turn = shim.referee_turn([{"role": "system", "content": msg},
                              {"role": "user", "content": "Did the man die?"}])
    if turn is None or turn[0] != p["surface"].strip() or turn[1] != p["bottom"].strip():
        bad += 1
check(f"all {len(puzzles)} puzzles extract exactly", bad, 0)

print("a plain tagged pair (our own probes):")
turn = shim.referee_turn([
    {"role": "system", "content": "You are the referee.\n<Surface>\nA man asks for water and the "
     "bartender points a gun at him; he says thank you and leaves.\n<Bottom>\nThe man had hiccups "
     "and the shock cured them, which the bartender intended.\n"},
    {"role": "user", "content": "Did the man have hiccups?"}])
check("tagged form still parses", turn is not None and "hiccups" in (turn[1] or ""), True)

print("instructions only, no instance:")
only_rules = TPL.split("## Question Content")[0]
check("refused", shim.referee_turn([{"role": "system", "content": only_rules},
                                    {"role": "user", "content": "Did he die?"}]), None)

print("an instance whose story is a scrap of the rules:")
check("refused by plausible_story",
      shim.plausible_story("and you are given the", "Judging Rules blah"), False)

sys.exit(1 if fails else 0)
