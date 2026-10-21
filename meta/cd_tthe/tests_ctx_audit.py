"""Self-test for the context arm's audit: the legal shape passes, six smuggling attempts fail."""
import sys
sys.path.insert(0, "/Users/zarreboln/Desktop/ar_harness/meta/cd_tthe")
from ctx_audit import text_only

OK = '''"""doc."""
from ctx_harness import SPCtxHarness


class Harness(SPCtxHarness):
    """notes."""
    PLAYBOOK = """
    - ask broad before narrow
    """
'''
BAD = {
 "a helper function": OK.replace('    PLAYBOOK = """', '    def helper(self):\n        return 1\n    PLAYBOOK = """'),
 "control flow in the class": OK.replace('    PLAYBOOK = """', '    if True:\n        pass\n    PLAYBOOK = """'),
 "computed playbook": '''from ctx_harness import SPCtxHarness\n\n\nclass Harness(SPCtxHarness):\n    PLAYBOOK = "a" * 3\n''',
 "extra import": OK.replace("from ctx_harness import SPCtxHarness", "import re\nfrom ctx_harness import SPCtxHarness"),
 "overriding solve": OK.replace('    PLAYBOOK = """', '    def solve(self):\n        return ""\n    PLAYBOOK = """'),
 "top-level statement": OK + "\nX = 1\n",
 "wrong base class": OK.replace("SPCtxHarness):", "object):").replace("import SPCtxHarness", "import SPCtxHarness"),
}
v = text_only(OK)
print(("PASS" if not v else "FAIL " + str(v)), "legal shape")
fails = 0
for name, src in BAD.items():
    v = text_only(src)
    ok = bool(v)
    fails += not ok
    print(("PASS" if ok else "FAIL"), f"rejects {name}", ("" if ok else "<- slipped through"))
# the committed seed must pass
seed = open("/Users/zarreboln/Desktop/ar_harness/meta/cd_tthe/agents_sp/ctx_bare.py").read()
v = text_only(seed)
print(("PASS" if not v else "FAIL " + str(v)), "committed seed agents_sp/ctx_bare.py")
sys.exit(1 if fails else 0)
