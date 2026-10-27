"""Remove UoT from a copy of AR-Bench so no proposer can read it.

The loop gives the proposer full Bash with no sandbox, exactly as upstream Meta-Harness does, so
the only reliable way to keep one of the published baselines out of its sight is for the text not
to exist in the tree it can reach. Moving a file is not enough -- a proposer read the sealed splits
directory on its eighth command. This deletes:

  * the UoT runner and the helpers only it reaches (spans verified by ast reachability)
  * the prompt templates only those helpers use (verified unreferenced elsewhere in the repo)
  * the `elif method == "uot"` dispatch branch and main's simulate_depth parameter
  * the README rows and the shell scripts that name the method

Everything the other methods and our own three environments need is untouched. The full tree is
kept outside the loop's working area for our own baseline reruns.
"""
import ast, os, re, subprocess, sys

ROOT = sys.argv[1]

EVAL_SPANS = {                       # (file, first line, last line) of the UoT-only defs
    "arbench/reasoner/gn/gn_evaluator.py": (293, 544),
    "arbench/reasoner/dc/dc_evaluator.py": (818, 1114),
    "arbench/reasoner/sp/sp_evaluator.py": (568, 1029),
}
PROMPT_DROPS = {
    "arbench/reasoner/gn/prompt.py": ["SYSTEMPROMPT", "MULTI_GUESS_PROMPT",
                                      "MULTI_GUESS_PROMPT_WITH_ANSWERS", "FEEDBACKPROMPT"],
    "arbench/reasoner/dc/prompt.py": ["uot_generate_questions_template",
                                      "uot_update_answer_set_template", "uot_conclusion_template"],
    "arbench/reasoner/sp/prompt.py": ["uot_initial_answer_set_prompt",
                                      "uot_candidate_questions_prompt",
                                      "uot_answer_set_reduction_prompt",
                                      "uot_update_answer_set_prompt", "uot_final_answer_prompt"],
}


def indent_of(line):
    return len(line) - len(line.lstrip())


def cut_elif_branch(lines):
    """Drop the `elif method == "uot":` branch, however many lines its call spans."""
    for i, l in enumerate(lines):
        if re.match(r'\s*elif\s+method\s*==\s*["\']uot["\']\s*:', l):
            base = indent_of(l)
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or indent_of(lines[j]) > base):
                j += 1
            return lines[:i] + lines[j:], True
    return lines, False


def main():
    # 1. the evaluators
    for rel, (a, b) in EVAL_SPANS.items():
        p = os.path.join(ROOT, rel)
        lines = open(p).read().split("\n")
        assert "uot" in "\n".join(lines[a - 1:b]).lower(), rel
        lines = lines[:a - 1] + lines[b:]
        lines, cut = cut_elif_branch(lines)
        assert cut, f"{rel}: dispatch branch not found"
        src = "\n".join(lines)
        # main's simulate_depth exists only for this method
        src = re.sub(r",\s*\n?\s*simulate_depth:\s*int\s*=\s*\d+", "", src)
        src = re.sub(r"\s*simulate_depth:\s*int\s*=\s*\d+\s*,", "", src)
        open(p, "w").write(src)
        print(f"  {rel}: 删 {b - a + 1} 行 + 分发分支")

    # 2. the prompt templates
    for rel, names in PROMPT_DROPS.items():
        p = os.path.join(ROOT, rel)
        tree = ast.parse(open(p).read())
        lines = open(p).read().split("\n")
        drop = set()
        for n in tree.body:
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) \
                    and n.targets[0].id in names:
                drop.update(range(n.lineno, n.end_lineno + 1))
        # nothing outside this file may still reference them
        for name in names:
            hits = subprocess.run(["grep", "-rlw", name, ROOT, "--include=*.py",
                                   "--include=*.sh"], capture_output=True, text=True).stdout.split()
            hits = [h for h in hits if os.path.abspath(h) != os.path.abspath(p)
                    and "__pycache__" not in h]
            assert not hits, f"{name} still used by {hits}"
        open(p, "w").write("\n".join(l for i, l in enumerate(lines, 1) if i not in drop))
        print(f"  {rel}: 删 {len(names)} 个模板 ({len(drop)} 行)")

    # 3. README and the launch scripts
    p = os.path.join(ROOT, "README.md")
    out = []
    for l in open(p):
        if re.search(r"simulate_depth", l):
            continue
        l = re.sub(r"<br>Uncertainty[ _-]of[ _-]Thought(\s*\(UoT\))?", "", l)
        l = re.sub(r"Uncertainty[ _-]of[ _-]Thought(\s*\(UoT\))?<br>", "", l)
        out.append(l)
    open(p, "w").write("".join(out))
    print("  README.md: 删 simulate_depth 行 + 方法名")

    for name in ["dc", "gn", "sp"]:
        p = os.path.join(ROOT, f"script/evaluation/{name}_evaluation.sh")
        if not os.path.exists(p):
            continue
        out = []
        for l in open(p):
            if re.search(r"SIMULATE_DEPTH|simulate_depth", l):
                continue
            l = re.sub(r",\s*uot\b", "", l)
            l = l.replace('METHOD="uot"', 'METHOD="zero_shot"')
            out.append(l)
        open(p, "w").write("".join(out))
    print("  script/evaluation/*.sh: METHOD -> zero_shot, 删 simulate_depth")


    # spellings the spans and the first regexes did not cover
    p = os.path.join(ROOT, "README.md"); s = open(p).read()
    open(p, "w").write(re.sub(r"(?i)<br>uncertainty of thoughts?", "", s))
    p = os.path.join(ROOT, "arbench/reasoner/sp/sp_evaluator.py"); s = open(p).read()
    s = re.sub(r'\n\s*"uot":\s*propose_template,.*', "", s)
    open(p, "w").write(s.replace(", Simulate Depth: {simulate_depth}", ""))
    p = os.path.join(ROOT, "arbench/reasoner/dc/dc_evaluator.py"); s = open(p).read()
    open(p, "w").write(s.replace(", Simulate Depth: {simulate_depth}", ""))
    print("  README/sp/dc: 残留拼法")

    # AppleDouble junk from an old rsync off a Mac; never importable, only breaks compileall
    n = 0
    for d, _, fs in os.walk(ROOT):
        for f in fs:
            if f.startswith("._"):
                os.remove(os.path.join(d, f)); n += 1
    print(f"  删掉 {n} 个 ._* 文件")

    for d, dirs, _ in os.walk(ROOT):
        for x in list(dirs):
            if x == "__pycache__":
                subprocess.run(["rm", "-rf", os.path.join(d, x)])
                dirs.remove(x)
    print("  清掉所有 __pycache__")


main()
