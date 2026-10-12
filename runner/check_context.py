"""Is the failure really the model's, or did the harness hide the history?

A 95% dead-guess rate only means "cannot track its own belief" if the belief
was actually in front of it. Three things have to hold: the full transcript is
resent every turn, the feedback is in it, and nothing was truncated by
max-model-len. If any fails, the finding is about the harness, not the model.
"""

import json
import sys


def main(path, max_model_len=4096):
    recs = json.load(open(path))
    r = recs[0]
    c = r["conversation"]
    print(f"一局消息数 {len(c)}；角色序列前 9: {[m['role'] for m in c[:9]]}")
    print("\n--- 前几条实际内容 ---")
    for m in c[1:7]:
        txt = m["content"].replace("\n", " ")[:100]
        print(f"  [{m['role']:<9}] {txt}")

    # Rebuild the prompt as it stood on each turn and size it.
    worst = 0
    for rec in recs:
        n = sum(len(m["content"]) for m in rec["conversation"])
        worst = max(worst, n)
    approx = worst // 4
    print(f"\n最长一局的完整 transcript: {worst} 字符 ≈ {approx} tokens "
          f"(max-model-len {max_model_len})")
    print(f"  余量: {max_model_len - approx} tokens "
          f"{'—— 安全' if approx < max_model_len * 0.8 else '—— 偏紧，可能截断'}")

    # Does the feedback actually appear in the transcript the model sees?
    fb = [m for m in c if m["role"] == "user"
          and "digits are present" in m["content"]]
    print(f"\n反馈消息出现在上下文里的次数: {len(fb)} "
          f"(应约等于轮数 {len(r['guess_list']) - 1})")

    # Did the model ever re-guess something the transcript already showed?
    g = [x[0] for x in r["guess_list"]]
    dup = len(g) - len(set(g))
    print(f"示例局: {len(g)} 次猜测, 其中 {dup} 次是重复 "
          f"(历史就在眼前)")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4096)
