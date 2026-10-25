"""Measure information self-lock: how often the questioner re-asks what it already asked.

The behaviour this quantifies was first seen by hand in the SP referee log, where the reasoner sent
four near-identical questions in a row and the judge returned Unknown to each at 0.88-0.94
confidence. It had committed to one hypothesis and kept rewording it instead of changing axis.

A question is normalised to its set of content words, and two questions count as near-duplicates at
Jaccard >= 0.8 on those sets. Reported per run:

  distinct    distinct questions as a fraction of questions asked (1.0 = never repeats)
  exact dup   questions that repeat an earlier one character for character
  run>=3      episodes containing three or more near-duplicates in a row
  worst run   the longest such run seen in the run

Usage: question_repetition.py md <file.json> [more.json ...]
       question_repetition.py jsonl <referee.jsonl>
"""
import json
import re
import sys

STOP = set("a an the is are was were do does did did you your have has had any of in on at to and "
           "or if there been be with that this it he she they them his her for from as by "
           "whether not no yes".split())
WORD = re.compile(r"[a-z0-9']+")


def norm(q):
    q = re.sub(r"^\s*q\s*[:.]\s*", "", (q or "").strip(), flags=re.I)
    return frozenset(w for w in WORD.findall(q.lower()) if w not in STOP)


def jaccard(a, b):
    if not a or not b:
        return 1.0 if a == b else 0.0
    return len(a & b) / len(a | b)


def containment(a, b):
    """How much of the smaller question is inside the larger one.

    Jaccard misses the form of self-lock actually observed, where the reasoner keeps the question
    and appends clauses to it -- "...potentially leading to her death" then "...and was there any
    recent change in her medication regimen that might have contributed to her sudden death?".
    The added words push Jaccard below any sensible threshold while the original question is still
    being asked verbatim inside the new one. Containment catches exactly that.
    """
    if not a or not b:
        return 1.0 if a == b else 0.0
    return len(a & b) / min(len(a), len(b))


def episode_stats(questions, thresh=0.8, cthresh=0.9):
    sets = [norm(q) for q in questions]
    n = len(questions)
    distinct_exact = len(set(q.strip() for q in questions))
    worst = worst_c = 1
    cur = cur_c = 1
    for i in range(1, n):
        cur = cur + 1 if jaccard(sets[i], sets[i - 1]) >= thresh else 1
        worst = max(worst, cur)
        cur_c = cur_c + 1 if containment(sets[i], sets[i - 1]) >= cthresh else 1
        worst_c = max(worst_c, cur_c)
    near = sum(1 for i in range(n)
               if any(jaccard(sets[i], sets[j]) >= thresh for j in range(i)))
    contained = sum(1 for i in range(n)
                    if any(containment(sets[i], sets[j]) >= cthresh for j in range(i)))
    return n, distinct_exact, near, worst, contained, worst_c


def report(label, episodes):
    tot = dup_exact = near = contained = 0
    worsts, worsts_c, runs3, runs3_c = [], [], 0, 0
    for qs in episodes:
        if not qs:
            continue
        n, de, nr, w, ct, wc = episode_stats(qs)
        tot += n
        dup_exact += n - de
        near += nr
        contained += ct
        worsts.append(w)
        worsts_c.append(wc)
        runs3 += w >= 3
        runs3_c += wc >= 3
    if not worsts:
        print("%-30s (no questions found)" % label)
        return
    print("%-30s asked %5d | jaccard: rep %.2f run>=3 %3d/%-3d worst %2d | contained: rep %.2f "
          "run>=3 %3d/%-3d worst %2d | exact-dup %4d"
          % (label, tot,
             near / max(tot, 1), runs3, len(worsts), max(worsts),
             contained / max(tot, 1), runs3_c, len(worsts_c), max(worsts_c),
             dup_exact))


def main():
    mode = sys.argv[1]
    if mode == "md":
        for path in sys.argv[2:]:
            try:
                d = json.load(open(path))
            except Exception:
                print("%-34s (unreadable)" % path.split("/")[-1])
                continue
            eps = []
            for r in d:
                eps.append([m["content"] for m in (r.get("record") or [])
                            if m.get("role") == "assistant" and m.get("content", "").strip().lower().startswith("q")])
            report(path.split("/")[-1], eps)
    else:
        qs = []
        for line in open(sys.argv[2]):
            try:
                qs.append(json.loads(line).get("question") or "")
            except Exception:
                pass
        # the referee log is one flat stream, so treat it as a single episode
        report(sys.argv[2].split("/")[-1] + " (flat stream)", [qs])


if __name__ == "__main__":
    main()
