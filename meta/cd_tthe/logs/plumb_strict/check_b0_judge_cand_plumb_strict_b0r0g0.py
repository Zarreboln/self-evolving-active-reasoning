import json, subprocess, sys, os
from pathlib import Path
HERE = Path('/Users/zarreboln/Desktop/ar_harness/meta/cd_tthe')
name = 'cand_plumb_strict_b0r0g0'; idx = [156, 11, 250, 128, 98]; solver = {'base_url': None, 'model': 'Qwen/Qwen2.5-7B-Instruct', 'mock': 'sequential', 'seed': 0, 'cache_dir': '/Users/zarreboln/Desktop/ar_harness/meta/cd_tthe/logs/plumb_strict/solver_cache'}; arm = 'scratch'
out_dir = Path('/Users/zarreboln/Desktop/ar_harness/meta/cd_tthe/logs/plumb_strict') / "checks" / 'b0_judge_cand_plumb_strict_b0r0g0'; out_dir.mkdir(parents=True, exist_ok=True)
for i in idx:
    task = out_dir / f"task_{i}.json"; res = out_dir / f"res_{i}.json"
    task.write_text(json.dumps({"harness": name, "index": i, "output": str(res), "solver": solver, "arm": arm, "budget": 18}))
    try:
        subprocess.run([sys.executable, str(HERE / "solve_worker.py"), "--task", str(task)], timeout=120, check=False)
    except subprocess.TimeoutExpired:
        print(f"instance {i}: TIMEOUT after 120s"); continue
    if not res.exists():
        print(f"instance {i}: no result (worker died)"); continue
    r = json.loads(res.read_text()); s = r.get("signals") or {}
    if True:
        print(f"instance {i}: status={r['status']} parsed_final_guess={s.get('parsed_guess')} queries_used={s.get('n_queries')} calls={r.get('solver_calls')}")
    else:
        print(f"instance {i}: status={r['status']} health={s.get('health')} queries={s.get('n_queries')} "
              f"uniq_end={s.get('uniq_end')} alive_end={s.get('n_alive_end')} zero_info={s.get('zero_info')} "
              f"repeat={s.get('repeat')} eff={s.get('eff')} calls={r.get('solver_calls')}")
    if r.get("error"): print("   error:", r["error"][-400:].replace("\n", " | "))
