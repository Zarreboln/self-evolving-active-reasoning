# `bath` — the Slurm cluster the baselines run on

Two nodes, eight H100 80GB each, shared with four other accounts. Login node `ml-16-login-001`
(192.222.48.222), `ssh bath`. This directory holds everything that runs there: the serving job, the
baseline drivers, and the sharding tools that make a single card usable.

Our layout on the cluster mirrors these files at `~/code`, with `~/results`, `~/logs`, `~/venv` and
`~/hf_cache` beside it. The Jev referee shim (`runner/jev_referee_shim.py`) runs on the **login
node**, not under Slurm: it needs no GPU, so the whole card goes to the reasoner.

## What runs what

| File | Role |
|---|---|
| `serve_7b.sbatch` | one vLLM server, one H100, publishes `~/results/<tag>.endpoint` |
| `run_sharded.sh` | one baseline combo, `NSHARD` concurrent workers over contiguous slices |
| `run_all_sharded.sh` | all 15 combos (3 tasks × BALAR's 5 methods), skipping those already at 100 |
| `shard_tool.py` | split a test set into blocks; merge the workers' logs back exactly |
| `check_shard_roundtrip.py` | proves the merge reproduces a sequential run record for record |
| `sweep_baselines.sh` | re-enters the queue until no combo is short |
| `count_incomplete.py` | how many of the 15 combos are not yet at 100 |
| `run_baselines.sh`, `run_all_baselines.sh` | the original sequential runner, kept for reference |

## AR-Bench's evaluator is sequential, and that wastes the card

There is no `ThreadPool`, `concurrent.futures`, `asyncio` or `multiprocessing` anywhere in
`arbench/`. Every method is a plain `for i in tqdm(range(len(logs), len(dataset)))` loop, and inside
one instance the 25 turns are sequential too, so the served model sees exactly **one request in
flight**. On an H100, `gn/proactive_cot` measured **121 s per instance — 3 h 22 m for 100**, with the
GPU reporting 100 % utilization the whole time. That number means "a kernel is resident", not
"the card is busy"; a 7B model decoding one stream leaves most of an H100 idle.

Both `--data_path` and `--output_path` are flags, so concurrency does not require editing the
benchmark. `run_sharded.sh` gives each of 16 workers a contiguous slice of the test set and its own
log file, and `shard_tool.py merge` concatenates them with `idx` rewritten to the global index.

Measured on the same card, same server, same method:

| | sequential | 16 shards |
|---|---|---|
| per instance | 121 s | 125–150 s |
| in 5 minutes | 2.5 instances | 24 instances |
| 100 instances | 3 h 22 m | ~17 min |

Per-instance latency rises about 20 %, which is the whole cost of running sixteen at once: vLLM
batches them almost for free. Roughly **11× throughput**.

Contiguous slices are what make this safe to report. Shard *j* holds `[offset_j, offset_j + n_j)`,
each record's `idx` is local to the slice it ran on, so the merge sets `idx = offset_j + position`
and concatenating in shard order reproduces the single-process file. `check_shard_roundtrip.py`
verifies this against `gn_few_shot.json`, a known-good sequential run: identical record for record.
A merge with any shard short refuses to write, so a partial file can never be mistaken for a result.

## What the second role has to be, per task

The four tasks do not share a second model, and the difference decides which judge can serve them.

| Task | Second role | What it has to produce | Who serves it |
|---|---|---|---|
| `gn` | none | the feedback is a pure function of the guess | nobody |
| `sp` | referee | one of Yes / No / Unknown | **Jev** (typed choice, calibrated) |
| `dc` | a suspect | a sentence in character (`respond_template`) | **Kimi** (`kimi-k3`) |
| `md` | a patient | a sentence from the case's fact list | **Kimi** (`kimi-k3`) |

Only SP's role is a typed decision, which is the shape Jev answers; GN needs no judge at all. DC and
iCraft-MD both need free text, and the Jev shim refuses anything that is not a referee turn — by
design, so a silent fallback cannot put a different model's words in a transcript. Pointing DC at
the shim would have failed every turn, which is what the per-task blocks in `run_sharded.sh` fix.

Kimi is OpenAI-compatible, so it needs no shim, but **`kimi-k3` accepts only `temperature=1` and
`top_p=0.95`** and rejects anything else with a 400; AR-Bench's 0.7 has to be overridden. Model
choice matters for cost, not just quality: on the same three patient turns, `kimi-k2.6` spent
1488 / 697 / 2434 completion tokens on one-sentence answers where `kimi-k3` spent 141 / 131 / 407,
about seven times less, at the same fidelity. The key lives in `~/.moonshot_token` (0600) and is
read into the environment rather than passed as an argument.

## Traps

- **QOS, not `--time`, is what gets a job rejected.** `h100_1` allows `dev, short, mid, long,
  lowprio, devr, reserved` and **not** `cpu`. A job that lands on `QOS=cpu` sits PENDING forever
  with `Job's QOS not permitted to use this partition`; it is being refused, not queued. Ask for the
  QOS explicitly: `sbatch --qos=mid --time=24:00:00`. Maxwall is `dev` 8 h, `short` 12 h, `mid` 24 h,
  `long` 3 d. Account `tmp_l` is allowed all of them.
- **`TimeLimit` is wall-clock.** It runs from the moment the job gets the node, whether or not the
  GPU is doing anything. Idle time inside a serving job is spent budget.
- **Hand over before the clock runs out.** Both drivers pick whichever published endpoint answers
  `/health`, so a replacement server is found automatically — but an *in-flight* combo keeps the URL
  it started with. Start the replacement, then retire the expiring job's `.endpoint` file so later
  combos cannot bind to it, and only cancel the old job once nothing is bound to it.
- **The system `nvcc` is CUDA 12.0 and torch is cu130.** Every runtime JIT path has to stay off, or
  vLLM dies compiling: `VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0
  VLLM_ATTENTION_BACKEND=FLASH_ATTN`. `ninja` and `fire` are needed and are not in the system image.
- **Never overwrite a shell script that is running.** `bash` reads a script incrementally by byte
  offset, so replacing the file in place (which is what `scp` does) makes the running shell resume
  at that offset in the new contents. `gn/proactive_cot` lost its merge exactly this way: all 16
  shards finished clean, and the next line read as `unexpected EOF while looking for matching '"'`.
  Copy to a temporary name and `mv` it, which is atomic and leaves the running process on the old
  inode; the data survives either way, so recover by running `shard_tool.py merge` by hand.
- **`pgrep`/`pkill -f` match their own command line.** They have killed the ssh session running them.
  List with `ps … | grep '[p]attern'` and kill explicit PIDs.
- **Background work needs `setsid`**, or it dies with the ssh connection.

## Reproduce

```bash
sbatch --qos=mid --time=24:00:00 \
  --export=ALL,REPO=Qwen/Qwen2.5-7B-Instruct,TAG=qwen25_7b serve_7b.sbatch
python ~/code/jev_referee_shim.py --port 8099 --log ~/logs/referee.jsonl   # login node
setsid nohup bash run_all_sharded.sh > ~/logs/all_baselines.out 2>&1 &
setsid nohup bash sweep_baselines.sh > ~/logs/sweep.out 2>&1 &
```
