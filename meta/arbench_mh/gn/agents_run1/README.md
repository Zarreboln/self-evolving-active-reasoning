# run1's evolved Guess Number harnesses

The fifteen harnesses the first evolution run produced, plus the two seeds it started from.
They were recovered from the cluster after `meta_harness.py --fresh` cleared the shared
`gn/agents/` directory, where they had been living untracked: the loop writes candidates there
and git only ever tracked the seeds, so for a week these existed on two servers and nowhere else.

They are kept out of `gn/agents/` on purpose. A later run's proposer lists that directory to see
what it may build on, and `verified_menu.py`, `packed_group_echo.py` and `plan_table_endgame.py`
tell it what the first run found before it has looked at anything. Each run gets its own agents
directory; this one is the archive.
