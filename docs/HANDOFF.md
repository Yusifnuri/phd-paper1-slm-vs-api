# Handoff (written 2026-11-21 ~18:30 Istanbul)

For a fresh Claude session continuing this project. Read this, then `docs/PLAN.md` and `docs/research/2026-11-21-verified-findings.md`. The user (PhD student, Istanbul Aydın University) wants replies in **Turkish**, code/commits/paper text in **English**. Rules: never invent numbers, versions, prices or citations (say when unsure); no Claude co-author or attribution trailers in commits (the user's repos forbid them); never overwrite the MSc thesis repo `/Users/yusifnuri/slm-benchmark` (frozen at tag `msc-thesis` = commit `5497c5b`). Hugging Face and API keys are never pasted into chat or stored in the repo.

## Where things are

| Thing | Where |
|---|---|
| This repo (GitHub: `Yusifnuri/phd-paper1-slm-vs-api`, `origin`; `main` and the three branches below pushed 2026-11-21; pilot results in `results/pilot-2026-11-21/`; pilot-train branch is now at `426d98e`) | `/Users/yusifnuri/phd-paper1-slm-vs-api`, branch `main` (docs only) |
| Step 2: evaluation + statistics | branch `claude/eval-harness-3f9302`, commit `ae0aabd`, worktree under `.claude/worktrees/` |
| Step 4: API arm harness | branch `claude/api-harness-827c78`, commit `6743026` (fixes its schema test) |
| Step 3: pilot training harness | branch `claude/pilot-train-226845`, commit `3a5c3a3` |
| None of these is merged into `main` | merging needs the user's OK; a scratch merge of eval + api ran 106 tests green |

## Server (university H200 portal)

- 1 × NVIDIA H200 NVL, 143,771 MiB; Ubuntu 24.04 container, user `jovyan`, passwordless sudo, 32 cores, 96 GB RAM. The container's own disk is wiped on every restart, **but `~/vault/Defaukt` (sic, typo in the name) is a persistent NFS mount** (`/mnt/pool/users/yusifnuri`, shared 4 TB pool, ≈595 GB free on 2026-11-21, writable by `jovyan`; the user confirmed it is not wiped). It also holds the user's older project `Animation_Studio`: never touch that. Our folder is `~/vault/Defaukt/paper1/`. Keep code, finished adapters, result files and (to avoid re-downloading ≈40 GB of models every week) the Hugging Face cache there (`HF_HOME=~/vault/Defaukt/hf_cache`), but write training checkpoints to the local disk because NFS is slower, and keep pushing adapters to the Hub as a second copy. Backed up there on 2026-11-21: `paper1/pilot` (code at `3a5c3a3`, without checkpoints).
- Credits: 200 per week, reset every Sunday; H200 = 15 credits/hour (≈13 h/week). The server bills while it runs: stop it in the portal when idle.
- Reach it with `ssh jovyan@h200-madrid` (Tailscale, tailnet SSH rule set to `accept`, works non-interactively). Code lives in `~/paper1/pilot` (copied with `tar | ssh`, not git).
- Installed there: torch 2.9.0+cu128, transformers 4.57.3, peft 0.18.0, trl 0.26.2, bitsandbytes 0.49.0, vllm 0.13.0, datasets 4.4.1. Missing: flash_attn, seqeval, codecarbon. `requirements/train.txt` on the pilot branch pins NEWER versions (torch 2.15, transformers 5.24) that were never installed: do not run `scripts/bootstrap_h200.sh` blindly, it would try to install them.
- After a server restart the Tailscale node and its IP reset (state lives in `/tmp`). `~/vault/Defaukt/paper1/start_h200.sh` (untested) restarts tailscaled with its state kept in the vault and brings up Tailscale SSH + sshd; the user runs it once from the portal terminal. Expect a changed host key on the Mac (`ssh-keygen -R h200-madrid`, then verify the fingerprint with `ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub` on the server) and delete the old `h200-madrid` machine in the Tailscale admin page first, or the new one becomes `h200-madrid-1`.
- Hugging Face login on the server worked for user `yusifnuri` (needs redoing after every restart: `hf auth login`). Adapters must be pushed to the Hub after every run (`--hub-repo yusifnuri/<name>`, private).
- The server's locale is not UTF-8: run Python with `PYTHONUTF8=1`. One pilot test (`test_no_test_split_reference_in_sources`) fails there for that reason only.

## Pilot status

- Smoke test (N=50, 10 steps): Phi-4-mini-instruct and Qwen3-4B-Instruct-2507 ran OK (train 1.8 s and 2.2 s, peak 14.0 and 17.0 GB). OLMo-3-7B-Instruct was still downloading; whether transformers 4.57.3 supports it was unverified.
- Real pilot started ~18:55 Istanbul on the server (12 runs: 3 models × lora/qlora/dora/rslora, N=1000, seed 42, log `~/paper1/pilot/pilot.log`, results `runs/pilot.jsonl`, adapters to the private Hub repo `yusifnuri/paper1-pilot-adapters`). First result: Phi-4-mini LoRA N=1000 trained in 18 s with 0.749 validation accuracy. The server runs code at `3a5c3a3`; the pilot-train session has since committed `426d98e` (pins matched to the server's real versions, an architecture check) which has NOT been copied to the server yet: copy it only after the grid finishes.
- **Pilot finished: 12/12 runs ok** (N=1000, seed 42, validation accuracy; train s / peak GB): Phi-4-mini lora .749 18s 15.6, qlora .704 25s 15.3, dora .750 23s 18.5, rslora .791 18s 15.6; Qwen3-4B lora .754 29s 19.3, qlora .746 42s 18.4, dora .757 41s 23.0, rslora .811 28s 19.3; OLMo-3-7B lora .735 36s 29.6, qlora .724 43s 26.3, dora **0.000** 57s 37.6, rslora .777 36s 29.6. All adapters pushed to the Hub. Single seed: differences between techniques are within noise until the 3-seed curve exists. DoRA overhead in training time ≈ +28% (Phi), +41% (Qwen) vs LoRA; peak memory ≈ +19% / +19% / +27% (OLMo).
- **OLMo-3 DoRA is a pipeline failure, not a result**: every output is `!!!!!!!…` (degenerate generation, likely NaN/diverged training or a DoRA+bf16 problem on this architecture; not yet investigated). Exclude that cell, check `first_train_loss`/`final_train_loss` in `runs/pilot.jsonl`.
- 4–12% of validation outputs are unparseable for the other runs: the models invent plausible intent names that are not among the 77 labels (e.g. `card_payment_not_received_by_cardholder`); the parser is fine. They count as wrong (thesis rule); constrained decoding or label scoring is a possible later change.
- The queue script `queue_curve.sh` never fired: it waited on the pilot's PID, which stayed a defunct zombie (`kill -0` succeeds on zombies). Lesson: never wait on a PID; `pgrep -f` also matches its own ssh command line. The curve was started directly with `run_curve.sh` (log `curve.log`, results `runs/curve.jsonl`, prints `CURVE_DONE`).
- **Curve finished: 36/36 ok** (BANKING77 validation, LoRA, 3 seeds; results in `runs/curve.jsonl`, backed up to `~/vault/Defaukt/paper1/runs_backup/`). Accuracy mean±SD by N = 50 / 200 / 1000 / 5000: Phi-4-mini .039±.001 / .084±.012 / .726±.020 / .897±.006; Qwen3-4B .057±.006 / .171±.006 / .757±.011 / .898±.002; OLMo-3-7B .054±.004 / .155±.005 / .736±.004 / .903±.007. Net training time for all 36 runs was only 0.42 h (+0.11 h load/eval); N=5000 train seconds: Phi 86, Qwen 135, OLMo 182.
- **Confound to fix before any H1 claim:** the fixed 3 epochs give only ≈9 optimizer steps at N=50 (effective batch 16), and 90–95% of outputs are unparseable at N ≤ 200, so small-N accuracy mostly measures undertraining, not label budget. Use a minimum number of steps (or tune epochs per N on validation) for all arms, and report it.
- Compute re-estimate, rough: BANKING77-type short tasks cost ≈0.5 GPU-h per 36 runs, far below the earlier 125–280 h estimate; the long-input task (BillSum, ≈2.6k tokens) will dominate. Naive scaling from the measured ≈5,000 tok/s: ≈2–3 h per model per seed for the four N levels, i.e. ≈20+ h for summarisation LoRA alone; short-sequence tok/s may not transfer, so measure it before planning.
- (Superseded) Queued ~19:15 Istanbul on the server (`~/paper1/pilot/queue_curve.sh`, log `curve.log`, prints `CURVE_DONE`): after the pilot grid ends it runs the BANKING77 label-budget curve, LoRA × 3 models × N ∈ {50,200,1000,5000} × seeds 42/43/44 = 36 runs into `runs/curve.jsonl`, adapters to the same Hub repo. Validation split only, exploratory (no pre-registration exists yet). The user must stop the server in the portal afterwards (billing is not stopped from inside), and back up `runs/` to `~/vault/Defaukt/paper1/`.
- (Original instruction, for a rerun) `cd ~/paper1/pilot && PYTHONUTF8=1 nohup python3 scripts/run_pilot.py --hub-repo yusifnuri/paper1-pilot-adapters > pilot.log 2>&1 &` (default: all registry models × techniques at N=1000, seed 42). Never run two grids at once: it corrupts the timing measurements. Read `runs/pilot.jsonl`, then recompute the GPU-hour estimate in `docs/PLAN.md`.

## Open decisions (Step 0, user)

NER dataset (Few-NERD, share-alike, or Broad Twitter Corpus, Twitter terms), financial dataset (SEntFiN or Twitter Financial News), replacement for code generation (Spider text-to-SQL, CC BY-SA, licence not yet verified), API arm (Claude Haiku 5.5, GPT-6 Luna, Gemini 3.1 Flash-Lite), two annotators for the 200-summary judge check, an empty GitHub repo `phd-paper1-slm-vs-api` (`gh` is not installed). Then write `docs/PREREGISTRATION.md` before any confirmatory result exists.

## Things that went wrong, so they are not repeated

- Never overwrite files in the user's `~/.ssh` without checking (an old key was overwritten on 2026-11-21).
- Never disable SSH host-key checking; the safety classifier blocks it.
- Verify a session's claims before reporting them (the integration test caught a real schema mismatch between eval and api).
