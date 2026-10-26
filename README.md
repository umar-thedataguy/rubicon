# Business Entity Resolution — reproducible pipeline

Links every Source-1 business to its matching Source-2 / Source-3 records (0, 1 or many), optimised for
the challenge metric: per-entity F0.5, macro-averaged, singletons included.

```
 S1, S2, S3 ─► normalise ─► multi-view blocking ─► ~90 pair features ─► GBDT stage 1 (OOF)
                (lexicons)   (TF-IDF x4 [+dense])    (string, TF-IDF,        │
                                                      IDF-token, numeric,    ▼
                                                      acronym, ranks)   context features
                                                                        (competition, support)
            [optional GPU]  cross-encoder score ──────────────────────►     │
            [optional GPU]  LLM select score   ──────────────────────►  GBDT stage 2 (OOF)
                                                                             │
                                  exclusivity + expected-F0.5 decoding ◄─────┘
                                                │
                     output/matching_results.tsv   output/candidate_pairs.tsv
```

## 0. Quick start (one command per step)

```bash
cd business_entity_resolution_pipeline/code/business_entity_resolution
bash run.sh install      # once
bash run.sh preflight    # must say GO
bash run.sh check        # normalisation report -> norm_check/
bash run.sh smoke        # first 300k rows per file -> output_dev/ (not a submission)
bash run.sh full         # everything -> output/matching_results.tsv, output/candidate_pairs.tsv
bash run.sh validate     # must say PASS before uploading
bash run.sh package TEAMNAME
```
The dataset is found automatically (e.g. `../student_resource/dataset`); set `DATA_DIR=/path/to/dataset`
if it is elsewhere. Outputs, `work/` and `logs/` go to the `business_entity_resolution_pipeline/` folder
(set `OUT_ROOT` to use a bigger disk). If a run stops, repeat the same command: it resumes.

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10–3.12
pip install -r requirements.txt                          # CPU pipeline (required)
pip install -r requirements-gpu.txt                      # only for the optional GPU stages
```

Data layout: `dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv` and `dataset/test/test_source{1,2,3}.tsv`.

## 2. Run (from `code/business_entity_resolution/`)

Built for the full challenge scale (~2.2M / 10M train and ~1.7M / 10M test records). It detects CPUs,
RAM and GPU and sizes itself (worker processes, DuckDB memory, training subset); no edits are needed
when moving between machines. Every stage saves its result in `--work_dir` and is skipped on re-run,
so an interrupted run resumes where it stopped.

```bash
# 0) checks on the data (sampled, minutes): transliteration / normalisation + data forensics
python -m src.check_normalization --data_dir ../../dataset --out_dir ../../work/norm_check
python -m src.eda --data_dir ../../dataset --out ../../work/eda_report.md

# 1) smoke test on the first 300k rows of every file (NOT a submission): checks the environment,
#    prints blocking recall, OOF score and timings you can extrapolate from
python -m src.run_pipeline --data_dir ../../dataset --out_dir ../../output_dev --work_dir ../../work_dev --dev_rows 300000

# 2) full run -> output/matching_results.tsv + output/candidate_pairs.tsv
python -m src.run_pipeline --data_dir ../../dataset --out_dir ../../output --work_dir ../../work

# 3) validate with the organisers' script before every upload
python3 utils/validate_submission.py --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

Stages (log in `work/log.txt`): **A** parallel normalisation → **B** DuckDB blocking (prints
BLOCKING RECALL and the ceiling macro F0.5 on train) → **C** text statistics → **D** features for a
training subset, stage-1 GBDT, context features, stage-2 GBDT, decision tuning (prints OOF macro F0.5)
→ **E** parallel scoring of every test S1 entity → **F** decoding and output files.

Useful flags: `--train_entities N` (default: auto from RAM), `--n_jobs N`, `--n_folds N` (default 3),
`--redo_from {blocking,features,train,test}` (recompute from a stage after changing code or settings),
`--dev_rows N` (smoke test).

Optional GPU stages (use the pairs files written by a full run; then rebuild features so the new
scores are merged):

```bash
python -m src.cross_encoder --data_dir ../../dataset --work_dir ../../work     # -> work/ce_{train,test}.tsv
python -m src.run_pipeline ... --ce_train ../../work/ce_train.tsv --ce_test ../../work/ce_test.tsv --redo_from features
python -m src.llm_select --data_dir ../../dataset --work_dir ../../work         # stretch
python -m src.run_pipeline ... --extra_score llm_pick:../../work/llm_train.tsv:../../work/llm_test.tsv --redo_from features
```

## 3. Outputs

| file | content |
|---|---|
| `output/matching_results.tsv` | final matches (the leaderboard file), one row per test S1, lists sorted by probability |
| `output/candidate_pairs.tsv` | every pair the final model scored (the blocking output), so matches ⊆ candidates |
| `work/report.json` | hardware, sizes, OOF macro F0.5 (+ singleton split), decision rule, counts |
| `work/log.txt` | full log incl. blocking recall and predicted-match rates per country |
| `work/oof_errors.tsv` | top 50k OOF errors with names and addresses |
| `work/importance_stage{1,2}.tsv`, `work/decision_grid_stage2.tsv` | diagnostics |

## 4. How it scales

* Normalisation: all CPU cores, 500k-row parquet parts.
* Blocking: DuckDB (out-of-core, all cores, spills to `work/duckdb_tmp`): rare-token, sound-key,
  glued-name-prefix and address keys; oversized blocks skipped; top-k by key weight; re-ranked with
  string similarity using weights learned from the training labels; ~30 candidates per S1.
* Training: a random subset of train S1 entities (auto-sized from RAM, 100k–800k) with all their
  candidates; negatives subsampled for speed and re-weighted so probabilities stay calibrated.
* Test: every S1 entity, streamed in chunks of 20k entities across all cores.
* Disk: plan ~30–60 GB free for `work/` at full scale. RAM: 32 GB+ recommended; 16 GB works with a
  smaller automatic training subset.

## 5. Compliance

* **Models** (all ≤ 8B params, MIT / Apache-2.0): LightGBM (MIT), `intfloat/multilingual-e5-base` (MIT),
  `microsoft/mdeberta-v3-base` (MIT) or `xlm-roberta-base` (MIT), `Qwen/Qwen2.5-7B-Instruct`
  (Apache-2.0, 7.6B). Not used: Qwen3-8B (8.2B params), Llama / Gemma (other licences).
* **No external data**: no geocoding, registries, APIs or web data. Normalisation uses a small
  hand-written abbreviation lexicon in `src/normalize.py` (plus substitutions mined from the provided
  training labels by `src/eda.py`) and a hand-written Unicode-table transliterator in `src/translit.py`
  (no transliteration library or dictionary). TF-IDF statistics are fitted on the provided train / test files only.
* **No identifier features**: entity IDs are never used as model inputs.

## 6. Files

```
src/config.py        all knobs (k per view, folds, LightGBM params, decision grid)
src/io_utils.py      robust TSV I/O, unquoted writers, local validation
src/normalize.py     country-agnostic normalisation + US/IN/FR lexicons, skeletons, DBA/domain split
src/translit.py      rule-based Indic-script -> Latin transliteration (Devanagari, Tamil, Kannada, ...)
run.sh               one-command runner (install, preflight, check, smoke, full, validate, package)
src/preflight.py     GO / NO-GO check of Python, libraries, dataset files, RAM, disk
src/hw.py            hardware detection and automatic sizing
src/scale_prep.py    parallel, resumable normalisation to parquet
src/scale_blocking.py  DuckDB key blocking, learned re-ranking, recall report
src/features.py      pairwise features on streamed pair rows
src/context.py       stage-2 collective features (competition, support)
src/model.py         S1-grouped K-fold LightGBM, OOF predictions, weighted negative sampling
src/decision.py      exclusivity + Bayes-optimal expected-F0.5 subset selection, fast exact metric
src/metrics.py       exact challenge metric + breakdowns
src/run_pipeline.py  orchestrator (resumable stages A-F)
src/legacy_run_pipeline_small.py  earlier in-memory version (small data only, kept for reference)
src/eda.py           data forensics + substitution miner
src/check_normalization.py  transliteration / normalisation checks + lexicon suggestions
src/embed_blocker.py optional dense view        (GPU)
src/cross_encoder.py optional Ditto-style CE    (GPU)
src/llm_select.py    optional LLM select stage  (GPU, stretch)
tools/make_synthetic.py  synthetic look-alike data for smoke tests
```
