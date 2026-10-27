# PatchConsensus

**Dual-Axis Query Consensus Fusion of Heterogeneous Foundation Encoders for Artifact-Free Patch-Based Skin Lesion Segmentation**

Reference implementation, configuration, reproduction procedures, logs, checkpoints and a numerical validation suite for the PatchConsensus paper.

PatchConsensus is an encoder-only segmentation framework that treats every *(encoder, patch)* pair as an information source. Its core module, **Dual-Axis Consensus Queries (DACQ)**, runs one shared set of mask queries through the last blocks of several frozen foundation encoders (DINOv3, PanDerm, SAM 3) and drives them to agreement across encoders and across overlapping patches with an unrolled **consensus ADMM** whose dual variables score source reliability. Label patches are merged with **minimum-variance weights** estimated label-free from pairwise disagreement (a three-cornered-hat estimator). A distilled student keeps inference to one pass.

The paper proves that *encoder* fusion attenuates only content-locked error, *patch* fusion only position-locked error, neither attenuates both, and that source reliability is identifiable from disagreement only when at least three encoders observe overlapping patches.

---

## Two ways to run this repo

This release is deliberately split so that the parts you **can verify without a GPU, dataset, or pretrained weights** are separated from the parts that reproduce the paper's benchmark numbers.

### Tier A — verifiable now (CPU only, no data, no weights)
A pure numpy/scipy core implements every closed-form result of the paper, and a validation suite checks each one numerically. This is what demonstrates the theory end to end and produces real log files and a checkpoint.

```bash
pip install -e .                       # installs numpy, scipy, pyyaml only
bash scripts/reproduce_all.sh quick    # runs everything below, writes logs/
```

What Tier A runs and produces:

| Step | Script | Output |
| --- | --- | --- |
| Theorem 1, Prop 2/3, Lemma 1 | `experiments/validate_theory.py` | `logs/validation/theory_validation.{md,json}` — **11/11 checks pass on every seed** |
| Assumption tests A1–A5 (Table VI) | `experiments/noise_model_validation.py` | `logs/tables/table_VI.*` (A3 lands borderline at ≈0.093, as in the paper) |
| End-to-end synthetic inference (Algorithm 1) | `experiments/smoke_test.py` | `logs/checkpoints/smoke_student.npz` + `logs/tables/table_smoke.*` |
| Unit tests | `tests/` | 32 tests |
| Paper tables I–VII as reference logs | `experiments/emit_reference_tables.py` | `logs/tables/table_*.{md,csv,json}` |

### Tier B — full paper numbers (A100 80 GB + datasets + encoder weights)
```bash
pip install -e '.[train]'              # adds torch, timm, etc.
bash scripts/reproduce_all.sh full     # prepares data, trains 5 seeds, evaluates
```
See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for obtaining the datasets and weights, the exact commands, and the expected cost (~130 GPU-h for the teacher, ~45 per student).



---

## Repository layout

```
patchconsensus/
├── patchconsensus/              # the library
│   ├── patch.py                 # patch grid, location codes        (Eq 1,2,5)
│   ├── noise_model.py           # dual noise model + sampler         (Eq 3, A1–A5)
│   ├── theory.py                # Theorem 1 closed forms             (Eq 18,19)
│   ├── reliability.py           # 3-cornered-hat, Σ, min-var weights (Eq 12–15)
│   ├── dacq.py                  # consensus-ADMM core                (Eq 7–10, Prop 3)
│   ├── metrics.py               # Dice/IoU/HD95/BIoU/ECE/E_art       (Section VI-C)
│   ├── model.py                 # PatchConsensus teacher + DACQBlock (Algorithm 1)  [torch]
│   ├── student.py               # one-pass distilled student         (Eq 17)         [torch]
│   ├── losses.py                # EoMT loss + consensus + distill     (Eq 16,17)     [torch]
│   ├── encoders/                # DINOv3 / PanDerm / SAM 3 + LoRA + bridges [torch]
│   ├── data/                    # datasets, transforms, split generation
│   └── utils/                   # config, logging, checkpointing, seeding
├── configs/                     # YAML configs (default, datasets, teacher, students, ablations)
├── scripts/                     # train_teacher, distill_student, evaluate, make_splits, prepare_data, reproduce_all.sh
├── experiments/                 # validate_theory, noise_model_validation, smoke_test, run_ablation, aggregate_seeds, emit_reference_tables
├── tests/                       # pytest unit tests
├── logs/                        # generated: runs/, tables/, validation/, checkpoints/
└── docs/                        # REPRODUCE.md, THEORY.md
```

The library has two layers: a **pure core** (`patch`, `noise_model`, `theory`, `reliability`, `dacq`, `metrics`, `utils`) importable with only numpy/scipy, and a **torch layer** (`model`, `student`, `losses`, `encoders`) that imports torch lazily. `import patchconsensus` never requires torch.

## Equation → code map
See [`docs/THEORY.md`](docs/THEORY.md) for a line-by-line mapping of every equation and theorem to the function that implements and tests it.

## Method at a glance

1. Resize to 896×896; cut into K=9 patches of side S=448, stride τ=224 (so interior pixels are seen P=4 ways per encoder). Each patch gets a location code (Eq 5).
2. Three frozen ViT-L encoders read each patch to their last `L2=4` blocks; LoRA adapters are the only encoder parameters trained.
3. DACQ runs `L2` consensus-ADMM blocks over the `M·K` sources: each forms a local query from the consensus `z` and dual `u_i` through its encoder block (Eq 7), the consensus is a reliability-weighted mean (Eq 8), the dual accumulates disagreement (Eq 9), and reliability `ρ_i` is a softmax of `−‖u_i‖²/T` (Eq 10).
4. Each source reads a mask with the shared query (Eq 11); three label-free disagreements give the source variances via a three-cornered-hat estimator (Eq 12–13); these assemble Σ (Eq 14) and the minimum-variance fusion weights (Eq 15).
5. The teacher is distilled into a single-encoder, one-pass student (Eq 17).

## License
Code: MIT (see `LICENSE`). Pretrained weights and datasets are under their own licenses / data-use agreements and are **not** redistributed here.

## Citation
See `CITATION.cff`.
