# Distill

**Distill a large open LLM into a small local model that turns free-text radiology reports into structured, evidence-grounded findings, and prove it works with a rigorous evaluation harness.**

> **Status:** in development. Results below will be filled in as each stage is completed. Nothing in this repository is intended for clinical use.

---

## What it does

Input: a chest X-ray report, written as free text by a radiologist.

```text
Heart size is mildly enlarged. No pneumothorax. Possible small left pleural effusion.
```

Output: structured JSON, where every finding carries a **verbatim quote** from the report as evidence.

```json
[
  {"finding": "cardiomegaly",     "status": "present",   "severity": "mild",  "laterality": null,   "evidence": "Heart size is mildly enlarged"},
  {"finding": "pneumothorax",     "status": "absent",    "severity": null,    "laterality": null,   "evidence": "No pneumothorax"},
  {"finding": "pleural effusion", "status": "uncertain", "severity": "small", "laterality": "left", "evidence": "Possible small left pleural effusion"}
]
```

If an `evidence` string does not appear in the report, the finding was invented. That turns hallucination from a vague worry into a measurable number.

## Why it exists

- **Privacy.** Hospitals often cannot send patient text to external APIs. A 1–3B model runs on a single consumer GPU or a laptop, so the text never leaves the building.
- **Cost.** A small tuned model serves requests at a fraction of the per-request price of a large hosted model.
- **Trust.** "It looked right on five examples" is not evidence. Every claim here is backed by a locked test set, confidence intervals and paired statistical tests.
- **Reproducible for $0.** Every model used is open-weight, and every step runs on free tiers or a 6 GB laptop GPU, so anyone can rerun the results.

## How it works

1. **Evaluate first.** A hand-labelled golden test set is built and locked before any training. Every model, including the large teacher, is measured by the same harness.
2. **Distill (SFT).** A large open-weight teacher model (Apache-2.0 licensed) labels the training split. Labels that fail automatic checks are filtered out, and a small open model is fine-tuned on the rest with LoRA.
3. **Refine (GRPO).** Reinforcement learning with a reward the harness can compute: valid schema, grounded evidence, correct findings.
4. **Verify.** An LLM judge from a different model family than the teacher handles fuzzy matches ("enlarged heart" = "cardiomegaly") and is itself checked against human labels using Cohen's kappa. Regressions are caught in CI with significance tests.
5. **Ship.** The final model is quantized, served locally and benchmarked for latency and cost.

Full details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the system design, and [docs/OVERVIEW.md](docs/OVERVIEW.md) for a plain-language explanation of the whole project.

## Results

| Model | Finding F1 (95% CI) | Status accuracy | Ungrounded evidence | Cost / 1k reports | p50 latency |
|---|---|---|---|---|---|
| Rule-based baseline | – | – | – | – | – |
| Large open teacher model (few-shot) | – | – | – | – | – |
| Small base model (prompted) | – | – | – | – | – |
| + LoRA SFT | – | – | – | – | – |
| + GRPO | – | – | – | – | – |

*To be filled in as each stage lands. All numbers will be reported on the locked test set.*

## Roadmap

- [ ] **Phase 0: Data and schema.** Dataset exploration, output schema, annotation guideline, golden test set
- [ ] **Phase 1: Eval harness.** Graders, model runner, statistics, baselines
- [ ] **Phase 2: Supervised fine-tuning.** Distilled training data, LoRA SFT, ablations
- [ ] **Phase 3: GRPO.** Reward design, RL training, comparison against SFT
- [ ] **Phase 4: Reliability.** Checked LLM judge, robustness and drift, CI regression gate
- [ ] **Phase 5: Ship.** Quantization, local serving, model card, public release

## Data

Development uses the publicly available, de-identified **Indiana University chest X-ray report collection (Open-i)**. The data is **not** included in this repository. Download it from its original source and follow its terms of use.

## Disclaimer

This is a research and engineering project. It is **not a medical device**, has not been clinically validated, and must not be used for diagnosis or patient care.

## License

[MIT](LICENSE)
