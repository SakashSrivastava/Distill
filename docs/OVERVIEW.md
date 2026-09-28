# Distill, explained

This document explains the whole project from the ground up. You do not need a background in medicine or machine learning to follow it. Technical terms are explained when they first appear, and there is a glossary at the end.

---

## 1. The one-paragraph version

Radiologists write their findings as free-text reports. Hospitals, researchers and software systems need that information as structured data, like a table that says "pleural effusion: present, left side, small". Large AI models from companies like OpenAI, Anthropic or Google can do this conversion well, but they are expensive at scale, and hospitals often are not allowed to send patient text to them. Distill trains a **small AI model that runs locally** to do the same job. It also builds a **measurement system** that proves how well the small model works compared with the large one, down to how often it makes things up.

---

## 2. The problem

### 2.1 What a radiology report looks like

After a chest X-ray, a radiologist writes a report like this:

> **Findings:** The heart size is mildly enlarged. The lungs are clear. No pneumothorax. There is possible small left pleural effusion.
>
> **Impression:** Mild cardiomegaly. Cannot exclude small left effusion.

This is easy for a doctor to read, but hard for a computer to use. You cannot easily ask a database "show me every patient with a left pleural effusion in the last year" when the answer is buried in sentences.

### 2.2 What structured data looks like

The same report, converted into structured form:

| Finding | Status | Side | Severity | Evidence (quoted from report) |
|---|---|---|---|---|
| cardiomegaly | present | – | mild | "The heart size is mildly enlarged" |
| pneumothorax | absent | – | – | "No pneumothorax" |
| pleural effusion | uncertain | left | small | "There is possible small left pleural effusion" |

Structured data like this can be searched, counted, used to spot trends, used to flag urgent cases, or used to train other AI systems, such as models that read the X-ray images themselves.

### 2.3 Why this is harder than it looks

A simple keyword search fails in several ways:

- **Negation.** "No pneumothorax" contains the word "pneumothorax", but it means the patient does *not* have one. Radiology reports are full of negations, because doctors routinely rule things out.
- **Uncertainty.** "Possible", "cannot exclude", "may represent" and "likely" all signal different levels of doubt. Treating "cannot exclude effusion" as either a yes or a no is wrong.
- **Synonyms.** "Enlarged heart", "cardiomegaly" and "increased cardiac silhouette" describe the same thing.
- **Details spread across sentences.** The side, size and location of a finding may be mentioned separately from the finding itself.
- **Varied writing styles.** Every radiologist and every hospital writes a little differently.

---

## 3. Why not just use a big AI model?

Large models like GPT, Claude or Gemini handle all of the above well. But three problems come with them:

1. **Privacy.** Patient reports are sensitive. Many hospitals cannot legally or contractually send them to an outside company's servers.
2. **Cost.** A hospital system produces millions of reports. Paying a per-request API price for each one adds up quickly.
3. **Control.** The provider can change or retire the model at any time. A system built on it can change behaviour overnight without anyone touching the code.

A small model (around 1–3 billion parameters, compared with hundreds of billions for the largest models) can run on one ordinary graphics card or even a laptop. The text never leaves the hospital, the cost per report is tiny, and the model never changes unless you change it.

The catch: out of the box, small models are much worse at this task. The project is about closing that gap, and about **proving** how much of it has been closed.

---

## 4. The key idea: no evidence, no finding

The most dangerous failure for an AI model here is **hallucination**: stating something confidently that is not true. A model that reports a collapsed lung that the radiologist never mentioned is worse than useless.

Distill handles this with one rule. **Every finding must include a word-for-word quote from the report that supports it.**

This makes hallucination easy to check. If the model says "pneumothorax: present" and quotes "small right pneumothorax", a program can check that the quote actually appears in the report. If it doesn't, the finding was invented. What used to be a vague worry ("does it make things up?") becomes an exact number: the percentage of findings with evidence that is not in the report.

The same check is reused in training, as part of the score the model is rewarded for (Section 6.3).

---

## 5. Measure first, then train

Most AI projects train a model first and figure out how to test it afterwards. Distill does it the other way round. The measuring system (the **evaluation harness**) is built first, because without it there is no honest way to say whether any training step helped.

### 5.1 The golden test set

A set of about 150–200 reports is labelled **by hand**, carefully, following a written rulebook (the annotation guideline) that says how to handle tricky cases. These human labels are the "answer key".

The test portion of this set is then **locked**. A digital fingerprint (a hash) of the file is recorded, and the harness refuses to produce results if the file has changed. This prevents the most common way of fooling yourself: tweaking the test until the numbers look good.

### 5.2 Grading an answer

Each model output is checked in three steps, from cheapest to most expensive:

1. **Is it valid?** Does the output follow the required format?
2. **Is it grounded?** Does every evidence quote actually appear in the report?
3. **Is it correct?** Do the findings match the answer key? This is measured with:
   - **Precision:** of the findings the model reported, how many were real?
   - **Recall:** of the real findings, how many did the model catch?
   - **F1:** a single score combining the two.

### 5.3 Using an AI to help grade, carefully

Sometimes code cannot tell whether two answers match. Is "enlarged heart" the same as "cardiomegaly"? For these cases a large AI model acts as a **judge**. It is deliberately a different model from the one that created the training labels, because AI judges tend to favour answers written in their own style.

But a judge is itself an AI and can be wrong, so it is tested before it is trusted. The project owner hand-grades a batch of cases, the judge grades the same batch, and the two are compared.

The comparison uses a statistic called **Cohen's kappa** rather than a simple "percentage agreed", because the simple percentage can be misleading. If 80% of cases are "match", two graders who both mostly say "match" will agree most of the time by luck alone. Kappa subtracts that luck. For example, 85% raw agreement in that situation works out to a kappa of only about 0.53, which is moderate at best.

The judge is also tested for known habits of AI judges, such as preferring whichever answer is shown first, or preferring longer answers.

### 5.4 Is a difference real, or just luck?

Suppose model A scores 82% and model B scores 85% on 200 test reports. Is B better? Not necessarily. With 200 examples, random variation alone can move the score by around 5 percentage points either way.

So every headline number in this project comes with a **confidence interval** (a range the true score probably falls in), and comparisons between models use **paired statistical tests**, which account for the fact that both models were tested on exactly the same reports. A claim like "the fine-tuned model is better" is only made when the test says the difference is unlikely to be luck.

### 5.5 Slices, not just averages

An average can hide a weakness. A model might be excellent on common findings and terrible on negated ones. Results are therefore also broken down by type: negated findings, uncertain findings, rare findings, short reports and long reports.

---

## 6. Training the small model

### 6.1 Borrowing the big model's knowledge (distillation)

Hand-labelling thousands of reports would take far too long. Instead, a large AI model labels the training reports. The project uses an **open-weight** model for this, one whose makers publish it for anyone to download and use under a permissive licence. That matters for two reasons: the terms of many commercial AI services forbid using their outputs to train other models, and using open models means anyone can rerun the whole project without paying anything. This is called **knowledge distillation**: a big "teacher" model's knowledge is passed to a small "student" model through the examples it produces. It is also where the project's name comes from.

The teacher makes mistakes too, so its labels are run through the same graders from Section 5.2. Labels with invalid format or evidence not found in the report are thrown away before the student ever sees them.

### 6.2 Supervised fine-tuning with LoRA

The student is first trained to copy the teacher's (filtered) answers. This is **supervised fine-tuning (SFT)**.

Retraining every parameter of even a small model is expensive, so the project uses **LoRA** (Low-Rank Adaptation). Instead of changing the whole model, LoRA adds a small set of extra parameters, often under 1% of the model's size, and trains only those. It is much cheaper, needs far less memory, and usually works nearly as well as full retraining.

Experiments (**ablations**) then test which choices matter. How big should the LoRA adapter be? Does using half the training data hurt much? Does filtering the teacher's labels actually help?

### 6.3 Reinforcement learning with GRPO

Copying the teacher has a limit: the student learns the teacher's mistakes and style. The second training stage uses **reinforcement learning**, where the model learns from a score rather than from examples to copy.

The method is **GRPO** (Group Relative Policy Optimization). For each report the model writes several answers. Each answer is scored automatically by the same code as the graders:

- Is the format valid?
- Is every evidence quote really in the report?
- Do the findings match the known labels?

The model is then nudged towards the answers that scored better than the rest of their group.

A known danger here is **reward hacking**: the model finding a cheap trick to score well without doing the task. For example, if the score only punished made-up evidence, the model could learn to report nothing at all and never be punished. The score is designed to block this, since missing real findings costs points too. Training progress is always checked against the independent test harness, not just the training score.

---

## 7. Making it reliable

- **Robustness testing.** Reports are rewritten in small ways (reworded negations, reordered sentences, abbreviations) to check that the model understands meaning rather than memorizing phrasing.
- **Drift detection.** Over time, the reports a system sees can change, for example with a new hospital or a new writing style. The project includes a check that flags incoming reports that look unlike the training data, a warning sign that accuracy may drop.
- **Automatic regression checks (CI).** Every time the code changes, an automated pipeline on GitHub re-runs the tests and a fast version of the evaluation. If performance drops by more than random variation can explain, the change is blocked.

---

## 8. Shipping it

The final model is compressed (**quantized**) so it uses less memory and runs faster, then served locally on an ordinary laptop graphics card with 6 GB of memory. The project reports how fast it runs, how much memory it needs, and what it costs per thousand reports, next to the same figures for the large teacher model.

The end result is a single table: accuracy, hallucination rate, cost and speed for every model, with confidence intervals.

---

## 9. What this project is not

- **Not a medical device.** It has not been clinically validated and must not be used to make decisions about patients.
- **Not a model that reads images.** It works only on the text of reports, not on the X-rays themselves.
- **Not general-purpose.** It is built for chest X-ray reports. Other kinds of reports would need new labels and testing.
- **Limited by its data.** The development data comes from a single public collection. Performance on reports from other hospitals has to be measured, not assumed.

---

## 10. Glossary

| Term | Meaning |
|---|---|
| **Ablation** | An experiment that changes or removes one ingredient to see how much it matters. |
| **Annotation guideline** | A written rulebook for labelling data consistently. |
| **Bootstrap** | A statistical method that resamples the test set many times to estimate how much a score could vary by chance. |
| **CI (continuous integration)** | Automated checks that run every time code changes. |
| **Cohen's kappa** | A measure of how well two graders agree, after removing agreement expected by chance. 1 is perfect, 0 is no better than chance. |
| **Confidence interval** | A range that the true value very probably lies within, e.g. "84% ± 4%". |
| **Distillation** | Training a small model using the outputs of a larger, stronger one. |
| **Drift** | A change over time in the data a system receives, which can quietly reduce its accuracy. |
| **F1 score** | A single score combining precision and recall. |
| **Fine-tuning** | Further training of an existing model on a specific task. |
| **Golden set** | A carefully hand-labelled set of examples used as the answer key. |
| **GRPO** | A reinforcement learning method that compares several answers to the same input and favours the better ones. |
| **Grounding** | Tying each claim to evidence in the source text. |
| **Hallucination** | An AI model stating something that is not supported by its input. |
| **LLM** | Large language model, an AI model trained on large amounts of text. |
| **LLM-as-judge** | Using a language model to grade other models' outputs. |
| **LoRA** | A cheap fine-tuning method that trains a small add-on instead of the whole model. |
| **Negation** | A statement that something is absent, e.g. "no pneumothorax". |
| **Paired test** | A statistical test for comparing two models evaluated on the same examples. |
| **Parameters** | The numbers inside a model that are adjusted during training. Their count is a rough measure of model size. |
| **Precision** | Of the findings a model reported, the fraction that were correct. |
| **Quantization** | Storing a model's numbers at lower precision so it is smaller and faster. |
| **Recall** | Of the real findings, the fraction the model found. |
| **Reinforcement learning** | Training a model by rewarding good outputs rather than showing it correct examples. |
| **Reward hacking** | A model exploiting a flaw in its reward to score well without doing the task. |
| **SFT** | Supervised fine-tuning: training a model to reproduce example answers. |

---

## 11. Medical terms used in examples

| Term | Meaning |
|---|---|
| **Cardiomegaly** | An enlarged heart. |
| **Pleural effusion** | Fluid around the lungs. |
| **Pneumothorax** | Air between the lung and chest wall, which can make the lung collapse. |
| **Laterality** | Which side of the body: left, right or both (bilateral). |
| **Findings / Impression** | The two main sections of a radiology report: detailed observations, then the radiologist's summary. |
