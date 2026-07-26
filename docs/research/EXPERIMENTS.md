# IlhaEmbed — research and experiment log

Running record of what was tried, what the evidence said, and what changed as a result. Entries are
append-only: a superseded conclusion stays, with the evidence that overturned it, because the
reasoning that failed is the part that is expensive to rediscover.

Early entries use the internal historical name `intake-embedder`. That name describes the
original product experiment, not the current public model or repository structure. The released
model is IlhaEmbed; historical names below are preserved only where needed to interpret recorded
runs.
`CODER-TW` (see `../README.md`) is the earlier surface-form/Taigi model; it remains the comparison
baseline, not a dependency.

---

## 2026-07-18 — Diagnosis: the objective never matched the task

**Question.** Field reports say text discrimination is poor. Is that a model-quality problem (fix by
swapping in a stronger model) or something else?

**What production actually asks the model to do**, read from
`modules/intake/column_resolution/semantic_embedding.py`:

1. classify a messy cell fragment into one of 14 auto-classifiable clinical categories, by cosine
   against a centroid built from 3–12 seed exemplars, accepted only above 0.70 with a 0.10 margin;
2. read a column label's concept (0.85 / 0.05);
3. resolve a free-text condition to an ICD code.

**Finding — objective mismatch.** No model in the stack was ever trained on decision (1).
`bge-small-zh-v1.5` is a general retrieval encoder; CODER-TW was trained with InfoNCE on
surface→canonical synonym pairs. Both optimize "related texts are near each other". Neither
optimizes "a category's members cluster tightly around their centroid and far from other
categories" — which is the decision actually made at runtime, and the one whose thresholds the
0.6–0.7 band complaints are about.

Consequence: a like-for-like model swap is not expected to fix the band, because the band is a
property of an objective nobody trained. This reframed the task from *evaluate and adopt* to
*train for the real objective*.

**Finding — the discrimination failure is antonym collapse.** `code_labels.jsonl` documents the
shipped behaviour case by case: querying a condition retrieves its opposite at 0.861, and a urine
finding outranks the disease of the same name. Chinese clinical antonyms differ by one polarity
morpheme and share every other character, so an embedding dominated by character overlap places
them adjacent. Character overlap is a lexical accident; that it currently drives the ranking IS the
"resolution" complaint, stated precisely.

---

## 2026-07-18 — Data: what the repo already contains, and what it does not

**Available to the original experiment through its licensed terminology cache**
(the cache is not part of this public repository):

| Source | Rows | Usable as | Chinese |
|---|---:|---|---|
| `icd10_cm_2023` | 96,804 | `condition` + EN↔ZH pairs | yes |
| `icd10_pcs_2023` | 78,531 | `procedure` + EN↔ZH pairs | yes |
| `snomed_ct` | 523,502 | category via semantic tag | **no** (0 rows) |
| `loinc_nhi` | 24,014 | `lab_value` + EN↔ZH pairs | yes |
| `icd10_cm_index_2023` | 71,002 | lay term → code | EN only |
| `icd_term_bridge` | 204,960 | EN↔ZH token alignment | bridge |

A code system is a category-labelled vocabulary by construction, which converts ~200k shipped rows
into supervision for decision (1) with no hand labelling.

**Finding — coverage collapses on real files.** Mining the 46 corpus fixtures yielded 2,812 distinct
term candidates. **Only 326 (11.6%) appear anywhere in the code systems above.** The other 88% is
Taiwanese community-health working vocabulary that no public code system publishes: heavily
abbreviated hospital names, community care sites named after their village, membership and
case-lifecycle states, phone-follow-up phrases, and national-insurance lab item names absent from
the LOINC Chinese displays.

This is the quantitative form of the diagnosis. The taxonomy categories covered by code systems
(condition / procedure / medication / lab_value) carry 12,006 / 12,006 / 12,006 / 958 examples after
capping. The other twelve carry **3–8 seed exemplars each** — and those twelve are exactly the ones
reported as failing (a department reading below the gate at ~0.64; a follow-up phrase that needed a
regex alias). Training only on public terminology therefore cannot reach the reported failures.
Hence `field_vocabulary.py`: corpus-frequency-selected working terms, labelled by category.

---

## 2026-07-18 — Hard negatives: two corrections before anything was trained

Teaching minimal-pair opposites apart is the direct answer to antonym collapse. Two iterations were
needed, both caught by reading the mined output rather than by reasoning about it.

**Correction 1 — laterality floods the axis.** The first miner flipped a polarity morpheme and kept
the pair when the flipped form existed. Output was overwhelmingly left/right variants of
sentence-length surgical descriptions (207 pairs, nearly all laterality). Laterality does flip
meaning, but it decorates tens of thousands of procedure strings and would drown the axis that
actually fails — a value's clinical polarity — in near-duplicate surgery names. Operators type short
terms into cells, not sentence-length descriptions. Fix: drop laterality from the polarity list, bound
mined terms to ≤12 characters, and add a general character-overlap miner so the result does not depend
on having enumerated every way Chinese expresses an opposite.

**Correction 2 — overlap alone conflates synonyms with opposites.** The overlap miner immediately
produced pairs that must NOT be pushed apart: two spellings of the same lipid analyte differing by one
homophone character, alongside genuine different-concept pairs (three distinct tumour-marker assays;
two different stroke-severity scores; two different program years). Training on the mixed set would
teach the model to separate a term from its own spelling variant — destroying exactly the surface
robustness that is the point.

Character similarity cannot settle "same concept or not". The code system already answers it: **same
code ⇒ synonym (pull together); disjoint codes ⇒ different concept (push apart); no code ⇒ not
adjudicable, never mined**. Rebuilt the term index to carry concept codes and gated both miners on
that. This also yields 25,286 same-code synonym pairs as positive supervision for free.

Result: 3,912 code-adjudicated hard negatives, 25,286 synonym pairs, 7,370 generated rejection
negatives (measurements, dates, identifiers, marks — shape-generated, never copied from client data).

**Lesson.** Both corrections came from printing samples of the mined data, not from inspecting the
mining logic. Neither would have been visible in an aggregate count, and both would have silently
degraded the trained model.

---

## 2026-07-19 — Head-to-head: CODER-TW does not beat bge, and both fail identically

Run on the GPU box over the held-out production labels, same seeds and same scoring for both models.

| | bge-small-zh | CODER-TW v2 int8 |
|---|---:|---:|
| value→category top1 (n=218) | 0.229 | 0.229 |
| value→category top5 | 0.257 | 0.252 |
| AUC | 0.735 | 0.724 |
| separation (pos p5 − neg p95) | −0.125 | −0.191 |
| recall at the production gate | 0.138 | 0.101 |
| fitted zero-false-positive threshold | 0.664 | 0.404 |
| ICD condition top1 (n=13) | **0.538** | 0.308 |
| latency per fragment, 1 CPU thread | **1.38 ms** | 7.50 ms |
| dimensions / deployed size | 512 / ~23 MB int8 | 768 / 170 MB |

**Verdict: do not adopt CODER-TW.** It matches bge on the category task, is worse on separation, worse
on ICD conditions, 5.4x slower and 7x larger. Its surface-form training does not transfer to the
decision production makes.

### RETRACTED — the value→category rows above are measured against an impossible metric

**Do not cite the value→category numbers in this table.** `run_eval.py` took all 218 semantic labels
as positives, but 162 of them are labelled `residue`, which is not a taxonomy category and is
excluded from the prototype set. Those 162 can never win top1, so the metric ceiling was
56/218 = 0.257 and both models scored 0.229 against it. The contamination is not confined to top1:
AUC and the separation gap both treat those 162 refuse-me rows as positives that should score high,
which inverts their contribution.

Retracted with it, and stated plainly because it was published before it was checked: **"recall at
the production gate is 0.138, so seven in eight true positives are refused" is wrong.** Most of what
that gate was refusing is residue, and refusing residue is correct behaviour, not a defect. The
user's complaint is real, but that number did not measure it.

The **tie at 0.229 is therefore weak evidence**, not the confirmation it was written up as: both
models failed the same 162 impossible rows, so the agreement is largely arithmetic. The
objective-mismatch diagnosis still rests on the argument from the training objective and on run 1's
behaviour — not on this.

**What survives unaffected**, because neither depends on the contaminated label split:
- ICD condition accuracy (n=13, its own labels): bge 0.538 vs CODER-TW 0.308.
- Cost: 1.38 ms vs 7.50 ms per fragment, 512 vs 768 dimensions, ~23 MB vs 170 MB.

The adopt/reject verdict stands on those two alone, and comfortably: CODER-TW is less accurate on the
one clean accuracy measurement while costing 5.4x the latency and 7x the size. A corrected
value→category run — residue moved to the negative side where it belongs, readability gate applied —
is queued.

**Lesson.** The subagent flagged the ~23% as possibly a metric artifact and declined to explain it
away; the write-up did not wait for that flag to be resolved before building a headline claim on it.
A number that looks like it confirms the hypothesis is exactly the number to check hardest.

---

## 2026-07-19 — Training run 1: the objective conflict, measured

Base `bge-small-zh-v1.5`, 3 epochs, four objectives (category SupCon + synonym InfoNCE + hard
negatives + rejection).

| | base | ep1 | ep2 | ep3 |
|---|---:|---:|---:|---:|
| category top1 | 0.716 | **0.767** | 0.750 | 0.750 |
| top-vs-runner-up margin | 0.125 | 0.189 | 0.212 | **0.214** |
| negatives p95 | 0.833 | 0.884 | 0.916 | **0.945** |
| separation | −0.385 | −0.541 | −0.607 | **−0.688** |
| antonym pair similarity (mean) | 0.867 | 0.884 | 0.865 | — |

**The baseline reproduces the shipped defect exactly.** `高血壓` vs `低血壓` scores **0.861** — the
same figure recorded in the production notes. The probe measures the real failure, not a proxy.

**Result: the category objective works, and its side effect is over-routing.** Margin rises
monotonically (0.125 → 0.214): categories genuinely separate from each other. But negatives climb
with it (0.833 → 0.945), so separation degrades every epoch. Pulling 12,006 condition terms into one
cluster inflates that cluster until held-out non-clinical fragments fall inside it. Category cohesion
and rejection are in direct tension, and with an unbalanced corpus cohesion wins on count alone.

**Antonyms did not improve**, and the reason is visible in the loss: the hard-negative term sat near
0.037, i.e. the mined pairs were already easy. The miner produced long procedure pairs the model
separates anyway, while the short clinical opposites that actually collide were absent — a bare
polarity term is usually not itself a billable display, so mining from code displays cannot reach
them. Worse, SupCon *actively pulls antonyms together*, because 高血壓 and 低血壓 are both
`condition`.

**Changes for run 2** (each traceable to a number above):
1. `PER_CATEGORY_CAP = 2000` — deny any category the ability to win by count.
2. Rejection weight 0.5 → 2.0, ceiling 0.35 → 0.25 — push junk clearly below any plausible gate.
3. Hard-negative weight 0.5 → 2.0, plus 20 curated field polarity pairs upsampled 40x, deliberately
   disjoint from the 10 held-out probes so the probes still measure generalization.
4. Evaluation now applies the runtime's Han-character readability gate and scores positives by top
   similarity, matching how negatives are scored. Run 1's negative set was largely bare numbers the
   runtime never scores, so part of that separation figure measured a failure the product does not have.

---

## 2026-07-19 — Training run 2: better on every axis, and still not fixed

Same base and objectives, with run 1's four corrections applied. Evaluation now uses the runtime's
readability gate, so the held-out split is 101 positives / 130 negatives (run 1 scored 116/238,
inflated by bare numbers the runtime never scores). Base numbers therefore shift slightly; only
within-run trends are comparable across runs.

| | base | ep1 | ep2 | ep3 |
|---|---:|---:|---:|---:|
| category top1 | 0.703 | **0.782** | 0.743 | 0.733 |
| margin | 0.120 | 0.152 | 0.177 | **0.189** |
| negatives p95 | 0.828 | 0.875 | 0.884 | **0.896** |
| separation | −0.305 | −0.468 | −0.508 | −0.503 |
| antonym similarity | 0.867 | 0.873 | 0.857 | **0.850** |

**Run 2 beats run 1 on every axis**, so the corrections were the right ones:

| | run 1 | run 2 |
|---|---:|---:|
| best top1 | 0.767 | **0.782** |
| negatives p95 drift | +0.112 | **+0.068** |
| separation drift | −0.303 | **−0.198** |
| antonym movement | −0.002 (flat) | **−0.017** |

The per-category cap cut rejection degradation by ~40%, and antonym similarity finally moved in the
right direction instead of sitting flat. **But neither target defect is fixed.** Negatives still climb
every epoch, and 0.850 antonym similarity is nowhere near separated. Both runs peak at epoch 1 and
decline after, so more epochs are not the answer either.

**Root cause, from the loss values rather than from theory:** the four objectives are on
incommensurable scales. In a typical step, `category≈2.9`, `pair≈2.9`, `reject≈0.09`,
`hardneg≈0.023`. Raising the hard-negative and rejection WEIGHTS from 0.5 to 2.0 could not close a
hundred-fold gap in magnitude — 2.0 x 0.023 is still noise beside 2.9. The hinge losses are also
mostly satisfied (`relu(sim − 0.7)` is zero for every already-separated pair), so they contribute
gradient only from the minority of pairs that are genuinely hard, and the curated polarity pairs are
just 800 of 4,712 mined pairs, sampled 32 at a time.

The diagnosis is structural, not a tuning matter: **an additive hinge cannot compete with a
contrastive loss for control of the geometry.**

**Run 3 design (not yet run):** move hard negatives and rejection INTO the contrastive denominator
instead of adding them alongside it. Each anchor's InfoNCE denominator gets its mined opposite and a
sampled rejection term as explicit negatives, so separating them is how the primary loss is minimized
rather than a side constraint competing with it. Also stop at one epoch, or lower the learning rate,
since both runs peak there.

Client workbooks are read in place; only term TYPES leave, never row values. The corpus miner's
name/ID/phone/measurement filter is heuristic and **demonstrably imperfect** — reviewing its output
found residual personal-name fragments and a named residential building. So mined corpus output is
git-ignored, and only the human-reviewed vocabulary in `field_vocabulary.py` is committed. Review,
not filtering, is what makes derived vocabulary safe to keep.

---

## 2026-07-19 — Base-model sweep: provenance and quality point the same way

Prompted by a compliance question. BAAI is a PRC state-backed institute, and this product's customers
include public hospitals, which fall under Taiwan's cybersecurity regime for critical infrastructure.
CODER-TW is PRC-origin too, so the whole semantic layer shared the exposure. Whether an Apache-2.0
open-weight file counts as a restricted "product" is unsettled and needs legal input, but procurement
checklists tend to be origin-based and binary, so the practical risk stands regardless.

Same recipe, seed and data; only the base model changes. One epoch each.

| base | origin | licence | int8 | base top1 | ep1 top1 | ep1 separation |
|---|---|---|---:|---:|---:|---:|
| **jina-embeddings-v2-base-zh** | **Germany** | Apache-2.0 | 132 MB | **0.861** | pending | pending |
| bge-small-zh-v1.5 | PRC | MIT | 23 MB | 0.703 | 0.782 | -0.468 |
| paraphrase-multilingual-MiniLM | Germany | Apache-2.0 | 117 MB | 0.604 | 0.713 | -0.450 |
| ckiplab/bert-tiny-chinese | Taiwan | **GPL-3.0** | 11 MB | 0.663 | 0.693 | **-0.316** |

**The result inverts the expected trade-off.** Jina's *untrained* baseline (0.861) beats every other
candidate's *fine-tuned* result, including bge's 0.782. Choosing a non-PRC base is not a quality
concession here -- it is the strongest model tested. Unsurprising in hindsight: jina is a 132M
bilingual model built for retrieval, measured against a 24M one. The earlier framing treated them as
peers because both are "small Chinese embedders", which they are not.

**Two findings that outlive this decision:**

1. **Vocabulary dominates small-model footprint.** Only **20.7%** of jina's 60,516-token vocabulary is
   touched by the entire domain corpus (110,786 strings spanning ICD/PCS/LOINC/SNOMED and the field
   vocabulary). Pruning the embedding matrix to what the domain uses removes 37M of 132M parameters,
   essentially losslessly in-domain. The multilingual candidates are large for the same reason and
   *worse* on Chinese: their 250k XLM-R vocabulary is 96M of their 117M parameters, and it fragments
   Taiwanese clinical terms more (29 tokens vs 26 for a Chinese-native vocabulary, 23 for jina).
2. **Top1 is not the product metric.** ckip-tiny has the *lowest* top1 but by far the *best*
   separation (-0.316 vs bge's -0.468), meaning it over-routes least -- and over-routing is the defect
   the field actually reports. An 11 MB model beating a 24 MB one on the axis that matters most is a
   standing warning against ranking candidates on accuracy alone.

**Licence note.** CKIP is GPL-3.0 and Traditional-Chinese native: the best linguistic fit, but
copyleft. Its corpus lineage (ZhWiki + CNA, ASBC for segmentation) carries a second constraint --
training our own model on research-licensed corpora moves the restriction from the weights to the
data rather than removing it.

**Two process failures in this sweep, both self-inflicted:**
- Candidates shipping `.bin` checkpoints (CKIP) fail to load under torch 2.5.1 post-CVE-2025-32434.
  Converting to safetensors with `weights_only=True` is the documented fix. The sweep silently skipped
  CKIP for a full round because stderr was discarded by `2>/dev/null` -- a swallowed error reads
  exactly like a model that produced no result.
- The jina probe ran concurrently with a ckip-base run already holding 15.3 GB of the 4080's 16.4 GB,
  so it could not allocate and produced nothing. The runbook says one heavy job at a time. Reading
  that empty result as "the model failed" nearly wrote off the best candidate in the field.

---

## 2026-07-19 — The sweep's real finding: fine-tuning HURTS a strong base

Jina's fine-tuned result came in below its own starting point.

| base | int8 | base top1 | ep1 top1 | delta |
|---|---:|---:|---:|---:|
| **jina-embeddings-v2-base-zh** | 132 MB | **0.861** | 0.762 | **-0.099** |
| ckip-base | 101 MB | 0.743 | 0.792 | +0.049 |
| bge-small-zh | 23 MB | 0.703 | 0.782 | +0.079 |
| paraphrase-multilingual-MiniLM | 117 MB | 0.604 | 0.713 | +0.109 |
| ckip-tiny | 11 MB | 0.663 | 0.693 | +0.030 |

**The deltas are monotonic in base strength.** The weaker the starting model, the more this training
helps; the stronger it is, the less — and past a point, training actively destroys value. MiniLM
starts worst and gains most (+0.109); jina starts best and loses most (-0.099).

**So the single best configuration measured in this entire project is jina with NO fine-tuning at
all** (0.861), well above every trained model including our own best (0.792). That is worth stating
plainly: the training pipeline built over the previous runs currently SUBTRACTS value from a good
base. It was designed and tuned against bge, where it genuinely helped, and the gain was read as a
property of the method when it was substantially a property of the weak starting point.

**Most likely cause: the learning rate.** 2e-5 is reasonable for adapting a small
general-purpose encoder and far too aggressive for a 132M model already trained for retrieval —
the category objective overwrites pretrained geometry faster than it adds task structure
(catastrophic forgetting). Supporting evidence: jina's margin barely moves (0.186 -> 0.182) while
top1 falls, i.e. the fine-tune is not sharpening the decision, it is degrading the representation.
Its antonym similarity DID improve (0.887 -> 0.833), so the hard negatives work; the damage is
elsewhere.

**What this changes:**
1. The immediate deployable option is jina **untrained**, with thresholds re-fitted to its own score
   distribution. That alone moves the category decision from 0.703 (today's shipped base) to 0.861
   with no training risk at all.
2. Fine-tuning a strong base is a separate, gentler problem: much lower learning rate (5e-6 or
   below), fewer steps, possibly freezing lower layers, and validating against the base's own score
   rather than against bge's.
3. Vocabulary pruning (132 MB -> ~95 MB, only 20.7% of the vocabulary is used in-domain) is
   orthogonal to all of this and still applies.

**Lesson.** A training recipe validated on one base does not transfer to a better base, and the
improvement it shows there is partly a measure of how weak that base was. Every gain reported in
runs 1 and 2 needs re-reading in this light: they were real against bge, and they were not evidence
that the method is good in general.

---

## 2026-07-19 — Correction: the jargon benchmark, and a verdict reversed

**The metric being optimized was the wrong one.** Every number up to this point was taxonomy top1 --
a proxy. The judgment the product is graded on is whether a model reads what staff actually write: a
blood culture written as the Taiwanese sounding-out of the English, a Taigi metaphor for stroke, a
clipped department name, an English dosing abbreviation. That was never measured.

Institution abbreviations are deliberately EXCLUDED from this benchmark: the repo already measured and
rejected an embedder for them (`tw-facility-abbreviation-matching.md`) because they have structural
anchors that deterministic scorers exploit better. Benchmarking a model on work it should not do would
argue for the wrong design.

| model | size | peak RSS | latency | jargon top1 | slang top1 |
|---|---:|---:|---:|---:|---:|
| **CODER-TW int8** | 179 MB | 892 MB | 11.9 ms | **0.795** (contaminated) / **0.433** held-out | **0.871** (contaminated) |
| jina-zh | 161M | 7,900 MB | 52.9 ms | 0.317 | 0.081 |
| ours-v2-bge | 24M | 1,329 MB | 6.2 ms | 0.315 (contaminated) | 0.097 (contaminated) |
| bge-small-zh | 24M | 1,332 MB | 6.1 ms | 0.261 | 0.048 |
| minilm | 118M | 1,617 MB | 11.3 ms | 0.200 | 0.016 |
| ckip-tiny | 12M | 1,287 MB | 2.5 ms | 0.185 | 0.048 |
| ckip-base | 102M | 1,643 MB | 40.0 ms | 0.160 | 0.032 |

**CODER-TW was rejected earlier on the wrong evidence.** That verdict rested on ICD accuracy and cost
-- neither of which is what the model was built for. On the task it WAS built for it leads by a wide
margin, and its cost profile (11.9 ms, 892 MB) is deployable, unlike jina's. The general lesson: a
purpose-built model dismissed on a general benchmark was never actually evaluated.

**The second finding is worse for our own work.** `ours-v2-bge` trained on exactly the same 1,652
jargon pairs and reached 0.315 on that training data, where CODER reached 0.795 on the same data. Same
data, same idea, 2.5x apart. The category objective (37k rows) was drowning the jargon signal (1,652
pairs), and upsampling recycled pairs adds no information.

**Everything on-prem reads roughly one jargon term in three.** That is the reported complaint,
measured at last, and it is much larger than any gap between base-model candidates. Choosing between
0.26 and 0.32 was arguing about the wrong decimal.

---

## 2026-07-19 — Capacity, isolated: data volume was not the whole story

Matched the reference model's training setup as closely as possible and varied only size.

| | CODER-TW | ours (ckip-tiny) |
|---|---:|---:|
| training pairs | 119,242 | **119,742** |
| recipe | InfoNCE, specialized x8 | same |
| held-out split | 15% | same, same seed |
| **parameters** | **~110M** | **11.5M** |
| **held-out jargon top1** | **0.433** | **0.287** |

| epoch | base | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| held-out top1 | 0.178 | 0.239 | 0.251 | 0.259 | **0.287** |
| top5 | 0.279 | 0.336 | 0.393 | 0.433 | **0.437** |

Loading the 108k bulk pairs that had been sitting unused on disk lifted a small model from 0.178 to
0.287 (+61%), and it had not converged at epoch 4. But with data and recipe matched, a 9.5x smaller
model reaches about two thirds of the reference accuracy. **Data volume was a real cause and not the
only one; capacity contributes.**

These are the first uncontaminated numbers for our own model in this project. Every jargon figure
reported before this was measured on training data and was an upper bound.

Next: extend training (not converged), add a 24M point, and a 102M point capacity-matched to the
reference -- if the matched size reaches ~0.433, capacity explains the gap and a non-PRC model of that
class inherits the capability. Distillation is the lever after that.

---

## 2026-07-19 — Capacity explains the whole gap, and a Taiwan-origin model matches the reference

Data, recipe, held-out split and seed matched; only parameter count varied.

| parameters | model | held-out jargon top1 | top5 |
|---:|---|---:|---:|
| 11.5M | ckip-tiny, 10 ep | 0.324 | 0.478 |
| 24M | bge-small-zh, 6 ep | 0.364 | 0.482 |
| **102M** | **ckip-base, 4 ep** | **0.433** | **0.591** |
| ~110M | CODER-TW (reference) | **0.433** | 0.615 |

**At matched capacity the number is identical.** Nothing was left in the reference model's recipe
that we had failed to copy: once the 108k bulk pairs were loaded and the size matched, our training
reproduced its accuracy exactly. The earlier 0.287 was a small model, not a wrong method.

This also produces the artifact the provenance question wanted: **a Taiwan-origin base (Academia
Sinica) trained on our own corpus reaches the reference model's accuracy**, with no PRC-origin
weights anywhere in it. Its licence (GPL-3.0) is copyleft.

### Published domain models do not transfer to this problem

| model | size | latency | jargon top1 |
|---|---:|---:|---:|
| our 24M | 24M | 6.2 ms | **0.364** |
| SapBERT-UMLS-all-lang (XLM-R) | 278 MB | 38.2 ms | 0.334 |
| BioLORD-2023-M | 278 MB | 39.3 ms | 0.229 |

Both are state of the art on international biomedical entity linking and both lose to a 24M model
trained here, at 6x the size and 6x the latency. The literature detour was still worth taking -- it
named the techniques and confirmed the task shape -- but the conclusion is that UMLS-derived,
English-centric training does not reach Taiwanese clinical shorthand. **The vocabulary that fails in
the field is local, and only local data covers it.** That also retires the UMLS licensing question:
the resource would not have solved the problem it was being considered for.

### Taiwan's own FHIR guides are a real bilingual source

Harvested the official implementation guides (TW Core IG, 衛福部; TWIDIR, 疾管署):

- 14,221 unique concepts, of which **326 are bilingual pairs** and 10,504 are Chinese-only.
- The bilingual pairs are precisely the categories that fail: professional roles and departments
  (語言治療師 / Speech and language therapist, 護理師 / Registered nurse, 家醫科 / Family practice).

Two harvesting bugs, both worth remembering because both produced plausible-looking output:
1. These guides put the Chinese term directly in `display`, not in a `designation`. A harvester that
   assumes `display` is English files every Chinese name in the English column -- it reported 4
   bilingual pairs instead of 326. Decide a string's language by looking at it, not by its field.
2. A `ConceptMap` element and its target are the SAME concept in two languages. Emitting them as two
   rows discards the pairing that is the entire value of the map.

---

## 2026-07-23 — Distillation earns its place, and a licence reality check (IlhaEmbed)

Two decisions were reversed by RUNNING the eval this round, not by reasoning:

1. **jina was re-tested and dropped.** An independent reproduction of jina-v2-base-zh
   untrained on the SAME split gave **specialized (jargon) top1 = 0.348**, corroborating
   the earlier 0.317 jargon figure — jina is near the *bottom* on the real task, not the
   top. The 0.861 it "won" with was the taxonomy PROXY metric this log already repudiated.
   The lesson repeats: rank on jargon top1, never on the proxy.

2. **Distillation finally beats from-scratch — the fix was the unlabelled pool.** Teacher =
   ckip-base retrained here to **best-epoch 0.421** (ep2 peak; note train_gpu.py saved the
   *last* epoch 0.401, so best-epoch checkpointing was added — a real defect, ship the best
   not the last). Student = ckip-tiny (11M, hidden 312). Dim-agnostic **relational**
   distillation (match the pairwise cosine matrix, 768d→312d) over the **204,707-string
   unlabelled pool**, plus supervised InfoNCE on the 1,652 specialized pairs.

   | model | params | jargon top1 |
   |---|---:|---:|
   | ckip-tiny before distill | 11M | 0.174 |
   | ckip-tiny from-scratch (curve baseline) | 11M | 0.324 |
   | **ckip-tiny distilled (best ep7/9)** | 11M | **0.348** |
   | bge-small from-scratch | 24M | 0.364 |
   | teacher ckip-base | 102M | 0.421 |

   **0.348 > 0.324 (+7.4%)**, approaching 24M-from-scratch 0.364 at <½ the params. This
   directly fixes the earlier 102M→11.5M distillation that merely TIED at 0.320 because the
   teacher spoke only on labelled pairs — the edge is dark knowledge on UNLABELLED data.
   Modest win, not a rout (11M cannot reach the 102M teacher); 0.348 is still
   suggest-with-review tier.

**Naming.** The distilled model is **IlhaEmbed** (brand *Ilha* ← Ilha Formosa, sovereign;
category *Embed*; clinical meaning in the model card, not the id). Artifact: `ilha-embed`.

**Licence note.** A distilled model's licence follows its own base, not the teacher's: since
IlhaEmbed-11M is initialised from ckip-tiny, it inherits ckip-tiny's **GPL-3.0**, not a
permissive licence. A permissive base was still an open question at this point — the only
~24M permissive option tested (bge-small, MIT) is PRC-origin and failed the sovereignty
criterion above. Unresolved; next question.

---

## 2026-07-23 — The permissive model wins outright: ckip → Granite-311M beats the teacher

A comprehensive scan of 2026 embedding models (filtered by licence × provenance × size ×
Chinese) and an empirical untrained bake-off on OUR held-out jargon task settled the student
base — not specs or leaderboards. Bake-off (untrained jargon top1, permissive/non-PRC only):

| base | params | jargon@1 | sep | licence | origin |
|---|---:|---:|---:|---|---|
| **IBM Granite-311M-R2** | 312M | **0.356** | -0.017 | Apache-2.0 | IBM (US) |
| jina-v2-zh (ref) | 161M | 0.348 | -0.057 | Apache-2.0 | Germany |
| bge-small-zh (ref) | 24M | 0.304 | -0.074 | MIT | PRC |
| Granite-97M-R2 | 97M | 0.291 | -0.029 | Apache-2.0 | IBM (US) |
| multilingual-e5-small | 118M | 0.247 | -0.018 | MIT | MS/MSRA-Beijing |
| paraphrase-MiniLM-L12 | 118M | 0.211 | -0.175 | Apache-2.0 | Germany |

Granite-311M won (best jargon AND best separation). The 97M was too small (0.291 < jina).
Gemma skipped (HF manual-gate friction). Adoption concern about IBM is moot: we distil + rebrand
to IlhaEmbed, downstream adopts IlhaEmbed not Granite.

**Distillation ckip-base (0.421) → Granite-311M** (relational over 204k pool + specialized
InfoNCE, gentle lr 1.5e-5, best-epoch save because a strong base degrades under aggressive FT):

| epoch | base | 1 | 2 | **3** | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| jargon top1 | 0.336 | 0.437 | 0.453 | **0.466** | 0.449 | 0.429 | 0.429 |

**Best jargon top1 = 0.466 — the STUDENT BEATS THE TEACHER (0.421)** and is the best result in
the project. Distillation transferred ckip's jargon knowledge onto a stronger multilingual
substrate; the peak-then-drop (ep3→ep4) confirms the strong-base degradation lesson, caught by
best-epoch save.

**Result.** IlhaEmbed = the Granite-distilled model: **Apache-2.0 · non-PRC (IBM/US base) ·
0.466 (best) · 311M (int8 ~150MB, deployable)**. It is strictly better than the GPL ckip-base
(0.421) on every axis but size. ckip-base is demoted to teacher/research artifact. Licence: the
student is initialised from Granite (Apache) with ckip only as a teacher signal (outputs, not
weights) → IlhaEmbed is Apache-2.0. Artifact: `ilha-embed-granite` (1.2 GB fp32; quantise next).

---

## 2026-07-24 — Shrink to 97M, and an honest accounting of the eval

Two things settled here: the deployable size, and — more importantly — a correction to how the
headline number was being reported.

**Vocab pruning: 311M → a 97M-class deployable, no quality loss.** The Granite base ships a 180k
multilingual BPE vocab; Traditional-Chinese clinical text uses ~25.5k of it. Pruning to those
tokens (merge-closure preserved, byte fallback kept, embedding table remapped) gave a **37 MB INT8
ONNX** (was 117 MB), and the real intake routing came out **bit-identical**. The 97M and 311M
distillations measured essentially equal on the held-out jargon task, so the smaller 97M ships
(`weemed/IlhaEmbed` on HF).

**Register-balanced held-out (self-matches excluded), the protocol the HF card reports:**

| register | IlhaEmbed-97M | IlhaEmbed-311M | bge-small (ref) |
|---|---:|---:|---:|
| slang (n=63) | 0.841 | 0.857 | 0.00 |
| abbrev (n=399) | 0.827 | 0.815 | 0.00 |
| apposition (n=371) | 0.871 | 0.876 | 0.31 |
| **3-register macro** | **0.846** | 0.849 | 0.10 |
| taigi (n=779) | 0.003 | 0.000 | 0.00 |
| ptt (n=514) | 0.033 | 0.045 | 0.00 |
| **all-register macro** | **0.515** | 0.518 | — |

**The two low registers are eval bugs, not model results — and this is the correction that
mattered.** For a while the model card headlined **0.85** (the 3-register macro) while not
disclosing that taigi and ptt were excluded, nor that the all-register macro is **0.52**:

- **taigi 0.00 = self-match bug**: the query term also sits in the retrieval pool as another
  entry's canonical, so it matches itself (cosine 1.0) and never reaches the Taigi gold. This
  measures the bug, not the model. Fixing this eval is now the **top TODO** — the whole
  positioning is Taigi, so an un-measurable Taigi register is unacceptable.
- **ptt 0.03 = noisy/mislabeled source pairs** (e.g. 本來→出血性中風 is a bad pair). Not a valid
  benchmark yet.

**Protocol note (do not compare across entries naively).** The 0.85/0.52 here is the per-register,
self-match-excluded macro. The earlier `jargon top1 0.466` (2026-07-23) is the combined-pool
retrieval protocol — a *different, harder* measurement. They are not directly comparable; unifying
the eval protocol into one reported number is a TODO. Report each with its protocol, never one as
if it were the other.

**HF model-card correction (2026-07-24, live).** apposition was overstated as 0.90 (real 0.871 →
0.87); the card now names the taigi self-match bug explicitly and shows the full 0.52 all-register
macro. Honesty over headline: 0.85 on the three registers we can measure cleanly, 0.52 on
everything including the two we cannot yet.

### 2026-07-24 (same day) — CORRECTION: taigi is a transliteration task, not a self-match bug; the definitive eval

Investigating the taigi 0.00 (instead of asserting the earlier "self-match bug" hypothesis) showed
the hypothesis was **wrong**. Reading the actual `taigi_med_lexicon.tsv`: its gold is a **Han → Tâi-lô
romanization** (中風 → tiong-hong). That is a **transliteration** task, not semantic retrieval — a
meaning embedder correctly scores ~0 mapping a Han term to a romanized string (it is the ASR /
romanization model's job). Self-match affects only **31/779** surfaces and those were already masked;
fixing it would not move the ~0. So taigi is not a "bug to fix into a real number" — it is a register
that does not belong in a semantic benchmark.

**The definitive model-card eval** is now `card_eval_cpu.py` (CPU-only, committed to the repo; runs
while the GPU trains). It keeps the same combined pool but reports taigi as a labeled *diagnostic*
excluded from the macro, and strips the un-stripped `漢字/台羅` header row (taigi n 779 → 778):

| model | slang | abbrev | apposition | **semantic macro** | taigi (diagnostic) |
|---|---:|---:|---:|---:|---:|
| **IlhaEmbed-97M** | 0.855 | 0.817 | 0.895 | **0.856** | 0.000 |
| jina-v2-base-zh | 0.048 | 0.140 | 0.447 | 0.212 | 0.003 |
| ckip-base | 0.000 | 0.013 | 0.361 | 0.125 | 0.001 |
| bge-small-zh | 0.000 | 0.003 | 0.332 | 0.111 | 0.000 |

**What this corrects in the entry above:** (1) the "self-match bug, real number coming" framing is
withdrawn — taigi is transliteration, reported as a diagnostic, not folded in. (2) apposition is
**0.895** on the definitive eval (the earlier 0.871, and the "→0.87" card edit, came from a different
run/pool; the canonical number is 0.90-class). (3) the **0.52 "all-register macro" is retired** — it
averaged a transliteration task and noisy ptt pairs into a semantic benchmark, which is a meaningless
number; report the semantic macro (**0.856**) and show taigi's 0.00 separately with its reason. The
embedder's Taigi clinical capability is the `slang` register (0.86). Next benchmark to add: a real
Taigi-colloquial → standard-Han-concept (Han→Han, semantic) set.

## 2026-07-24 (same day) — the real Taigi-semantic register: `taigi_semantic_register.tsv`

Built the benchmark the entry above named as missing. It is one shared research
dataset, not a per-model artifact, and is not redistributed in this repository.

**Distinction that matters (do not repeat the earlier mistake):** `taigi_med_lexicon.tsv` maps
Han → Tâi-lô **romanization** (中風 → tiong-hong) — a transliteration task, correctly scored ~0 as
"semantic" because it isn't semantics. `taigi_semantic_register.tsv` is a different construction:
every pair is **Han → Han** — a Taiwanese-colloquial written form mapped to the standard clinical
Han concept it names (斷腦筋 → 中風, 皮蛇 → 帶狀皰疹, 胃藥 → 制酸劑). This is genuine semantic
retrieval and belongs in the semantic macro alongside slang/abbrev/apposition, unlike the
romanization lexicon.

**Size: 141 pairs across 66 clinical concepts.** Two provenance families:

1. **itaigi_TGB (134 pairs)** — derived from `data/itaigi_med_lexicon_full.tsv` +
   `data/itaigi_med_lexicon.tsv` (itaigi crowd-sourced 台語漢字辭典, `華語 <-> 台語漢字` pairs with
   community up/down votes). Filtered to pure-Han surface+canonical (drops romanized-loanword
   entries like `bi-tá-bín` → 維他命 — same transliteration class as taigi_med_lexicon, excluded
   for the same reason), net votes ≥ 1, and — the part that took the real work — a **hand-reviewed
   whitelist of ~60 clinical concepts**. High vote counts were NOT sufficient on their own: entries
   like 逐工→每天 (net=73), 誠無閒→很忙 (net=14), 白菜→小白菜 (net=41), 保生大帝→大道公 (net=7) are
   legitimate itaigi dictionary entries but not clinical vocabulary, and were excluded despite
   strong votes. Reversed-direction near-duplicates (`手術<->開刀` existed as separate entries in
   both directions in the source) were collapsed to the direction matching this register's
   contract (colloquial → standard). Capped at 6 surfaces per canonical concept so common concepts
   (懷孕, 感冒, 氣喘) don't dominate the pool.
2. **Internal clinical-narrative source (7 pairs)** — mined from a proprietary, non-redistributable
   internal case-narrative corpus (not part of this public release). Each pair is a SHORT colloquial
   expression verified recurring across multiple independent case notes (not a single anecdote) —
   e.g. `胃藥` used as the umbrella term for acid-suppressant therapy; `尿酸藥` / `降尿酸藥物` used
   interchangeably for the same drug class; `血壓高` used as the patient-reported description of a
   hypertension diagnosis. This subset (5% of the register) is excluded from the public data release;
   the itaigi_TGB family above is the fully public, reproducible majority.

Example pairs: 斷腦筋→中風, 胃藥→制酸劑, 皮蛇→帶狀皰疹, 洗腰子→洗腎, 愛睏藥仔→安眠藥, 著猴→疳積,
擋落屎藥→止瀉劑, 尿酸藥→降尿酸藥物.

**Excluded as transliteration:** every itaigi row whose 台語漢字 field was a romanized loanword
transcription instead of Han characters (`bi-tá-bín`, `ba̋i-khín`, `ha̋i-khé-khá-kuh`, `mi-sín`,
`le-bóng`) — same class taigi_med_lexicon already established as non-semantic, excluded on the
same grounds even though their target concepts (維他命, 細菌, 肺結核) are clinical.

**Known gap — a sovereignty-corpus application, still pending.** A larger, professionally-curated
Taiwanese clinical corpus is the intended primary source for this register; the application to
access it is under review, not yet usable. It would add: (1) scale beyond itaigi's incidental medical coverage (itaigi is a general dictionary;
medicine is a side effect, not its focus), (2) register closer to real 2026 pharmacist/patient
speech rather than itaigi's more literary/archaic entries (楊梅瘡, 疳瘡 for 梅毒 are historically
real but rarer in a modern clinical conversation), (3) an independent confidence signal beyond
itaigi's vote count, which measures dictionary consensus, not clinical-register accuracy. When it
clears review: extend this register, don't replace it — the two source families keep independent
provenance so before/after stays comparable.

**Eval harness:** `card_eval_taigi_semantic.py`, same combined-pool/top-1/self-match-excluded
protocol as `card_eval_cpu.py`'s three semantic registers, so numbers are directly comparable to
slang/abbrev/apposition. Also reports by confidence tier (high/medium/low, derived from itaigi net
votes; the internal-narrative pairs are high/medium by evidence strength).

**Result (CPU, GPU host, same day):**

| model | overall top1 | high tier (n=68) | medium tier (n=36) | low tier (n=37) |
|---|---:|---:|---:|---:|
| **IlhaEmbed (pruned 97M)** | **0.943** | 0.971 | 0.917 | 0.919 |
| jina-v2-base-zh | 0.638 | — | — | — |
| ckip-base | 0.560 | — | — | — |
| bge-small-zh | 0.504 | — | — | — |

n=141, pool=66. Next to the three known registers, this is the **strongest**, not the weak point:

| register | slang | abbrev | apposition | **taigi_semantic** |
|---|---:|---:|---:|---:|
| IlhaEmbed-97M | 0.855 | 0.817 | 0.895 | **0.943** |

**Caveat, stated plainly (do not read 0.943 as "Taigi solved"):** this is a real, hand-reviewed
141-pair register, but it is a v0, not the full intended register — see the MODA sovereignty
corpus gap above. The high tier (net votes ≥10, 68/141 pairs) is itaigi crowd-vote strength
filtered through a hand-reviewed clinical whitelist, not an independent clinical-accuracy signal;
a vote-strong itaigi entry and genuine 2026 clinical-register frequency are correlated, not
identical. The number can move in either direction once the MODA corpus clears review and the
register is extended — likely down if it adds harder, rarer colloquialisms less represented in
itaigi's own vote distribution, since itaigi's high-vote entries are, almost by construction, the
more commonly recognized ones.

Full source documentation and the exact exclusion rules live alongside the (non-public) data file;
the itaigi_TGB filtering method is documented in full above.

**What is NOT in this public release:** the `taigi_semantic_register.tsv` data file itself, and the
other specialized/scarce pair files (`med_slang.tsv`, `abbr_dict_v1.tsv`, `appos_pairs.tsv`,
`taigi_med_lexicon.tsv`) that `card_eval_cpu.py` reads — see `SOURCES.md` for why (third-party /
institutional copyright on the raw text) and the top-level `README.md` for what is released instead
(trained weights + the method used to build and evaluate them).
