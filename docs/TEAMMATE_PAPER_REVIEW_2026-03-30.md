# Teammate Paper Review Notes (2026-03-30)

Source reviewed:
- `C:\Users\zack\Documents\AwesomeGNN\GNN_Final_Project\main.tex`

Scope of this note:
- Record optimization opportunities only.
- Do not modify the teammate draft yet.

## Overall Assessment

The draft has enough material for a solid course-project paper, but it currently reads more like a combined project report than a tightly scoped academic paper. The main issue is not lack of content. The main issue is that the paper mixes evidence of different quality levels and gives too much space to exploratory and status-style material.

## Highest-Priority Optimization Points

### 1. Clarify the single main paper claim

Current issue:
- The draft effectively contains two different paper centers:
  - `Experiment: Ablation on Visual Encoder`
  - `Experiment: Ablation on Representation`
- The second one is structurally stronger and more controlled, but the first one occupies substantial space and is written as if it has equal evidential status.

Why this matters:
- The reader cannot immediately tell which experiment carries the paper's main conclusion.
- This weakens the perceived rigor of the whole draft.

Recommended direction:
- Make the controlled `PointMaze patch/CLS × deterministic/Gaussian` study the clear main paper.
- Reframe the broader encoder comparison as exploratory or preliminary evidence rather than a co-equal main experiment.

### 2. Align claims with evidence strength

Current issue:
- The abstract and conclusion currently describe the encoder-comparison experiment in stronger terms than the body fully supports.
- Experiment 1 relies heavily on training and rollout curves plus qualitative summary, while the planning-side evidence is less compact and less formalized than Experiment 2.

Why this matters:
- It creates overclaim risk.
- A reader may feel the paper promises a fair benchmark but delivers a partial exploratory comparison.

Recommended direction:
- Narrow the abstract and conclusion so that the strongest claims come only from the controlled PointMaze matrix.
- Keep V-JEPA2 / VFM-VAE results, but explicitly mark them as exploratory if they were not run under the same level of control.

### 3. Remove project-status writing from the main body

Current issue:
- `What Has Not Been Demonstrated Yet`
- `Encoder-3 and Encoder-4 Status`

Why this matters:
- These sections read like internal progress notes rather than paper discussion.
- Environment blockers, Python-version friction, and memory notes are useful for engineering documentation, but not for a polished final draft.

Recommended direction:
- Delete these sections from the main body, or compress them into one short sentence in `Future Work` or an appendix-style note.

### 4. Reduce figure overload from raw training diagnostics

Current issue:
- The draft includes many low-level training and rollout plots in `figures/`.
- These plots are useful during experimentation but currently dominate too much of Experiment 1.

Why this matters:
- The paper spends many figure slots on diagnostics rather than on synthesis.
- Readers should be able to identify the final comparative takeaway without reading multiple 2x2 curve grids.

Recommended direction:
- Replace most low-level plots with one or two summary figures:
  - one compact representation-learning summary
  - one compact planning comparison summary
- Move extra diagnostic curves to appendix or omit them.

### 5. Make the evidence hierarchy explicit

Current issue:
- The draft currently mixes:
  - controlled formal matrix evidence
  - larger-sample but asymmetric follow-up
  - cross-environment sanity checks
  - exploratory encoder comparisons

Why this matters:
- These are not the same type of evidence.
- Without a clear hierarchy, the reader may overinterpret weaker sections or underappreciate stronger ones.

Recommended direction:
- Explicitly separate:
  - primary evidence
  - supporting evidence
  - sanity checks
  - exploratory evidence

## Medium-Priority Optimization Points

### 6. Tighten the Introduction

Current issue:
- The introduction tries to introduce both the broad encoder agenda and the controlled ablation agenda at once.
- It also repeats some motivation that later reappears in the experiment sections.

Recommended direction:
- Keep the introduction to:
  - problem
  - why DINO-WM is the right backbone
  - paper questions
  - what this draft actually completes
  - contributions

### 7. Strengthen the Results section as a results section

Current issue:
- Some parts of the results read more like narrative interpretation than evidence presentation.
- The strongest controlled evidence is in the formal matrix, but the surrounding sections sometimes dilute it.

Recommended direction:
- Results should prioritize:
  - tables
  - summary figures
  - concise empirical statements
- Push broader interpretation to `Discussion`.

### 8. Narrow general statements

Current issue:
- Some conclusions are stated too generally, for example implying broader superiority of one representation family over another.

Why this matters:
- The completed evidence is PointMaze-centered and budget-specific.

Recommended direction:
- Use narrower wording such as:
  - `under the current PointMaze setting`
  - `in this controlled DINO-WM setup`
  - `the current evidence suggests`

## Lower-Priority Optimization Points

### 9. Clean LaTeX and style issues

Observed issues:
- Duplicate package imports:
  - `graphicx`
  - `hyperref`
- Inconsistent subsection capitalization:
  - `motivation`
- Encoding artifacts such as:
  - `â€”`
  - `â€™`
- A few awkward sentence joins around figure references.

Recommended direction:
- Do one cleanup pass after the structure is settled.

### 10. Improve title and framing precision

Current issue:
- The current title tries to cover both representation comparison and uncertainty in one line.

Recommended direction:
- After the paper center is finalized, simplify the title so it matches the actual main evidence.

## Recommended Revision Order

1. Decide the single paper center.
2. Reclassify other material as supporting or exploratory evidence.
3. Remove project-status sections from the main body.
4. Replace multiple raw diagnostic plots with fewer summary figures.
5. Rewrite abstract and conclusion to match the actual evidence hierarchy.
6. Do a final language and LaTeX cleanup pass.

## Practical Recommendation

If time is limited, the highest-value move is not adding more content. It is restructuring the paper so that the strongest completed experiment becomes unmistakably central, and everything else is clearly labeled by evidence strength.

## External Review Cross-Check (Gemini Feedback Integrated)

An additional external review was received after the initial note above. Most of its strongest points are consistent with the current assessment, but it also adds several concrete cleanup items that should be tracked explicitly.

### Points that confirm the current review

- The draft still reads partly like a `lab notebook` or progress report rather than a finalized paper.
- The results presentation exposes too many engineering and status details in the main text.
- The paper should present only completed and interpretable evidence in its main tables and figures.
- The title, abstract, and body are not yet perfectly aligned in scope.

### New concrete items from the external review

#### A. Citation and reference consistency

Observed concern:
- Numeric citation references in the teammate draft appear mismatched with the actual bibliography entries.
- Some in-text bracketed references point to the wrong papers.

Why this matters:
- This is a high-visibility professionalism issue.
- Even a strong paper looks unreliable if its citations do not resolve correctly.

Action note:
- Do a dedicated bibliography pass before any final submission.
- Verify every in-text citation against the actual reference list after the structure is stabilized.

#### B. Broken cross-references and figure-text links

Observed concern:
- The draft contains at least one broken figure reference such as `??`.
- Some figure-introduction sentences also have missing spaces or awkward joins.

Why this matters:
- Broken references immediately signal an unfinished draft.

Action note:
- Run a final LaTeX compile-and-fix pass specifically for:
  - figure references
  - table references
  - section references
  - spacing around references

#### C. Table language is too informal

Observed concern:
- Terms like `shard B only`, `stalled`, and `optional unfinished` are too notebook-like for the main paper.

Why this matters:
- These labels are informative for internal tracking but read as incomplete or unpolished in a paper.

Action note:
- In the final paper, either:
  - exclude incomplete groups from the main table, or
  - summarize them in cleaner terms such as `partial coverage` or `not included in aggregate`.

#### D. Engineering environment details should be trimmed

Observed concern:
- Too much paper space is spent on environment fixes such as WSL-specific Hydra changes and manual MuJoCo setup.

Why this matters:
- These details are useful for reproducibility notes, but they do not belong at full length in the main scientific narrative unless they materially change the experiment.

Action note:
- Compress environment/debugging notes into a short reproducibility paragraph.
- Move detailed setup friction to repository documentation if needed.

#### E. Decoder asymmetry should be sharpened as a fairness issue

Observed concern:
- The patch runs and CLS runs are not fully symmetric if decoder-side losses or gradients influence training.

Why this matters:
- This is not just an implementation note; it affects how fair the ablation is.

Action note:
- Explicitly verify whether decoder-side loss contributes gradients to the learned transition stack in the patch baseline.
- If yes, the paper must describe the comparison as partially asymmetric rather than fully controlled.

#### F. Metric justification should be stronger

Observed concern:
- The paper correctly moves from `success_rate` to `mean_state_dist`, but the justification can be made more rigorous.

Why this matters:
- Replacing the headline metric is a methodological choice and should be defended more explicitly.

Action note:
- Explain more clearly why `mean_state_dist` is a better discriminator for PointMaze under metric saturation, rather than only saying that `success_rate` is coarse.

#### G. CLS conclusion should remain slightly softer

Observed concern:
- The conclusion that CLS is surprisingly competitive is interesting, but larger-sample CLS evidence is still incomplete.

Why this matters:
- The paper should not overstate confidence beyond the completed evidence.

Action note:
- Use wording such as:
  - `preliminary but consistent with the current controlled matrix`
  - `suggests competitiveness under the present budget`
  - `should be validated by more symmetric larger-sample follow-up`

## Updated Priority View After External Review

After incorporating the external review, the priority order is:

1. Fix the paper center and evidence hierarchy.
2. Remove or compress lab-notebook material from the main text.
3. Resolve citation, cross-reference, and formatting correctness issues.
4. Tighten fairness language around decoder asymmetry.
5. Strengthen the metric rationale for `mean_state_dist`.
6. Soften any claims that currently exceed the completed evidence.

## Additional Reviewer Synthesis (Integrated 2026-03-30)

Another review round reinforced the same overall diagnosis: the paper already has a credible course-project core, but the evidence strength, narrative scope, and writing finish are not yet fully aligned.

### Main additions from this round

#### 1. Claims are still slightly stronger than the completed evidence

Examples called out:
- `systematic representation study`
- `reproducible insights`
- `fair comparison`

Recommended direction:
- Reframe the paper more explicitly as:
  - `a controlled course-project study on PointMaze`, or
  - `a preliminary but controlled ablation study within the DINO-WM framework`

#### 2. Encoder-comparison results need a harder quantitative summary

Observed issue:
- Experiment 1 currently relies too much on curves plus narrative interpretation.
- The planning-side evidence needs a clearer summary artifact.

Recommended direction:
- Add a dedicated summary table for Experiment 1, ideally including:
  - train / val latent loss
  - visual rollout error
  - mean state distance
  - success rate
  - mean ± std across seeds when available

#### 3. Statistical strength should be described more proactively

Observed issue:
- The draft already admits that:
  - `3 epochs` is short,
  - `n_evals = 3` is weak,
  - the larger-sample follow-up is asymmetric.

Recommended direction:
- State more directly that the conclusions are trend-level rather than benchmark-level.
- Explain why the current budget is still sufficient to reveal the main qualitative pattern.

#### 4. The patch-vs-CLS comparison should be framed as planning-side only

Observed issue:
- Decoder asymmetry is not just an implementation quirk; it changes how fair the comparison is.

Recommended direction:
- Explicitly state that patch-vs-CLS conclusions are limited to planning-side behavior and do not support reconstruction-side claims.

#### 5. Main text should not carry unfinished-project energy

Observed issue:
- The draft still includes traces of project-log style writing and partially unfinished artifacts.

Recommended direction:
- Keep the core narrative focused on completed evidence.
- Downgrade branch-status material, incomplete runs, and engineering blockers to appendix / future work / notes.

#### 6. Main-paper figures should be more synthesis-oriented

Observed issue:
- Many current figures are training diagnostics rather than paper-first summary figures.

Recommended direction:
- Main body should prioritize:
  - one encoder-comparison summary table,
  - one main ablation table,
  - one or two synthesis figures for rollout / planning.
- Extra training curves can move to appendix.
