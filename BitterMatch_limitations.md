# BitterMatch: Known Limitations and Planned Response

**Status:** internal working note
**Scope:** the "new ligands" scenario of the public BitterMatch release (Margulis et al., *J. Cheminform.* 14:45, 2022), as published at `github.com/YuliSl/BitterMatch`
**Last revised:** July 2026

---

## How to read this document

BitterMatch predicts which bitter taste receptors a molecule will activate. Think of it as a recommender system in the style of a streaming service, except that the "users" are receptors and the "films" are chemical compounds. The model looks at which compounds a receptor has responded to in the past, finds new compounds that resemble those, and scores the match.

Each section below opens with a plain description of what goes wrong, followed by the measurement that establishes it. Every number quoted was reproduced locally on the authors' own public dataset with their own code, so none of this depends on our internal compound library. Where we could not establish something firmly, that is stated rather than glossed over.

A short glossary sits at the end.

---

## Part 1. Issues that distort the reported performance figures

### L1. The decision threshold is an undocumented constant

**What it is.** The evaluation notebook converts a probability into a yes or no call by comparing it against the fixed number `0.5248`. That constant appears exactly once in the entire repository, in `new_ligands-eval.ipynb`, and nothing in the codebase derives it. It is a residue of one particular model that the authors trained on one particular random split.

**What we measured.** Rerunning the published evaluation reproduces the authors' pipeline exactly: of 252 ligand and receptor pairs, only 10 clear the threshold. All 10 are correct, which is where the headline precision of 1.00 comes from, and 28 true associations are missed. The threshold turns out to sit at the 96th percentile of the score distribution.

Repeating the fit over 15 random splits, with data and hyperparameters held constant, the same constant drifts between the 92nd and the 96th percentile. Recall at that fixed value ranged from 0.19 to 0.43 with a standard deviation of 0.07, and precision from 0.83 to 0.95. Our internal figures of 0.45 recall and 0.85 precision fall inside that band, which means they are not currently distinguishable from seed variation.

**Why this matters.** A precision of 1.00 computed from 10 predictions carries a 95% confidence interval of roughly 0.72 to 1.00 under the Wilson method. It is not evidence of a precise model. It is evidence that the operating point was placed far out on the tail. Once we retrain on regenerated descriptors, the score distribution shifts, the same constant lands at a different percentile, and any comparison against the published figures becomes meaningless. This applies equally to the 0.45 recall and 0.85 precision we obtained internally.

**Our response.** We will calibrate the threshold on the training partition by selecting the point on the precision and recall curve that meets a stated precision target, then persist that value alongside the model artefact. Reporting will lead with average precision and the full curve rather than a single operating point.

**Rationale.** A threshold is a business decision about the acceptable cost of a false lead, and it must be traceable to that decision rather than inherited from someone else's model.

### L2. Performance rests on one split of a small validation set

**What it is.** The published validation set contains 12 compounds. After removing orphan receptors and the murine subset, 252 pairs remain, of which 38 are positive. The notebook evaluates a single train and test split.

**What we measured.** Holding the data and hyperparameters constant and varying only the random seed across 15 repetitions of the 61 compound holdout, average precision averaged 0.689 with a standard deviation of 0.049 and a range of 0.604 to 0.745. That spread exceeds every modelling improvement we were able to demonstrate.

**Why this matters.** The original paper averaged over 100 repetitions and reported bootstrap confidence intervals. The released notebook does not, so a single run gives no sense of how much of an observed change is real. Our internal figure of 0.45 recall and 0.85 precision came from one seed, which means we currently cannot distinguish a genuine effect from ordinary variation.

**Our response.** All future comparisons will run over at least 20 repeated ligand grouped splits and report the mean together with a dispersion estimate. Single run numbers will not be quoted in any decision document.

**Rationale.** With a dataset of a few hundred compounds, seed variation is not a nuisance to be tidied away. It is the dominant source of uncertainty and belongs in the headline figure.

---

## Part 2. Issues in how the features are built

### L3. The neighbourhood features scale with how well studied a receptor is

**What it is.** For each compound and receptor pair the model computes how chemically similar that compound is to the receptor's known activators. The published formulation adds those similarities up rather than averaging them. A receptor with 167 recorded ligands therefore accumulates a much larger value than one with 6, before any chemistry is taken into account.

**What we measured.** The feature summarising similarity to activating ligands correlates with the receptor's ligand count at a Spearman coefficient of 0.98 in the linear fingerprint version and 0.89 in the 2D version. Between 48% and 58% of the variance in those features is attributable to which receptor a pair belongs to rather than which compound.

**Why this matters.** The paper introduces these features with the stated aim of avoiding dependence on dataset size, and the training notebook describes them in its own documentation as average similarities. Equations 5 and 7 of the paper nonetheless specify unnormalised sums, and the code implements the sums. The intent and the implementation appear to have diverged.

**Our response.** We will replace the sums with counts normalised averages and add two contrast features, the difference between mean similarity to activators and mean similarity to non activators, and the equivalent difference for the nearest neighbour features. Train and evaluation paths must be changed together and all saved model artefacts retired.

**Rationale.** Normalising restores the property the authors intended and makes the resulting feature importances interpretable. In our trial the contrast feature became the single most informative input, which is a far more defensible basis for explaining a prediction to a chemist than an unnormalised sum.

**Caveat worth stating.** Normalisation on its own moved average precision from 0.608 to 0.612 and reduced the correlation with receptor popularity only from 0.93 to 0.91. We regard this as a correctness and interpretability fix, not a performance fix, and we should not promise otherwise.

### L4. Two feature columns carry each other's names

**What it is.** In `similarity.py` the numerical arrays are assembled in the order W1, W0, M1, M0 while the column labels are written as W0, W1, M1, M0. The column called W0 therefore holds the values for W1 and vice versa.

**What we measured.** The column labelled `W0` correlates positively with receptor ligand count at 0.98, which is only possible if it actually holds W1.

**Why this matters.** Prediction is unaffected, because training and inference share the same mislabelled function. Any interpretation is affected. Reading a feature importance plot from this code without knowing about the swap leads to the conclusion that similarity to *non* activating compounds drives the model, which reverses the actual finding.

**Our response.** Correct the labels as part of the L3 rework and revisit any conclusion previously drawn from feature importance output.

### L5. Four receptor similarity matrices are loaded and never used

**What it is.** The evaluation notebook reads `Columns_similarity_matrix.csv`, `Columns_identity_matrix.csv`, `Col_Similarity_Bindingsite.csv` and `Col_Identity_Bindingsites.csv`. None of the four is referenced again anywhere in the new ligands pipeline. The corresponding training notebook does not load them at all.

**Why this matters for us specifically.** We invested effort in regenerating these matrices and in adapting `read_receptor_similarity` to handle the changed diagonal orientation produced by the newer Schrödinger release. That work has no effect on the new ligands scenario. It matters only for the "filling the gaps" scenario, which does consume these matrices. The paper is consistent on this point, noting that receptor neighbourhood features cannot be computed when a compound has no known associations, so the omission is deliberate rather than accidental.

**Our response.** Remove the unused loads from the new ligands notebook and carry the corrected reader forward only into the filling the gaps workflow, where we will verify the orientation fix on a symmetric submatrix before trusting it.

---

## Part 3. Issues in how the model is fitted

### L6. The learning rate compresses the output scale

**What it is.** The model is fitted with a learning rate of 0.001 over 1000 boosting rounds. Gradient boosting builds its answer as a sum of many small corrections, and the learning rate controls the size of each one. At 0.001 the corrections are small enough that the model never travels far from its starting point.

**What we measured.** Predicted probabilities span 0.09 to 0.65 on the public validation data. The model never expresses high confidence in anything. This ceiling proved remarkably stable: across all 15 random splits the highest score produced by any model fell between 0.632 and 0.653, so the compression is a property of the configuration rather than an accident of one run. Raising the learning rate to 0.05 opens the range to 0.001 through 0.99. Under five fold ligand grouped cross validation restricted to the training compounds, average precision was 0.728 at a rate of 0.001, peaked around 0.749 near rates of 0.01 to 0.03, and fell to 0.707 at 0.3. The standard deviation across folds was about 0.05 throughout.

**Why this matters, stated carefully.** The ranking quality is essentially flat across two orders of magnitude of learning rate. The differences we observed sit inside one cross validation standard deviation, so we should not claim that raising the rate makes the model more accurate. What the low rate demonstrably does is squash the probability scale, and that has two practical consequences. Any fixed absolute threshold becomes an arbitrary percentile, which is the mechanism behind L1. And a heatmap of these scores looks uniformly pale, which is exactly the presentation problem we encountered internally.

Early stopping is informative here as a second line of evidence. At a rate of 0.001 the optimal number of rounds under early stopping was around 50 rather than 1000, indicating that the published configuration overshoots by its own criterion without gaining anything.

**Our response.** Move to a learning rate near 0.03 with early stopping on a held aside fold, which landed at roughly 250 rounds in our trials. Present this as a calibration and usability change, not an accuracy claim.

**Rationale.** We want a score that a colleague can read as a confidence, and a threshold that means the same thing next quarter. Neither is achievable when the entire output lives inside a narrow band.

---

## Part 4. Consequences for how the predictions behave

### L7. Most of the signal describes receptors, not compounds

**What it is.** When we decompose the variance of the predicted scores, 76% of it is explained by which receptor a pair involves. The remaining quarter reflects which compound. Put plainly, the model's strongest instinct is that TAS2R14 responds to a lot of things, so anything might activate TAS2R14.

**What we measured.** Average predicted score per receptor correlates with that receptor's training ligand count at a Pearson coefficient of 0.99. Within any one receptor the spread of scores across compounds has a standard deviation of about 0.04, against 0.11 between receptors.

**But the model is not merely a popularity table.** Against a baseline that predicts each receptor's historical hit rate and ignores chemistry entirely, BitterMatch improves global average precision from 0.387 to 0.608. Asked to rank receptors for a single molecule, which is precisely the task our heatmap represents, it reaches 0.661 against the baseline's 0.474. Asked to rank molecules for a single receptor, it reaches 0.449 against a chance level of 0.189.

**Why this matters.** The chemistry contributes real information, but it is a correction applied on top of a strong prior rather than the dominant term. A heatmap of raw scores will therefore look like a portrait of BitterDB's screening history, with faint compound specific structure superimposed. That is not a defect to be eliminated, since broadly tuned receptors genuinely are more likely to respond. It is a presentation problem and a communication risk.

**Our response.** Report the heatmap in two panels. The raw calibrated probability answers "how likely is this pair" and remains the correct quantity for prioritising experiments. Alongside it we will show the score expressed relative to the receptor's prior, which isolates the compound specific evidence and is the panel a chemist should read when asking what is unusual about this molecule. Neither panel is presented without the other.

**Rationale.** Showing only the raw scores invites the reader to rediscover the ligand counts of BitterDB. Showing only the relative scores hides the fact that a weak signal on a promiscuous receptor may still be the best experimental bet.

### L8. There is no check on whether a compound falls within the model's competence

**What it is.** Nothing in the pipeline asks whether a query compound resembles anything the model was trained on. Every compound receives a confident looking score regardless.

**What we measured.** Using each compound's maximum similarity to any known activator as a coverage proxy, we split 61 validation compounds into quartiles and evaluated separately.

| Coverage quartile | Coverage range | Average precision | Prior baseline | Per compound receptor ranking |
|---|---|---|---|---|
| Lowest | 0.04 to 0.29 | 0.400 | 0.271 | 0.603 |
| Second | 0.30 to 0.62 | 0.463 | 0.363 | 0.625 |
| Third | 0.64 to 0.81 | 0.885 | 0.604 | 0.737 |
| Highest | 0.82 to 1.00 | 0.896 | 0.527 | 0.975 |

Coverage correlates with per compound average precision at a Spearman coefficient of 0.55. Applying a gate at a coverage of 0.15 retains 35 of 41 evaluable compounds at an average precision of 0.79, while the six rejected compounds score 0.40.

**A nuance we should not suppress.** Coverage correlates only weakly, at 0.14, with the *gain* over the prior baseline. Low coverage compounds are harder in absolute terms, yet the model still adds roughly the same increment above the prior for them. So these predictions are less reliable rather than entirely uninformative, and discarding them outright would waste signal.

**Why this matters for us.** Our internal library consists largely of proprietary scaffolds with limited representation in BitterDB. If coverage is systematically low, predictions will collapse towards the receptor prior across the board, which is a plausible explanation for the flat heatmap we observed and for scores that peak around 0.6.

**Our response.** Introduce a three tier triage rather than a binary gate.

1. **Within domain**, coverage at or above roughly 0.3. Report the calibrated probability and act on it.
2. **Marginal**, coverage between roughly 0.1 and 0.3. Report the ranking of receptors but suppress the absolute probability, flag the compound, and require a nearest neighbour listing so a chemist can judge whether the analogy holds.
3. **Outside domain**, coverage below roughly 0.1. Report the receptor prior explicitly labelled as such, state that the model contributed no compound specific evidence, and route the compound to a broad experimental panel instead of a targeted one.

Thresholds will be recalibrated on our own data rather than adopted from the public numbers above, since coverage distributions are library dependent.

**Rationale.** The most expensive failure mode is not a wrong prediction. It is a wrong prediction that looked as trustworthy as a right one. Making the model decline to answer converts a silent error into a visible one.

---

## Part 5. Smaller findings and latent risks

### L9. Test set membership depends on an implicit type conversion

New compounds enter the pipeline without a `test` column. Concatenation leaves that column as missing, and a later cast to boolean converts missing values to `True`. This is how the new compounds come to be marked for prediction. It works under the current pandas version, though nothing documents the intent and a future change to missing value casting would silently empty the prediction set. We will replace it with an explicit assignment.

### L10. A code path in the similarity reader cannot execute

`read_receptor_similarity` branches on a `from_file` argument, then unconditionally reads from disk on the following line, overwriting the result. Passing a DataFrame with `from_file=False` raises a type error. We confirmed this by calling it. Low impact given L5, but it should be repaired before the filling the gaps work begins.

### L11. Ligand filtering differs between training and evaluation

The evaluation notebook removes compounds with no human associations while the training notebook does not. On the public data this removes nothing, so no discrepancy is currently observable. It becomes a genuine inconsistency if our internal training set contains such compounds, because the neighbourhood features would then be computed over different reference populations at fit time and at prediction time. We will align the two paths in advance of that becoming a problem.

### L12. The chemical descriptors contribute less than expected

The authors report that ligand properties show low gain in both scenarios, and our feature importance output agrees. The model leans on receptor properties and on the similarity derived features. This tempers expectations for the regenerated 277 column descriptor set. It may improve matters, but the published architecture does not use descriptors heavily, and the honest prediction is that the effect will be modest.

---

## Priority of work

| Order | Item | Effort | Expected benefit |
|---|---|---|---|
| 1 | L8 coverage diagnostic on our library | Low | Determines whether anything else is worth doing |
| 2 | L1 threshold calibration | Low | Makes every reported metric interpretable |
| 3 | L2 repeated splits | Low | Establishes whether observed changes are real |
| 4 | L6 learning rate and early stopping | Low | Usable score range and readable heatmaps |
| 5 | L7 dual panel reporting | Medium | Prevents the popularity prior being read as chemistry |
| 6 | L3 and L4 feature rework | Medium | Correctness and interpretability, modest accuracy effect |
| 7 | L5, L9, L10, L11 cleanups | Low | Removes latent failure modes |

The ordering is deliberate. The first item can invalidate the rest, so it goes first. Items two through four cost little and repair the measurement apparatus, without which we cannot tell whether item six helped.

---

## What we are not proposing to change

We are not replacing the XGBoost classifier or the overall recommender framing. The evidence indicates the architecture extracts genuine chemical signal, roughly 0.61 average precision against 0.39 for a receptor prior, and we have no reason yet to believe a different model class would do better on a few hundred compounds. Nor are we proposing to remove the receptor popularity signal, since broadly tuned receptors really do respond more often and suppressing that would degrade a correct prior. The problem described in L7 is one of presentation and interpretation rather than of the underlying quantity.

---

## Glossary

**Association matrix.** A grid recording which compounds were tested against which receptors, and whether they responded. Most cells are empty because most combinations were never tested.

**Average precision.** A summary of ranking quality that does not depend on any particular cutoff. Higher is better, and the baseline for comparison is the fraction of pairs that are positive, about 0.15 to 0.19 here.

**Applicability domain.** The region of chemical space where a model's predictions can reasonably be trusted, usually defined by proximity to the training data.

**Calibration.** Whether a stated probability of 0.7 corresponds to something happening about 70% of the time. A model can rank well yet be poorly calibrated.

**Orphan receptor.** A receptor with no known ligands, excluded from both training and inference.

**Prior.** The base rate for a receptor, meaning the fraction of tested compounds that activated it, used here as a baseline that ignores chemistry entirely.
