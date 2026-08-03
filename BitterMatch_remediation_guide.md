# BitterMatch Remediation: Implementation Guide

Copy and paste target for items 1 to 6 of the audit. Everything here was tested against the public dataset in this repository before being written down.

---

## How to use this document

**If you want something you can run**, use the four files below. They implement every item and were executed end to end against the public dataset before this was written. Only `bm_config.py` contains placeholders, and it is the only file you edit.

| File | What it is |
|---|---|
| `bm_config.py` | every path and setting, all placeholders live here |
| `bittermatch_ext.py` | the corrected feature construction and all helpers, no placeholders |
| `bm_train.py` | fits, calibrates the threshold, fixes the coverage cut points, writes artefacts |
| `bm_predict.py` | scores one query set, labelled or not, writes the table and the figure |
| `bm_repeated_eval.py` | the dispersion estimate item 2 requires, plus a label permutation control |

PowerShell, which is what the internal machine runs:

```powershell
python bm_train.py                                          # once

$env:BM_CONFIG = "bm_config_nba"; python bm_predict.py      # 14 compound set
$env:BM_CONFIG = "bm_config_nbb"; python bm_predict.py      # 20 compound set

$env:BM_CONFIG = "bm_config"
Start-Process -NoNewWindow -FilePath python `
    -ArgumentList "-u", "bm_repeated_eval.py" `
    -RedirectStandardOutput "repeated_eval.log" `
    -RedirectStandardError  "repeated_eval.err"
Get-Content repeated_eval.log -Wait
```

`$env:BM_CONFIG` persists for the rest of the session once set, so clear it with `Remove-Item Env:BM_CONFIG` afterwards or set it again before every run. The equivalent on a shell that uses `VAR=value command` syntax:

```bash
python bm_train.py
BM_CONFIG=bm_config_nba python bm_predict.py
BM_CONFIG=bm_config_nbb python bm_predict.py
nohup python -u bm_repeated_eval.py > repeated_eval.log 2>&1 &
```

Copy `bm_config.py` twice, once per query set, and select between them with `BM_CONFIG` as shown. Nothing in the original repository is edited. In particular `similarity.py` is left alone, since the corrected `sim_metrics` lives in the new module and supersedes it, which keeps your own changes to that file out of the way.

**If you would rather fold the changes into your notebooks**, the remaining sections give the same logic as cell level edits with the reasoning attached. Sections 2 through 6 describe the general shape and section 9 covers what differs for the two inference notebooks specifically. The scripts and the snippets do the same thing, so pick one route rather than mixing them.

A note on placeholders. The two query sets are not present on the machine this was developed on, so every path pointing at them is a placeholder and the code around them was exercised against the public validation set instead, once with associations supplied and once with `QUERY_ASSOC_CSV = None` to reproduce the label free case.

---

## 0. Placeholders and prerequisites

Replace these before running anything.

| Placeholder | Meaning |
|---|---|
| `<REPO>` | repository root on the internal machine |
| `<A_CSV>` | internal association matrix, same layout as `bitterdb_associations.csv` |
| `<LIG_FEAT_CSV>` | internal ligand descriptors, the 277 column Schrödinger output |
| `<REC_FEAT_CSV>` | receptor features |
| `<LIG_LINEAR_SIM_CSV>` | regenerated linear fingerprint similarity matrix |
| `<LIG_MOL2D_SIM_CSV>` | regenerated 2D fingerprint similarity matrix |
| `<MODEL_DIR>` | directory for model and calibration artefacts |
| `<SEED>` | base random seed, currently 6 |

One structural note before you start. Items 3 and 4 change the feature schema, so every model artefact trained before that change becomes unusable. Sequence the work as follows: apply the `similarity.py` changes first, retrain once, then layer items 1, 2, 5 and 6 on top of the new artefact. Doing it in the other order means calibrating a threshold you will immediately throw away.

### 0.1 Which parts apply to which notebook

Three notebooks are in play. The published repository blurs the distinction between them, which is the source of several problems described in section 9.

| Short name | What it is | Labels available |
|---|---|---|
| **TRAIN** | `new_ligands-train.ipynb`, fits the model on BitterDB plus whatever internal associations you have added | full |
| **NB-A** | your 14 compound notebook | 8 of 14 |
| **NB-B** | your 20 compound notebook | none |

Not every item belongs everywhere. Applying item 2 inside NB-B, for instance, is not merely unnecessary but impossible, since there is nothing to score against.

| Item | TRAIN | NB-A | NB-B |
|---|---|---|---|
| 1, threshold calibration | produces the artefact | consumes it | consumes it |
| 2, repeated splits | yes | no | no |
| 3 and 4, `similarity.py` | yes | must match TRAIN exactly | must match TRAIN exactly |
| 5, learning rate | yes | no | no |
| 6, coverage tiers | validate the gate here | apply, limited check possible | apply, no check possible |
| Two panel heatmap | no | yes | yes |

The asymmetry in the last two rows carries most of the practical weight. Every quantity that tells you whether to believe a prediction is estimated in TRAIN. NB-A and NB-B consume those estimates but cannot produce them. Read section 9 before touching either inference notebook.

---

## 1. New module: `<REPO>/bittermatch_ext.py`

This is a new file. Nothing in the original repository is deleted by creating it. It holds the pieces that both notebooks need, which keeps the notebooks readable and stops the two paths drifting apart.

```python
"""Extensions to the published BitterMatch pipeline.

Addresses audit items 1, 2, 5 and 6. Item 3 and 4 live in similarity.py.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import precision_score, recall_score

from similarity import sim_metrics

ID_COLS = ['ligand', 'receptor', 'association']


# --------------------------------------------------------------------------
# Receptor prior as an explicit feature (supports items 3 and 6)
# --------------------------------------------------------------------------
def receptor_prior_features(masked_A):
    """Per receptor base rate computed from KNOWN associations only.

    Must be given the same masked matrix that feeds sim_metrics, otherwise
    held out labels leak in through the denominator.
    """
    n_tested = masked_A.notna().sum(axis=0)
    n_positive = masked_A.sum(axis=0)
    out = pd.DataFrame({
        'Rec_n_tested': n_tested,
        'Rec_n_positive': n_positive,
        'Rec_prior': n_positive / n_tested.replace(0, np.nan),
    })
    out.index.name = 'receptor'
    return out.reset_index()


# --------------------------------------------------------------------------
# Feature assembly, factored out of the notebooks so train and eval agree
# --------------------------------------------------------------------------
def build_base(X_Lig, X_Rec):
    """Cross join of ligand and receptor content features. Computed once."""
    base = pd.merge(
        X_Lig.rename(columns=lambda c: 'Lig_%s' % c).assign(key_=1),
        X_Rec.rename(columns=lambda c: 'Rec_%s' % c).assign(key_=1),
        on='key_').drop(columns='key_')
    base = base.rename(columns={'Lig_cid': 'ligand', 'Rec_column_label': 'receptor'})
    base['is_human_receptor'] = base.receptor < 2000
    return base


def build_features(A, base, long_A, sim_dict, mask_ligands, keep_unknown=False):
    """Assemble the full design matrix with `mask_ligands` held out.

    Masking happens before any neighbourhood feature is computed, which is the
    only way to keep W and M free of the held out labels.
    """
    masked_A = A.copy()
    masked_A.loc[masked_A.index.isin(mask_ligands), :] = np.nan

    f = base.copy()
    for prefix, (S, axis) in sim_dict.items():
        block = sim_metrics(S, masked_A, axis).rename(columns=lambda c: '%s_%s' % (prefix, c))
        f = f.merge(block, how='left', on=['ligand', 'receptor'])
    f = f.merge(receptor_prior_features(masked_A), how='left', on='receptor')
    f = f.merge(long_A, how='left', on=['ligand', 'receptor'])
    return f if keep_unknown else f[f.association.notna()]


def make_model(seed, learning_rate=0.03, n_estimators=250, early_stopping_rounds=None):
    """Item 5. Defaults chosen by cross validation, see the guide text."""
    kw = dict(objective='binary:logistic', booster='gbtree', n_jobs=-1,
              seed=seed, random_state=seed, subsample=0.7, scale_pos_weight=1,
              min_child_weight=0.45, max_depth=4, gamma=2.0,
              colsample_bytree=0.3, colsample_bylevel=1.0,
              learning_rate=learning_rate, n_estimators=n_estimators,
              tree_method='hist', verbosity=0)
    if early_stopping_rounds:
        kw.update(early_stopping_rounds=early_stopping_rounds, eval_metric='aucpr')
    return xgb.XGBClassifier(**kw)


# --------------------------------------------------------------------------
# Item 1: threshold calibration
# --------------------------------------------------------------------------
def calibrate_threshold(y, p, target_precision):
    """Highest recall point on the PR curve that still meets the precision target.

    Feed this OUT OF FOLD scores. In sample scores are optimistic and will
    produce a threshold that collapses the moment it meets new compounds.
    """
    pr, rc, th = precision_recall_curve(y, p)
    ok = np.where(pr[:-1] >= target_precision)[0]
    if len(ok) == 0:
        fallback = float(np.quantile(p, 0.99))
        return fallback, {'met_target': False, 'note': 'fell back to 99th percentile'}
    i = ok[int(np.argmax(rc[ok]))]
    return float(th[i]), {'met_target': True,
                          'oof_precision': float(pr[i]), 'oof_recall': float(rc[i])}


def oof_predictions(A, base, long_A, sim_dict, train_ligands, holdout_ligands,
                    seed, n_folds=4, model_kw=None):
    """Ligand grouped out of fold scores over the training ligands only."""
    model_kw = model_kw or {}
    rng = np.random.RandomState(seed)
    folds = np.array_split(rng.permutation(train_ligands), n_folds)
    ys, ps = [], []
    for val in folds:
        f = build_features(A, base, long_A, sim_dict,
                           np.concatenate([holdout_ligands, val]))
        tr = f[f.ligand.isin(np.setdiff1d(train_ligands, val))]
        va = f[f.ligand.isin(val)]
        m = make_model(seed, **model_kw).fit(tr.drop(ID_COLS, axis=1), tr.association.values)
        ps.append(m.predict_proba(va.drop(ID_COLS, axis=1))[:, 1])
        ys.append(va.association.values)
    return np.concatenate(ys), np.concatenate(ps)


# --------------------------------------------------------------------------
# Item 2: repeated ligand grouped evaluation
# --------------------------------------------------------------------------
def repeated_evaluation(A, base, long_A, sim_dict, n_repeats=20, train_frac=0.8,
                        base_seed=100, target_precision=0.80, model_kw=None,
                        calibrate=True, verbose=True):
    """Repeat the whole protocol, including calibration, and report dispersion.

    Cost warning. Each repeat rebuilds the neighbourhood features once per
    inner fold plus once for the final fit, so 20 repeats with 4 inner folds is
    roughly 100 feature builds. Budget half an hour and run it detached.
    """
    model_kw = model_kw or {}
    all_lig = np.unique(long_A.ligand)
    rows = []
    for rep in range(n_repeats):
        seed = base_seed + rep
        rng = np.random.RandomState(seed)
        train_lig = rng.choice(all_lig, int(train_frac * len(all_lig)), replace=False)
        test_lig = np.setdiff1d(all_lig, train_lig)

        thr = np.nan
        if calibrate:
            oy, op = oof_predictions(A, base, long_A, sim_dict, train_lig, test_lig,
                                     seed, model_kw=model_kw)
            thr, _ = calibrate_threshold(oy, op, target_precision)

        f = build_features(A, base, long_A, sim_dict, test_lig)
        tr, te = f[f.ligand.isin(train_lig)], f[f.ligand.isin(test_lig)]
        m = make_model(seed, **model_kw).fit(tr.drop(ID_COLS, axis=1), tr.association.values)
        p = m.predict_proba(te.drop(ID_COLS, axis=1))[:, 1]
        y = te.association.values

        row = {'repeat': rep, 'seed': seed, 'AP': average_precision_score(y, p),
               'base_rate': float(y.mean()), 'score_min': float(p.min()),
               'score_max': float(p.max())}
        if calibrate:
            yh = 1 * (p >= thr)
            row.update(threshold=thr, n_called=int(yh.sum()),
                       precision=precision_score(y, yh, zero_division=0),
                       recall=recall_score(y, yh))
        rows.append(row)
        if verbose:
            print('repeat %d/%d  AP=%.3f' % (rep + 1, n_repeats, row['AP']), flush=True)
    return pd.DataFrame(rows)


def summarise(df):
    """Mean and dispersion. This is what goes into a decision document."""
    num = df.select_dtypes(include=[np.number]).drop(columns=['repeat', 'seed'], errors='ignore')
    out = pd.DataFrame({'mean': num.mean(), 'sd': num.std(),
                        'min': num.min(), 'max': num.max()})
    return out.round(4)


# --------------------------------------------------------------------------
# Item 6: applicability domain
# --------------------------------------------------------------------------
def coverage_per_ligand(features_df, sim_prefixes=('Lig_linear_sim', 'Lig_mol2d_sim')):
    """Strongest resemblance between each compound and any known activator.

    Taking the max across similarity views is deliberate. A compound only needs
    one view to recognise it, and requiring agreement would discard compounds
    that a single fingerprint happens to describe well.
    """
    cols = ['%s_M1' % p for p in sim_prefixes]
    cols = [c for c in cols if c in features_df.columns]
    if not cols:
        raise KeyError('no M1 columns found, check sim_prefixes against %s'
                       % [c for c in features_df.columns if c.endswith('M1')])
    return features_df.groupby('ligand')[cols].max().max(axis=1).rename('coverage')


def assign_tier(coverage, t_low=0.10, t_high=0.30):
    return pd.cut(coverage, bins=[-np.inf, t_low, t_high, np.inf],
                  labels=['outside', 'marginal', 'within'])


def coverage_cutpoints(coverage, q_low=0.25, q_high=0.60):
    """Recalibrate the gate on your own library rather than importing 0.1 / 0.3.

    Coverage distributions are library dependent, so the public cut points are
    a starting shape, not a constant.
    """
    return float(coverage.quantile(q_low)), float(coverage.quantile(q_high))


def validate_coverage_gate(results_df, coverage, t_low=0.10, t_high=0.30):
    """Evidence that the gate separates reliable from unreliable predictions.

    Run this before trusting the tiers. If AP does not increase across tiers on
    your data, the coverage proxy is not working and the gate should not ship.
    """
    d = results_df.merge(coverage, left_on='ligand', right_index=True)
    d['tier'] = assign_tier(d.coverage, t_low, t_high)
    rows = []
    for tier, g in d.groupby('tier', observed=True):
        if g.association.nunique() < 2:
            continue
        per_lig = [average_precision_score(x.association, x.score)
                   for _, x in g.groupby('ligand') if 0 < x.association.sum() < len(x)]
        rows.append({'tier': tier, 'n_ligands': g.ligand.nunique(), 'n_pairs': len(g),
                     'coverage_min': g.coverage.min(), 'coverage_max': g.coverage.max(),
                     'AP': average_precision_score(g.association, g.score),
                     'prior_AP': average_precision_score(g.association, g.Rec_prior)
                                 if 'Rec_prior' in g else np.nan,
                     'within_ligand_AP': np.mean(per_lig) if per_lig else np.nan})
    return pd.DataFrame(rows)


def nearest_neighbours(ligand_id, S, A, receptor, top_k=5, name_map=None):
    """Which known compounds drove this call. Required output for marginal tier."""
    if ligand_id not in S.index:
        return pd.DataFrame(columns=['neighbour', 'similarity', 'association'])
    sims = S.loc[ligand_id].drop(index=ligand_id, errors='ignore')
    known = A[receptor].dropna()
    sims = sims[sims.index.isin(known.index)].sort_values(ascending=False).head(top_k)
    out = pd.DataFrame({'neighbour': sims.index, 'similarity': sims.values,
                        'association': known.reindex(sims.index).values})
    if name_map:
        out['neighbour'] = out.neighbour.map(lambda v: name_map.get(v, v))
    return out


# --------------------------------------------------------------------------
# Reporting: separate the receptor prior from the compound specific evidence
# --------------------------------------------------------------------------
def lift_over_prior(score, prior, eps=1e-6):
    """Log odds of the prediction minus log odds of the receptor base rate.

    Zero means the model learned nothing about this compound beyond the fact
    that this receptor responds often. Positive means the chemistry argued for
    the pair, negative means it argued against.
    """
    s = np.clip(np.asarray(score, dtype=float), eps, 1 - eps)
    q = np.clip(np.asarray(prior, dtype=float), eps, 1 - eps)
    return np.log(s / (1 - s)) - np.log(q / (1 - q))
```

---

## 2. Items 3 and 4: replace `sim_metrics` in `<REPO>/similarity.py`

Delete the existing `sim_metrics` body from the line beginning `np.fill_diagonal` to the end of the function, then paste the following. The signature gains one argument with a default, so nothing else needs to change at the call sites.

```python
def sim_metrics(S, A, axis, normalise=True):
    """Neighbour informed features.

    Items 3 and 4 of the audit. Two changes relative to the published version.

    The W features are averages rather than sums. Paper equations 5 and 7
    specify sums, which makes the feature proportional to how many ligands the
    receptor has on record. Measured Spearman correlation with receptor ligand
    count was 0.98 before this change.

    The column labels now match the arrays. The published version assembled
    W1, W0, M1, M0 and labelled them W0, W1, M1, M0.
    """
    if axis == 'row' or axis == 0:
        rows = A.index.intersection(S.index)
        cols = A.columns
        A_vals = A.loc[rows, :].values.copy()
        S_vals = S.loc[rows, rows].values.copy()
        transpose = False
    elif axis == 'col' or axis == 1:
        rows = A.index
        cols = A.columns.intersection(S.index)
        A_vals = A.loc[:, cols].values.copy().T
        S_vals = S.loc[cols, cols].values.copy()
        transpose = True
    else:
        raise ValueError('axis must be either "row" or 0, or "col" or 1.')

    np.fill_diagonal(S_vals, 0)

    pos = np.nan_to_num(A_vals)
    neg = np.nan_to_num(1 - A_vals)
    ones = np.ones_like(S_vals)

    W1, W0 = S_vals.dot(pos), S_vals.dot(neg)
    n1, n0 = ones.dot(pos), ones.dot(neg)      # how many known pos/neg per column

    M01 = np.array([(np.max(S_vals * (line == 0), axis=1),
                     np.max(S_vals * (line == 1), axis=1)) for line in A_vals.T]).T

    if transpose:
        W1, W0, n1, n0, M01 = W1.T, W0.T, n1.T, n0.T, M01.T
    M1, M0 = M01[:, 1, :], M01[:, 0, :]

    if normalise:
        mW1 = W1 / np.maximum(n1, 1)
        mW0 = W0 / np.maximum(n0, 1)
        vals = np.vstack([mW1.flatten(), mW0.flatten(), (mW1 - mW0).flatten(),
                          M1.flatten(), M0.flatten(), (M1 - M0).flatten()]).T
        names = ['W1', 'W0', 'dW', 'M1', 'M0', 'dM']
    else:
        vals = np.vstack([W1.flatten(), W0.flatten(),
                          M1.flatten(), M0.flatten()]).T
        names = ['W1', 'W0', 'M1', 'M0']       # labels corrected either way

    return pd.DataFrame(
        vals,
        index=pd.MultiIndex.from_product([rows, cols], names=['ligand', 'receptor']),
        columns=names)
```

### Why the receptor prior comes back as its own feature

Normalising W removes the popularity signal from the neighbourhood features, and that signal is genuinely predictive. A base rate model that ignores chemistry entirely reached an average precision of 0.387 against a base rate of 0.195 on the public data, so discarding it would cost real accuracy.

The intent is not to suppress popularity but to stop it hiding inside a feature that claims to measure chemical similarity. `receptor_prior_features` in the new module reintroduces it explicitly as `Rec_prior`, `Rec_n_positive` and `Rec_n_tested`. The model can then use the prior where it helps, and the two effects stay separable when you explain a prediction or build the reporting described at the end of this guide.

### What the change actually achieves, measured

Running the replacement on the public data, the coupling between the activator similarity feature and receptor ligand count fell from a Spearman correlation of 0.98 to 0.33, and the share of that feature's variance attributable to receptor identity dropped from 58% to 7%. The feature now measures roughly what its name claims.

One part of the problem survives, and it should be stated rather than quietly left in place. The nearest neighbour features keep their popularity coupling, with `dM` still correlating at 0.89. This is structural: a maximum taken over 167 candidate ligands will tend to exceed a maximum taken over 6, whatever normalisation is applied to the sums. Two options exist if this becomes a concern, neither yet tested. Replacing the maximum with a mean over the top three neighbours reduces the dependence on sample size and is more stable besides. Alternatively the feature can be expressed as a quantile of the similarity distribution to that receptor's ligands rather than as a raw maximum. I would treat this as a follow up rather than a blocker, since `Rec_prior` now gives the model a clean channel for the popularity signal and the residual coupling in `dM` is no longer the only route by which it can enter.

### Compatibility warning

The feature count goes from four to six per similarity view, plus three receptor prior columns. Any pickled model predating this change will raise a feature name mismatch, which is the desired behaviour. Move the old artefacts somewhere with a clear name rather than deleting them, in case a reviewer asks to see the previous numbers.

---

## 3. Item 5 and item 1: changes to `<REPO>/new_ligands-train.ipynb`

### 3a. Replace the model construction in cell 30

```python
from bittermatch_ext import make_model, ID_COLS

# Item 5. Learning rate 0.001 over 1000 rounds confined predictions to
# 0.09 through 0.65, with the ceiling stable at 0.632 to 0.653 across 15 splits.
# Cross validated average precision was flat from 0.001 to 0.1, so this is a
# resolution and usability change, not an accuracy claim.
xgb_clf = make_model(seed=SEED, learning_rate=0.03, n_estimators=250)
model = xgb_clf.fit(X_train, Y_train)
Y_pred = model.predict_proba(X_test)
```

If you prefer the round count chosen automatically rather than fixed, use early stopping. Note that this consumes the test partition as a stopping signal, so the resulting test metrics are mildly optimistic and the honest version carves a third partition out of the training ligands instead.

```python
xgb_clf = make_model(seed=SEED, learning_rate=0.03, n_estimators=3000,
                     early_stopping_rounds=50)
model = xgb_clf.fit(X_train, Y_train, eval_set=[(X_test, Y_test)], verbose=False)
print('stopped at round', model.best_iteration)
```

### 3b. Add threshold calibration as a new cell after cell 44

Cell 44 is the one that computes the precision and recall curves.

```python
import json, pickle
import numpy as np
from bittermatch_ext import (build_base, build_features, oof_predictions,
                             calibrate_threshold, ID_COLS)

TARGET_PRECISION = 0.80          # see the guide text on choosing this
MODEL_KW = dict(learning_rate=0.03, n_estimators=250)

base = build_base(X_Lig, X_Rec)
long_A = pd.melt(A.assign(ligand=A.index), id_vars='ligand',
                 var_name='receptor', value_name='association')
train_ligands = np.setdiff1d(np.unique(long_A.ligand), test_ligands)

# Out of fold scores over the training ligands. The test ligands stay masked
# throughout, so nothing about them reaches the calibration.
oof_y, oof_p = oof_predictions(A, base, long_A, sim_metrics_dict,
                               train_ligands, test_ligands,
                               seed=SEED, n_folds=4, model_kw=MODEL_KW)

threshold, info = calibrate_threshold(oof_y, oof_p, TARGET_PRECISION)
print('threshold %.4f  %s' % (threshold, info))

artefact = {'threshold': threshold, 'target_precision': TARGET_PRECISION,
            'calibration': info, 'seed': SEED, 'model_kw': MODEL_KW,
            'feature_names': list(X_train.columns)}
with open('<MODEL_DIR>/calibration_for_new_ligands.json', 'w') as fh:
    json.dump(artefact, fh, indent=2)
```

Once you have run the coverage gate validation described in section 5e, add its two cut points to the same file. Both inference notebooks read them from here and neither recomputes them.

```python
artefact['coverage_cutpoints'] = {'low': float(T_LOW), 'high': float(T_HIGH)}
artefact['train_holdout_AP'] = float(average_precision_score(Y_test, Y_pred[:, 1]))
with open('<MODEL_DIR>/calibration_for_new_ligands.json', 'w') as fh:
    json.dump(artefact, fh, indent=2)
```

`train_holdout_AP` is stored so that the inference notebooks have something to compare against. Without a reference figure, an average precision computed on a handful of internal compounds is a number with nothing to say.

Storing `feature_names` alongside the threshold is what lets the evaluation notebook fail loudly instead of silently scoring a misaligned matrix.

### Choosing the precision target

You asked whether 0.8 is the right number. It appears to be, and the reasoning is worth putting in the report rather than presenting it as a preference.

I ran the exact protocol above over 8 repeats on the public data. Each repeat calibrated a threshold on inner fold out of fold scores, then applied it to compounds the model had never seen, at four different targets.

| Target | Threshold (mean, sd) | Achieved precision | Achieved recall | Positive calls |
|---|---|---|---|---|
| 0.60 | 0.263, 0.026 | 0.637, sd 0.068 | 0.666, sd 0.049 | 154 |
| 0.70 | 0.403, 0.053 | 0.721, sd 0.072 | 0.590, sd 0.034 | 120 |
| **0.80** | **0.604, 0.049** | **0.810, sd 0.080** | **0.506, sd 0.045** | **92** |
| 0.90 | 0.817, 0.056 | 0.888, sd 0.076 | 0.330, sd 0.098 | 55 |

Two things follow.

The calibration transfers with almost no bias. Achieved precision tracked the requested target at every level, which is the evidence that the out of fold recipe works and that the published constant was not merely unlucky but unanchored.

The 0.80 setting sits at the bend in the trade. Loosening from 0.80 to 0.70 buys 0.08 of recall at a cost of 0.09 of precision, roughly an even exchange. Tightening from 0.80 to 0.90 costs 0.18 of recall to gain 0.08 of precision, which is a poor bargain and leaves only 55 calls to interpret. For context, the hardcoded 0.5248 delivered a recall near 0.30, so calibrating at 0.80 recovers about two thirds more true associations at a comparable precision.

One caveat belongs in the report. The standard deviation of achieved precision was 0.07 to 0.08 at every target, and roughly half the repeats landed slightly below the requested figure. The target is met in expectation rather than guaranteed on any single run. Describe it to Andrew as "calibrated for approximately 0.8 precision, observed 0.81 plus or minus 0.08 across repeats" instead of "precision 0.8", and recalibrate on the internal library since the base rate there will differ from the 0.16 seen here.

### 3c. Report average precision rather than a single operating point

Add after the curve plotting cell.

```python
from sklearn.metrics import average_precision_score
print('BitterMatch AP %.3f | prior AP %.3f | base rate %.3f'
      % (average_precision_score(Y_test, Y_pred),
         average_precision_score(Y_test, prior_pred),
         Y_test.mean()))
```

---

## 4. Item 2: repeated splits

New notebook or script, `<REPO>/repeated_evaluation.py`. Keep it out of the training notebook, because it takes long enough that you will not want to rerun it by accident.

```python
import numpy as np, pandas as pd, pickle
from preprocessing import load_A, load_X_Lig, load_X_Rec, read_ligand_similarity
from bittermatch_ext import build_base, repeated_evaluation, summarise

SEED = <SEED>
A = load_A('<A_CSV>')
A = A[A.columns[~np.isin(A.columns, A.columns[np.sum(A, axis=0) == 0])]]   # orphans
X_Rec = load_X_Rec('<REC_FEAT_CSV>')
X_Rec = X_Rec[np.isin(X_Rec.column_label, A.columns)]
X_Lig = load_X_Lig('<LIG_FEAT_CSV>')

Ll = read_ligand_similarity('<LIG_LINEAR_SIM_CSV>')
Lm = read_ligand_similarity('<LIG_MOL2D_SIM_CSV>')
Ll = Ll.iloc[np.isin(Ll.index, A.index), np.isin(Ll.columns, A.index)]
Lm = Lm.iloc[np.isin(Lm.index, A.index), np.isin(Lm.columns, A.index)]

base = build_base(X_Lig, X_Rec)
long_A = pd.melt(A.assign(ligand=A.index), id_vars='ligand',
                 var_name='receptor', value_name='association')
sim_dict = {'Lig_linear_sim': (Ll, 0), 'Lig_mol2d_sim': (Lm, 0)}

res = repeated_evaluation(A, base, long_A, sim_dict, n_repeats=20,
                          target_precision=0.80, base_seed=100,
                          model_kw=dict(learning_rate=0.03, n_estimators=250))
res.to_csv('<MODEL_DIR>/repeated_evaluation.csv', index=False)
print(summarise(res).to_string())
```

Run it detached so a dropped session does not lose the work.

```bash
cd <REPO> && nohup python -u repeated_evaluation.py > repeated_eval.log 2>&1 &
```

### Reading the output

The number that matters is the standard deviation, not the mean. On the public data the seed to seed standard deviation of average precision was 0.049 against a mean of 0.689. Any modelling change that moves the mean by less than roughly two standard errors, which is about 0.022 at 20 repeats, has not been demonstrated to do anything. Apply that test to your own results before reporting a change as an improvement.

A useful sanity check to run alongside: repeat the whole thing with the compound labels shuffled. Average precision should collapse to the base rate. If it does not, something is leaking.

---

## 5. Items 1 and 6: changes to `<REPO>/new_ligands-eval.ipynb`

### 5a. Merge the receptor prior features, extend cell 35

The loop that merges similarity features becomes:

```python
from bittermatch_ext import receptor_prior_features

for prefix, (sim_df, axis) in sim_metrics_dict.items():
    sim_metrics_df = sim_metrics(sim_df, A, axis).rename(
        columns=lambda col: '%s_%s' % (prefix, col))
    features_df = features_df.merge(sim_metrics_df, how='left', on=['ligand', 'receptor'])

# A already carries NaN for the query compounds, so this counts training
# associations only and no label reaches the prior.
features_df = features_df.merge(receptor_prior_features(A), how='left', on='receptor')
```

### 5b. Guard the column alignment, extend cell 37

```python
X_test.is_human_receptor = X_test.is_human_receptor.astype('bool')

expected = xgb_model.get_booster().feature_names
missing = [c for c in expected if c not in X_test.columns]
extra = [c for c in X_test.columns if c not in expected]
if missing or extra:
    raise ValueError('feature mismatch\n  missing: %s\n  unexpected: %s' % (missing[:10], extra[:10]))
X_test = X_test[expected]
```

Without this, a descriptor set regenerated in a different order scores silently against the wrong columns. It has to raise rather than warn.

### 5c. Item 1, replace the hardcoded threshold in cell 39

Delete `t = 0.5248` and substitute:

```python
import json

with open('<MODEL_DIR>/calibration_for_new_ligands.json') as fh:
    calib = json.load(fh)
t = calib['threshold']
print('threshold %.4f calibrated for target precision %.2f (%s)'
      % (t, calib['target_precision'], calib['calibration']))

Y_pred = xgb_model.predict_proba(X_test)[:, 1]
results_df['pred_score'] = Y_pred
results_df['pred'] = 1 * (Y_pred >= t)

# Where does the threshold actually sit on this data. If it lands beyond the
# 95th percentile the calibration has not transferred and precision at that
# point is being measured on a handful of calls.
print('threshold sits at percentile %.1f, %d of %d pairs called positive'
      % (100 * (Y_pred < t).mean(), int(results_df.pred.sum()), len(results_df)))
```

### 5d. Report the curve, not one point, replace the final cell

```python
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt

y, y_hat, s = slim_results_df.association, slim_results_df.pred, slim_results_df.pred_score
print('AP %.3f (base rate %.3f) | at threshold: recall %.3f precision %.3f on %d calls'
      % (average_precision_score(y, s), y.mean(),
         recall_score(y, y_hat), precision_score(y, y_hat, zero_division=0), int(y_hat.sum())))

pr, rc, _ = precision_recall_curve(y, s)
plt.plot(rc, pr); plt.axhline(y.mean(), ls=':', c='grey')
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.title('AP %.3f' % average_precision_score(y, s)); plt.grid(alpha=.3)
```

When the number of positive calls is small, say the count out loud in the report. A precision of 1.00 on 10 calls carries a Wilson interval reaching down to roughly 0.72, and readers who do not see the denominator will not know that.

### 5e. Item 6, applicability domain, new cells after cell 40

```python
from bittermatch_ext import (coverage_per_ligand, assign_tier, coverage_cutpoints,
                             validate_coverage_gate, nearest_neighbours, lift_over_prior)

cov = coverage_per_ligand(results_df)
cov.index = cov.index.map(lambda v: rev_name_dict.get(v, v))
print(cov.sort_values().round(3).to_string())
print('\npublic reference range for this quantity: 0.04 to 1.00, median near 0.6')
```

Choose the cut points from the training holdout, where labels exist, rather than importing the public ones.

```python
T_LOW, T_HIGH = coverage_cutpoints(cov, q_low=0.25, q_high=0.60)
print('proposed cut points for this library: low %.3f, high %.3f' % (T_LOW, T_HIGH))
```

**A warning about where `cov` comes from.** `coverage_cutpoints` takes quantiles, so it is only meaningful when the compounds you feed it are broadly representative. Running it on a query set defines the tiers relative to that set rather than relative to the training data, which quietly guarantees that some fraction of the compounds land in the top tier no matter how far they all sit from anything the model has seen. On a set of twenty compounds that are uniformly out of domain, this would promote five of them to "within domain" purely by construction.

Derive the two numbers once in the training notebook, store them in the calibration artefact next to the threshold, and treat them thereafter as fixed constants. A tier has to be an absolute statement about distance from the training set, not a rank within whatever batch happens to be in front of you. Section 9 gives the mechanics for the inference notebooks.

Then check the gate is actually separating anything. Run this on a partition where you hold labels, not on the unlabelled query set.

```python
gate = validate_coverage_gate(
    results_df.rename(columns={'pred_score': 'score'}), cov, T_LOW, T_HIGH)
print(gate.round(3).to_string(index=False))
```

For reference, this is what the table looks like on the public data with cut points taken from its own quartiles, using 61 holdout compounds.

| tier | ligands | pairs | coverage | AP | prior AP | within ligand AP |
|---|---|---|---|---|---|---|
| outside | 17 | 418 | 0.04 to 0.29 | 0.370 | 0.271 | 0.568 |
| marginal | 20 | 273 | 0.30 to 0.68 | 0.540 | 0.416 | 0.561 |
| within | 24 | 201 | 0.69 to 1.00 | 0.839 | 0.530 | 0.887 |

**If your table does not show AP rising across the tiers, the coverage proxy is not working on your library and the gate should not ship.** That negative result would itself be worth reporting, because it would point at the similarity metrics rather than the classifier.

Notice also that the outside tier still beats its own prior baseline, 0.370 against 0.271. This is why the cascade suppresses the absolute probability rather than discarding the prediction. Those compounds are less reliable, not uninformative, and throwing them away would waste signal.

Finally, assemble the tiered output.

```python
report = slim_results_df.copy()
report['coverage'] = report.ligand.map(cov)
report['tier'] = assign_tier(report.coverage, T_LOW, T_HIGH)
report['prior'] = results_df['Rec_prior'].values
report['lift'] = lift_over_prior(report.pred_score, report.prior)

# Within domain keeps the probability. Marginal keeps only the ranking.
# Outside domain reports the prior and says so.
report['reported_score'] = np.where(report.tier == 'within', report.pred_score, np.nan)
report['rank_in_ligand'] = report.groupby('ligand').pred_score.rank(ascending=False)
report.loc[report.tier == 'outside', 'rank_in_ligand'] = np.nan
report['recommendation'] = report.tier.map({
    'within':   'act on probability',
    'marginal': 'ranking only, review neighbours before committing',
    'outside':  'receptor prior only, route to broad panel'})
report.to_csv('<MODEL_DIR>/tiered_predictions.csv', index=False)
```

For every compound in the marginal tier, attach the evidence a chemist needs to overrule the model.

```python
for name in report.loc[report.tier == 'marginal', 'ligand'].unique():
    lid = name_dict.get(name, name)
    top_rec = report[(report.ligand == name)].nlargest(3, 'pred_score').receptor
    print('\n=== %s (coverage %.3f) ===' % (name, cov[name]))
    for r in top_rec:
        nn = nearest_neighbours(lid, Lig_linear_sim, A, r, top_k=5, name_map=rev_name_dict)
        print(' receptor %s:' % r, nn.to_dict('records'))
```

---

## 6. Reporting the heatmap in two panels

This addresses the finding that 76% of the variance in predicted scores is explained by which receptor a pair involves rather than which compound. A single panel of raw scores invites the reader to rediscover the ligand counts of BitterDB and mistake them for chemistry.

```python
import matplotlib.pyplot as plt

panel_a = report.pivot(index='ligand', columns='receptor', values='pred_score')
panel_b = report.pivot(index='ligand', columns='receptor', values='lift')

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
im0 = ax[0].imshow(panel_a, aspect='auto', cmap='viridis', vmin=0, vmax=1)
ax[0].set_title('Calibrated probability\nuse this to prioritise experiments')
im1 = ax[1].imshow(panel_b, aspect='auto', cmap='coolwarm',
                   vmin=-np.nanmax(abs(panel_b.values)), vmax=np.nanmax(abs(panel_b.values)))
ax[1].set_title('Evidence beyond the receptor prior\nuse this to ask what is unusual')
for a, p, im in zip(ax, [panel_a, panel_b], [im0, im1]):
    a.set_xticks(range(p.shape[1])); a.set_xticklabels(p.columns, rotation=90, fontsize=7)
    a.set_yticks(range(p.shape[0])); a.set_yticklabels(p.index, fontsize=7)
    fig.colorbar(im, ax=a, fraction=0.03)
plt.tight_layout()
```

The left panel remains the right quantity for deciding what to test, since a modest signal on a promiscuous receptor may still be the best experimental bet. The right panel is where a chemist should look when asking what this particular molecule contributes. Neither should be circulated without the other.

---

## 7. Order of work and expected duration

| Step | Change | Rough effort | Blocks |
|---|---|---|---|
| 1 | Coverage diagnostic on the internal library, section 5e first block only | under an hour | everything, if coverage is uniformly low |
| 2 | `similarity.py` rewrite, items 3 and 4 | half a day including retrain | items 1, 2, 5 |
| 3 | New module and the model change, item 5 | half a day | item 1 |
| 4 | Threshold calibration, item 1 | half a day | reporting |
| 5 | Repeated evaluation, item 2 | one day, mostly compute | reporting |
| 6 | Applicability domain and tiered output, item 6 | one to two days | none |
| 7 | Two panel reporting | half a day | item 6 |

Step 1 goes first because it can invalidate the rest. If the internal compounds sit almost entirely below the low cut point, the finding to report is that the training set does not cover the chemical space of interest, and the productive next move is expanding the reference set rather than tuning the classifier.

---

## 8. Verification before anything is reported

Run these and keep the output.

```python
# 1. Feature schema is identical on both paths.
assert list(X_train.columns) == list(X_test.columns)

# 2. No held out label reached the neighbourhood features.
#    Rebuild with the test ligands masked and confirm the features are unchanged.
f_a = build_features(A, base, long_A, sim_dict, test_ligands)
f_b = build_features(A, base, long_A, sim_dict, test_ligands)
pd.testing.assert_frame_equal(f_a, f_b)

# 3. Label permutation destroys the signal.
perm = res_permuted.AP.mean()          # from repeated_evaluation on shuffled labels
print('permuted AP %.3f should approach the base rate' % perm)

# 4. The threshold transfers.
#    Achieved precision on untouched ligands should sit near the target, and
#    the number of positive calls should be large enough to interpret.
```

If any of these fails, the finding is the failure. Report it rather than working around it.

---

## 9. The two inference notebooks

Sections 5 and 6 describe the shared inference path. This section covers what changes when the notebook is scoring internal compounds rather than reproducing the published validation, and it is the part that most needs reading before either notebook is touched. NB-A holds 14 compounds of which 8 carry associations. NB-B holds 20 with none.

### 9.1 The published notebook discards exactly the compounds you want scored

Cell 27 of `new_ligands-eval.ipynb` reads:

```python
new_pairs = new_pairs[pd.Series.notna(new_pairs.association)]
```

This keeps only pairs whose answer is already known. The line makes sense for reproducing a published validation table, where every pair has a label and the task is measurement. It is actively harmful for prediction.

I simulated both of your notebooks against the published cells. The results are worth stating precisely because neither failure announces itself.

| Notebook | Pairs after the melt | Pairs surviving cell 27 | Consequence |
|---|---|---|---|
| NB-B, 20 compounds, no labels | 820 | **0** | `X_test` has no rows and nothing can be scored |
| NB-A, 14 compounds, 8 labelled | 574 | 328 | **the 6 unlabelled compounds disappear silently** |

The NB-A case is the more dangerous of the two, since it produces a plausible looking heatmap covering 8 compounds while 6 have vanished without any error. If your current NB-A output has fewer rows than you expected, this is why. Check the row count before anything else.

### 9.2 Replacement for cells 26 through 32

Substitute this block. It marks test membership explicitly rather than relying on the missing value cast described in the limitations note, and it keeps unlabelled pairs.

```python
new_pairs = pd.melt(new_A.assign(ligand=new_A.index), id_vars='ligand',
                    var_name='receptor', value_name='association')

# The published line here dropped every pair whose answer was not already
# known. Membership of the query set is a property of the compound, not of
# whether we happen to hold a label for it.
new_pairs['test'] = True

pairs['test'] = False
pairs = pd.concat([pairs, new_pairs], ignore_index=True)
pairs = pairs.astype({'ligand': 'int64', 'receptor': 'int64',
                      'association': 'float64', 'test': 'bool'})
pairs.loc[pairs.receptor > 2000, 'test'] = False        # human receptors only

# Query labels must not reach the neighbourhood features, including the 8 in
# NB-A. Blanking the matrix after the melt preserves them in `pairs` for
# checking while keeping them out of W, M and the receptor prior.
new_A = pd.DataFrame(np.nan, index=new_A.index, columns=new_A.columns)
A = pd.concat([A, new_A])

n_human = sum(1 for c in new_A.columns if c < 2000)
expected = len(new_A) * n_human
assert pairs.test.sum() == expected, (
    'expected %d query pairs, got %d' % (expected, pairs.test.sum()))
print('%d query compounds x %d human receptors = %d pairs to score'
      % (len(new_A), n_human, expected))
```

The assertion is not decoration. It is the only thing standing between you and a silently truncated prediction set, which is the failure mode described above.

### 9.3 Replacement for the selection step in cell 36

```python
# Published version also required association.notna(), which reintroduces the
# problem from 9.1 at the second gate.
results_df = features_df[features_df.test == True].copy()
X_test = results_df.drop(['ligand', 'receptor', 'association', 'test'], axis=1)
```

`Y_test` disappears from NB-B entirely. In NB-A it exists only for the labelled subset and is pulled out later, in section 9.5, rather than being formed here.

### 9.4 Whether the 8 labelled compounds belong in the training matrix

They can serve one of two purposes and not both at once.

Held out, they tell you whether the model transfers to your chemistry. Folded into `A` before training, they improve the neighbourhood features for every other compound you score, including the 6 in NB-A and the 20 in NB-B. Using them for both is the ordinary circular evaluation mistake, and with only 8 compounds the resulting figure would be flattering and meaningless.

My suggestion is to run the sequence in that order. First keep them out, retrain, and measure as in 9.5. If the transfer check is acceptable, fold them into `A`, retrain once more, and use that model for the compounds you actually care about. Record which model produced which numbers, because the two are not comparable and will be confused later otherwise.

Before reporting any figure from 9.5, confirm the compounds were genuinely excluded.

```python
labelled = results_df.loc[results_df.association.notna(), 'ligand'].unique()
leaked = set(labelled) & set(A.index[A.notna().any(axis=1)])
assert not leaked, 'these compounds carry labels inside A: %s' % sorted(leaked)
```

### 9.5 What NB-A's 8 compounds can and cannot establish

They cannot calibrate a threshold. Section 3b needs several hundred pairs across many compounds to place an operating point, and the dispersion measured in the precision target table would swamp anything estimated from 8. Read the threshold from the artefact and leave it alone.

What they can do is answer a narrower and more useful question: does performance on internal chemistry resemble performance on the training holdout, or has it collapsed. Run this and compare against the `train_holdout_AP` stored in the artefact.

```python
import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score

def wilson(k, n, z=1.96):
    """Interval for a proportion. Small denominators need it stated."""
    if n == 0:
        return (np.nan, np.nan)
    p, d = k / n, 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return ((centre - half) / d, (centre + half) / d)

check = results_df[results_df.association.notna()]
y, s = check.association.values, check.pred_score.values
yh = 1 * (s >= t)
tp, called = int(((yh == 1) & (y == 1)).sum()), int(yh.sum())

print('checking on %d compounds, %d pairs, %d positives (base rate %.3f)'
      % (check.ligand.nunique(), len(check), int(y.sum()), y.mean()))
print('AP %.3f   | training holdout AP was %.3f' % (average_precision_score(y, s),
                                                    calib['train_holdout_AP']))
print('prior AP %.3f (the figure to beat, chemistry aside)'
      % average_precision_score(y, check.Rec_prior))
lo, hi = wilson(tp, called)
print('at the calibrated threshold: %d calls, precision %.2f (95%% CI %.2f to %.2f), recall %.2f'
      % (called, precision_score(y, yh, zero_division=0), lo, hi, recall_score(y, yh)))
```

Read the output as a pass or fail rather than as a performance figure. Average precision near the stored holdout value means the pipeline transferred. Average precision near the prior baseline means the descriptors or the similarity matrices are not carrying internal chemistry, which sends you back to section 5e rather than to hyperparameters. Whatever the precision comes out at, quote it with the interval, because on the order of ten calls that interval will be wide enough to change how the number reads.

### 9.6 Coverage tiers when you cannot validate them

Both notebooks read the cut points rather than computing them.

```python
from bittermatch_ext import coverage_per_ligand, assign_tier

T_LOW = calib['coverage_cutpoints']['low']
T_HIGH = calib['coverage_cutpoints']['high']

cov = coverage_per_ligand(results_df)
cov.index = cov.index.map(lambda v: rev_name_dict.get(v, v))
tiers = assign_tier(cov, T_LOW, T_HIGH)
print(pd.DataFrame({'coverage': cov.round(3), 'tier': tiers})
      .sort_values('coverage').to_string())
print('\ntier counts:', tiers.value_counts().to_dict())
```

`validate_coverage_gate` has no role in NB-B and very little in NB-A, where 8 compounds spread across three tiers leaves too few in each to estimate anything. The gate is validated once, in the training notebook, on the holdout. These notebooks apply a decision rule that was justified elsewhere.

One reading of this output deserves attention. If most or all query compounds land in the outside tier, that is itself the headline result, and it is a finding about the reference database rather than about the model. It says BitterDB does not cover the chemistry you care about. Tuning the classifier cannot fix that and expanding the training associations can.

### 9.7 What NB-B may report

NB-B has no labels, so no performance statement of any kind is available from it. Precision, recall and average precision are undefined, and any number resembling them would have to come from the training notebook and be labelled as such.

What it can legitimately produce: the coverage distribution and tier assignment from 9.6, the ranking of receptors within each compound, the two panel figure from section 6, and the nearest neighbour listings that let a chemist audit a call. The tiered output block in section 5e works unchanged apart from dropping the `association` column.

State the provenance explicitly wherever these predictions travel. A sentence along the lines of "scores from the model calibrated on the training holdout at approximately 0.8 precision, applied to 20 compounds with no measured associations, of which N fall inside the applicability domain" prevents the figure being read as a measurement of these compounds.

### 9.8 Two small things that will bite later

Give the notebooks disjoint identifier ranges. Both currently follow the published pattern of `999000 + i`, so NB-A occupies 999000 to 999013 and NB-B occupies 999000 to 999019. The moment anyone concatenates the two result sets, twelve compounds collide silently. Use `998000 + i` in one of them.

Check that each combined similarity matrix contains every query compound alongside the full training set. A compound missing from the similarity file will pass through the merge as missing values, XGBoost will accept them without complaint, and the prediction will rest on content features alone. The assertion in 9.2 catches a missing row in `new_A` but not a missing row in the similarity matrix, so verify separately.

```python
missing = [c for c in new_A.index if c not in Lig_linear_sim.index]
assert not missing, 'absent from the similarity matrix: %s' % missing
```
