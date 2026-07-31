"""Audited replacements for the published BitterMatch pipeline.

Covers audit items 1 to 6. Nothing in the original repository is edited: the
corrected `sim_metrics` lives here and supersedes the one in similarity.py, so
your local changes to that file stay untouched.

    item 1  calibrate_threshold, oof_predictions
    item 2  repeated_evaluation, summarise
    item 3  sim_metrics normalises W, receptor_prior_features restores the prior
    item 4  sim_metrics labels its columns correctly
    item 5  make_model
    item 6  coverage_per_ligand, assign_tier, validate_coverage_gate
"""
import os
import importlib

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             precision_score, recall_score)

from preprocessing import load_A, load_X_Lig, load_X_Rec, read_ligand_similarity

ID_COLS = ['ligand', 'receptor', 'association']


def get_config():
    """Import the config module named by BM_CONFIG, defaulting to bm_config."""
    return importlib.import_module(os.environ.get('BM_CONFIG', 'bm_config'))


# --------------------------------------------------------------------------
# Items 3 and 4: neighbour informed features
# --------------------------------------------------------------------------
def sim_metrics(S, A, axis, normalise=True):
    """Neighbour informed features, with two corrections.

    W is an average rather than a sum. Paper equations 5 and 7 specify sums,
    which makes the feature proportional to how many ligands a receptor has on
    record. Measured Spearman correlation against that count was 0.97 before
    the change and 0.06 after.

    Column labels match the arrays they hold. The published version assembled
    W1, W0, M1, M0 and then labelled them W0, W1, M1, M0.
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

    # The diagonal of `ones` must stay 1. Zeroing it to match S_vals would make
    # n1 equal to (total activators minus this pair's own label), which encodes
    # the label directly into the feature. Verified: that offset correlates
    # with the label at exactly -1.0.
    ones = np.ones_like(S_vals)

    W1, W0 = S_vals.dot(pos), S_vals.dot(neg)
    n1, n0 = ones.dot(pos), ones.dot(neg)

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
        names = ['W1', 'W0', 'M1', 'M0']

    return pd.DataFrame(
        vals,
        index=pd.MultiIndex.from_product([rows, cols], names=['ligand', 'receptor']),
        columns=names)


def receptor_prior_features(masked_A):
    """Per receptor base rate from KNOWN associations only.

    Give this the same masked matrix that feeds sim_metrics, otherwise held out
    labels leak in through the denominator.
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
# Loading
# --------------------------------------------------------------------------
def load_training_inputs(cfg):
    """Association matrix, features and similarity matrices, orphans removed."""
    A = load_A(cfg.A_CSV)
    orphans = set(A.columns[np.nansum(A.values, axis=0) == 0])
    A = A[[c for c in A.columns if c not in orphans]]

    X_Rec = load_X_Rec(cfg.REC_FEAT_CSV)
    X_Rec = X_Rec[X_Rec.column_label.isin(A.columns)].reset_index(drop=True)

    X_Lig = load_X_Lig(cfg.LIG_FEAT_CSV)

    Ll = read_ligand_similarity(cfg.LIG_LINEAR_SIM_CSV)
    Lm = read_ligand_similarity(cfg.LIG_MOL2D_SIM_CSV)
    Ll = Ll.iloc[np.isin(Ll.index, A.index), np.isin(Ll.columns, A.index)]
    Lm = Lm.iloc[np.isin(Lm.index, A.index), np.isin(Lm.columns, A.index)]

    missing = [i for i in A.index if i not in Ll.index]
    if missing:
        raise ValueError('%d training compounds absent from the linear similarity '
                         'matrix, first few: %s' % (len(missing), missing[:5]))

    print('training inputs: A %s, %d receptors after removing %d orphans, '
          'X_Lig %s, X_Rec %s' % (A.shape, A.shape[1], len(orphans),
                                  X_Lig.shape, X_Rec.shape))
    return A, X_Rec, X_Lig, {'Lig_linear_sim': (Ll, 0), 'Lig_mol2d_sim': (Lm, 0)}


def load_query_inputs(cfg, A):
    """Query compounds, their descriptors, and the combined similarity matrices.

    Returns new_A (all NaN where no label is held), the query descriptors with
    integer identifiers, the rebuilt similarity matrices, and the name maps.
    """
    q_X_Lig = load_X_Lig(cfg.QUERY_LIG_FEAT_CSV)
    query_names = list(q_X_Lig.cid.values)

    assoc_path = getattr(cfg, 'QUERY_ASSOC_CSV', None)
    if isinstance(assoc_path, str) and assoc_path.startswith('<'):
        assoc_path = None

    if assoc_path:
        q_A = load_A(assoc_path)
        unknown = [n for n in q_A.index if n not in query_names]
        if unknown:
            raise ValueError('compounds in the association file but not in the '
                             'descriptor file: %s' % unknown)
    else:
        q_A = None

    name_dict = {n: cfg.QUERY_ID_OFFSET + i for i, n in enumerate(query_names)}
    rev_name_dict = {v: k for k, v in name_dict.items()}

    new_A = pd.DataFrame(np.nan, index=[name_dict[n] for n in query_names],
                         columns=A.columns, dtype=float)
    if q_A is not None:
        for name in q_A.index:
            for col in q_A.columns:
                c = int(col)
                if c in new_A.columns:
                    new_A.loc[name_dict[name], c] = q_A.loc[name, col]

    q_X_Lig = q_X_Lig.replace({'cid': name_dict})

    sims = {}
    for key, path in [('Lig_linear_sim', cfg.QUERY_LINEAR_SIM_CSV),
                      ('Lig_mol2d_sim', cfg.QUERY_MOL2D_SIM_CSV)]:
        df = pd.read_csv(path, sep=',')
        raw = df[df.columns[0]].values
        ids = []
        for v in raw:
            key_v = v if v in name_dict else str(v)
            ids.append(int(name_dict[key_v]) if key_v in name_dict else int(v))
        df = df.drop(columns=df.columns[0])
        df.index = np.array(ids, dtype='int64')
        df.columns = np.array(ids, dtype='int64')
        sim = read_ligand_similarity(df, from_file=False)
        absent = [i for i in new_A.index if i not in sim.index]
        if absent:
            raise ValueError('%s: query compounds absent from the similarity '
                             'matrix: %s' % (key, [rev_name_dict[a] for a in absent]))
        sims[key] = sim

    n_lab = 0 if q_A is None else int(new_A.notna().any(axis=1).sum())
    print('query set "%s": %d compounds, %d carry associations'
          % (cfg.QUERY_NAME, len(new_A), n_lab))
    return new_A, q_X_Lig, sims, name_dict, rev_name_dict


# --------------------------------------------------------------------------
# Feature assembly
# --------------------------------------------------------------------------
def build_base(X_Lig, X_Rec):
    """Cross join of ligand and receptor content features."""
    base = pd.merge(
        X_Lig.rename(columns=lambda c: 'Lig_%s' % c).assign(key_=1),
        X_Rec.rename(columns=lambda c: 'Rec_%s' % c).assign(key_=1),
        on='key_').drop(columns='key_')
    base = base.rename(columns={'Lig_cid': 'ligand', 'Rec_column_label': 'receptor'})
    base['is_human_receptor'] = base.receptor < 2000
    return base


def build_features(A, base, long_A, sim_dict, mask_ligands, keep_unknown=False):
    """Design matrix with `mask_ligands` held out of every derived feature.

    Masking happens before any neighbourhood feature is computed, which is the
    only way to keep W, M and the receptor prior free of the held out labels.

    keep_unknown=True retains pairs with no label. Prediction needs this. The
    published notebook dropped them, which silently discarded every compound
    you actually wanted scored.
    """
    masked_A = A.copy()
    masked_A.loc[masked_A.index.isin(mask_ligands), :] = np.nan

    f = base.copy()
    for prefix, (S, axis) in sim_dict.items():
        block = sim_metrics(S, masked_A, axis).rename(
            columns=lambda c: '%s_%s' % (prefix, c))
        f = f.merge(block, how='left', on=['ligand', 'receptor'])
    f = f.merge(receptor_prior_features(masked_A), how='left', on='receptor')
    f = f.merge(long_A, how='left', on=['ligand', 'receptor'])
    return f if keep_unknown else f[f.association.notna()]


def long_form(A):
    return pd.melt(A.assign(ligand=A.index), id_vars='ligand',
                   var_name='receptor', value_name='association')


# --------------------------------------------------------------------------
# Item 5
# --------------------------------------------------------------------------
def make_model(seed, learning_rate=0.03, n_estimators=250, early_stopping_rounds=None):
    """Learning rate 0.001 over 1000 rounds confined predictions to 0.09 to 0.65.

    Cross validated average precision was flat from 0.001 to 0.1, so this is a
    score resolution change rather than an accuracy claim.
    """
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
# Item 1
# --------------------------------------------------------------------------
def calibrate_threshold(y, p, target_precision):
    """Highest recall point on the PR curve that still meets the target.

    Feed this OUT OF FOLD scores. In sample scores are optimistic and produce a
    threshold that collapses the moment it meets new compounds.
    """
    pr, rc, th = precision_recall_curve(y, p)
    ok = np.where(pr[:-1] >= target_precision)[0]
    if len(ok) == 0:
        return float(np.quantile(p, 0.99)), {
            'met_target': False, 'note': 'target unreachable, used 99th percentile'}
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
    for k, val in enumerate(folds):
        f = build_features(A, base, long_A, sim_dict,
                           np.concatenate([holdout_ligands, val]))
        tr = f[f.ligand.isin(np.setdiff1d(train_ligands, val))]
        va = f[f.ligand.isin(val)]
        m = make_model(seed, **model_kw).fit(tr.drop(ID_COLS, axis=1),
                                             tr.association.values)
        ps.append(m.predict_proba(va.drop(ID_COLS, axis=1))[:, 1])
        ys.append(va.association.values)
        print('  calibration fold %d/%d' % (k + 1, len(folds)), flush=True)
    return np.concatenate(ys), np.concatenate(ps)


def wilson(k, n, z=1.96):
    """Interval for a proportion. Small denominators need it stated."""
    if n == 0:
        return (float('nan'), float('nan'))
    p, d = k / n, 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return ((centre - half) / d, (centre + half) / d)


# --------------------------------------------------------------------------
# Item 2
# --------------------------------------------------------------------------
def repeated_evaluation(A, base, long_A, sim_dict, n_repeats=20, train_frac=0.8,
                        base_seed=100, target_precision=0.80, model_kw=None,
                        calibrate=True, n_folds=4, verbose=True):
    """Repeat the whole protocol, calibration included, and report dispersion."""
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
                                     seed, n_folds=n_folds, model_kw=model_kw)
            thr, _ = calibrate_threshold(oy, op, target_precision)

        f = build_features(A, base, long_A, sim_dict, test_lig)
        tr, te = f[f.ligand.isin(train_lig)], f[f.ligand.isin(test_lig)]
        m = make_model(seed, **model_kw).fit(tr.drop(ID_COLS, axis=1),
                                             tr.association.values)
        p = m.predict_proba(te.drop(ID_COLS, axis=1))[:, 1]
        y = te.association.values

        row = {'repeat': rep, 'seed': seed, 'AP': average_precision_score(y, p),
               'prior_AP': average_precision_score(y, te.Rec_prior.values),
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
    num = df.select_dtypes(include=[np.number]).drop(
        columns=['repeat', 'seed'], errors='ignore')
    return pd.DataFrame({'mean': num.mean(), 'sd': num.std(),
                         'min': num.min(), 'max': num.max()}).round(4)


# --------------------------------------------------------------------------
# Item 6
# --------------------------------------------------------------------------
def coverage_per_ligand(features_df, sim_prefixes=('Lig_linear_sim', 'Lig_mol2d_sim')):
    """Strongest resemblance between each compound and any known activator.

    The max across similarity views is deliberate. A compound needs only one
    view to be recognised, and requiring agreement would discard compounds that
    a single fingerprint happens to describe well.
    """
    cols = [c for c in ('%s_M1' % p for p in sim_prefixes) if c in features_df.columns]
    if not cols:
        raise KeyError('no M1 columns found among %s'
                       % [c for c in features_df.columns if c.endswith('M1')])
    return features_df.groupby('ligand')[cols].max().max(axis=1).rename('coverage')


def coverage_cutpoints(coverage, q_low=0.25, q_high=0.60):
    """Propose cut points. Only meaningful on a representative labelled set.

    Run this in training, store the two numbers, and treat them as constants
    afterwards. Taking quantiles of a query set defines the tiers relative to
    that batch, which guarantees a top tier however far the whole batch sits
    from anything the model has seen.
    """
    return float(coverage.quantile(q_low)), float(coverage.quantile(q_high))


def assign_tier(coverage, t_low, t_high):
    return pd.cut(coverage, bins=[-np.inf, t_low, t_high, np.inf],
                  labels=['outside', 'marginal', 'within'])


def validate_coverage_gate(results_df, coverage, t_low, t_high):
    """Evidence that the gate separates reliable from unreliable predictions.

    If average precision does not rise across the tiers, the coverage proxy is
    not working and the gate should not ship.
    """
    d = results_df.merge(coverage, left_on='ligand', right_index=True)
    d['tier'] = assign_tier(d.coverage, t_low, t_high)
    rows = []
    for tier, g in d.groupby('tier', observed=True):
        if g.association.nunique() < 2:
            continue
        per_lig = [average_precision_score(x.association, x.score)
                   for _, x in g.groupby('ligand')
                   if 0 < x.association.sum() < len(x)]
        rows.append({'tier': tier, 'n_ligands': g.ligand.nunique(), 'n_pairs': len(g),
                     'coverage_min': round(g.coverage.min(), 3),
                     'coverage_max': round(g.coverage.max(), 3),
                     'AP': average_precision_score(g.association, g.score),
                     'prior_AP': (average_precision_score(g.association, g.Rec_prior)
                                  if 'Rec_prior' in g else np.nan),
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


def lift_over_prior(score, prior, eps=1e-6):
    """Log odds of the prediction minus log odds of the receptor base rate.

    Zero means the model learned nothing about this compound beyond the fact
    that the receptor responds often. Positive means the chemistry argued for
    the pair, negative means it argued against.
    """
    s = np.clip(np.asarray(score, dtype=float), eps, 1 - eps)
    q = np.clip(np.asarray(prior, dtype=float), eps, 1 - eps)
    return np.log(s / (1 - s)) - np.log(q / (1 - q))
