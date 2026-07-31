"""Score a query set of compounds with the calibrated model.

Audit items 1 and 6 on the inference side, plus the two panel figure.

Works whether the query compounds carry associations, carry some, or carry
none. Set QUERY_ASSOC_CSV to None in the config for a set with no labels.

    BM_CONFIG=bm_config_nba python bm_predict.py
    BM_CONFIG=bm_config_nbb python bm_predict.py

Writes into OUT_DIR:
    <QUERY_NAME>_predictions.csv     one row per pair, the table view
    <QUERY_NAME>_heatmap.png         two panels, probability and lift
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_score, recall_score

from bittermatch_ext import (get_config, load_training_inputs, load_query_inputs,
                             build_base, build_features, long_form,
                             coverage_per_ligand, assign_tier, nearest_neighbours,
                             lift_over_prior, wilson, ID_COLS)

cfg = get_config()
os.makedirs(cfg.OUT_DIR, exist_ok=True)

print('=' * 74)
print('1. loading model, calibration and data')
print('=' * 74)
with open(os.path.join(cfg.MODEL_DIR, 'bittermatch_model.pkl'), 'rb') as fh:
    model = pickle.load(fh)
with open(os.path.join(cfg.MODEL_DIR, 'bittermatch_calibration.json')) as fh:
    calib = json.load(fh)
t = calib['threshold']
T_LOW = calib['coverage_cutpoints']['low']
T_HIGH = calib['coverage_cutpoints']['high']
print('threshold %.4f, calibrated for target precision %.2f' % (t, calib['target_precision']))
print('coverage cut points %.3f / %.3f, gate validated in training: %s'
      % (T_LOW, T_HIGH, calib['gate_validated']))

A, X_Rec, X_Lig, _ = load_training_inputs(cfg)
new_A, q_X_Lig, sim_dict, name_dict, rev_name_dict = load_query_inputs(cfg, A)

print('\n' + '=' * 74)
print('2. assembling the query set')
print('=' * 74)

# Labels, where they exist, travel in the long form only. The matrix that feeds
# the neighbourhood features is blank for every query compound, so nothing the
# model is asked to predict has contributed to its own features.
long_all = pd.concat([long_form(A), long_form(new_A)], ignore_index=True)
A_full = pd.concat([A, pd.DataFrame(np.nan, index=new_A.index, columns=new_A.columns)])

# The combined matrices cover training and query compounds together. Restrict
# them to the rows the design matrix will actually use.
sim_dict = {k: (S.iloc[np.isin(S.index, A_full.index), np.isin(S.columns, A_full.index)], 0)
            for k, S in sim_dict.items()}

X_Lig_all = pd.concat([X_Lig, q_X_Lig], ignore_index=True)
base = build_base(X_Lig_all, X_Rec)

query_ids = list(new_A.index)
n_human = sum(1 for c in new_A.columns if c < 2000)
expected = len(query_ids) * n_human

f = build_features(A_full, base, long_all, sim_dict, query_ids, keep_unknown=True)
results = f[f.ligand.isin(query_ids) & (f.receptor < 2000)].copy()

# The published notebook filtered on association.notna() here, which discarded
# every pair whose answer was not already known. On an unlabelled set that
# leaves nothing at all, and on a partly labelled set it drops the unlabelled
# compounds without raising anything.
assert len(results) == expected, (
    'expected %d query pairs (%d compounds x %d human receptors), got %d'
    % (expected, len(query_ids), n_human, len(results)))
print('%d compounds x %d human receptors = %d pairs to score'
      % (len(query_ids), n_human, expected))

print('\n' + '=' * 74)
print('3. scoring')
print('=' * 74)
X = results.drop(ID_COLS, axis=1)
expected_cols = model.get_booster().feature_names
missing = [c for c in expected_cols if c not in X.columns]
extra = [c for c in X.columns if c not in expected_cols]
if missing or extra:
    raise ValueError('feature mismatch against the trained model\n'
                     '  missing: %s\n  unexpected: %s' % (missing[:10], extra[:10]))
X = X[expected_cols]

results['score'] = model.predict_proba(X)[:, 1]
results['pred'] = 1 * (results.score >= t)
results['lift'] = lift_over_prior(results.score, results.Rec_prior)
print('score range [%.3f, %.3f], %d of %d pairs called positive, threshold at '
      'percentile %.1f' % (results.score.min(), results.score.max(),
                           int(results.pred.sum()), len(results),
                           100 * (results.score < t).mean()))
if (results.score < t).mean() > 0.95:
    print('NOTE: the threshold lands beyond the 95th percentile here, so very '
          'few calls are being made. Precision measured at this point would '
          'rest on a handful of pairs.')

print('\n' + '=' * 74)
print('4. applicability domain, item 6')
print('=' * 74)
cov = coverage_per_ligand(results)
tiers = assign_tier(cov, T_LOW, T_HIGH)
results['coverage'] = results.ligand.map(cov)
results['tier'] = results.ligand.map(tiers).astype(str)

tbl = pd.DataFrame({'compound': [rev_name_dict.get(i, i) for i in cov.index],
                    'coverage': cov.round(3).values,
                    'tier': tiers.astype(str).values}).sort_values('coverage')
print(tbl.to_string(index=False))
counts = tbl.tier.value_counts().to_dict()
print('\ntier counts: %s' % counts)
if counts.get('outside', 0) >= 0.5 * len(tbl):
    print('NOTE: at least half the query set sits outside the applicability '
          'domain. That is the headline result, and it is a statement about '
          'the reference database rather than about the model. Expanding the '
          'training associations helps here, tuning the classifier does not.')

results['reported_score'] = np.where(results.tier == 'within', results.score, np.nan)
results['rank_in_compound'] = results.groupby('ligand').score.rank(ascending=False)
results.loc[results.tier == 'outside', 'rank_in_compound'] = np.nan
results['recommendation'] = results.tier.map({
    'within': 'act on probability',
    'marginal': 'ranking only, review neighbours before committing',
    'outside': 'receptor prior only, route to broad panel'})

# ---------------------------------------------------------------------------
# 5. Transfer check, only where labels exist. This is not a calibration step.
# ---------------------------------------------------------------------------
check = results[results.association.notna()]
if len(check) and check.association.nunique() > 1:
    print('\n' + '=' * 74)
    print('5. transfer check on the labelled subset')
    print('=' * 74)
    y, s = check.association.values, check.score.values
    yh = 1 * (s >= t)
    tp, called = int(((yh == 1) & (y == 1)).sum()), int(yh.sum())
    lo, hi = wilson(tp, called)
    print('%d compounds, %d pairs, %d positives (base rate %.3f)'
          % (check.ligand.nunique(), len(check), int(y.sum()), y.mean()))
    print('AP %.3f | training holdout AP %.3f | receptor prior AP %.3f'
          % (average_precision_score(y, s), calib['train_holdout_AP'],
             average_precision_score(y, check.Rec_prior)))
    print('%d calls, precision %.2f (95%% CI %.2f to %.2f), recall %.2f'
          % (called, precision_score(y, yh, zero_division=0), lo, hi,
             recall_score(y, yh)))
    print('\nRead this as pass or fail, not as a performance figure. Near the '
          'training holdout means the pipeline transferred. Near the prior '
          'means the descriptors are not carrying your chemistry.')
else:
    print('\n(no labelled pairs in this query set, so no performance figure of '
          'any kind can be computed here)')

# ---------------------------------------------------------------------------
# 6. Nearest neighbour evidence for the marginal tier
# ---------------------------------------------------------------------------
marginal = [i for i in cov.index if tiers[i] == 'marginal']
if marginal:
    print('\n' + '=' * 74)
    print('6. nearest neighbours for marginal compounds')
    print('=' * 74)
    S_lin = sim_dict['Lig_linear_sim'][0]
    for lid in marginal:
        top = results[results.ligand == lid].nlargest(2, 'score')
        print('\n%s (coverage %.3f)' % (rev_name_dict.get(lid, lid), cov[lid]))
        for _, r in top.iterrows():
            nn = nearest_neighbours(lid, S_lin, A, int(r.receptor), top_k=3,
                                    name_map=rev_name_dict)
            print('  receptor %s, score %.3f -> %s'
                  % (int(r.receptor), r.score,
                     ', '.join('%s (sim %.2f, assoc %g)'
                               % (n.neighbour, n.similarity, n.association)
                               for n in nn.itertuples())))

# ---------------------------------------------------------------------------
# 7. Outputs
# ---------------------------------------------------------------------------
print('\n' + '=' * 74)
print('7. writing outputs')
print('=' * 74)
out = results[['ligand', 'receptor', 'association', 'score', 'reported_score',
               'pred', 'lift', 'Rec_prior', 'coverage', 'tier',
               'rank_in_compound', 'recommendation']].copy()
out['compound'] = out.ligand.map(lambda v: rev_name_dict.get(v, v))
csv_path = os.path.join(cfg.OUT_DIR, '%s_predictions.csv' % cfg.QUERY_NAME)
out.to_csv(csv_path, index=False)
print('table view -> %s' % csv_path)

# Two panels. The left one is the quantity for deciding what to test. The right
# one isolates what the chemistry contributed once the receptor base rate is
# taken out. Circulating either alone invites a misreading.
panel_a = out.pivot(index='compound', columns='receptor', values='score')
panel_b = out.pivot(index='compound', columns='receptor', values='lift')
order = cov.sort_values(ascending=False).index.map(lambda v: rev_name_dict.get(v, v))
panel_a, panel_b = panel_a.reindex(order), panel_b.reindex(order)
tier_of = {rev_name_dict.get(i, i): str(tiers[i]) for i in cov.index}
ylabels = ['%s  [%s]' % (n, tier_of[n]) for n in panel_a.index]

INK, MUTED, SURFACE = '#1f2328', '#6b7280', '#ffffff'
h = max(3.2, 0.30 * len(panel_a) + 1.9)
fig, axes = plt.subplots(1, 2, figsize=(15, h), constrained_layout=True)

lim = float(np.nanmax(np.abs(panel_b.values))) or 1.0
specs = [(panel_a, 'Blues', dict(vmin=0, vmax=1),
          'Calibrated probability', 'what to prioritise for testing'),
         (panel_b, 'RdBu_r', dict(vmin=-lim, vmax=lim),
          'Evidence beyond the receptor prior', 'what is unusual about this molecule')]

for ax, (panel, cmap, norm, title, sub) in zip(axes, specs):
    im = ax.imshow(panel.values, aspect='auto', cmap=cmap, **norm)
    ax.set_title('%s\n%s' % (title, sub), fontsize=10, color=INK, pad=8, loc='left')
    ax.set_xticks(range(panel.shape[1]))
    ax.set_xticklabels(panel.columns, rotation=90, fontsize=7, color=MUTED)
    ax.set_xlabel('TAS2R', fontsize=8, color=MUTED, labelpad=4)
    ax.set_yticks(range(panel.shape[0]))
    ax.set_yticklabels(ylabels, fontsize=7, color=MUTED)
    # A surface gap between cells rather than a border drawn around them.
    ax.set_xticks(np.arange(-.5, panel.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, panel.shape[0], 1), minor=True)
    ax.grid(which='minor', color=SURFACE, linewidth=1.4)
    ax.tick_params(which='both', length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cb.ax.tick_params(labelsize=7, length=0, colors=MUTED)
    cb.outline.set_visible(False)

fig.suptitle('%s  |  threshold %.3f calibrated for ~%.0f%% precision  |  '
             'tiers from the training holdout'
             % (cfg.QUERY_NAME, t, 100 * calib['target_precision']),
             fontsize=9, color=MUTED, y=1.02)
png_path = os.path.join(cfg.OUT_DIR, '%s_heatmap.png' % cfg.QUERY_NAME)
fig.savefig(png_path, dpi=170, bbox_inches='tight', facecolor='white')
print('figure     -> %s' % png_path)

print('\nProvenance line for any report these travel in:')
print('  scores from the model calibrated on the training holdout at '
      'approximately %.0f%% precision (observed dispersion around 0.08 across '
      'repeats), applied to %d compounds%s, of which %d fall inside the '
      'applicability domain.'
      % (100 * calib['target_precision'], len(query_ids),
         '' if len(check) else ' with no measured associations',
         counts.get('within', 0)))
