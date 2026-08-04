"""Fit BitterMatch, calibrate the decision threshold, fix the coverage tiers.

Audit items 1, 3, 4, 5 and the training half of item 6.

PowerShell:
    $env:BM_CONFIG = "bm_config"; python bm_train.py

bash:
    BM_CONFIG=bm_config python bm_train.py

Writes into MODEL_DIR:
    bittermatch_model.pkl                the fitted classifier
    bittermatch_calibration.json         threshold, cut points, reference figures
"""
import json
import os
import pickle

import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score

from bittermatch_ext import (get_config, model_kw, load_training_inputs,
                             build_base, build_features, long_form, make_model,
                             oof_predictions, calibrate_threshold,
                             coverage_per_ligand, coverage_cutpoints,
                             validate_coverage_gate, wilson, ID_COLS)

cfg = get_config()
os.makedirs(cfg.MODEL_DIR, exist_ok=True)
os.makedirs(cfg.OUT_DIR, exist_ok=True)
MODEL_KW = model_kw(cfg)

print('=' * 74)
print('1. loading')
print('=' * 74)
A, X_Rec, X_Lig, sim_dict = load_training_inputs(cfg)
base = build_base(X_Lig, X_Rec)
long_A = long_form(A)

# Ligand grouped split. Every association of a held out compound is hidden,
# which is what makes this a "new ligands" test rather than a gap filling one.
rng = np.random.RandomState(cfg.SEED)
all_lig = np.unique(long_A.ligand)
train_lig = rng.choice(all_lig, int(cfg.TRAIN_FRACTION * len(all_lig)), replace=False)
test_lig = np.setdiff1d(all_lig, train_lig)
print('%d compounds: %d train, %d holdout' % (len(all_lig), len(train_lig), len(test_lig)))

print('\n' + '=' * 74)
print('2. features and fit')
print('=' * 74)
f = build_features(A, base, long_A, sim_dict, test_lig)
tr = f[f.ligand.isin(train_lig)]
te = f[f.ligand.isin(test_lig)]
print('design matrix %s, train %d pairs, holdout %d pairs' % (f.shape, len(tr), len(te)))

model = make_model(cfg.SEED, **MODEL_KW).fit(tr.drop(ID_COLS, axis=1),
                                             tr.association.values)
p = model.predict_proba(te.drop(ID_COLS, axis=1))[:, 1]
y = te.association.values

holdout_AP = float(average_precision_score(y, p))
prior_AP = float(average_precision_score(y, te.Rec_prior.values))
print('holdout AP %.3f | receptor prior AP %.3f | base rate %.3f'
      % (holdout_AP, prior_AP, y.mean()))

# Training keeps the murine receptors, following the published notebook, but
# inference scores human receptors only. The mixed figure is therefore not the
# right reference for a query set, so store the human subset separately.
human = (te.receptor < 2000).values
holdout_AP_human = float(average_precision_score(y[human], p[human]))
print('holdout AP over human receptors only %.3f (%d of %d pairs) <- the figure '
      'bm_predict.py compares against' % (holdout_AP_human, human.sum(), len(y)))
print('score range [%.3f, %.3f]  (the published settings capped this near 0.65)'
      % (p.min(), p.max()))
if holdout_AP <= prior_AP:
    print('WARNING: the model does not beat a receptor base rate. Chemistry is '
          'contributing nothing on this data. Investigate before going further.')

print('\n' + '=' * 74)
print('3. threshold calibration, item 1')
print('=' * 74)
oof_y, oof_p = oof_predictions(A, base, long_A, sim_dict, train_lig, test_lig,
                               seed=cfg.SEED, n_folds=cfg.N_CALIB_FOLDS,
                               model_kw=MODEL_KW)
threshold, info = calibrate_threshold(oof_y, oof_p, cfg.TARGET_PRECISION)
print('threshold %.4f  %s' % (threshold, info))

yh = 1 * (p >= threshold)
tp, called = int(((yh == 1) & (y == 1)).sum()), int(yh.sum())
lo, hi = wilson(tp, called)
print('applied to the holdout: %d calls, precision %.3f (95%% CI %.2f to %.2f), '
      'recall %.3f' % (called, precision_score(y, yh, zero_division=0), lo, hi,
                       recall_score(y, yh)))
print('the threshold sits at percentile %.1f of the holdout scores'
      % (100 * (p < threshold).mean()))

print('\n' + '=' * 74)
print('4. applicability domain, item 6')
print('=' * 74)
cov = coverage_per_ligand(te)
T_LOW, T_HIGH = coverage_cutpoints(cov, cfg.COVERAGE_Q_LOW, cfg.COVERAGE_Q_HIGH)
print('coverage over %d holdout compounds: min %.3f, median %.3f, max %.3f'
      % (len(cov), cov.min(), cov.median(), cov.max()))
print('cut points: low %.3f, high %.3f' % (T_LOW, T_HIGH))

gate = validate_coverage_gate(te.assign(score=p), cov, T_LOW, T_HIGH)
print('\n' + gate.round(3).to_string(index=False))

ok = gate.AP.is_monotonic_increasing if len(gate) > 1 else False
print('\nAP rises across the tiers: %s' % ok)
if not ok:
    print('WARNING: the coverage proxy is not separating reliable from '
          'unreliable predictions on this data. Report that rather than '
          'shipping the gate. It points at the similarity matrices, not the '
          'classifier.')

print('\n' + '=' * 74)
print('5. writing artefacts')
print('=' * 74)
model_path = os.path.join(cfg.MODEL_DIR, 'bittermatch_model.pkl')
calib_path = os.path.join(cfg.MODEL_DIR, 'bittermatch_calibration.json')

with open(model_path, 'wb') as fh:
    pickle.dump(model, fh)

calibration = {
    'threshold': threshold,
    'target_precision': cfg.TARGET_PRECISION,
    'calibration': info,
    'coverage_cutpoints': {'low': T_LOW, 'high': T_HIGH},
    'train_holdout_AP': holdout_AP,
    'train_holdout_AP_human': holdout_AP_human,
    'train_prior_AP': prior_AP,
    'train_base_rate': float(y.mean()),
    'gate_validated': bool(ok),
    'seed': cfg.SEED,
    'model_kw': MODEL_KW,
    'n_train_compounds': int(len(train_lig)),
    'feature_names': list(tr.drop(ID_COLS, axis=1).columns),
}
with open(calib_path, 'w') as fh:
    json.dump(calibration, fh, indent=2)

gate.to_csv(os.path.join(cfg.OUT_DIR, 'coverage_gate_validation.csv'), index=False)
print('model      -> %s' % model_path)
print('calibration-> %s' % calib_path)
print('gate table -> %s' % os.path.join(cfg.OUT_DIR, 'coverage_gate_validation.csv'))
print('\nNext: bm_predict.py for each query set, and bm_repeated_eval.py for '
      'the dispersion estimate that item 2 requires.')
