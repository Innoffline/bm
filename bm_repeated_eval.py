"""Repeated ligand grouped evaluation. Audit item 2.

The released notebook reports one split. Varying only the seed moved average
precision by roughly 0.05 on the public data, which exceeds most modelling
changes worth arguing about, so a single run cannot tell you whether something
helped.

Slow by design. Each repeat rebuilds the neighbourhood features once per inner
fold plus once for the final fit. Run it detached:

    BM_CONFIG=bm_config nohup python -u bm_repeated_eval.py > repeated_eval.log 2>&1 &

Writes into OUT_DIR:
    repeated_evaluation.csv          one row per repeat
    repeated_evaluation_summary.csv  mean, sd, min, max
"""
import os

import numpy as np

from bittermatch_ext import (get_config, load_training_inputs, build_base,
                             long_form, repeated_evaluation, summarise)

cfg = get_config()
os.makedirs(cfg.OUT_DIR, exist_ok=True)

PERMUTE = os.environ.get('BM_PERMUTE', '0') == '1'

A, X_Rec, X_Lig, sim_dict = load_training_inputs(cfg)
base = build_base(X_Lig, X_Rec)
long_A = long_form(A)

if PERMUTE:
    # Negative control. Shuffling the labels should collapse average precision
    # to the base rate. Anything else means something is leaking.
    rng = np.random.RandomState(cfg.SEED)
    long_A = long_A.copy()
    mask = long_A.association.notna()
    long_A.loc[mask, 'association'] = rng.permutation(long_A.loc[mask, 'association'].values)
    A = long_A.pivot(index='ligand', columns='receptor', values='association')
    print('PERMUTED LABELS: this is the negative control, not a result')

res = repeated_evaluation(A, base, long_A, sim_dict,
                          n_repeats=cfg.N_REPEATS,
                          train_frac=cfg.TRAIN_FRACTION,
                          target_precision=cfg.TARGET_PRECISION,
                          n_folds=cfg.N_CALIB_FOLDS,
                          model_kw=dict(learning_rate=cfg.LEARNING_RATE,
                                        n_estimators=cfg.N_ESTIMATORS))

suffix = '_permuted' if PERMUTE else ''
res.to_csv(os.path.join(cfg.OUT_DIR, 'repeated_evaluation%s.csv' % suffix), index=False)
summary = summarise(res)
summary.to_csv(os.path.join(cfg.OUT_DIR, 'repeated_evaluation_summary%s.csv' % suffix))

print('\n' + '=' * 74)
print(summary.to_string())
print('=' * 74)

sd = res.AP.std()
sem2 = 2 * sd / np.sqrt(len(res))
print('\nAP %.3f, sd %.3f over %d repeats.' % (res.AP.mean(), sd, len(res)))
print('A change smaller than about %.3f has not been demonstrated to do '
      'anything. Apply that test before reporting any improvement.' % sem2)
if PERMUTE:
    print('Base rate was %.3f. Permuted AP should be close to it.' % res.base_rate.mean())
else:
    print('Run the negative control too: BM_PERMUTE=1 python bm_repeated_eval.py')
