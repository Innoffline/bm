"""BitterMatch configuration. Replace every <...> value, then run the scripts.

All three scripts read their settings from here, so this is the only file you
need to edit. To keep more than one query set configured at a time, copy this
file (bm_config_nba.py, bm_config_nbb.py) and select one at run time:

    BM_CONFIG=bm_config_nba python bm_predict.py

Run order:
    1. python bm_train.py           # fits the model, calibrates, writes artefacts
    2. python bm_predict.py         # scores a query set, once per config
    3. python bm_repeated_eval.py   # dispersion estimate, slow, optional but expected
"""

# --------------------------------------------------------------------------
# Output locations. Created automatically if absent.
# --------------------------------------------------------------------------
MODEL_DIR = '<MODEL_DIR>'          # e.g. 'Trained_Models'
OUT_DIR = '<OUT_DIR>'              # e.g. 'Results'

# --------------------------------------------------------------------------
# Training inputs. The reference set the model learns from.
# --------------------------------------------------------------------------
A_CSV = '<A_CSV>'                          # association matrix, ligands x receptors
REC_FEAT_CSV = '<REC_FEAT_CSV>'            # receptor features
LIG_FEAT_CSV = '<LIG_FEAT_CSV>'            # ligand descriptors, 'cid' column
LIG_LINEAR_SIM_CSV = '<LIG_LINEAR_SIM_CSV>'
LIG_MOL2D_SIM_CSV = '<LIG_MOL2D_SIM_CSV>'

# --------------------------------------------------------------------------
# Query set. Used by bm_predict.py only.
#
# QUERY_ASSOC_CSV may be None. Set it to None for a set with no measured
# associations at all. Where a file is given it may cover only some of the
# compounds, and the rest are predicted without being scored.
#
# The two similarity files must be COMBINED matrices covering the training
# compounds and the query compounds together, exactly as the published
# 'Ligand_linear_limilarity_combined.csv' does.
# --------------------------------------------------------------------------
QUERY_NAME = '<QUERY_NAME>'                    # label used in filenames, e.g. 'nba_14'
QUERY_LIG_FEAT_CSV = '<QUERY_LIG_FEAT_CSV>'
QUERY_ASSOC_CSV = '<QUERY_ASSOC_CSV_OR_None>'  # literal None if no labels exist
QUERY_LINEAR_SIM_CSV = '<QUERY_LINEAR_SIM_CSV>'
QUERY_MOL2D_SIM_CSV = '<QUERY_MOL2D_SIM_CSV>'

# Distinct per query set, otherwise identifiers collide when results are merged.
# Suggested: 999000 for the first set, 998000 for the second.
QUERY_ID_OFFSET = 999000

# --------------------------------------------------------------------------
# Model and protocol settings. The defaults are the audited values.
# --------------------------------------------------------------------------
SEED = 6
TARGET_PRECISION = 0.80    # item 1, see the guide on why 0.8
LEARNING_RATE = 0.03       # item 5, was 0.001
N_ESTIMATORS = 250         # item 5, was 1000
TRAIN_FRACTION = 0.8
N_CALIB_FOLDS = 4          # inner folds used to calibrate the threshold
N_REPEATS = 20             # item 2, repeats in bm_repeated_eval.py

# Coverage quantiles used to propose the applicability domain cut points.
# These are turned into absolute values in bm_train.py and then frozen.
COVERAGE_Q_LOW = 0.25
COVERAGE_Q_HIGH = 0.60
