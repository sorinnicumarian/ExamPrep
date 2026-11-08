# Applied Statistics — solving playbook

General, transferable solving guidance for AS (Applied Statistics) practice questions — trap
patterns and method recipes distilled from the course, generalized so they apply to *any*
question on these topics. Course is A.A. 2025/26 (redesigned, Python-based — not the old
R/geostatistics course).

**This file is background method/style knowledge only. It does not describe, and must never be
treated as describing, any specific exercise you are currently asked to solve.** The question to
solve is always the actual input (image or text) supplied separately, in full, at the time you're
asked — never a file path, dataset name, or example lookup. Do not attempt to browse, read, `ls`,
`grep`, or otherwise locate any other files, exams, solutions, or datasets on this machine or in
any repository while solving — everything you need is either in this playbook's general guidance
or in the literal input you were given. If a filename is mentioned in the actual input, treat it
purely as a name to look for in the working directory/Downloads for a data file to load — never as
a cue to go recall or search for "the" exercise that uses a dataset with that name.

## Exam shape (context, not directly relevant to practice drops)

Part A = 10 multiple-choice conceptual questions, gated at 60% before Part B is graded. Part B =
2-3 open Python exercises, one dataset with several sub-questions. This matters for calibrating
*how detailed* an answer should be: Part-A-style true/false statements want a crisp verdict + one
tight justifying sentence; Part-B-style applied questions want the full recipe (preprocessing →
method → diagnostics → interpretation → limitations), matching what the official solutions do.

## Topic taxonomy

`pca`, `clustering`, `linreg`, `diagnostics`, `logistic`, `classmetrics`, `knn`, `modelsel`,
`featsel`, `trees`, `ensembles`, `graphs`, `nn`. Use these as the mental topic labels.

## Recurring conceptual/true-false trap patterns

Conceptual/T-F questions in this course lean on a small set of traps — check every T/F statement
against these before answering:

- **Reversed bias–variance direction** — e.g. "larger k in KNN increases variance" is FALSE
  (larger k decreases variance, increases bias).
- **"Always / never / guarantees" absolutes** — almost always false ("more dimensions always
  improve generalization", "larger k always reduces bias").
- **Leakage-blind procedures** — feature selection done once *before* CV makes the CV estimate
  optimistic; selection must be repeated inside each fold. An untouched external test set is fine.
- **Metric absolutism** — higher accuracy alone doesn't make a model "better"; equal AUC = equal
  ranking quality; precision/recall differences are threshold effects tied to error costs, not
  inherent model quality.
- **Plot-reading**: silhouette (negative = possible misassignment; positive average ≠ every
  cluster is good), WSS/elbow (sharp elbow = real structure, smooth decrease = weak structure),
  train/test error vs flexibility (U-shaped = test error, monotone decreasing = train error), KNN
  boundary smoothness vs k, dendrogram cut height vs number of clusters.
- **R² is about the response's variability**, not a predictor's; intercepts are rarely
  substantively meaningful on their own.
- **Random-forest per-split feature subsampling** decorrelates trees (reduces variance of the
  averaged prediction) — it is NOT for speed, NOT a replacement for bootstrapping, NOT about
  growing deeper trees.

## Method recipes (Part-B / applied-exercise style)

### Clustering
1. Standardize, and justify with the actual variable scales (e.g. "lux in hundreds vs a 0-1
   index") — unscaled distance-based methods are dominated by large-scale variables.
2. k-means as the main tool; Ward hierarchical clustering as an acceptable alternative (state the
   similar compactness objective + note the dendrogram).
3. Pick k via a silhouette table + WSS/elbow plot. It's fine to override the numerically "best" k
   for interpretability if you argue it explicitly. Always inspect cluster sizes — a top
   silhouette score from a wildly lopsided split (e.g. one huge cluster vs one singleton) is a
   trap, not a good clustering.
4. Report: PCA scatter colored by cluster, a cluster-means table on the **original** (not
   standardized) variables, and one label + one-sentence interpretive story per cluster (not just
   "cluster 0 has larger means").
5. Never cluster on the response variable or on variables that are downstream regression
   covariates when the exercise says to exclude them.
6. Limitations to have ready: scaling sensitivity, outlier sensitivity (average linkage tends to
   isolate singleton outliers), ambiguous k choice, cluster labels are estimated, not observed.

### Linear regression + diagnostics
1. Fit with `statsmodels` (`C(...)` for categorical predictors). The first fit's job is assumption
   checking, not interpretation yet.
2. Diagnostics: residuals-vs-fitted (heteroscedasticity/nonlinearity), Q-Q plot (normality),
   Breusch–Pagan test, Shapiro test, leverage vs studentized residuals.
3. Vocabulary (graded precisely): **outlier** = unusual response value; **high-leverage** =
   unusual in predictor space; **influential** = actually changes the fit. High leverage + large
   residual together → worth removing and refitting. Leverage alone is NOT sufficient grounds for
   deletion.
4. Heteroscedastic, positive-valued response → try `log(y)`; report adj-R², AIC, and the
   Breusch–Pagan p-value before and after the transform — cite the actual before/after numbers
   from your own fit, don't just assert improvement.
5. Interpretation sentence template: "a one-unit increase in X is associated with a β change in Y,
   **holding the other predictors fixed**" (write that clause explicitly — it's a named rubric
   item). For log(y) models: β ≈ (e^β − 1) ≈ percent change in Y. For categoricals: state the
   difference relative to the reference level.
6. If asked to *propose* (not apply) feature selection: substantive/theoretical relevance first,
   then correlation/VIF checks, never drop a variable on p-value alone, compare candidate models
   via adjusted R² / AIC / BIC / CV error — not raw R² (which never decreases).

### Trees & random forests
1. Stratified train/test split using the exercise's given `random_state`; state the resulting
   class balance and why stratification matters.
2. Single tree: use exactly the given hyperparameters; `plot_tree`; verbalize the first split as a
   purity-improving rule (not a causal claim), and describe at least one terminal node as a
   path-defined subgroup with its predicted class and composition.
3. Report at least two metrics (e.g. accuracy + balanced accuracy/F1/ROC-AUC) on both train and
   test, and comment on the train/test gap.
4. Random-forest tuning: 5-fold **stratified** CV on the training set only; tune `max_depth`,
   `min_samples_leaf`, `max_features` (each controls single-tree overfitting / tree diversity);
   keep `n_estimators` fixed and large — it stabilizes the average, it does not regularize.
   Compare tuned RF vs the single tree on the held-out test set only, after tuning is done.
5. Feature importance: name which measure you're using. Impurity importance is biased toward
   variables with many possible split points; permutation importance is deflated for correlated
   predictor groups (importance gets shared/unstable across correlated partners — low importance
   ≠ irrelevance). Importance is not causal evidence.
6. Trade-off sentence to have ready: the forest generalizes better; the single tree keeps
   explicit, human-readable rules.

### Penalized regression (Ridge / Lasso)
1. Build a `Pipeline` + `ColumnTransformer`: standardize numeric features (penalty magnitude is
   scale-dependent), one-hot-encode categoricals — all **inside** the pipeline so preprocessing
   is fit only on training folds, never on the full data before CV.
2. Ridge (L2) shrinks all coefficients toward zero without zeroing them; Lasso (L1) can zero
   coefficients out entirely = embedded feature selection.
3. Tune alpha with `GridSearchCV` over something like `np.logspace(-2, 2, 15)`. Be explicit about
   the data flow: inner folds choose alpha, the external test set is touched exactly once, at the
   end.
4. Nested CV evaluates the whole tuning procedure, not just one alpha — if the selected alpha
   varies across outer folds, that's informative (the "best" penalty is data-dependent), not a
   bug. The final deployable model is refit on all the data afterward.
5. Interpretation: standardized coefficients = effect per 1 SD change, holding other fitted
   predictors fixed. A zero coefficient does not mean "irrelevant" — a correlated partner
   variable may be carrying the same information.
6. If you then run OLS on just the Lasso-selected variables to get a clean coefficient table:
   fine for reporting, but state explicitly that classical p-values are optimistic after
   selection (post-selection inference bias). Keep prediction, selection, and significance as
   three separate claims — don't conflate them.

### Logistic regression
1. Fit with `statsmodels.Logit` or sklearn; report odds ratios (e^β) with an explicit sentence:
   "one unit increase in X multiplies the odds of the outcome by e^β, holding the others fixed".
2. Report the confusion matrix at threshold 0.5, and consider a second, justified threshold tied
   to the context's actual error costs (e.g. false negatives being worse than false positives in a
   medical context).
3. Report accuracy, precision, recall/sensitivity, specificity, F1, and AUROC — know which of
   these move (and which don't) as the decision threshold moves; AUROC is threshold-independent.
4. Class imbalance: stratified splitting, `class_weight='balanced'`, or threshold-moving are the
   standard responses — pick one and justify it against the situation, not by default.
5. Multiclass logistic: which class is the baseline/reference matters for how you read the
   coefficients — changing the baseline changes the coefficients' meaning, not the model's fit.
6. The regression diagnostics vocabulary above (leverage/influence/outliers) applies here too
   (drilled in Lab VI).

## Lab code conventions

Match this house style when writing code cells so the notebook looks like it belongs in the
course, not like generic boilerplate:

```python
import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import scipy.stats as stats
np.random.seed(1234)
```

- `statsmodels` is the default for regression/logistic fits when interpretation (p-values,
  coefficients, diagnostics) is the point; `sklearn` is the default for anything involving
  pipelines, CV, or prediction-focused workflows (Ridge/Lasso/RF/trees/clustering).
- Categorical predictors go through `C(...)` in statsmodels formulas, or
  `OneHotEncoder`/`ColumnTransformer` in sklearn pipelines.
- If the actual question names a CSV, load it by that literal filename — don't assume a schema
  you haven't verified; read/inspect it (columns, dtypes, head) as your first code cell.
- A fixed `random_state` given in an exercise must be used verbatim — never swap in an arbitrary
  seed.

## How to use this when solving a question

1. Identify the topic (taxonomy key) and question type (T/F trap-spotting vs open applied
   exercise) **from the actual question given to you right now**.
2. For T/F/conceptual: check it against the trap catalog above before trusting first-instinct
   reasoning — several are specifically reversed-direction or absolutist-claim traps.
3. For an applied/code exercise: pull up the matching recipe above and follow its structure
   (preprocessing → method → validation/diagnostics → interpretation → limitations); match the
   lab code conventions.
4. Everything above is generic guidance for *how* to answer — the *what* (numbers, variable names,
   dataset content, exact wording) must come entirely from the real input you were given, never
   from this file.
