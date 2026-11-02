import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import altair as alt
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import phik  # noqa: F401  (registers DataFrame.phik_matrix / .significance_matrix)
    import polars as pl
    import shap
    import statsmodels.formula.api as smf
    from lorepy import loreplot
    from scipy import stats
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import (
        RepeatedStratifiedKFold,
        StratifiedKFold,
        cross_val_predict,
        cross_validate,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from statsmodels.stats.anova import anova_lm
    from statsmodels.stats.multitest import multipletests

    return (
        LogisticRegression,
        RandomForestClassifier,
        RepeatedStratifiedKFold,
        StandardScaler,
        StratifiedKFold,
        alt,
        anova_lm,
        clone,
        confusion_matrix,
        cross_val_predict,
        cross_validate,
        hierarchy,
        loreplot,
        make_pipeline,
        mo,
        multipletests,
        np,
        pd,
        permutation_importance,
        pl,
        plt,
        shap,
        smf,
        squareform,
        stats,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Palmer Penguins: what distinguishes the species and the sexes?

    This notebook analyses body measurements of 344 penguins of three species (Adelie,
    Chinstrap, Gentoo), sampled on three islands in the Palmer Archipelago, Antarctica,
    between 2007 and 2009. It asks two questions:

    1. **How do the sexes differ** once species is taken into account?
    2. **Can species be predicted** from four body measurements, and which measurements matter?

    **Outline**

    1. The data
    2. Interactive exploration
    3. How the variables relate (φK correlation with clustering)
    4. Sex differences within species (two-way ANOVA)
    5. Species along each measurement (lorepy)
    6. Predicting species (cross-validated classifier)
    7. Which measurements drive the prediction (permutation importance, SHAP)
    8. Discussion

    Data: [palmerpenguins](https://allisonhorst.github.io/palmerpenguins/) (CC0), downloaded by
    `scripts/download_data.py`. Except in Section 2, every analysis uses all complete cases and
    ignores the species filter.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The data

    Each row is one penguin: species, island, sex, year of sampling, and four continuous
    measurements: bill length and bill depth (mm), flipper length (mm) and body mass (g).
    Rows with any missing value are dropped.
    """)
    return


@app.cell(hide_code=True)
def _(mo, pl):
    data_path = mo.notebook_dir().parent / "data" / "penguins.csv"
    mo.stop(
        not data_path.exists(),
        mo.callout(
            mo.md("No data yet. Run `uv run python scripts/download_data.py`, then re-run."),
            kind="warn",
        ),
    )
    raw = pl.read_csv(data_path, null_values="NA")
    penguins = raw.drop_nulls()
    mo.md(f"Loaded **{raw.height}** rows; **{penguins.height}** complete cases kept.")
    return (penguins,)


@app.cell
def _(mo, penguins):
    species_by_island = (
        penguins.group_by("species", "island")
        .len()
        .pivot(on="island", index="species", values="len")
        .fill_null(0)
        .sort("species")
    )
    mo.vstack(
        [
            mo.md(
                f"**Complete cases by species and island** "
                f"({penguins['year'].min()}–{penguins['year'].max()})"
            ),
            species_by_island,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
    **Conclusion.** 333 of the 344 penguins are complete cases. All 11 dropped rows lack sex
    (two of them also lack every measurement), so little data is lost. The sampling design
    is not fully crossed: Adelie were sampled on all three islands, but Gentoo only on Biscoe
    and Chinstrap only on Dream. Island and species are therefore largely confounded, and
    island effects cannot be separated from species effects. Within each species the sexes
    are close to balanced (73/73, 34/34, 61/58 males/females).
    """),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Interactive exploration

    A first look at the raw measurements. Choose species and axes, and drag on the scatter
    plot to inspect individual penguins. **The species filter applies to this section only.**
    The linear fit at the end pools the selected species; the conclusion below shows why
    that can be misleading.
    """)
    return


@app.cell(hide_code=True)
def _(mo, penguins):
    numeric_cols = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
    species_picker = mo.ui.multiselect(
        options=sorted(penguins["species"].unique().to_list()),
        value=sorted(penguins["species"].unique().to_list()),
        label="Species",
    )
    x_axis = mo.ui.dropdown(numeric_cols, value="flipper_length_mm", label="X")
    y_axis = mo.ui.dropdown(numeric_cols, value="body_mass_g", label="Y")
    mo.hstack([species_picker, x_axis, y_axis], justify="start", gap=2)
    return numeric_cols, species_picker, x_axis, y_axis


@app.cell(hide_code=True)
def _(penguins, pl, species_picker):
    selected = penguins.filter(pl.col("species").is_in(species_picker.value))
    return (selected,)


@app.cell(hide_code=True)
def _(alt, mo, selected, x_axis, y_axis):
    _chart = (
        alt.Chart(selected)
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x=alt.X(x_axis.value, scale=alt.Scale(zero=False)),
            y=alt.Y(y_axis.value, scale=alt.Scale(zero=False)),
            color="species",
            tooltip=["species", "island", "sex", x_axis.value, y_axis.value],
        )
        .properties(height=380)
    )
    scatter = mo.ui.altair_chart(_chart)
    mo.vstack([mo.md("### Scatter (drag to select points)"), scatter])
    return (scatter,)


@app.cell(hide_code=True)
def _(mo, scatter, selected):
    _rows = scatter.value if scatter.value.height > 0 else selected
    mo.vstack([mo.md(f"### Selected rows ({_rows.height})"), mo.ui.table(_rows)])
    return


@app.cell(hide_code=True)
def _(mo, pl, selected):
    summary = (
        selected.group_by("species")
        .agg(
            pl.len().alias("n"),
            pl.col("body_mass_g").mean().round(0).alias("mean_mass_g"),
            pl.col("flipper_length_mm").mean().round(1).alias("mean_flipper_mm"),
            pl.col("bill_length_mm").mean().round(1).alias("mean_bill_mm"),
        )
        .sort("species")
    )
    mo.vstack([mo.md("### Summary by species"), summary])
    return


@app.cell(hide_code=True)
def _(mo, np, selected, stats, x_axis, y_axis):
    mo.stop(selected.height < 3, mo.md("Select more data for statistics."))
    _x = selected[x_axis.value].to_numpy()
    _y = selected[y_axis.value].to_numpy()
    _fit = stats.linregress(_x, _y)
    _r = np.corrcoef(_x, _y)[0, 1]

    mo.md(
        f"""
        ### Linear fit

        `{y_axis.value} ~ {x_axis.value}` pooled over the selected species:
        slope = {_fit.slope:.3f}, intercept = {_fit.intercept:.1f},
        r = {_r:.3f}, R² = {_fit.rvalue**2:.3f}, p = {_fit.pvalue:.2e}
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
    **Conclusion.** The species form distinct clusters. Gentoo are heavier, with longer
    flippers and shallower bills; Adelie and Chinstrap are similar in size and differ mainly
    in bill length. Because of this, fits that pool species can mislead:

    - Across all three species, bill length and bill depth are *negatively* correlated
      (r = −0.23), yet within each species the correlation is *positive* (r = 0.39 Adelie,
      0.65 Chinstrap, 0.65 Gentoo). This is a Simpson's paradox.
    - Flipper length and body mass are correlated within species (r = 0.46–0.71), but pooling
      inflates this to r = 0.87.

    Every later analysis therefore accounts for species.
    """),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. How the variables relate

    To see the overall structure, we compute the Phi_K (φK) correlation between every pair
    of variables. [φK](https://phik.readthedocs.io) handles categorical variables (species,
    island, sex, year) and continuous ones alike. It ranges from 0 (independent) to 1 (fully
    dependent) and has no sign. Hover a cell to see its significance as a Z-score
    (Z > 3 ≈ p < 0.001).

    Variables are ordered by hierarchical clustering on the distance 1 − φK (average linkage,
    optimal leaf ordering). Branch height is the distance at which groups merge.
    """)
    return


@app.cell(hide_code=True)
def _(numeric_cols, pd, penguins):
    # Phi_K on all complete cases (ignores the species filter); year is treated as categorical.
    _df = pd.DataFrame(penguins.to_dict(as_series=False))
    phik_corr = _df.phik_matrix(interval_cols=numeric_cols)
    phik_sig = _df.significance_matrix(interval_cols=numeric_cols)
    return phik_corr, phik_sig


@app.cell(hide_code=True)
def _(hierarchy, np, phik_corr, squareform):
    # Hierarchical clustering of the variables on φK distance (1 − φK), average linkage.
    _dist = 1 - phik_corr.to_numpy()
    _dist = (_dist + _dist.T) / 2
    np.fill_diagonal(_dist, 0)
    phik_linkage = hierarchy.linkage(
        squareform(_dist, checks=False), method="average", optimal_ordering=True
    )
    phik_dendro = hierarchy.dendrogram(
        phik_linkage, labels=list(phik_corr.columns), no_plot=True
    )
    return (phik_dendro,)


@app.cell(hide_code=True, expand_output=True)
def _(alt, np, phik_corr, phik_dendro, phik_sig, pl):
    _order = phik_dendro["ivl"]  # leaf order of the tree
    _vars = list(phik_corr.columns)
    _n = len(_vars)
    _size, _tree = 420, 90

    _long = pl.DataFrame(
        {
            "var1": [a for a in _vars for _ in _vars],
            "var2": [b for _ in _vars for b in _vars],
            "phik": phik_corr.to_numpy().ravel(),
            "z": phik_sig.to_numpy().ravel(),
        }
    ).with_columns(pl.col("phik").round(2), pl.col("z").round(1))
    _base = alt.Chart(_long).encode(
        x=alt.X("var2:N", sort=_order, title=None, axis=alt.Axis(labelAngle=-40)),
        y=alt.Y("var1:N", sort=_order, title=None, axis=alt.Axis(orient="right")),
    )
    _heat = _base.mark_rect().encode(
        color=alt.Color("phik:Q", scale=alt.Scale(domain=[0, 1], scheme="blues"), title="φK"),
        tooltip=["var1", "var2", "phik", alt.Tooltip("z", title="significance (Z)")],
    )
    _text = _base.mark_text(fontSize=11).encode(
        text=alt.Text("phik:Q", format=".2f"),
        color=alt.condition("datum.phik > 0.6", alt.value("white"), alt.value("black")),
    )

    # Dendrogram segments: scipy places leaf i at 10*i + 5, i.e. the centre of heatmap band i.
    _segs = pl.DataFrame(
        [
            {"pos": _xs[_i], "pos2": _xs[_i + 1], "h": _ys[_i], "h2": _ys[_i + 1]}
            for _xs, _ys in zip(phik_dendro["icoord"], phik_dendro["dcoord"])
            for _i in range(3)
        ]
    )
    _pos = alt.Scale(domain=[0, 10 * _n], nice=False, zero=False)
    _hmax = float(np.max(phik_dendro["dcoord"])) * 1.05
    _top = (
        alt.Chart(_segs)
        .mark_rule(color="#555")
        .encode(
            x=alt.X("pos:Q", scale=_pos, axis=None),
            x2="pos2",
            y=alt.Y("h:Q", scale=alt.Scale(domain=[0, _hmax], nice=False), axis=None),
            y2="h2",
        )
        .properties(width=_size, height=_tree)
    )
    _left = (
        alt.Chart(_segs)
        .mark_rule(color="#555")
        .encode(
            y=alt.Y("pos:Q", scale=alt.Scale(domain=[0, 10 * _n], nice=False, zero=False, reverse=True), axis=None),
            y2="pos2",
            x=alt.X("h:Q", scale=alt.Scale(domain=[0, _hmax], nice=False, reverse=True), axis=None),
            x2="h2",
        )
        .properties(width=_tree, height=_size)
    )
    _corner = alt.Chart(_segs).mark_text().properties(width=_tree, height=_tree)
    phik_clustermap = (
        alt.vconcat(
            alt.hconcat(_corner, _top, spacing=0),
            alt.hconcat(_left, (_heat + _text).properties(width=_size, height=_size), spacing=0),
            spacing=0,
        )
        .configure_view(strokeWidth=0)
    )

    phik_clustermap
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
    **Conclusion.** Three groups emerge:

    - **Species and island** merge first (φK = 0.92), reflecting the sampling design described
      in Section 1.
    - **Body size** (body mass, flipper length, bill depth; φK 0.80–0.82 among them) joins
      that group, because these measurements are largely determined by species
      (φK 0.74–0.81).
    - **Bill length and sex** form a separate branch. Sex is unrelated to species and island
      (φK = 0) because the sexes are balanced, yet it is strongly associated with body mass
      and bill depth (φK = 0.75). Sex differences therefore have to be studied *within*
      species (Section 4).

    Year is nearly unrelated to everything. The one exception is a weak link with flipper
    length (φK = 0.33, Z ≈ 3.3), which could be a real change over time or a change in how
    flippers were measured between field seasons.
    """),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Sex differences within species

    Males and females are compared with a two-way ANOVA, `feature ~ species * sex`, using
    Type II sums of squares, which suit this nearly balanced design.

    - The **sex** term tests the sex difference after accounting for species.
    - The **species × sex** interaction tests whether that difference varies between species.
      When it does, interpret the per-species comparisons (Welch t-tests, Holm-corrected for
      three tests) rather than the overall sex term.

    Pick the measurement to analyse below; the table at the end of the section summarises
    all four.
    """)
    return


@app.cell(hide_code=True)
def _(anova_lm, pd, penguins, smf):
    def fit_sex_anova(feature):
        """Type II two-way ANOVA of `feature ~ species * sex` on all complete cases."""
        df = pd.DataFrame(penguins.select("species", "sex", feature).to_dict(as_series=False))
        return anova_lm(smf.ols(f"{feature} ~ C(species) * C(sex)", data=df).fit(), typ=2)

    return (fit_sex_anova,)


@app.cell(hide_code=True)
def _(mo, numeric_cols):
    anova_feature = mo.ui.dropdown(numeric_cols, value="body_mass_g", label="Feature")
    anova_feature
    return (anova_feature,)


@app.cell(hide_code=True)
def _(anova_feature, fit_sex_anova, mo, multipletests, penguins, pl, stats):
    _feat = anova_feature.value
    _aov = fit_sex_anova(_feat)
    _ss_res = _aov.loc["Residual", "sum_sq"]
    _terms = {"C(species)": "species", "C(sex)": "sex", "C(species):C(sex)": "species × sex"}
    anova_table = pl.DataFrame(
        {
            "term": list(_terms.values()),
            "df": [int(_aov.loc[_k, "df"]) for _k in _terms],
            "F": [round(_aov.loc[_k, "F"], 2) for _k in _terms],
            "p": [f"{_aov.loc[_k, 'PR(>F)']:.2e}" for _k in _terms],
            "partial_eta_sq": [
                round(_aov.loc[_k, "sum_sq"] / (_aov.loc[_k, "sum_sq"] + _ss_res), 3)
                for _k in _terms
            ],
        }
    )

    _rows = []
    for _sp in sorted(penguins["species"].unique().to_list()):
        _sub = penguins.filter(pl.col("species") == _sp)
        _m = _sub.filter(pl.col("sex") == "male")[_feat].to_numpy()
        _f = _sub.filter(pl.col("sex") == "female")[_feat].to_numpy()
        _t = stats.ttest_ind(_m, _f, equal_var=False)
        _ci = _t.confidence_interval()
        _rows.append(
            {
                "species": _sp,
                "n_male": len(_m),
                "n_female": len(_f),
                "mean_male": round(_m.mean(), 1),
                "mean_female": round(_f.mean(), 1),
                "diff_m_minus_f": round(_m.mean() - _f.mean(), 1),
                "ci95_low": round(_ci.low, 1),
                "ci95_high": round(_ci.high, 1),
                "p": _t.pvalue,
            }
        )
    _p_holm = multipletests([_r["p"] for _r in _rows], method="holm")[1]
    sex_contrasts = pl.DataFrame(_rows).with_columns(
        pl.col("p").map_elements(lambda v: f"{v:.2e}", return_dtype=pl.String),
        p_holm=pl.Series([f"{v:.2e}" for v in _p_holm]),
    )

    _p_int = _aov.loc["C(species):C(sex)", "PR(>F)"]
    _verdict = (
        f"The sex effect **differs between species** (interaction p = {_p_int:.2e}), "
        "so interpret the per-species differences rather than the overall sex term."
        if _p_int < 0.05
        else f"No evidence the sex effect differs between species (interaction p = {_p_int:.2f}); "
        "the `sex` row is the species-adjusted sex effect."
    )
    mo.vstack(
        [
            mo.md(f"**ANOVA (Type II)** for `{_feat}`, n = {penguins.height}"),
            anova_table,
            mo.md(_verdict),
            mo.md("**Male − female within each species** (Welch t-test, Holm-adjusted)"),
            sex_contrasts,
        ]
    )
    return


@app.cell(hide_code=True)
def _(fit_sex_anova, mo, numeric_cols, penguins, pl):
    _species = sorted(penguins["species"].unique().to_list())
    _rows = []
    for _f in numeric_cols:
        _a = fit_sex_anova(_f)
        _ss_res = _a.loc["Residual", "sum_sq"]
        _ss_sex = _a.loc["C(sex)", "sum_sq"]
        _row = {
            "feature": _f,
            "F_sex": round(_a.loc["C(sex)", "F"], 1),
            "p_sex": f"{_a.loc['C(sex)', 'PR(>F)']:.1e}",
            "partial_eta_sq_sex": round(_ss_sex / (_ss_sex + _ss_res), 2),
            "p_interaction": f"{_a.loc['C(species):C(sex)', 'PR(>F)']:.1e}",
        }
        for _sp in _species:
            _d = penguins.filter(pl.col("species") == _sp)
            _male = _d.filter(pl.col("sex") == "male")[_f].mean()
            _female = _d.filter(pl.col("sex") == "female")[_f].mean()
            _row[f"{_sp} M vs F (%)"] = round(100 * (_male / _female - 1), 1)
        _rows.append(_row)
    sex_overview = pl.DataFrame(_rows)
    mo.vstack(
        [
            mo.md(
                "**All four measurements at a glance** (independent of the picker; "
                "the last three columns show how much larger males are than females, in %)"
            ),
            sex_overview,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
    **Conclusion.** Males are larger than females in every species and on every measurement
    (sex term p < 10⁻²³ for all four). The effect is largest for body mass (partial η² = 0.54;
    males are 12–20% heavier) and smallest for flipper length (partial η² = 0.27; 2–4% longer).

    - **Bills:** the sex difference is similar across species (interaction p = 0.10 for length,
      0.67 for depth), so one species-adjusted effect describes it: male bills are about
      8–10% longer and deeper.
    - **Body mass and flipper length:** the size of the difference depends on species
      (interaction p = 2 × 10⁻⁴ and 0.006). In body mass, Chinstrap are the least dimorphic
      (+412 g, +12%), while Adelie (+675 g, +20%) and Gentoo (+805 g, +17%) are the most.
    """),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Species along each measurement

    Before building a classifier, it helps to see how well each measurement separates the
    species on its own. A [loreplot](https://github.com/raeslab/lorepy) fits a multinomial
    logistic regression of a category on a single measurement and shows the predicted
    prevalence of each class as stacked areas; white dots are the observed penguins. The
    second picker can also show island or sex.
    """)
    return


@app.cell(hide_code=True)
def _(mo, numeric_cols):
    lore_x = mo.ui.dropdown(numeric_cols, value="bill_length_mm", label="X feature")
    lore_y = mo.ui.dropdown(["species", "island", "sex"], value="species", label="Prevalence of")
    mo.hstack([lore_x, lore_y], justify="start", gap=2)
    return lore_x, lore_y


@app.cell(hide_code=True)
def _(lore_x, lore_y, loreplot, pd, penguins, plt):
    _df = pd.DataFrame(penguins.select(lore_y.value, lore_x.value).to_dict(as_series=False))
    _n_classes = _df[lore_y.value].nunique()
    # Tableau 10, Altair's default scheme, so species colours match the scatter above.
    _palette = ["#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b"][:_n_classes]
    _fig, _ax = plt.subplots(figsize=(8, 4))
    loreplot(_df, lore_x.value, lore_y.value, ax=_ax, color=_palette, scatter_kws={"s": 12})
    _ax.set_xlabel(lore_x.value)
    _ax.set_ylabel("Predicted prevalence")
    _ax.legend(title=lore_y.value, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
    **Conclusion.** No single measurement separates all three species:

    - **Bill length** separates Adelie (short bills, below about 42 mm) from the other two,
      but Chinstrap and Gentoo overlap.
    - **Bill depth, flipper length and body mass** each pick out Gentoo (shallow bills, long
      flippers, heavy birds) but barely separate Adelie from Chinstrap.

    The measurements complement each other: bill length combined with any of the
    Gentoo-specific measurements should separate all three species. Section 6 tests this.
    """),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Predicting species

    A classifier predicts species from the four continuous measurements. Two models are
    available:

    - multinomial **logistic regression** on standardised features (a simple linear model);
    - a **random forest** with 300 trees (non-linear).

    Performance is estimated with stratified 5-fold cross-validation repeated 10 times. The
    confusion matrix uses out-of-fold predictions from a single 5-fold split, so every penguin
    is predicted by a model that never saw it during training.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    species_model_choice = mo.ui.dropdown(
        ["Logistic regression", "Random forest"], value="Logistic regression", label="Model"
    )
    species_model_choice
    return (species_model_choice,)


@app.cell(hide_code=True)
def _(
    LogisticRegression,
    RandomForestClassifier,
    RepeatedStratifiedKFold,
    StandardScaler,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    make_pipeline,
    mo,
    numeric_cols,
    penguins,
    pl,
    species_model_choice,
):
    species_X = penguins.select(numeric_cols).to_numpy().astype(float)
    species_y = penguins["species"].to_numpy()
    species_model = (
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        if species_model_choice.value == "Logistic regression"
        else RandomForestClassifier(n_estimators=300, random_state=0)
    )
    _metrics = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "f1_macro": "Macro F1"}
    _cv = cross_validate(
        species_model,
        species_X,
        species_y,
        cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0),
        scoring=list(_metrics),
        n_jobs=-1,
    )
    cv_scores = pl.DataFrame(
        {
            "metric": list(_metrics.values()),
            "mean": [round(_cv[f"test_{_m}"].mean(), 3) for _m in _metrics],
            "sd": [round(_cv[f"test_{_m}"].std(), 3) for _m in _metrics],
        }
    )
    species_cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    species_oof = cross_val_predict(species_model, species_X, species_y, cv=species_cv_folds)
    mo.vstack([mo.md(f"**{species_model_choice.value}**: cross-validated performance (5-fold × 10 repeats)"), cv_scores])
    return species_X, species_cv_folds, species_model, species_oof, species_y


@app.cell(hide_code=True)
def _(alt, confusion_matrix, mo, penguins, pl, species_oof, species_y):
    _labels = sorted(set(species_y))
    _cm = confusion_matrix(species_y, species_oof, labels=_labels)
    _cm_long = pl.DataFrame(
        {
            "true": [t for t in _labels for _ in _labels],
            "predicted": [p for _ in _labels for p in _labels],
            "n": _cm.ravel(),
            "row_pct": (100 * _cm / _cm.sum(axis=1, keepdims=True)).round(1).ravel(),
        }
    )
    _base = alt.Chart(_cm_long).encode(
        x=alt.X("predicted:N", sort=_labels, title="Predicted", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("true:N", sort=_labels, title="True"),
    )
    _cm_chart = (
        _base.mark_rect().encode(
            color=alt.Color("row_pct:Q", scale=alt.Scale(domain=[0, 100], scheme="blues"), title="% of true"),
            tooltip=["true", "predicted", "n", alt.Tooltip("row_pct", title="% of true class")],
        )
        + _base.mark_text(fontSize=14).encode(
            text="n:Q",
            color=alt.condition("datum.row_pct > 60", alt.value("white"), alt.value("black")),
        )
    ).properties(width=260, height=260)
    misclassified = penguins.with_columns(predicted=pl.Series(species_oof)).filter(
        pl.col("species") != pl.col("predicted")
    )
    mo.vstack(
        [
            mo.md(f"**Confusion matrix** (out-of-fold predictions, {len(species_y)} penguins)"),
            _cm_chart,
            mo.md(f"**Misclassified penguins ({misclassified.height})**"),
            misclassified,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
    **Conclusion.** Species can be predicted almost perfectly from body measurements.

    | Model | Accuracy | Balanced accuracy | Misclassified (out-of-fold) |
    |---|---|---|---|
    | Logistic regression | 0.988 ± 0.012 | 0.982 | 4 of 333 |
    | Random forest | 0.978 ± 0.016 | 0.974 | 6 of 333 |

    The simpler logistic regression does slightly better, which suggests the species are
    close to linearly separable and the forest's extra flexibility does not help. With
    logistic regression Gentoo are never misclassified. All errors are between Adelie and
    Chinstrap, mostly Chinstrap females from Dream whose bills fall in the Adelie range:
    the same overlap seen in Section 5.
    """),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Which measurements drive the prediction

    Two complementary methods are applied to the model selected in Section 6:

    - **Permutation importance** shuffles one feature in held-out data and measures the drop
      in accuracy. It is computed out-of-sample, within each cross-validation fold.
    - **SHAP** splits each penguin's predicted probability into contributions from each
      feature, so it also shows *which species* a feature helps identify and *in which
      direction*. It is computed on a model fitted to all data.

    Both methods can understate correlated features. If two features carry the same
    information, the model can rely on either one, and shuffling one leaves the other to
    carry the signal.
    """)
    return


@app.cell(hide_code=True)
def _(
    alt,
    clone,
    mo,
    numeric_cols,
    permutation_importance,
    pl,
    species_X,
    species_cv_folds,
    species_model,
    species_y,
):
    # Permutation importance on each held-out fold: drop in accuracy when one feature is shuffled.
    _rows = []
    for _fold, (_tr, _te) in enumerate(species_cv_folds.split(species_X, species_y)):
        _fitted = clone(species_model).fit(species_X[_tr], species_y[_tr])
        _pi = permutation_importance(
            _fitted, species_X[_te], species_y[_te], n_repeats=20, random_state=0, n_jobs=-1
        )
        _rows += [
            {"feature": _f, "fold": _fold, "importance": _v}
            for _f, _v in zip(numeric_cols, _pi.importances_mean)
        ]
    perm_importance = (
        pl.DataFrame(_rows)
        .group_by("feature")
        .agg(pl.col("importance").mean().alias("mean"), pl.col("importance").std().alias("sd"))
        .with_columns(lo=pl.col("mean") - pl.col("sd"), hi=pl.col("mean") + pl.col("sd"))
        .sort("mean", descending=True)
    )
    _sort = perm_importance["feature"].to_list()
    _bars = alt.Chart(perm_importance).mark_bar(color="#4c78a8").encode(
        x=alt.X("mean:Q", title="Drop in accuracy when shuffled"),
        y=alt.Y("feature:N", sort=_sort, title=None),
        tooltip=[alt.Tooltip("mean", format=".3f"), alt.Tooltip("sd", format=".3f")],
    )
    _err = alt.Chart(perm_importance).mark_rule().encode(
        x="lo:Q", x2="hi:Q", y=alt.Y("feature:N", sort=_sort)
    )
    mo.vstack(
        [
            mo.md(
                "### Permutation importance (out-of-sample)\n\n"
                "Mean drop in held-out accuracy when a feature is shuffled; lines ± 1 sd across folds."
            ),
            (_bars + _err).properties(width=420, height=160),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    RandomForestClassifier,
    alt,
    clone,
    mo,
    np,
    numeric_cols,
    pl,
    shap,
    species_X,
    species_model,
    species_y,
):
    # SHAP on a model fitted to all data: per-penguin contribution of each feature
    # to each class probability.
    species_model_full = clone(species_model).fit(species_X, species_y)
    if isinstance(species_model_full, RandomForestClassifier):
        _explainer = shap.TreeExplainer(species_model_full, feature_names=numeric_cols)
    else:
        _explainer = shap.Explainer(
            species_model_full.predict_proba,
            shap.maskers.Independent(species_X, max_samples=len(species_X)),
            feature_names=numeric_cols,
        )
    species_shap = _explainer(species_X)  # shape: (penguins, features, classes)
    _classes = list(species_model_full.classes_)
    _vals = species_shap.values
    _n, _p, _k = _vals.shape
    _x_scaled = (species_X - species_X.min(0)) / (species_X.max(0) - species_X.min(0))
    shap_long = pl.DataFrame(
        {
            "species": np.repeat(_classes, _n * _p),
            "feature": np.tile(numeric_cols, _n * _k),
            "shap": np.concatenate([_vals[:, :, _c].ravel() for _c in range(_k)]),
            "feature_value": np.tile(_x_scaled.ravel(), _k),
            "jitter": np.random.default_rng(0).uniform(-0.35, 0.35, _n * _p * _k),
        }
    )
    _mean_abs = shap_long.group_by("species", "feature").agg(pl.col("shap").abs().mean().alias("mean_abs_shap"))
    _order = (
        _mean_abs.group_by("feature").agg(pl.col("mean_abs_shap").sum()).sort("mean_abs_shap", descending=True)["feature"].to_list()
    )
    _bar = (
        alt.Chart(_mean_abs)
        .mark_bar()
        .encode(
            x=alt.X("mean_abs_shap:Q", title="mean |SHAP| (probability)"),
            y=alt.Y("feature:N", sort=_order, title=None),
            color=alt.Color("species:N", scale=alt.Scale(scheme="tableau10")),
            tooltip=["species", "feature", alt.Tooltip("mean_abs_shap", format=".3f")],
        )
        .properties(width=420, height=160)
    )
    _swarm = (
        alt.Chart(shap_long)
        .mark_circle(size=14, opacity=0.6)
        .encode(
            x=alt.X("shap:Q", title="SHAP value (change in class probability)"),
            y=alt.Y("feature:N", sort=_order, title=None),
            yOffset=alt.YOffset("jitter:Q", scale=alt.Scale(domain=[-0.5, 0.5])),
            color=alt.Color("feature_value:Q", scale=alt.Scale(scheme="redblue", reverse=True), title="feature value (scaled)"),
            tooltip=["species", "feature", alt.Tooltip("shap", format=".3f")],
        )
        .properties(width=260, height=200)
        .facet(column=alt.Column("species:N", title=None))
    )
    mo.vstack(
        [
            mo.md(
                "### SHAP (model fitted on all data)\n\n"
                "Top: mean |SHAP| per feature, split by species. Bottom: one dot per penguin; "
                "colour is the feature's value (red = high, blue = low)."
            ),
            _bar,
            _swarm,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
    **Conclusion.**

    - **Bill length is the most important feature** in both models (accuracy drops by about
      0.31 when it is shuffled). It separates Adelie from Chinstrap: long bills push the
      prediction towards Chinstrap and away from Adelie.
    - **Bill depth is the main signal for Gentoo** (shallow bills mean Gentoo), helped by
      flipper length and body mass.
    - **Flipper length and body mass are interchangeable.** Logistic regression relies on body
      mass (importance 0.10) and barely uses flipper length (0.03); the random forest does the
      reverse (0.13 vs 0.02), with a large spread across folds. The two are strongly correlated
      (φK = 0.82) and carry the same "large penguin = Gentoo" information, so neither ranking
      means one is biologically more informative than the other.

    This matches Section 5: bill length handles the Adelie/Chinstrap split, and bill depth or
    body size identifies Gentoo.
    """),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Discussion

    **Main findings**

    1. **Species dominates the variation.** Species accounts for most of the variation in every
       body measurement and is almost completely confounded with island. Analyses that pool
       species can give misleading results, such as the reversed sign of the bill length–bill
       depth correlation (Section 2).
    2. **Sexual dimorphism is consistent but not uniform.** Males are larger on every
       measurement in every species. The bill difference is similar across species (about
       8–10%), but the body-mass difference ranges from 12% in Chinstrap to 20% in Adelie
       (Section 4).
    3. **Species is highly predictable.** Four measurements identify species with about 99%
       cross-validated accuracy using a linear model. Bill length separates Adelie from
       Chinstrap; bill depth and body size identify Gentoo (Sections 5–7).

    **Limitations**

    - *Confounding by design.* Island and species cannot be separated, so neither the ANOVA
      nor the classifier can tell whether an effect comes from species or from location.
    - *Missing data.* The 11 penguins without a recorded sex are excluded. That is too few to
      change the results much, but if sex was hard to determine for some kinds of birds (for
      example small or young ones), the excluded penguins may differ systematically.
    - *ANOVA assumptions.* Normality and equal variance of the residuals were not checked.
      With 34–73 penguins per group the F-test is fairly robust, and the per-species Welch
      tests do not assume equal variances.
    - *Model choice.* The two models are compared on the same cross-validation that estimates
      their performance, so the winner's score is slightly optimistic. With only two
      candidates and a one-point gap this matters little; nested cross-validation would
      remove the bias.
    - *Feature importance.* SHAP values describe the fitted model, not causal effects, and
      correlated features (flipper length, body mass) share credit in ways that depend on
      the model.
    - *Year.* The weak association between year and flipper length (Section 3) was not
      followed up.

    **Possible next steps**

    - Add sex as a predictor, or fit separate models per sex, to see whether dimorphism blurs
      the Adelie/Chinstrap boundary.
    - Check the ANOVA residuals.
    - Look at flipper length across years within each species to decide whether the trend is
      biological or a change in measurement protocol.
    """)
    return


if __name__ == "__main__":
    app.run()
