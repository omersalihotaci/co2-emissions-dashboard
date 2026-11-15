"""CEN445 - Exploratory Data Visualization Dashboard
Global CO2 Emissions, Energy and Economy (Our World in Data)

Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from preprocess import FUEL_LABELS, FUELS, load_and_clean

st.set_page_config(page_title="Global CO2 Explorer", page_icon="🌍", layout="wide")

# --------------------------------------------------------------------------
# Shared visual language: one template, one continent palette everywhere
# --------------------------------------------------------------------------
TEMPLATE = "plotly_white"
CONTINENT_COLORS = {
    "Asia": "#E45756", "Europe": "#4C78A8", "North America": "#F58518",
    "South America": "#54A24B", "Africa": "#B279A2", "Oceania": "#72B7B2",
}
FUEL_COLORS = {
    "Coal": "#3B3B3B", "Oil": "#9C755F", "Gas": "#4C78A8",
    "Cement": "#BAB0AC", "Flaring": "#F58518", "Other / unattributed": "#D3D3D3",
}
METRICS = {
    "co2_per_capita": "CO₂ per capita (t)",
    "co2": "Total CO₂ (Mt)",
    "co2_per_gdp": "CO₂ per GDP (kg per $)",
    "energy_per_capita": "Energy per capita (kWh)",
    "share_global_co2": "Share of global CO₂ (%)",
    "gdp_per_capita": "GDP per capita ($, 2011 PPP)",
    "population": "Population",
    "methane_per_capita": "Methane per capita (t CO₂e)",
}
LAST_GDP_YEAR = 2022


def insight(text: str):
    st.info(f"**Insight:** {text}", icon="💡")


@st.cache_data
def get_data():
    return load_and_clean()


@st.cache_data(show_spinner=False)
def run_kmeans(Xs: np.ndarray, k: int):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Xs)
    return labels, silhouette_score(Xs, labels)


@st.cache_data(show_spinner=False)
def silhouette_curve(Xs: np.ndarray, ks=tuple(range(2, 9))):
    return [run_kmeans(Xs, k)[1] for k in ks]


@st.cache_data(show_spinner="Training model…")
def fit_regressor(X: pd.DataFrame, y: pd.Series, groups: pd.Series, model_name: str):
    """Train on 75% of the COUNTRIES and predict the other 25% (never seen in training)."""
    tr, te = next(GroupShuffleSplit(test_size=0.25, random_state=42).split(X, y, groups=groups))
    model = (RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1)
             if model_name == "Random Forest" else LinearRegression())
    model.fit(X.iloc[tr], y.iloc[tr])
    pred = model.predict(X.iloc[te])
    if model_name == "Random Forest":
        imp = model.feature_importances_
    else:  # standardised coefficient magnitude
        imp = np.abs(model.coef_) * X.iloc[tr].std().values
    return tr, te, pred, pd.Series(imp, index=X.columns)


df, report = get_data()

# --------------------------------------------------------------------------
# Sidebar - global interactive components
# --------------------------------------------------------------------------
st.sidebar.title("🌍 Filters")
min_y, max_y = int(df["year"].min()), int(df[df["energy_per_capita"].notna()]["year"].max())
year_range = st.sidebar.slider("Year range", min_y, max_y, (min_y, max_y))
focus_year = year_range[1]
continents = st.sidebar.multiselect(
    "Continents", list(CONTINENT_COLORS), default=list(CONTINENT_COLORS)
)
metric = st.sidebar.selectbox(
    "Map metric", list(METRICS)[:5], format_func=METRICS.get
)
log_scale = st.sidebar.checkbox("Logarithmic scales", value=True)
top_n = st.sidebar.slider("Top-N countries", 5, 20, 10)
st.sidebar.caption(
    f"Snapshot charts use the last year of the range (**{focus_year}**). "
    f"GDP data ends in {LAST_GDP_YEAR}."
)

if not continents:
    st.warning("Select at least one continent.")
    st.stop()

fdf = df[df["year"].between(*year_range) & df["continent"].isin(continents)]
snap = fdf[fdf["year"] == focus_year]
st.sidebar.download_button(
    "⬇️ Download filtered data (CSV)", fdf.to_csv(index=False).encode("utf-8"),
    file_name="co2_filtered.csv", mime="text/csv", width="stretch",
)

# --------------------------------------------------------------------------
# Header + KPIs
# --------------------------------------------------------------------------
st.title("Global CO₂ Emissions, Energy & Economy")
st.caption(
    "Exploratory dashboard on the Our World in Data CO₂ dataset · "
    f"{df['country'].nunique()} countries · {df['year'].min()}–{df['year'].max()}"
)
with st.expander("ℹ️ How to use this dashboard", expanded=False):
    st.markdown("""
- **Sidebar filters** (year range, continents, log scale, Top-N, map metric) update every chart.
- **Tabs:** *Data & Preprocessing* shows how the raw data was cleaned · *Geography & Trends* (charts 1–3) ·
  *Energy Sources & Relationships* (charts 4–6) · *Economy & Machine Learning* (charts 7–9).
- **Every chart is interactive:** hover for tooltips, drag to zoom, double-click to reset, click legend items to hide or
  isolate a series, press ▶ on animated charts, click a sunburst slice to drill down, drag Sankey nodes, and drag along a
  parallel-coordinates axis to brush-filter.
- Each chart has a 💡 **Insight** box with the main takeaway.
""")

k1, k2, k3, k4 = st.columns(4)
first = fdf[fdf["year"] == year_range[0]]["co2"].sum()
last = snap["co2"].sum()
k1.metric(f"Total CO₂ {focus_year}", f"{last/1000:,.1f} Gt",
          f"{(last/first-1)*100:+.0f}% vs {year_range[0]}" if first else None)
k2.metric("Countries in view", snap["country"].nunique())
top_row = snap.nlargest(1, "co2")
k3.metric("Largest emitter", top_row["country"].iloc[0] if len(top_row) else "–")
w = snap.dropna(subset=["co2_per_capita"])
k4.metric("Weighted CO₂ per capita",
          f"{(w['co2'].sum()*1e6/w['population'].sum()):.2f} t" if len(w) else "–")

tab0, tab1, tab2, tab3 = st.tabs([
    "🧹 Data & Preprocessing",
    "🗺️ Geography & Trends",
    "⚡ Energy Sources & Relationships",
    "🤖 Economy & Machine Learning",
])

# ==========================================================================
# TAB 0 - Data quality / preprocessing
# ==========================================================================
with tab0:
    st.subheader("Dataset & cleaning steps")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"""
- **Raw file:** {report['raw_shape'][0]:,} rows × {report['raw_shape'][1]} selected columns (79 in the source file)
- **Removed aggregates** (World, continents, income groups, shipping…): {report['aggregates_removed']:,} rows
- **Filtered to {df['year'].min()}+** → {report['after_filter_shape'][0]:,} rows
- **No continent** (dropped): {', '.join(report['no_continent'])}
- **Linear interpolation** of population, GDP and energy *inside* each country's series (no extrapolation)
- **Fuel breakdown:** missing → 0, remainder kept as *Other / unattributed* so totals add up
- **Rows without CO₂** dropped: {report['no_co2_rows']}
- **Derived:** GDP per capita, fuel shares (%)
- **Outliers (IQR rule):** CO₂ per capita > {report['iqr_upper']:.1f} t → {report['outlier_rows']} rows flagged, **kept** (they are real)
- **Final:** {report['clean_shape'][0]:,} rows × {report['clean_shape'][1]} columns
""")
    with c2:
        miss = pd.DataFrame({
            "Before": report["missing_before"], "After": report["missing_after"]
        }).dropna().reset_index(names="column").melt("column", var_name="stage", value_name="missing %")
        fig = px.bar(miss, x="missing %", y="column", color="stage", barmode="group",
                     orientation="h", template=TEMPLATE, title="Missing values before vs after cleaning",
                     color_discrete_map={"Before": "#BAB0AC", "After": "#4C78A8"})
        fig.update_layout(height=430, yaxis_title=None, legend_title=None)
        st.plotly_chart(fig, width="stretch")
    st.caption(
        "GDP stays ~31% missing: the source has no GDP after 2022 and none for some small territories. "
        "GDP-based charts and models therefore use years ≤ 2022."
    )
    fig = px.box(fdf[fdf["co2_per_capita"] > 0], x="continent", y="co2_per_capita", color="continent", points="outliers",
                 hover_name="country", hover_data=["year"], template=TEMPLATE,
                 color_discrete_map=CONTINENT_COLORS, log_y=log_scale,
                 title="Outlier check: CO₂ per capita by continent (IQR whiskers)",
                 labels={"co2_per_capita": METRICS["co2_per_capita"], "continent": ""})
    fig.add_hline(y=report["iqr_upper"], line_dash="dot", annotation_text="global IQR upper fence")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")
    with st.expander("Most frequently flagged outlier countries / raw data preview"):
        a, b = st.columns([1, 2])
        a.dataframe(report["outlier_countries"].rename("flagged years").head(15))
        b.dataframe(fdf.head(200), height=380)

# ==========================================================================
# TAB 1 - Member 1: Geography & Trends
# ==========================================================================
with tab1:
    # --- Chart 1: animated choropleth map --------------------------------
    st.subheader("1 · How emissions spread across the world over time")
    map_df = fdf.dropna(subset=[metric]).sort_values("year")
    lo, hi = map_df[metric].quantile([0.02, 0.98])
    color_col = metric
    if log_scale and metric != "share_global_co2":
        map_df = map_df.assign(_log=np.log10(map_df[metric].clip(lower=max(lo, 1e-3))))
        color_col, lo, hi = "_log", np.log10(max(lo, 1e-3)), np.log10(hi)
    fig = px.choropleth(
        map_df, locations="iso_code", color=color_col, hover_name="country",
        hover_data={metric: ":,.2f", "year": True, "iso_code": False, color_col: False},
        animation_frame="year", range_color=(lo, hi), color_continuous_scale="YlOrRd",
        template=TEMPLATE, labels={metric: METRICS[metric]},
        title=f"{METRICS[metric]} by country{' (log colour scale)' if color_col == '_log' else ''}",
    )
    cbar = dict(title="")
    if color_col == "_log":  # label the log colour bar with real values
        ticks = [v for v in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]
                 if lo <= np.log10(v) <= hi]
        cbar = dict(title="", tickvals=np.log10(ticks), ticktext=[f"{t:,g}" for t in ticks])
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=50, b=0), coloraxis_colorbar=cbar)
    fig.update_geos(showframe=False, projection_type="natural earth", bgcolor="rgba(0,0,0,0)",
                    showcoastlines=False)
    st.plotly_chart(fig, width="stretch")
    insight("Per-capita emissions are highest in the Gulf petro-states (Qatar, Bahrain, Kuwait), "
            "Australia, the US and Canada, while most of Sub-Saharan Africa stays below 1 t per person. "
            "Press ▶ to watch Asia darken after 2000.")

    c1, c2 = st.columns(2)
    # --- Chart 2: sunburst -------------------------------------------------
    with c1:
        st.subheader(f"2 · Who emits the CO₂? ({focus_year})")
        sb = snap[snap["co2"] > 0]
        fig = px.sunburst(
            sb, path=["continent", "country"], values="co2", color="continent",
            color_discrete_map=CONTINENT_COLORS, template=TEMPLATE,
            hover_data={"co2_per_capita": ":.2f"},
            title="Continent → country, size = total CO₂ (click to zoom)",
        )
        fig.update_traces(hovertemplate="<b>%{label}</b><br>CO₂: %{value:,.0f} Mt"
                                        "<br>Share of parent: %{percentParent:.1%}<extra></extra>")
        fig.update_layout(height=520, margin=dict(t=50, l=0, r=0, b=0))
        st.plotly_chart(fig, width="stretch")
        insight("Asia alone produces more than half of all CO₂, and China is roughly half of Asia. "
                "Click a continent to drill down.")

    # --- Chart 3: line chart with range slider -----------------------------
    with c2:
        st.subheader("3 · Top emitters over time")
        line_metric = st.radio("Rank by", ["co2", "co2_per_capita", "share_global_co2"],
                               format_func=METRICS.get, horizontal=True, key="line_metric")
        big_only = st.checkbox("Only countries with ≥ 1 million people", value=True,
                               help="Hides micro-states whose per-capita values are extreme but globally tiny.")
        pool = snap[snap["population"] >= 1e6] if big_only else snap
        top = pool.nlargest(top_n, line_metric)["country"]
        ts = fdf[fdf["country"].isin(top)]
        fig = px.line(ts, x="year", y=line_metric, color="country", template=TEMPLATE, markers=False,
                      log_y=log_scale, labels={line_metric: METRICS[line_metric], "year": "Year"},
                      category_orders={"country": list(top)},
                      title=f"Top {top_n} countries by {METRICS[line_metric]} in {focus_year}")
        fig.update_layout(height=520, hovermode="x unified", legend_title=None,
                          xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fig, width="stretch")
        insight("With ‘Total CO₂’: China overtook the United States in 2006 and now emits more than "
                "twice as much; India passed Russia to become #3. The US and EU members trend downward.")

# ==========================================================================
# TAB 2 - Member 2: Energy sources & relationships
# ==========================================================================
with tab2:
    # --- Chart 4: Sankey ---------------------------------------------------
    st.subheader(f"4 · From fuel to continent: where CO₂ comes from ({focus_year})")
    src_cols = FUELS + ["other_co2"]
    flows = snap.groupby("continent")[src_cols].sum()
    fuels = [FUEL_LABELS[c] for c in src_cols]
    conts = list(flows.index)
    nodes = fuels + conts
    s, t, v, col = [], [], [], []
    for i, fc in enumerate(src_cols):
        for j, ct in enumerate(conts):
            val = flows.loc[ct, fc]
            if val > 0:
                s.append(i); t.append(len(fuels) + j); v.append(val)
                c = CONTINENT_COLORS[ct].lstrip("#")
                col.append(f"rgba({int(c[0:2],16)},{int(c[2:4],16)},{int(c[4:6],16)},0.45)")
    fig = go.Figure(go.Sankey(
        node=dict(label=nodes, pad=18, thickness=18,
                  color=[FUEL_COLORS[f] for f in fuels] + [CONTINENT_COLORS[c] for c in conts],
                  hovertemplate="%{label}: %{value:,.0f} Mt<extra></extra>"),
        link=dict(source=s, target=t, value=v, color=col,
                  hovertemplate="%{source.label} → %{target.label}: %{value:,.0f} Mt<extra></extra>"),
    ))
    fig.update_layout(template=TEMPLATE, height=480, title="CO₂ by fuel source flowing to continents (Mt) — drag nodes to rearrange")
    st.plotly_chart(fig, width="stretch")
    insight("Coal is the single largest source globally and almost all of it is burned in Asia. "
            "Europe and the Americas emit mainly from oil and gas.")

    c1, c2 = st.columns(2)
    # --- Chart 5: radar / polar fuel mix ----------------------------------
    with c1:
        st.subheader(f"5 · Fuel-mix fingerprints ({focus_year})")
        default = [c for c in ["China", "United States", "India", "Germany", "Turkey", "Qatar"]
                   if c in set(snap["country"])]
        picks = st.multiselect("Countries to compare", sorted(snap["country"].unique()),
                               default=default, key="radar")
        share_cols = [c + "_share" for c in src_cols]
        rad = snap[snap["country"].isin(picks)].melt(
            id_vars="country", value_vars=share_cols, var_name="source", value_name="share")
        rad["source"] = rad["source"].str.replace("_share", "").map(FUEL_LABELS)
        fig = px.line_polar(rad, r="share", theta="source", color="country", line_close=True,
                            template=TEMPLATE, title="Share of national CO₂ by source (%)")
        fig.update_traces(fill="toself", opacity=0.55)
        fig.update_layout(height=500, legend_title=None,
                          polar=dict(radialaxis=dict(ticksuffix="%", range=[0, 100])))
        st.plotly_chart(fig, width="stretch")
        insight("China and India are coal-shaped, Qatar is a gas economy, while the US and Germany "
                "have a balanced oil–gas–coal mix. Click legend items to isolate a country.")

    # --- Chart 6: correlation heatmap --------------------------------------
    with c2:
        st.subheader("6 · How indicators move together")
        corr_cols = ["co2_per_capita", "gdp_per_capita", "energy_per_capita", "co2_per_gdp",
                     "methane_per_capita", "coal_co2_share", "oil_co2_share", "gas_co2_share"]
        method = st.radio("Correlation method", ["spearman", "pearson"], horizontal=True,
                          help="Spearman is rank-based and robust to the heavy outliers.")
        cm = fdf[corr_cols].corr(method=method)
        nice = dict(zip(corr_cols, ["CO₂/cap", "GDP/cap", "Energy/cap", "CO₂/GDP",
                                    "Methane/cap", "Coal %", "Oil %", "Gas %"]))
        cm = cm.rename(index=nice, columns=nice)
        fig = px.imshow(cm, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                        template=TEMPLATE, aspect="auto", title=f"{method.capitalize()} correlation matrix")
        fig.update_layout(height=520)
        st.plotly_chart(fig, width="stretch")
        insight("Energy use per person is the strongest correlate of CO₂ per person, and wealth "
                "(GDP per capita) is close behind (Spearman ≈ 0.9). CO₂ per unit of GDP behaves differently: "
                "rich countries are not necessarily carbon-intensive per dollar.")

# ==========================================================================
# TAB 3 - Member 3: Economy & Machine Learning
# ==========================================================================
with tab3:
    gdf = fdf[fdf["year"] <= LAST_GDP_YEAR].dropna(subset=["gdp_per_capita", "co2_per_capita", "population"])
    ml_year = min(focus_year, LAST_GDP_YEAR)

    # --- Chart 7: animated bubble ----------------------------------------
    st.subheader("7 · Getting richer vs emitting more")
    if gdf.empty:
        st.warning("No GDP data in the selected range (GDP ends in 2022).")
    else:
        fig = px.scatter(
            gdf.sort_values("year"), x="gdp_per_capita", y="co2_per_capita", size="population",
            color="continent", hover_name="country", animation_frame="year", size_max=55,
            log_x=log_scale, log_y=log_scale, color_discrete_map=CONTINENT_COLORS, template=TEMPLATE,
            range_x=[gdf.gdp_per_capita.min() * 0.8, gdf.gdp_per_capita.max() * 1.2],
            range_y=[max(gdf.co2_per_capita.min(), 0.01) * 0.8 if log_scale else 0,
                     gdf.co2_per_capita.max() * 1.2],
            labels={"gdp_per_capita": METRICS["gdp_per_capita"], "co2_per_capita": METRICS["co2_per_capita"]},
            title="GDP per capita vs CO₂ per capita (bubble = population) — press ▶",
        )
        fig.update_layout(height=580, legend_title=None)
        st.plotly_chart(fig, width="stretch")
        insight("Countries line up along a clear upward diagonal: richer means more CO₂ per person. "
                "Over time China and India travel up-and-right, while many European countries move right "
                "but *down*, a sign of decoupling growth from emissions.")

    # --- Chart 8: K-Means + parallel coordinates (ML #1) ------------------
    st.subheader(f"8 · ML #1 — K-Means clustering of country profiles ({ml_year})")
    c1, c2 = st.columns([1, 3])
    feats = {
        "gdp_per_capita": "log GDP/cap", "co2_per_capita": "log CO₂/cap",
        "energy_per_capita": "log energy/cap", "coal_co2_share": "Coal %",
        "oil_co2_share": "Oil %", "gas_co2_share": "Gas %",
    }
    ml_conts = continents
    cl = df[(df["year"] == ml_year) & df["continent"].isin(ml_conts)].dropna(subset=list(feats)).copy()
    if len(cl) < 20:  # too few countries for meaningful clustering / train-test split
        ml_conts = list(CONTINENT_COLORS)
        cl = df[df["year"] == ml_year].dropna(subset=list(feats)).copy()
        st.warning("Only a handful of countries match the continent filter, so both ML models below "
                   "use all continents.")
    with c1:
        k = st.slider("Number of clusters (k)", 2, 8, 4)
        X = cl[list(feats)].copy()
        for c in ["gdp_per_capita", "co2_per_capita", "energy_per_capita"]:
            X[c] = np.log10(X[c].clip(lower=1e-3))
        Xs = StandardScaler().fit_transform(X)
        labels, sil = run_kmeans(Xs, k)
        cl["cluster"] = labels
        st.metric("Silhouette score", f"{sil:.2f}", help="−1…1, higher = better separated clusters")
        st.metric("Countries clustered", len(cl))
        ks = list(range(2, 9))
        sils = silhouette_curve(Xs)
        f2 = px.line(x=ks, y=sils, markers=True, template=TEMPLATE,
                     labels={"x": "k", "y": "silhouette"}, title="Choosing k")
        f2.add_vline(x=k, line_dash="dot")
        f2.update_layout(height=250, margin=dict(t=40, l=0, r=0, b=0))
        st.plotly_chart(f2, width="stretch")
    with c2:
        X_disp = X.rename(columns=feats).assign(cluster=cl["cluster"].values)
        palette = px.colors.qualitative.Bold
        stepped = []  # discrete (stepped) colour scale: one flat colour per cluster
        for i in range(k):
            stepped += [[i / k, palette[i]], [(i + 1) / k, palette[i]]]
        fig = px.parallel_coordinates(
            X_disp, color="cluster", dimensions=list(feats.values()),
            color_continuous_scale=stepped, range_color=(-0.5, k - 0.5),
            template=TEMPLATE, title="Country profiles coloured by cluster — drag along an axis to filter",
        )
        fig.update_layout(height=470, margin=dict(t=90),
                          coloraxis_colorbar=dict(title="cluster", tickvals=list(range(k))))
        st.plotly_chart(fig, width="stretch")
    summ = cl.groupby("cluster").agg(
        countries=("country", "count"),
        examples=("country", lambda s: ", ".join(cl.loc[s.index].nlargest(4, "population")["country"])),
        gdp_per_capita=("gdp_per_capita", "median"), co2_per_capita=("co2_per_capita", "median"),
        coal_share=("coal_co2_share", "median"), gas_share=("gas_co2_share", "median"),
    ).round(1)
    st.dataframe(summ, width="stretch")
    insight("Without any labels, K-Means (k = 4, 2022) finds four readable groups: low-income, low-emission "
            "countries whose small CO₂ comes almost only from oil (Nigeria, Ethiopia); oil-dominated "
            "middle-income economies (Brazil, Mexico, France); gas-heavy emitters (United States, Russia, Iran); "
            "and coal-heavy Asian economies (China, India, Japan). The silhouette score peaks at k = 2 "
            "(just rich vs poor), so k = 4 trades a little separation for a much more informative split.")

    # --- Chart 9: Regression (ML #2) ---------------------------------------
    st.subheader("9 · ML #2 — Can we predict a country's CO₂ per capita?")
    reg = df[(df["year"] <= LAST_GDP_YEAR) & df["continent"].isin(ml_conts)].dropna(
        subset=["gdp_per_capita", "energy_per_capita", "co2_per_capita"]).copy()
    reg = reg[reg["co2_per_capita"] > 0]
    feat_cols = ["log_gdp_pc", "log_energy_pc", "coal_co2_share", "gas_co2_share", "oil_co2_share", "year"]
    reg["log_gdp_pc"] = np.log10(reg["gdp_per_capita"])
    reg["log_energy_pc"] = np.log10(reg["energy_per_capita"].clip(lower=1))
    cont_dummies = pd.get_dummies(reg["continent"], prefix="cont", dtype=float)
    Xr = pd.concat([reg[feat_cols], cont_dummies], axis=1)
    yr = np.log10(reg["co2_per_capita"])

    c1, c2 = st.columns([1, 3])
    with c1:
        model_name = st.selectbox("Model", ["Random Forest", "Linear Regression"])
        use_energy = st.checkbox("Include energy use as a feature", value=True,
                                 help="Turn off to see how well wealth and fuel mix alone explain emissions.")
        if not use_energy:
            Xr = Xr.drop(columns=["log_energy_pc"])
        # Split by COUNTRY so the model is tested on countries it has never seen (no leakage
        # from neighbouring years of the same country).
        tr, te, pred, imp = fit_regressor(Xr, yr, reg["country"], model_name)
        st.metric("R² (test countries)", f"{r2_score(yr.iloc[te], pred):.2f}")
        st.metric("MAE", f"×{10**mean_absolute_error(yr.iloc[te], pred):.2f}",
                  help="Typical multiplicative error on CO₂ per capita (log₁₀ model)")
        st.caption(f"{len(tr):,} training rows · {len(te):,} test rows · "
                   f"{reg['country'].iloc[te].nunique()} unseen test countries")
    with c2:
        res = reg.iloc[te].assign(predicted=10**pred)
        fig = px.scatter(res, x="co2_per_capita", y="predicted", color="continent", hover_name="country",
                         hover_data=["year"], log_x=True, log_y=True, opacity=0.6,
                         color_discrete_map=CONTINENT_COLORS, template=TEMPLATE,
                         labels={"co2_per_capita": "Actual CO₂ per capita (t)", "predicted": "Predicted (t)"},
                         title=f"{model_name}: predicted vs actual on unseen countries")
        lim = [res[["co2_per_capita", "predicted"]].min().min(), res[["co2_per_capita", "predicted"]].max().max()]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(dash="dash", color="grey"),
                                 name="perfect prediction"))
        fig.update_layout(height=430, legend_title=None)
        st.plotly_chart(fig, width="stretch")
    imp = (imp / imp.sum()).sort_values()
    pretty = {"log_gdp_pc": "GDP per capita (log)", "log_energy_pc": "Energy per capita (log)",
              "coal_co2_share": "Coal share", "gas_co2_share": "Gas share", "oil_co2_share": "Oil share",
              "year": "Year"}
    imp.index = [pretty.get(i, i.replace("cont_", "Continent: ")) for i in imp.index]

    # Per-country prediction error on unseen countries (mean of log2 predicted/actual).
    err = res.assign(log2_err=np.log2(res["predicted"] / res["co2_per_capita"]))
    err = err.groupby("country")["log2_err"].mean().sort_values()
    n_show = min(8, len(err) // 2)
    err_show = pd.concat([err.head(n_show), err.tail(n_show)]).clip(-3, 3)

    b1, b2 = st.columns(2)
    with b1:
        fig = px.bar(x=imp.values, y=imp.index, orientation="h", template=TEMPLATE,
                     labels={"x": "relative importance", "y": ""},
                     title="Which features drive the prediction?")
        fig.update_layout(height=60 + 28 * max(len(imp), 10), showlegend=False)
        fig.update_yaxes(dtick=1)
        fig.update_traces(marker_color="#4C78A8", hovertemplate="%{y}: %{x:.1%}<extra></extra>")
        st.plotly_chart(fig, width="stretch")
    with b2:
        fig = px.bar(x=err_show.values, y=err_show.index, orientation="h", template=TEMPLATE,
                     color=err_show.values > 0,
                     color_discrete_map={True: "#E45756", False: "#4C78A8"},
                     labels={"x": "prediction error", "y": ""},
                     title="Hardest countries to predict (diverging error)")
        fig.update_traces(customdata=np.round(2 ** err_show.values, 2),
                          hovertemplate="%{y}: model says ×%{customdata} of the real value<extra></extra>")
        fig.update_xaxes(tickvals=[-3, -2, -1, 0, 1, 2, 3],
                         ticktext=["÷8", "÷4", "÷2", "exact", "×2", "×4", "×8"], range=[-3.2, 3.2])
        fig.update_yaxes(dtick=1)
        fig.update_layout(height=60 + 28 * max(len(imp), 10), showlegend=False)
        st.plotly_chart(fig, width="stretch")
    over =", ".join(err.tail(3).index[::-1])
    under = ", ".join(err.head(3).index)
    insight("With energy use included, the Random Forest reaches R² ≈ 0.96 on countries it never saw, and "
            "energy per person carries almost all the importance. Untick it: wealth alone still gives "
            "R² ≈ 0.88, with GDP per capita as the dominant driver. Emissions are mostly a story of how much "
            f"energy a society uses, which in turn follows income. For the current settings, the model "
            f"over-estimates {over} and under-estimates {under}.")

st.divider()
st.caption("Data: Our World in Data — CO₂ and Greenhouse Gas Emissions (github.com/owid/co2-data), "
           "based on the Global Carbon Project, Energy Institute and Maddison Project. CEN445 Assignment 2026–27.")
