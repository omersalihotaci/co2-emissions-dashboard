# Presentation notes (English) — Global CO₂ Explorer

> Suggested length: ~10–12 min total. Each member presents their **own three charts** live in the dashboard (the course assesses each member individually), so everyone should be able to explain the code behind their charts.

## 0. Opening (Ömer Salih, ~1 min)
- **Question:** *Who emits the world's CO₂, from which fuels, and how is it tied to wealth and energy use?*
- **Data:** Our World in Data CO₂ dataset: 219 countries, 1990–2024, 7,867 rows × 28 columns after cleaning.
- **Roadmap:** 4 tabs = data cleaning → geography & trends → energy sources & relationships → economy & machine learning.

## 1. Data & preprocessing (Ömer Salih, ~1.5 min) — *tab "Data & Preprocessing"*
- Raw file mixes **countries with aggregates** (World, "Asia", income groups…). Aggregates have no ISO code → removed 7,963 rows (otherwise totals are double-counted).
- **Missing values:** interpolation *inside* each country's series, no extrapolation; missing fuel columns → 0 with an "Other / unattributed" remainder so totals add up; GDP has no data after 2022 (kept honest: GDP charts stop at 2022).
- **Outliers:** IQR rule flags 471 rows (Qatar, Bahrain, Kuwait…). They are real, so we **flag and keep** them and use **log scales** instead of deleting.

## 2. Ömer Salih — Charts 1–3 (Geography & Trends, ~3 min)
| Chart | What to say |
|---|---|
| **1 Animated choropleth** | Press ▶. Gulf states, Australia, US, Canada are darkest; most of Africa < 1 t per person. Log colour scale because the distribution is very skewed. Metric dropdown changes the map. |
| **2 Sunburst** | Click *Asia* → Asia ≈ 62% of world CO₂, China ≈ 53% of Asia. Hierarchy = continent → country. |
| **3 Line chart + range slider** | China overtook the US in 2006; India passed Russia. Radio switches the ranking; the ≥ 1 M population checkbox hides micro-states. |

## 3. Erdoğan — Charts 4–6 (Energy Sources & Relationships, ~3 min)
| Chart | What to say |
|---|---|
| **4 Sankey** | Fuel → continent. Coal is the biggest source and ~84% of it is burned in Asia; drag nodes, hover flows. |
| **5 Radar** | Fuel-mix "fingerprints": China/India coal-shaped, Qatar ≈ 94% gas, US/Germany balanced. Pick countries in the multiselect. |
| **6 Correlation heatmap** | Energy/cap vs CO₂/cap ρ ≈ 0.95; GDP/cap ρ ≈ 0.90. We default to **Spearman** because of heavy outliers; switch to Pearson to compare. |

## 4. Abdullah — Charts 7–9 (Economy & ML, ~3.5 min)
| Chart | What to say |
|---|---|
| **7 Animated bubble** | GDP/cap vs CO₂/cap, bubble = population. China/India move up-right; UK, Germany, France, Sweden move right but **down** → decoupling. |
| **8 K-Means + parallel coordinates (ML #1)** | Standardised log-features, k slider, silhouette curve. 4 readable clusters (low-income oil-only / oil-based middle-income / gas-heavy / coal-heavy Asia). Honest note: silhouette peaks at k=2; we chose k=4 for interpretability. Brush an axis to filter. |
| **9 Regression (ML #2)** | Random Forest vs Linear Regression. **Grouped split by country** → tested on unseen countries (R² ≈ 0.96, ≈ 0.88 without energy). Feature importance + diverging error chart show where the model is wrong. |

## 5. Technical challenges (any member, ~1 min)
1. **Aggregates inside the data** → filter by ISO code, otherwise world totals are counted twice.
2. **Skewed distributions** → log scales, Spearman correlation, IQR flagging instead of deleting.
3. **Data leakage** → rows of the same country in train *and* test inflate R²; fixed with `GroupShuffleSplit` by country.
4. **Missing GDP after 2022** → interpolate only inside series, restrict GDP-based analysis to ≤ 2022 and say so in the UI.
5. **Performance** → ML steps cached with `st.cache_data` (first load ≈ 6 s, widget changes ≈ 1 s).
6. **Parallel-coordinates colours** → Plotly treats cluster ids as continuous; built a stepped colour scale for discrete clusters.

## 6. Closing (Abdullah, ~30 s)
- Main message: **emissions follow energy use, which follows income — but rich countries *can* decouple.**
- Limitations: GDP ends 2022; production-based CO₂ (not consumption-based); clusters are exploratory.
- Future work: consumption-based emissions, per-sector data, forecasting.

## Likely questions — be ready
- *Why Plotly and not matplotlib/seaborn?* Native hover, zoom, animation, and brushing; assignment requires interactivity in every chart.
- *Why keep outliers?* They are real (petro-states); deleting them would remove the most interesting countries.
- *Why log-transform before K-Means / regression?* GDP, energy and CO₂ per capita span orders of magnitude; without logs a few countries dominate the distance metric.
- *Why is the R² so high?* Energy use and CO₂ are physically linked; without the energy feature it drops to ≈ 0.88. That's why the checkbox exists.
- *Why silhouette = 0.34 only?* Countries form a continuum, not separated blobs; clusters are descriptive.
- *What is Spearman vs Pearson?* Rank vs linear correlation; Spearman is robust to outliers.
- *How are the 9 charts different?* Each answers a different question (where / who / when / which fuel / what mix / which indicators / growth vs emissions / which groups / can we predict).
