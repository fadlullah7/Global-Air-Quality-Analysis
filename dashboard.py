import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ── Load & Prep ───────────────────────────────────────────────────────────────
CSV_PATH = r"c:\Users\user\OneDrive\Documents\TUGAS SMT 4\visdat\kelompok uas\global_air_quality_data_10000.csv"

df = pd.read_csv(CSV_PATH)
df["Date"]       = pd.to_datetime(df["Date"])
df["Month"]      = df["Date"].dt.month
df["Month_Name"] = df["Date"].dt.strftime("%b")
df["Year"]       = df["Date"].dt.year

POLLUTANTS  = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
COUNTRIES   = sorted(df["Country"].unique().tolist())
CITIES      = sorted(df["City"].unique().tolist())
YEARS       = [str(y) for y in sorted(df["Year"].unique().tolist())]

MONTH_ORDER = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
COLOR_SEQ   = px.colors.qualitative.Bold

# ── Koordinat kota ────────────────────────────────────────────────────────────
CITY_COORDS = {
    "Bangkok":        (13.7563,  100.5018),
    "Beijing":        (39.9042,  116.4074),
    "Berlin":         (52.5200,   13.4050),
    "Cairo":          (30.0444,   31.2357),
    "Dubai":          (25.2048,   55.2708),
    "Istanbul":       (41.0082,   28.9784),
    "Johannesburg":   (-26.2041,  28.0473),
    "London":         (51.5074,   -0.1278),
    "Los Angeles":    (34.0522, -118.2437),
    "Madrid":         (40.4168,   -3.7038),
    "Mexico City":    (19.4326,  -99.1332),
    "Moscow":         (55.7558,   37.6173),
    "Mumbai":         (19.0760,   72.8777),
    "New York":       (40.7128,  -74.0060),
    "Paris":          (48.8566,    2.3522),
    "Rio de Janeiro": (-22.9068, -43.1729),
    "Seoul":          (37.5665,  126.9780),
    "Sydney":         (-33.8688, 151.2093),
    "Tokyo":          (35.6762,  139.6503),
    "Toronto":        (43.6532,  -79.3832),
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def empty_fig(msg="Tidak ada data untuk filter yang dipilih"):
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5,
                       xref="paper", yref="paper",
                       showarrow=False, font=dict(size=16, color="gray"))
    fig.update_layout(template="plotly_white")
    return fig

def apply_filters(data, sel_years, sel_months):
    out = data
    if sel_years:
        out = out[out["Year"].isin([int(y) for y in sel_years])]
    if sel_months:
        nums = [MONTH_ORDER.index(m) + 1 for m in sel_months]
        out = out[out["Month"].isin(nums)]
    return out

def plabel(sel_years, sel_months):
    y = ", ".join(sorted(sel_years)) if sel_years else "Semua Tahun"
    m = ", ".join(sel_months)        if sel_months else "Semua Bulan"
    return f"{y} | {m}"


# ════════════════════════════════════════════════════════════════════════════
# KPI Cards
# ════════════════════════════════════════════════════════════════════════════
def kpi_cards():
    avg_pm25   = df["PM2.5"].mean()
    max_pm25   = df["PM2.5"].max()
    worst_city = df.groupby("City")["PM2.5"].mean().idxmax()
    best_city  = df.groupby("City")["PM2.5"].mean().idxmin()
    total_rec  = len(df)
    s = ("display:inline-block;width:18%;margin:6px;padding:16px 10px;"
         "border-radius:12px;text-align:center;font-family:Arial,sans-serif;"
         "box-shadow:0 2px 8px rgba(0,0,0,.15);")
    def c(title, val, col):
        return (f'<div style="{s}background:{col};">'
                f'<div style="font-size:13px;color:#fff;opacity:.9;">{title}</div>'
                f'<div style="font-size:22px;font-weight:700;color:#fff;margin-top:6px;">{val}</div>'
                f'</div>')
    html  = '<div style="text-align:center;padding:10px;">'
    html += c(" Total Records",        f"{total_rec:,}",        "#3b82f6")
    html += c(" Rata-rata PM2.5",       f"{avg_pm25:.1f} µg/m³", "#f59e0b")
    html += c("⚠️ Maks PM2.5",            f"{max_pm25:.1f} µg/m³", "#ef4444")
    html += c("️ Kota Paling Tercemar", worst_city,              "#8b5cf6")
    html += c(" Kota Paling Bersih",    best_city,               "#10b981")
    html += '</div>'
    return html


# ════════════════════════════════════════════════════════════════════════════
# Tab 1 – Rankings: bar chart top N kota
# ════════════════════════════════════════════════════════════════════════════
def bar_city(pollutant, top_n, sel_years, sel_months):
    sub = apply_filters(df, sel_years, sel_months)
    if sub.empty: return empty_fig()
    grp = (sub.groupby("City")[pollutant].mean()
              .sort_values(ascending=False).head(int(top_n)).reset_index())
    grp.columns = ["Kota", pollutant]
    fig = px.bar(grp, x=pollutant, y="Kota", orientation="h",
                 color=pollutant, color_continuous_scale="Reds",
                 title=f"Top {top_n} Kota – {pollutant}  [{plabel(sel_years,sel_months)}]",
                 labels={pollutant: f"{pollutant} (µg/m³)"},
                 template="plotly_white")
    fig.update_layout(yaxis=dict(autorange="reversed"),
                      coloraxis_showscale=False, title_font_size=16)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# Tab 2 – Perbandingan Kota
# ════════════════════════════════════════════════════════════════════════════

# 2a. Radar chart – profil semua polutan per kota
def radar_kota(sel_cities, sel_years, sel_months):
    sub = apply_filters(df, sel_years, sel_months)
    sub = sub[sub["City"].isin(sel_cities)] if sel_cities else sub
    if sub.empty: return empty_fig()

    means = sub.groupby("City")[POLLUTANTS].mean()
    # normalisasi 0-100 agar skala antar polutan sebanding
    norm = (means - means.min()) / (means.max() - means.min() + 1e-9) * 100

    fig = go.Figure()
    for i, city in enumerate(norm.index):
        vals = norm.loc[city].tolist()
        vals += [vals[0]]  # tutup polygon
        cats  = POLLUTANTS + [POLLUTANTS[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill="toself",
            name=city,
            line=dict(color=COLOR_SEQ[i % len(COLOR_SEQ)])
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,100])),
        title=f"Radar Polutan per Kota (nilai dinormalisasi 0–100)  [{plabel(sel_years,sel_months)}]",
        template="plotly_white", title_font_size=15,
        legend=dict(orientation="h", y=-0.15)
    )
    return fig


# 2b. Grouped bar – bandingkan satu polutan antar kota yang dipilih
def grouped_bar_kota(pollutant, sel_cities, sel_years, sel_months):
    sub = apply_filters(df, sel_years, sel_months)
    sub = sub[sub["City"].isin(sel_cities)] if sel_cities else sub
    if sub.empty: return empty_fig()

    grp = sub.groupby("City")[pollutant].mean().reset_index()
    grp.columns = ["Kota", pollutant]
    grp = grp.sort_values(pollutant, ascending=False)

    fig = px.bar(grp, x="Kota", y=pollutant, color="Kota",
                 color_discrete_sequence=COLOR_SEQ,
                 title=f"Perbandingan {pollutant} Antar Kota  [{plabel(sel_years,sel_months)}]",
                 labels={pollutant: f"{pollutant} (µg/m³)", "Kota":"Kota"},
                 template="plotly_white")
    fig.update_layout(showlegend=False, xaxis_tickangle=-35, title_font_size=15)
    return fig


# 2c. Line tren bulanan per kota
def line_kota(pollutant, sel_cities, sel_years, sel_months):
    sub = apply_filters(df, sel_years, sel_months)
    sub = sub[sub["City"].isin(sel_cities)] if sel_cities else sub
    if sub.empty: return empty_fig()

    grp = sub.groupby(["City","Month"])[pollutant].mean().reset_index()
    fig = px.line(grp, x="Month", y=pollutant, color="City", markers=True,
                  title=f"Tren Bulanan {pollutant} per Kota  [{plabel(sel_years,sel_months)}]",
                  labels={pollutant: f"{pollutant} (µg/m³)", "Month":"Bulan", "City":"Kota"},
                  template="plotly_white", color_discrete_sequence=COLOR_SEQ)
    fig.update_xaxes(tickvals=list(range(1,13)), ticktext=MONTH_ORDER)
    fig.update_layout(title_font_size=15)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# Tab 3 – Composition
# ════════════════════════════════════════════════════════════════════════════
def pollutant_composition(chart_type, sel_city, sel_years, sel_months):
    sub = apply_filters(df, sel_years, sel_months)
    sub = sub[sub["City"] == sel_city] if sel_city else sub
    if sub.empty: return empty_fig()
    loc   = sel_city or "Global"
    means = sub[POLLUTANTS].mean().reset_index()
    means.columns = ["Polutan","Nilai"]
    if chart_type == "Pie Chart":
        fig = px.pie(means, names="Polutan", values="Nilai",
                     title=f"Komposisi Polutan – {loc}  [{plabel(sel_years,sel_months)}]",
                     color_discrete_sequence=COLOR_SEQ, template="plotly_white")
        fig.update_traces(textinfo="percent+label")
    else:
        fig = px.bar(means, x="Polutan", y="Nilai", color="Polutan",
                     title=f"Rata-rata Polutan – {loc}  [{plabel(sel_years,sel_months)}]",
                     labels={"Nilai":"Konsentrasi (µg/m³)"},
                     color_discrete_sequence=COLOR_SEQ, template="plotly_white")
        fig.update_layout(showlegend=False)
    fig.update_layout(title_font_size=16)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# Tab 4 – AQI Categories
# ════════════════════════════════════════════════════════════════════════════
def aqi_category_bar(sel_city, sel_pollutant, sel_years, sel_months):
    # Threshold kategori per polutan (standar US EPA / WHO)
    THRESHOLDS = {
        "PM2.5": [
            (12.0,   "Good"),
            (35.4,   "Moderate"),
            (55.4,   "Unhealthy (Sensitive)"),
            (150.4,  "Unhealthy"),
            (250.4,  "Very Unhealthy"),
            (float("inf"), "Hazardous"),
        ],
        "PM10": [
            (54,     "Good"),
            (154,    "Moderate"),
            (254,    "Unhealthy (Sensitive)"),
            (354,    "Unhealthy"),
            (424,    "Very Unhealthy"),
            (float("inf"), "Hazardous"),
        ],
        "NO2": [
            (53,     "Good"),
            (100,    "Moderate"),
            (360,    "Unhealthy (Sensitive)"),
            (649,    "Unhealthy"),
            (1249,   "Very Unhealthy"),
            (float("inf"), "Hazardous"),
        ],
        "SO2": [
            (35,     "Good"),
            (75,     "Moderate"),
            (185,    "Unhealthy (Sensitive)"),
            (304,    "Unhealthy"),
            (604,    "Very Unhealthy"),
            (float("inf"), "Hazardous"),
        ],
        "CO": [
            (4.4,    "Good"),
            (9.4,    "Moderate"),
            (12.4,   "Unhealthy (Sensitive)"),
            (15.4,   "Unhealthy"),
            (30.4,   "Very Unhealthy"),
            (float("inf"), "Hazardous"),
        ],
        "O3": [
            (54,     "Good"),
            (70,     "Moderate"),
            (85,     "Unhealthy (Sensitive)"),
            (105,    "Unhealthy"),
            (200,    "Very Unhealthy"),
            (float("inf"), "Hazardous"),
        ],
    }

    def cat(v, thresholds):
        for limit, label in thresholds:
            if v <= limit:
                return label
        return "Hazardous"

    sub = apply_filters(df, sel_years, sel_months)
    sub = sub[sub["City"] == sel_city].copy() if sel_city else sub.copy()
    if sub.empty: return empty_fig()

    thresholds = THRESHOLDS.get(sel_pollutant, THRESHOLDS["PM2.5"])
    sub["AQI Category"] = sub[sel_pollutant].apply(lambda v: cat(v, thresholds))
    loc    = sel_city or "Global"
    order  = ["Good","Moderate","Unhealthy (Sensitive)","Unhealthy","Very Unhealthy","Hazardous"]
    colors = {"Good":"#00e400","Moderate":"#ffff00","Unhealthy (Sensitive)":"#ff7e00",
              "Unhealthy":"#ff0000","Very Unhealthy":"#99004c","Hazardous":"#7e0023"}
    cnt = (sub["AQI Category"].value_counts()
                               .reindex(order, fill_value=0).reset_index())
    cnt.columns = ["Category","Count"]
    fig = px.bar(cnt, x="Category", y="Count", color="Category",
                 color_discrete_map=colors,
                 title=f"Kategori AQI ({sel_pollutant}) – {loc}  [{plabel(sel_years,sel_months)}]",
                 template="plotly_white", category_orders={"Category":order})
    fig.update_layout(showlegend=False, title_font_size=16)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# Tab 5 – World Map
# ════════════════════════════════════════════════════════════════════════════
def world_map(pollutant, sel_years, sel_months):
    sub = apply_filters(df, sel_years, sel_months)
    if sub.empty: return empty_fig()
    grp = sub.groupby("City")[pollutant].mean().reset_index()
    grp.columns = ["Kota", pollutant]
    grp["lat"] = grp["Kota"].map(lambda c: CITY_COORDS.get(c, (0,0))[0])
    grp["lon"] = grp["Kota"].map(lambda c: CITY_COORDS.get(c, (0,0))[1])

    fig = px.scatter_geo(
        grp, lat="lat", lon="lon",
        size=pollutant, color=pollutant,
        hover_name="Kota",
        hover_data={pollutant:":.1f", "lat":False, "lon":False},
        color_continuous_scale="RdYlGn_r",
        size_max=12,
        projection="natural earth",
        title=f"Peta Kota – {pollutant}  [{plabel(sel_years,sel_months)}]",
        labels={pollutant: f"{pollutant} (µg/m³)"},
        template="plotly_white",
    )
    fig.update_traces(marker=dict(line=dict(width=0.8, color="white")))
    fig.update_layout(
        title_font_size=16, height=550,
        geo=dict(showframe=False, showcoastlines=True,
                 coastlinecolor="LightGray",
                 showland=True, landcolor="#f0f0f0",
                 showocean=True, oceancolor="#cce5ff",
                 showcountries=True, countrycolor="white",
                 lonaxis=dict(range=[-180,180]),
                 lataxis=dict(range=[-70,80])),
        coloraxis_colorbar=dict(title=f"{pollutant}<br>(µg/m³)"),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════════════════
with gr.Blocks(title=" Global Air Quality Dashboard") as demo:

    gr.Markdown("#  Global Air Quality Dashboard\n"
                "**Dataset:** 10,000 records · 20 kota · 19 negara · Tahun 2023")
    gr.HTML(value=kpi_cards())
    gr.Markdown("---")

    # ── Tab 1: Rankings ───────────────────────────────────────────────────────
    with gr.Tab(" Rankings"):
        gr.Markdown("### Kota dengan rata-rata polutan tertinggi")
        with gr.Row():
            r_poll  = gr.Dropdown(POLLUTANTS, value="PM2.5", label="Polutan")
            r_topn  = gr.Slider(5, 20, value=10, step=1, label="Top N Kota")
        with gr.Row():
            r_year  = gr.CheckboxGroup(YEARS, value=YEARS, label=" Filter Tahun")
            r_month = gr.CheckboxGroup(MONTH_ORDER, value=MONTH_ORDER, label=" Filter Bulan")
        r_btn = gr.Button(" Tampilkan Ranking", variant="primary")
        r_out = gr.Plot()
        r_btn.click(fn=bar_city, inputs=[r_poll, r_topn, r_year, r_month], outputs=r_out)

    # ── Tab 2: Perbandingan Kota ──────────────────────────────────────────────
    with gr.Tab("️ Perbandingan Kota"):
        gr.Markdown("### Bandingkan kualitas udara antar kota")

        with gr.Row():
            p_cities = gr.CheckboxGroup(CITIES, value=CITIES[:6], label="Pilih Kota")
        with gr.Row():
            p_year  = gr.CheckboxGroup(YEARS, value=YEARS, label=" Filter Tahun")
            p_month = gr.CheckboxGroup(MONTH_ORDER, value=MONTH_ORDER, label=" Filter Bulan")

        # --- Radar chart ---
        gr.Markdown("#### ️ Radar – Profil Semua Polutan per Kota")
        p_radar_btn = gr.Button("Tampilkan Radar", variant="primary")
        p_radar_out = gr.Plot()
        p_radar_btn.click(fn=radar_kota,
                          inputs=[p_cities, p_year, p_month],
                          outputs=p_radar_out)

        gr.Markdown("---")

        # --- Grouped bar ---
        gr.Markdown("####  Bar – Perbandingan Satu Polutan")
        p_poll_bar = gr.Dropdown(POLLUTANTS, value="PM2.5", label="Polutan")
        p_bar_btn  = gr.Button("Tampilkan Bar Chart", variant="primary")
        p_bar_out  = gr.Plot()
        p_bar_btn.click(fn=grouped_bar_kota,
                        inputs=[p_poll_bar, p_cities, p_year, p_month],
                        outputs=p_bar_out)

        gr.Markdown("---")

        # --- Line tren per kota ---
        gr.Markdown("####  Tren Bulanan per Kota")
        p_poll_line = gr.Dropdown(POLLUTANTS, value="PM2.5", label="Polutan")
        p_line_btn  = gr.Button("Tampilkan Tren", variant="primary")
        p_line_out  = gr.Plot()
        p_line_btn.click(fn=line_kota,
                         inputs=[p_poll_line, p_cities, p_year, p_month],
                         outputs=p_line_out)

    # ── Tab 3: Composition ────────────────────────────────────────────────────
    with gr.Tab(" Composition"):
        gr.Markdown("### Proporsi setiap polutan per kota")
        with gr.Row():
            c_type  = gr.Radio(["Pie Chart","Bar Chart"], value="Pie Chart", label="Jenis Grafik")
            c_city  = gr.Dropdown(["(Global)"] + CITIES, value="(Global)", label="Kota")
        with gr.Row():
            c_year  = gr.CheckboxGroup(YEARS, value=YEARS, label=" Filter Tahun")
            c_month = gr.CheckboxGroup(MONTH_ORDER, value=MONTH_ORDER, label=" Filter Bulan")
        c_btn = gr.Button(" Tampilkan Grafik", variant="primary")
        c_out = gr.Plot()

        def comp_wrapper(ct, cc, y, m):
            return pollutant_composition(ct, None if cc=="(Global)" else cc, y, m)

        c_btn.click(fn=comp_wrapper, inputs=[c_type, c_city, c_year, c_month], outputs=c_out)

    # ── Tab 4: AQI ────────────────────────────────────────────────────────────
    with gr.Tab(" AQI Categories"):
        gr.Markdown("### Distribusi kategori AQI berdasarkan polutan (standar US EPA)")
        with gr.Row():
            a_city = gr.Dropdown(["(Global)"] + CITIES, value="(Global)", label="Kota")
            a_poll = gr.Dropdown(POLLUTANTS, value="PM2.5", label="Polutan")
        with gr.Row():
            a_year  = gr.CheckboxGroup(YEARS, value=YEARS, label=" Filter Tahun")
            a_month = gr.CheckboxGroup(MONTH_ORDER, value=MONTH_ORDER, label=" Filter Bulan")
        a_btn = gr.Button(" Tampilkan AQI", variant="primary")
        a_out = gr.Plot()

        def aqi_wrapper(cc, poll, y, m):
            return aqi_category_bar(None if cc=="(Global)" else cc, poll, y, m)

        a_btn.click(fn=aqi_wrapper, inputs=[a_city, a_poll, a_year, a_month], outputs=a_out)
        gr.Markdown("""
        ---
        ###  Standar Kategori AQI – US EPA (per polutan)

        | Kategori | PM2.5 (µg/m³) | PM10 (µg/m³) | NO2 (ppb) | SO2 (ppb) | CO (ppm) | O3 (ppb) | Warna |
        |---|---|---|---|---|---|---|---|
        | **Good** | 0–12 | 0–54 | 0–53 | 0–35 | 0–4.4 | 0–54 |  |
        | **Moderate** | 12.1–35.4 | 55–154 | 54–100 | 36–75 | 4.5–9.4 | 55–70 |  |
        | **Unhealthy (Sensitive)** | 35.5–55.4 | 155–254 | 101–360 | 76–185 | 9.5–12.4 | 71–85 |  |
        | **Unhealthy** | 55.5–150.4 | 255–354 | 361–649 | 186–304 | 12.5–15.4 | 86–105 |  |
        | **Very Unhealthy** | 150.5–250.4 | 355–424 | 650–1249 | 305–604 | 15.5–30.4 | 106–200 |  |
        | **Hazardous** | >250.4 | >424 | >1249 | >604 | >30.4 | >200 |  |

        > **PM2.5/PM10** = partikel halus · **NO2** = nitrogen dioksida · **SO2** = sulfur dioksida · **CO** = karbon monoksida · **O3** = ozon
        > Sumber: [US EPA AirNow](https://www.airnow.gov/aqi/aqi-basics/)
        """)
    # ── Tab 5: World Map ──────────────────────────────────────────────────────
    with gr.Tab("️ World Map"):
        gr.Markdown("### Bubble map – tingkat polutan per kota di seluruh dunia")
        w_poll = gr.Dropdown(POLLUTANTS, value="PM2.5", label="Polutan")
        with gr.Row():
            w_year  = gr.CheckboxGroup(YEARS, value=YEARS, label=" Filter Tahun")
            w_month = gr.CheckboxGroup(MONTH_ORDER, value=MONTH_ORDER, label=" Filter Bulan")
        w_btn = gr.Button("️ Tampilkan Peta", variant="primary")
        w_out = gr.Plot()
        w_btn.click(fn=world_map, inputs=[w_poll, w_year, w_month], outputs=w_out)

    gr.Markdown("---\n*Dashboard UAS Visualisasi Data · Global Air Quality (10,000 records)*")

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
