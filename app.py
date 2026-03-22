import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "yfinance", "plotly", "pandas", "numpy"], check=True)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import date

# ── Sayfa ayarları ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BIST Sinyal Tarayıcı",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════
# HİSSE LİSTESİ
# ════════════════════════════════════════════════════════════════════════════

BIST100_STOCKS = [
    "AGHOL", "AGROT", "AHGAZ", "AKBNK", "AKSA", "AKSEN", "ALARK", "ALFAS",
    "ALTNY", "ANSGR", "AEFES", "ANHYT", "ARCLK", "ARDYZ", "ASELS", "ASTOR",
    "AVPGY", "BTCIM", "BSOKE", "BERA", "BIMAS", "BRSAN", "BRYAT", "CCOLA",
    "CWENE", "CANTE", "CLEBI", "CIMSA", "DOHOL", "DOAS", "EFORC", "ECILC",
    "EKGYO", "ENJSA", "ENERY", "ENKAI", "EREGL", "EUPWR", "FROTO", "GSRAY",
    "GESAN", "GOLTS", "GRTHO", "GUBRF", "SAHOL", "HEKTS", "IEYHO", "ISMEN",
    "KRDMD", "KARSN", "KTLEV", "KCHOL", "KONTR", "KONYA", "KOZAL", "KOZAA",
    "LMKDC", "MAGEN", "MAVI", "MIATK", "MGROS", "MPARK", "OBAMS", "ODAS",
    "OTKAR", "OYAKC", "PASEU", "PGSUS", "PETKM", "RALYH", "REEDR", "RYGYO",
    "SASA", "SELEC", "SMRTG", "SKBNK", "SOKM", "TABGD", "TAVHL", "TKFEN",
    "TOASO", "TCELL", "TUPRS", "THYAO", "GARAN", "HALKB", "ISCTR", "TSKB",
    "TURSG", "SISE", "VAKBN", "TTKOM", "VESTL", "YKBNK", "CVKMD", "ZOREN",
    "PRKAB",
]

# Ek hisseler — BIST 100'de olmayanlar
CUSTOM_EXTRA = [
    "AKFYE", "ASGR", "ORGE", "HTTBT", "SDTTR",
    "OYYAT", "NETCAD", "VBTYZ", "EGEGY", "RYSAS",
    "TGSAS", "ATATP", "YEOTK", "KCAER",
]

ALL_STOCKS = list(BIST100_STOCKS)
for s in CUSTOM_EXTRA:
    if s not in ALL_STOCKS:
        ALL_STOCKS.append(s)
ALL_STOCKS = sorted(ALL_STOCKS)

# ════════════════════════════════════════════════════════════════════════════
# ALGORİTMA — SMOOTHED HEİKEN ASHİ  (EMA 10 / Smoothing 14)
# ════════════════════════════════════════════════════════════════════════════

EMA_LEN   = 10
SMOOTH_LEN = 14


def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def compute_signals(df):
    df = df.copy()
    o = ema(df["Open"],  EMA_LEN)
    c = ema(df["Close"], EMA_LEN)
    h = ema(df["High"],  EMA_LEN)
    l = ema(df["Low"],   EMA_LEN)

    haclose = (o + h + l + c) / 4
    haopen  = pd.Series(index=df.index, dtype=float)
    haopen.iloc[0] = (o.iloc[0] + c.iloc[0]) / 2
    for i in range(1, len(haopen)):
        haopen.iloc[i] = (haopen.iloc[i-1] + haclose.iloc[i-1]) / 2

    o2 = ema(haopen,  SMOOTH_LEN)
    c2 = ema(haclose, SMOOTH_LEN)

    col      = (c2 > o2).astype(int)
    col_prev = col.shift(1)

    df["sha_color"]    = col
    df["sha_o2"]       = o2
    df["sha_c2"]       = c2
    df["long_signal"]  = (col == 1) & (col_prev == 0)
    df["short_signal"] = (col == 0) & (col_prev == 1)
    return df


def get_latest_signals(df):
    today = pd.Timestamp.now(tz="UTC").normalize()

    def days_ago(dt):
        if dt is None:
            return None
        try:
            ts = pd.Timestamp(dt)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
            return (today - ts).days
        except Exception:
            return None

    long_dates  = df.index[df["long_signal"]].tolist()
    short_dates = df.index[df["short_signal"]].tolist()
    last_long   = long_dates[-1]  if long_dates  else None
    last_short  = short_dates[-1] if short_dates else None

    return {
        "last_long":       last_long,
        "last_long_days":  days_ago(last_long),
        "last_short":      last_short,
        "last_short_days": days_ago(last_short),
    }


# ════════════════════════════════════════════════════════════════════════════
# VERİ ÇEKME
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlc(ticker, period="1y"):
    try:
        df = yf.download(f"{ticker}.IS", period=period, interval="1d",
                         auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_signals(tickers):
    rows = []
    for ticker in tickers:
        df = fetch_ohlc(ticker, period="6mo")
        if df.empty or len(df) < 30:
            continue
        df = compute_signals(df)
        info = get_latest_signals(df)
        rows.append({
            "Hisse":            ticker,
            "Son Long":         info["last_long"].date()  if info["last_long"]  else None,
            "Long (Gün Önce)":  info["last_long_days"],
            "Son Short":        info["last_short"].date() if info["last_short"] else None,
            "Short (Gün Önce)": info["last_short_days"],
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# GRAFİK
# ════════════════════════════════════════════════════════════════════════════

def build_chart(df, ticker):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name="Mum",
        increasing_line_color="#00b050", decreasing_line_color="#ff0000",
        increasing_fillcolor="#00b050",  decreasing_fillcolor="#ff0000",
        opacity=0.7,
    ))

    long_df  = df[df["long_signal"]]
    short_df = df[df["short_signal"]]

    if not long_df.empty:
        fig.add_trace(go.Scatter(
            x=long_df.index, y=long_df["Low"] * 0.985,
            mode="markers",
            marker=dict(symbol="triangle-up", color="lime", size=14,
                        line=dict(color="darkgreen", width=1)),
            name="Long Sinyal",
            hovertemplate="%{x}<br>▲ LONG<extra></extra>",
        ))

    if not short_df.empty:
        fig.add_trace(go.Scatter(
            x=short_df.index, y=short_df["High"] * 1.015,
            mode="markers",
            marker=dict(symbol="triangle-down", color="red", size=14,
                        line=dict(color="darkred", width=1)),
            name="Short/Çıkış",
            hovertemplate="%{x}<br>▼ SHORT<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=f"<b>{ticker}</b> — Smoothed Heiken Ashi Sinyalleri",
                   font=dict(size=18)),
        xaxis_rangeslider_visible=False,
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis=dict(gridcolor="#1f2937"),
        yaxis=dict(gridcolor="#1f2937", title="Fiyat (TL)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=560,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
.stApp { background-color: #0e1117; }
.metric-box {
    background: #1a1d23; border-radius: 10px;
    padding: 14px 18px; text-align: center;
    border: 1px solid #2a2d35; margin-bottom: 4px;
}
.metric-label { color: #9ca3af; font-size: 13px; margin-bottom: 4px; }
.metric-value { font-size: 26px; font-weight: 700; }
.long-val  { color: #22c55e; }
.short-val { color: #ef4444; }
.tag-long  { background:#14532d; color:#86efac; border-radius:6px;
             padding:3px 10px; font-size:13px; display:inline-block; margin:2px; }
.tag-short { background:#7f1d1d; color:#fca5a5; border-radius:6px;
             padding:3px 10px; font-size:13px; display:inline-block; margin:2px; }
.block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("📈 BIST Sinyal")
    page = st.radio("Sayfa", ["📊 Sinyal Tarayıcı", "🔍 Hisse Detayı", "📰 Haberler & KAP"])
    st.divider()
    st.caption(f"🗓 Bugün: {date.today().strftime('%d.%m.%Y')}")
    st.caption(f"📋 Toplam hisse: **{len(ALL_STOCKS)}**")
    st.caption(f"• BIST 100: {len(BIST100_STOCKS)}")
    st.caption(f"• Ek hisse: {len([s for s in CUSTOM_EXTRA if s not in BIST100_STOCKS])}")
    st.divider()
    if st.button("🔄 Verileri Yenile", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# SAYFA 1 — SİNYAL TARAYICI
# ════════════════════════════════════════════════════════════════════════════

if page == "📊 Sinyal Tarayıcı":
    st.title("📊 BIST Sinyal Tarayıcı")
    st.markdown("*Smoothed Heiken Ashi — EMA 10 / Smoothing 14*")

    c1, c2, c3 = st.columns(3)
    with c1:
        signal_filter = st.selectbox("Sinyal Filtresi", ["Tümü", "Sadece Long", "Sadece Short"])
    with c2:
        day_filter = st.slider("Max. gün önce", 1, 90, 30)
    with c3:
        group_filter = st.selectbox("Hisse Grubu", ["Tümü", "BIST 100", "Ek Hisseler"])

    with st.spinner("Tüm hisseler taranıyor... (ilk açılış ~1 dk sürebilir)"):
        scan_df = fetch_all_signals(ALL_STOCKS)

    if scan_df.empty:
        st.warning("Veri çekilemedi. İnternet bağlantınızı kontrol edin.")
        st.stop()

    if group_filter == "BIST 100":
        scan_df = scan_df[scan_df["Hisse"].isin(BIST100_STOCKS)]
    elif group_filter == "Ek Hisseler":
        extra = [s for s in CUSTOM_EXTRA if s not in BIST100_STOCKS]
        scan_df = scan_df[scan_df["Hisse"].isin(extra)]

    long_df_f  = scan_df[(scan_df["Long (Gün Önce)"].notna())  & (scan_df["Long (Gün Önce)"]  <= day_filter)]
    short_df_f = scan_df[(scan_df["Short (Gün Önce)"].notna()) & (scan_df["Short (Gün Önce)"] <= day_filter)]

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Son {day_filter} Günde Long</div>'
                    f'<div class="metric-value long-val">{len(long_df_f)}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Son {day_filter} Günde Short</div>'
                    f'<div class="metric-value short-val">{len(short_df_f)}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Taranan Hisse</div>'
                    f'<div class="metric-value" style="color:#60a5fa">{len(scan_df)}</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Toplam Long Geçmişi</div>'
                    f'<div class="metric-value" style="color:#a78bfa">{scan_df["Long (Gün Önce)"].notna().sum()}</div></div>',
                    unsafe_allow_html=True)

    st.divider()

    tab1, tab2, tab3 = st.tabs(["🟢 Long Sinyaller", "🔴 Short Sinyaller", "📋 Tüm Hisseler"])

    def color_long(val):
        if pd.isna(val): return ""
        if val <= 3:  return "background-color:#14532d; color:#86efac"
        if val <= 7:  return "background-color:#1a3a1a; color:#4ade80"
        if val <= 14: return "background-color:#1c2a1a; color:#86efac"
        return ""

    def color_short(val):
        if pd.isna(val): return ""
        if val <= 3:  return "background-color:#7f1d1d; color:#fca5a5"
        if val <= 7:  return "background-color:#4a1414; color:#f87171"
        return ""

    with tab1:
        disp = long_df_f.sort_values("Long (Gün Önce)")[["Hisse", "Son Long", "Long (Gün Önce)"]].rename(
            columns={"Long (Gün Önce)": "Gün Önce"})
        st.dataframe(disp.style.applymap(color_long, subset=["Gün Önce"]),
                     use_container_width=True, hide_index=True)

    with tab2:
        disp_s = short_df_f.sort_values("Short (Gün Önce)")[["Hisse", "Son Short", "Short (Gün Önce)"]].rename(
            columns={"Short (Gün Önce)": "Gün Önce"})
        st.dataframe(disp_s.style.applymap(color_short, subset=["Gün Önce"]),
                     use_container_width=True, hide_index=True)

    with tab3:
        st.dataframe(scan_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# SAYFA 2 — HİSSE DETAYI
# ════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Hisse Detayı":
    st.title("🔍 Hisse Detayı")

    col_s, col_p = st.columns([3, 2])
    with col_s:
        selected = st.selectbox("Hisse Seçin", ALL_STOCKS,
                                index=ALL_STOCKS.index("THYAO") if "THYAO" in ALL_STOCKS else 0)
    with col_p:
        period_map = {"3 Ay": "3mo", "6 Ay": "6mo", "1 Yıl": "1y", "2 Yıl": "2y"}
        period = period_map[st.selectbox("Dönem", list(period_map.keys()), index=2)]

    with st.spinner(f"{selected} verisi çekiliyor..."):
        raw_df = fetch_ohlc(selected, period=period)

    if raw_df.empty:
        st.error(f"**{selected}** için veri çekilemedi.")
        st.stop()

    sig_df = compute_signals(raw_df)
    info   = get_latest_signals(sig_df)

    last_close = float(raw_df["Close"].iloc[-1])
    prev_close = float(raw_df["Close"].iloc[-2]) if len(raw_df) > 1 else last_close
    change_pct = (last_close - prev_close) / prev_close * 100

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        color = "long-val" if change_pct >= 0 else "short-val"
        sign  = "+" if change_pct >= 0 else ""
        st.markdown(f'<div class="metric-box"><div class="metric-label">Son Kapanış</div>'
                    f'<div class="metric-value {color}">{last_close:.2f} ₺</div>'
                    f'<div style="font-size:13px;color:#9ca3af">{sign}{change_pct:.2f}%</div></div>',
                    unsafe_allow_html=True)
    with m2:
        ld = info["last_long_days"]
        dt = info["last_long"].date().strftime("%d.%m.%Y") if info["last_long"] else "—"
        val = f"{ld} gün önce" if ld is not None else "—"
        st.markdown(f'<div class="metric-box"><div class="metric-label">Son Long Sinyali</div>'
                    f'<div class="metric-value long-val">{val}</div>'
                    f'<div style="font-size:12px;color:#9ca3af">{dt}</div></div>', unsafe_allow_html=True)
    with m3:
        sd = info["last_short_days"]
        dt_s = info["last_short"].date().strftime("%d.%m.%Y") if info["last_short"] else "—"
        val_s = f"{sd} gün önce" if sd is not None else "—"
        st.markdown(f'<div class="metric-box"><div class="metric-label">Son Short Sinyali</div>'
                    f'<div class="metric-value short-val">{val_s}</div>'
                    f'<div style="font-size:12px;color:#9ca3af">{dt_s}</div></div>', unsafe_allow_html=True)
    with m4:
        cur = sig_df["sha_color"].iloc[-1]
        label = "🟢 LONG" if cur == 1 else "🔴 SHORT"
        cls   = "long-val" if cur == 1 else "short-val"
        st.markdown(f'<div class="metric-box"><div class="metric-label">Güncel SHA Rengi</div>'
                    f'<div class="metric-value {cls}" style="font-size:20px">{label}</div></div>',
                    unsafe_allow_html=True)

    st.divider()
    st.plotly_chart(build_chart(sig_df, selected), use_container_width=True)

    with st.expander("📅 Sinyal Geçmişi"):
        long_dates  = sig_df.index[sig_df["long_signal"]].strftime("%d.%m.%Y").tolist()
        short_dates = sig_df.index[sig_df["short_signal"]].strftime("%d.%m.%Y").tolist()
        h1, h2 = st.columns(2)
        with h1:
            st.markdown("**🟢 Long Sinyalleri**")
            for d in reversed(long_dates):
                st.markdown(f'<span class="tag-long">▲ {d}</span>', unsafe_allow_html=True)
        with h2:
            st.markdown("**🔴 Short / Çıkış Sinyalleri**")
            for d in reversed(short_dates):
                st.markdown(f'<span class="tag-short">▼ {d}</span>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SAYFA 3 — HABERLER & KAP
# ════════════════════════════════════════════════════════════════════════════

elif page == "📰 Haberler & KAP":
    st.title("📰 Haberler & KAP Raporları")

    selected_n = st.selectbox("Hisse Seçin", ALL_STOCKS,
                              index=ALL_STOCKS.index("THYAO") if "THYAO" in ALL_STOCKS else 0)

    tab_kap, tab_news = st.tabs(["📑 KAP Bildirimleri", "📰 Haberler"])

    with tab_kap:
        kap_url = f"https://www.kap.org.tr/tr/Bildirim/Liste/{selected_n}"
        st.markdown(f"""
        <div style="background:#1a1d23;border-radius:10px;padding:20px;border:1px solid #2a2d35;margin-bottom:16px">
            <p style="color:#9ca3af;margin:0 0 12px 0;font-size:14px">
                KAP (Kamuyu Aydınlatma Platformu) — <b>{selected_n}</b> resmi bildirimleri
            </p>
            <a href="{kap_url}" target="_blank" style="
                display:inline-block;background:#2563eb;color:white;
                padding:10px 20px;border-radius:8px;text-decoration:none;
                font-weight:600;font-size:14px;">
                🔗 KAP'ta {selected_n} Bildirimlerini Görüntüle
            </a>
        </div>""", unsafe_allow_html=True)
        st.components.v1.iframe(kap_url, height=500, scrolling=True)

    with tab_news:
        isyat_url  = f"https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/sirket-karti.aspx?hisse={selected_n}#tab-6"
        mynet_url  = f"https://finans.mynet.com/borsa/hisseler/{selected_n.lower()}-hisse-senedi/"
        google_url = f"https://www.google.com/search?q={selected_n}+borsa+haber&tbm=nws"

        col1, col2, col3 = st.columns(3)
        for col, label, icon, url, sub in [
            (col1, "İş Yatırım",    "🏦", isyat_url,  "Şirket Kartı"),
            (col2, "Mynet Finans",  "📰", mynet_url,  "Haber Akışı"),
            (col3, "Google Haberler","🔍",google_url, "Son haberler"),
        ]:
            with col:
                st.markdown(f"""
                <a href="{url}" target="_blank" style="
                    display:block;background:#1a1d23;border:1px solid #374151;
                    border-radius:8px;padding:14px;text-decoration:none;text-align:center;">
                    <div style="font-size:24px">{icon}</div>
                    <div style="color:#e5e7eb;font-weight:600;margin:6px 0 4px">{label}</div>
                    <div style="color:#9ca3af;font-size:12px">{selected_n} {sub}</div>
                </a>""", unsafe_allow_html=True)

        st.divider()
        try:
            st.components.v1.iframe(mynet_url, height=480, scrolling=True)
        except Exception:
            st.info("Haber sayfası yüklenemedi. Yukarıdaki linklerden takip edebilirsiniz.")

    st.divider()
    st.markdown('<div style="text-align:center;color:#6b7280;font-size:12px">'
                '⚠️ Bu uygulama yatırım tavsiyesi içermez. Veriler Yahoo Finance üzerinden çekilmektedir.</div>',
                unsafe_allow_html=True)
