import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timezone

st.set_page_config(
    page_title="BIST Sinyal Tarayıcı",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════
# HİSSE LİSTESİ  —  BIST 100 (1 Nisan 2026) + Ek Hisseler
# ════════════════════════════════════════════════════════════════════════════

BIST100 = [
    "AGHOL","AGROT","AHGAZ","AKBNK","AKSA","AKSEN","ALARK","ALFAS","ALTNY","ANSGR",
    "AEFES","ANHYT","ARCLK","ARDYZ","ASELS","ASTOR","AVPGY","BTCIM","BSOKE","BERA",
    "BIMAS","BRSAN","BRYAT","CCOLA","CWENE","CANTE","CLEBI","CIMSA","DOHOL","DOAS",
    "EFORC","ECILC","EKGYO","ENJSA","ENERY","ENKAI","EREGL","EUPWR","FROTO","GSRAY",
    "GESAN","GOLTS","GRTHO","GUBRF","SAHOL","HEKTS","IEYHO","ISMEN","KRDMD","KARSN",
    "KTLEV","KCHOL","KONTR","KONYA","KOZAL","KOZAA","LMKDC","MAGEN","MAVI","MIATK",
    "MGROS","MPARK","OBAMS","ODAS","OTKAR","OYAKC","PASEU","PGSUS","PETKM","RALYH",
    "REEDR","RYGYO","SASA","SELEC","SMRTG","SKBNK","SOKM","TABGD","TAVHL","TKFEN",
    "TOASO","TCELL","TUPRS","THYAO","GARAN","HALKB","ISCTR","TSKB","TURSG","SISE",
    "VAKBN","TTKOM","VESTL","YKBNK","CVKMD","ZOREN","PRKAB","EGEEN","TTRAK","YEOTK",
]  # 100 hisse

EK_HISSELER = [
    "AKFYE","ASGR","ORGE","HTTBT","SDTTR",
    "OYYAT","NETCAD","VBTYZ","EGEGY","RYSAS",
    "TGSAS","ATATP","KCAER",
]  # BIST100'de olmayanlar

ALL_STOCKS = BIST100 + EK_HISSELER  # tekrar yok çünkü kontrol edildi

# ════════════════════════════════════════════════════════════════════════════
# ALGORİTMA — EMA 10 / Smoothing 14
# ════════════════════════════════════════════════════════════════════════════

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def compute_signals(df):
    df = df.copy()
    o = ema(df["Open"],  10)
    c = ema(df["Close"], 10)
    h = ema(df["High"],  10)
    l = ema(df["Low"],   10)

    haclose = (o + h + l + c) / 4
    haopen  = haclose.copy()
    haopen.iloc[0] = (o.iloc[0] + c.iloc[0]) / 2
    for i in range(1, len(haopen)):
        haopen.iloc[i] = (haopen.iloc[i-1] + haclose.iloc[i-1]) / 2

    o2 = ema(haopen,  14)
    c2 = ema(haclose, 14)

    col      = (c2 > o2).astype(int)
    col_prev = col.shift(1)

    df["long_signal"]  = (col == 1) & (col_prev == 0)
    df["short_signal"] = (col == 0) & (col_prev == 1)
    df["sha_color"]    = col
    df["o2"] = o2
    df["c2"] = c2
    return df

def gun_once(dt):
    """Bir tarihten bugüne kaç iş günü geçti (takvim günü değil iş günü)."""
    if dt is None:
        return None
    try:
        # timezone'u sil, sadece date karşılaştır
        sinyal_tarihi = pd.Timestamp(dt).date()
        bugun = datetime.now(timezone.utc).date()
        # iki tarih arasındaki iş günü sayısı
        is_gunleri = pd.bdate_range(start=sinyal_tarihi, end=bugun)
        return max(0, len(is_gunleri) - 1)  # sinyal günü dahil değil
    except Exception:
        return None

def get_signals_info(df):
    long_idx  = df.index[df["long_signal"]].tolist()
    short_idx = df.index[df["short_signal"]].tolist()
    last_long  = long_idx[-1]  if long_idx  else None
    last_short = short_idx[-1] if short_idx else None
    return {
        "last_long":       last_long,
        "last_long_days":  gun_once(last_long),
        "last_short":      last_short,
        "last_short_days": gun_once(last_short),
    }

# ════════════════════════════════════════════════════════════════════════════
# VERİ ÇEKME — yfinance
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlc(ticker, period="1y"):
    try:
        df = yf.download(
            f"{ticker}.IS",
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            show_errors=False,
        )
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df[["Open","High","Low","Close","Volume"]].dropna()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def tara_hepsini(tickers):
    rows = []
    for ticker in tickers:
        df = fetch_ohlc(ticker, period="6mo")
        if df.empty or len(df) < 40:
            continue
        df   = compute_signals(df)
        info = get_signals_info(df)
        rows.append({
            "Hisse":      ticker,
            "Son Long":   pd.Timestamp(info["last_long"]).date()  if info["last_long"]  else None,
            "Long (İG)":  info["last_long_days"],
            "Son Short":  pd.Timestamp(info["last_short"]).date() if info["last_short"] else None,
            "Short (İG)": info["last_short_days"],
        })
    return pd.DataFrame(rows)

# ════════════════════════════════════════════════════════════════════════════
# GRAFİK
# ════════════════════════════════════════════════════════════════════════════

def grafik_ciz(df, ticker):
    fig = go.Figure()

    # Mum grafiği
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name="Fiyat",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a",
        decreasing_fillcolor="#ef5350",
    ))

    # Long sinyalleri
    long_df = df[df["long_signal"]]
    if not long_df.empty:
        fig.add_trace(go.Scatter(
            x=long_df.index,
            y=long_df["Low"] * 0.982,
            mode="markers+text",
            text=["▲"] * len(long_df),
            textposition="bottom center",
            textfont=dict(color="#00e676", size=16),
            marker=dict(symbol="triangle-up", color="#00e676", size=12,
                        line=dict(color="#00c853", width=1)),
            name="Long",
            hovertemplate="<b>LONG</b><br>%{x|%d.%m.%Y}<extra></extra>",
        ))

    # Short sinyalleri
    short_df = df[df["short_signal"]]
    if not short_df.empty:
        fig.add_trace(go.Scatter(
            x=short_df.index,
            y=short_df["High"] * 1.018,
            mode="markers+text",
            text=["▼"] * len(short_df),
            textposition="top center",
            textfont=dict(color="#ff1744", size=16),
            marker=dict(symbol="triangle-down", color="#ff1744", size=12,
                        line=dict(color="#d50000", width=1)),
            name="Short/Çıkış",
            hovertemplate="<b>SHORT</b><br>%{x|%d.%m.%Y}<extra></extra>",
        ))

    fig.update_layout(
        title=f"<b>{ticker}</b>  —  Smoothed Heiken Ashi (EMA 10 / Smooth 14)",
        xaxis_rangeslider_visible=False,
        plot_bgcolor="#131722",
        paper_bgcolor="#131722",
        font=dict(color="#d1d4dc", size=12),
        xaxis=dict(gridcolor="#1e222d", showgrid=True),
        yaxis=dict(gridcolor="#1e222d", showgrid=True, title="Fiyat (TL)"),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        height=580,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig

# ════════════════════════════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
.stApp, [data-testid="stAppViewContainer"] { background:#0d1117; }
.mbox {
    background:#161b22; border:1px solid #30363d;
    border-radius:10px; padding:16px; text-align:center; margin-bottom:6px;
}
.mlabel { color:#8b949e; font-size:12px; margin-bottom:6px; }
.mval   { font-size:28px; font-weight:700; line-height:1.2; }
.green  { color:#3fb950; }
.red    { color:#f85149; }
.blue   { color:#58a6ff; }
.purple { color:#bc8cff; }
.tag-long  { background:#0d2818; color:#3fb950; border:1px solid #238636;
             border-radius:6px; padding:4px 12px; font-size:13px;
             display:inline-block; margin:3px; font-family:monospace; }
.tag-short { background:#2d1117; color:#f85149; border:1px solid #da3633;
             border-radius:6px; padding:4px 12px; font-size:13px;
             display:inline-block; margin:3px; font-family:monospace; }
section[data-testid="stSidebar"] { background:#161b22 !important; }
.block-container { padding-top:1.2rem !important; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📈 BIST Sinyal")
    st.markdown("*Smoothed Heiken Ashi*")
    st.divider()
    page = st.radio("", ["📊 Sinyal Tarayıcı", "🔍 Hisse Detayı"], label_visibility="collapsed")
    st.divider()
    st.caption(f"🗓 {datetime.now().strftime('%d.%m.%Y')}")
    st.caption(f"📋 BIST 100: **{len(BIST100)}** hisse")
    st.caption(f"➕ Ek hisse: **{len(EK_HISSELER)}**")
    st.caption(f"🔢 Toplam: **{len(ALL_STOCKS)}**")
    st.divider()
    if st.button("🔄 Verileri Yenile", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.caption("⚠️ Yatırım tavsiyesi değildir.")
    st.caption("İG = İş Günü")

# ════════════════════════════════════════════════════════════════════════════
# SAYFA 1 — SİNYAL TARAYICI
# ════════════════════════════════════════════════════════════════════════════

if page == "📊 Sinyal Tarayıcı":
    st.title("📊 Sinyal Tarayıcı")
    st.caption("İG = İş Günü  |  EMA 10  |  Smoothing 14")

    f1, f2, f3 = st.columns(3)
    with f1:
        gun_filtre = st.slider("Son kaç iş günü?", 1, 60, 20)
    with f2:
        grup = st.selectbox("Grup", ["Tümü", "BIST 100", "Ek Hisseler"])
    with f3:
        st.markdown("<br>", unsafe_allow_html=True)
        tara_btn = st.button("🔍 Tara", use_container_width=True)

    with st.spinner("Hisseler taranıyor... (ilk açılışta ~1-2 dk sürebilir)"):
        df_tara = tara_hepsini(tuple(ALL_STOCKS))

    if df_tara.empty:
        st.error("Veri çekilemedi.")
        st.stop()

    # Grup filtresi
    if grup == "BIST 100":
        df_tara = df_tara[df_tara["Hisse"].isin(BIST100)]
    elif grup == "Ek Hisseler":
        df_tara = df_tara[df_tara["Hisse"].isin(EK_HISSELER)]

    # Gün filtresi
    long_list  = df_tara[df_tara["Long (İG)"].notna()  & (df_tara["Long (İG)"]  <= gun_filtre)].sort_values("Long (İG)")
    short_list = df_tara[df_tara["Short (İG)"].notna() & (df_tara["Short (İG)"] <= gun_filtre)].sort_values("Short (İG)")

    # Metrikler
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="mbox"><div class="mlabel">Long Sinyal (son {gun_filtre} iş günü)</div>'
                    f'<div class="mval green">{len(long_list)}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="mbox"><div class="mlabel">Short Sinyal (son {gun_filtre} iş günü)</div>'
                    f'<div class="mval red">{len(short_list)}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="mbox"><div class="mlabel">Taranan Hisse</div>'
                    f'<div class="mval blue">{len(df_tara)}</div></div>', unsafe_allow_html=True)
    with m4:
        bugun_long = df_tara[df_tara["Long (İG)"] == 0]
        st.markdown(f'<div class="mbox"><div class="mlabel">Bugün Long Veren</div>'
                    f'<div class="mval purple">{len(bugun_long)}</div></div>', unsafe_allow_html=True)

    st.divider()

    # LONG ve SHORT yan yana
    col_long, col_short = st.columns(2)

    with col_long:
        st.markdown("### 🟢 Long Sinyaller")
        if long_list.empty:
            st.info("Bu filtrede long sinyal yok.")
        else:
            disp_l = long_list[["Hisse","Son Long","Long (İG)"]].copy()
            disp_l.columns = ["Hisse","Tarih","İş Günü Önce"]
            def renk_long(val):
                if pd.isna(val): return ""
                if val == 0: return "background:#0d4429;color:#3fb950;font-weight:700"
                if val <= 3: return "background:#0d2818;color:#3fb950"
                if val <= 7: return "background:#0a1f14;color:#56d364"
                return ""
            st.dataframe(
                disp_l.style.applymap(renk_long, subset=["İş Günü Önce"]),
                use_container_width=True,
                hide_index=True,
                height=min(600, 35 * len(disp_l) + 38),
            )

    with col_short:
        st.markdown("### 🔴 Short Sinyaller")
        if short_list.empty:
            st.info("Bu filtrede short sinyal yok.")
        else:
            disp_s = short_list[["Hisse","Son Short","Short (İG)"]].copy()
            disp_s.columns = ["Hisse","Tarih","İş Günü Önce"]
            def renk_short(val):
                if pd.isna(val): return ""
                if val == 0: return "background:#4d1212;color:#f85149;font-weight:700"
                if val <= 3: return "background:#2d1117;color:#f85149"
                if val <= 7: return "background:#1f0d0d;color:#ffa198"
                return ""
            st.dataframe(
                disp_s.style.applymap(renk_short, subset=["İş Günü Önce"]),
                use_container_width=True,
                hide_index=True,
                height=min(600, 35 * len(disp_s) + 38),
            )

    with st.expander("📋 Tüm Hisseler Özet"):
        st.dataframe(df_tara, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# SAYFA 2 — HİSSE DETAYI
# ════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Hisse Detayı":
    st.title("🔍 Hisse Detayı")

    c1, c2 = st.columns([3, 2])
    with c1:
        secilen = st.selectbox(
            "Hisse",
            ALL_STOCKS,
            index=ALL_STOCKS.index("THYAO") if "THYAO" in ALL_STOCKS else 0,
        )
    with c2:
        donem_map = {"3 Ay":"3mo","6 Ay":"6mo","1 Yıl":"1y","2 Yıl":"2y"}
        donem = donem_map[st.selectbox("Dönem", list(donem_map.keys()), index=2)]

    with st.spinner(f"{secilen} yükleniyor..."):
        ham_df = fetch_ohlc(secilen, period=donem)

    if ham_df.empty:
        st.error(f"**{secilen}** için veri alınamadı. Yahoo Finance'ta `.IS` uzantısıyla bulunamıyor olabilir.")
        st.stop()

    sig_df = compute_signals(ham_df)
    info   = get_signals_info(sig_df)

    # Metrikler
    son_fiyat  = float(sig_df["Close"].iloc[-1])
    prev_fiyat = float(sig_df["Close"].iloc[-2]) if len(sig_df) > 1 else son_fiyat
    degisim    = (son_fiyat - prev_fiyat) / prev_fiyat * 100
    isaret     = "+" if degisim >= 0 else ""
    f_renk     = "green" if degisim >= 0 else "red"

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="mbox"><div class="mlabel">Son Kapanış</div>'
                    f'<div class="mval {f_renk}">{son_fiyat:.2f} ₺</div>'
                    f'<div style="color:#8b949e;font-size:13px">{isaret}{degisim:.2f}%</div></div>',
                    unsafe_allow_html=True)
    with m2:
        ld  = info["last_long_days"]
        ldt = pd.Timestamp(info["last_long"]).strftime("%d.%m.%Y") if info["last_long"] else "—"
        lv  = f"{ld} iş günü önce" if ld is not None else "—"
        st.markdown(f'<div class="mbox"><div class="mlabel">Son Long Sinyali</div>'
                    f'<div class="mval green" style="font-size:20px">{lv}</div>'
                    f'<div style="color:#8b949e;font-size:12px">{ldt}</div></div>',
                    unsafe_allow_html=True)
    with m3:
        sd  = info["last_short_days"]
        sdt = pd.Timestamp(info["last_short"]).strftime("%d.%m.%Y") if info["last_short"] else "—"
        sv  = f"{sd} iş günü önce" if sd is not None else "—"
        st.markdown(f'<div class="mbox"><div class="mlabel">Son Short Sinyali</div>'
                    f'<div class="mval red" style="font-size:20px">{sv}</div>'
                    f'<div style="color:#8b949e;font-size:12px">{sdt}</div></div>',
                    unsafe_allow_html=True)
    with m4:
        sha = sig_df["sha_color"].iloc[-1]
        sha_lbl = "🟢 LONG" if sha == 1 else "🔴 SHORT"
        sha_cls = "green" if sha == 1 else "red"
        st.markdown(f'<div class="mbox"><div class="mlabel">Mevcut SHA Durumu</div>'
                    f'<div class="mval {sha_cls}" style="font-size:22px">{sha_lbl}</div></div>',
                    unsafe_allow_html=True)

    st.divider()

    # Grafik
    fig = grafik_ciz(sig_df, secilen)
    st.plotly_chart(fig, use_container_width=True)

    # Sinyal geçmişi
    with st.expander("📅 Tüm Sinyal Tarihleri"):
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**🟢 Long Sinyalleri**")
            long_tarihler = sig_df.index[sig_df["long_signal"]].strftime("%d.%m.%Y").tolist()
            for t in reversed(long_tarihler):
                st.markdown(f'<span class="tag-long">▲ {t}</span>', unsafe_allow_html=True)
        with g2:
            st.markdown("**🔴 Short Sinyalleri**")
            short_tarihler = sig_df.index[sig_df["short_signal"]].strftime("%d.%m.%Y").tolist()
            for t in reversed(short_tarihler):
                st.markdown(f'<span class="tag-short">▼ {t}</span>', unsafe_allow_html=True)
