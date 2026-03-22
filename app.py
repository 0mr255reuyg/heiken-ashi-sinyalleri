import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timezone, timedelta

st.set_page_config(
    page_title="BIST Sinyal Tarayıcı",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════
# HİSSE LİSTESİ
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
]

EK_HISSELER = [
    "AKFYE","ASGR","ORGE","HTTBT","SDTTR",
    "OYYAT","NETCAD","VBTYZ","EGEGY","RYSAS",
    "TGSAS","ATATP","KCAER",
]

ALL_STOCKS = BIST100 + [s for s in EK_HISSELER if s not in BIST100]

# ════════════════════════════════════════════════════════════════════════════
# VERİ ÇEKME
# ════════════════════════════════════════════════════════════════════════════

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlc(ticker, gun=365):
    try:
        now   = int(datetime.now(timezone.utc).timestamp())
        start = int((datetime.now(timezone.utc) - timedelta(days=gun)).timestamp())
        url   = (f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}.IS"
                 f"?period1={start}&period2={now}&interval=1d")
        r     = requests.get(url, headers=HEADERS, timeout=20)
        data  = r.json()
        res   = data["chart"]["result"][0]
        q     = res["indicators"]["quote"][0]
        idx   = pd.to_datetime(res["timestamp"], unit="s").normalize()
        df    = pd.DataFrame({
            "Open": q["open"], "High": q["high"],
            "Low":  q["low"],  "Close": q["close"],
        }, index=idx)
        return df.dropna()
    except Exception:
        return pd.DataFrame()

# ════════════════════════════════════════════════════════════════════════════
# ALGORİTMA
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
    o2 = ema(haopen, 14)
    c2 = ema(haclose, 14)
    col = (c2 > o2).astype(int)
    df["sha_color"]    = col
    df["long_signal"]  = (col == 1) & (col.shift(1) == 0)
    df["short_signal"] = (col == 0) & (col.shift(1) == 1)
    return df

def is_gunu_once(dt):
    if dt is None:
        return None
    try:
        t1 = pd.Timestamp(dt).normalize().tz_localize(None)
        t2 = pd.Timestamp(datetime.now()).normalize()
        return max(0, len(pd.bdate_range(t1, t2)) - 1)
    except Exception:
        return None

def get_signals_info(df):
    li = df.index[df["long_signal"]].tolist()
    si = df.index[df["short_signal"]].tolist()
    ll = li[-1] if li else None
    ls = si[-1] if si else None
    return {
        "last_long": ll, "last_long_days": is_gunu_once(ll),
        "last_short": ls, "last_short_days": is_gunu_once(ls),
    }

@st.cache_data(ttl=3600, show_spinner=False)
def tara_hepsini(tickers):
    rows = []
    for ticker in tickers:
        df = fetch_ohlc(ticker, gun=180)
        if df.empty or len(df) < 40:
            continue
        df   = compute_signals(df)
        info = get_signals_info(df)
        rows.append({
            "Hisse":     ticker,
            "Son Long":  pd.Timestamp(info["last_long"]).date()  if info["last_long"]  else None,
            "Long İG":   info["last_long_days"],
            "Son Short": pd.Timestamp(info["last_short"]).date() if info["last_short"] else None,
            "Short İG":  info["last_short_days"],
        })
    return pd.DataFrame(rows)

# ════════════════════════════════════════════════════════════════════════════
# GRAFİK — TradingView Lightweight Charts (JS, sıfır Python bağımlılığı)
# ════════════════════════════════════════════════════════════════════════════

def grafik_ciz(df, ticker):
    candles = []
    longs   = []
    shorts  = []

    for ts, row in df.iterrows():
        t = int(pd.Timestamp(ts).timestamp())
        candles.append({
            "time": t,
            "open":  round(float(row["Open"]),  4),
            "high":  round(float(row["High"]),  4),
            "low":   round(float(row["Low"]),   4),
            "close": round(float(row["Close"]), 4),
        })
        if row.get("long_signal"):
            longs.append({"time": t, "position": "belowBar", "color": "#00e676",
                          "shape": "arrowUp", "text": "L", "size": 1})
        if row.get("short_signal"):
            shorts.append({"time": t, "position": "aboveBar", "color": "#ff1744",
                           "shape": "arrowDown", "text": "S", "size": 1})

    markers = sorted(longs + shorts, key=lambda x: x["time"])

    html = f"""
    <div id="chart_{ticker}" style="width:100%;height:520px;background:#131722;border-radius:8px"></div>
    <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
    <script>
    (function() {{
        var el = document.getElementById('chart_{ticker}');
        var chart = LightweightCharts.createChart(el, {{
            width: el.offsetWidth,
            height: 520,
            layout: {{ background: {{ color: '#131722' }}, textColor: '#d1d4dc' }},
            grid: {{
                vertLines: {{ color: '#1e222d' }},
                horzLines: {{ color: '#1e222d' }},
            }},
            timeScale: {{ timeVisible: true, borderColor: '#2a2d3e' }},
            rightPriceScale: {{ borderColor: '#2a2d3e' }},
        }});
        var series = chart.addCandlestickSeries({{
            upColor: '#26a69a', downColor: '#ef5350',
            borderUpColor: '#26a69a', borderDownColor: '#ef5350',
            wickUpColor: '#26a69a', wickDownColor: '#ef5350',
        }});
        series.setData({json.dumps(candles)});
        series.setMarkers({json.dumps(markers)});
        window.addEventListener('resize', function() {{
            chart.applyOptions({{ width: el.offsetWidth }});
        }});
    }})();
    </script>
    """
    st.components.v1.html(html, height=540)

# ════════════════════════════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
.stApp,[data-testid="stAppViewContainer"]{background:#0d1117}
section[data-testid="stSidebar"]{background:#161b22!important}
.mbox{background:#161b22;border:1px solid #30363d;border-radius:10px;
      padding:16px;text-align:center;margin-bottom:6px}
.mlabel{color:#8b949e;font-size:12px;margin-bottom:6px}
.mval{font-size:26px;font-weight:700;line-height:1.3}
.green{color:#3fb950}.red{color:#f85149}
.blue{color:#58a6ff}.purple{color:#bc8cff}
.tag-l{background:#0d2818;color:#3fb950;border:1px solid #238636;
       border-radius:6px;padding:3px 10px;font-size:13px;
       display:inline-block;margin:2px;font-family:monospace}
.tag-s{background:#2d1117;color:#f85149;border:1px solid #da3633;
       border-radius:6px;padding:3px 10px;font-size:13px;
       display:inline-block;margin:2px;font-family:monospace}
.block-container{padding-top:1.2rem!important}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📈 BIST Sinyal")
    st.divider()
    page = st.radio("", ["📊 Sinyal Tarayıcı", "🔍 Hisse Detayı"], label_visibility="collapsed")
    st.divider()
    st.caption(f"🗓 {datetime.now().strftime('%d.%m.%Y')}")
    st.caption(f"BIST 100: **{len(BIST100)}** hisse")
    st.caption(f"Ek: **{len(EK_HISSELER)}** hisse")
    st.caption(f"Toplam: **{len(ALL_STOCKS)}** hisse")
    st.divider()
    if st.button("🔄 Yenile", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("İG = İş Günü")
    st.caption("⚠️ Yatırım tavsiyesi değildir.")

# ════════════════════════════════════════════════════════════════════════════
# SAYFA 1 — SİNYAL TARAYICI
# ════════════════════════════════════════════════════════════════════════════

if page == "📊 Sinyal Tarayıcı":
    st.title("📊 Sinyal Tarayıcı")
    st.caption("EMA 10  |  Smoothing 14  |  İG = İş Günü")

    f1, f2 = st.columns(2)
    with f1:
        gun_filtre = st.slider("Son kaç iş günü?", 1, 60, 20)
    with f2:
        grup = st.selectbox("Hisse Grubu", ["Tümü", "BIST 100", "Ek Hisseler"])

    with st.spinner("Hisseler taranıyor... (~1-2 dk)"):
        df_tara = tara_hepsini(tuple(ALL_STOCKS))

    if df_tara.empty:
        st.error("Veri çekilemedi.")
        st.stop()

    if grup == "BIST 100":
        df_tara = df_tara[df_tara["Hisse"].isin(BIST100)]
    elif grup == "Ek Hisseler":
        df_tara = df_tara[df_tara["Hisse"].isin(EK_HISSELER)]

    long_list  = df_tara[df_tara["Long İG"].notna()  & (df_tara["Long İG"]  <= gun_filtre)].sort_values("Long İG")
    short_list = df_tara[df_tara["Short İG"].notna() & (df_tara["Short İG"] <= gun_filtre)].sort_values("Short İG")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="mbox"><div class="mlabel">Long (son {gun_filtre} İG)</div>'
                    f'<div class="mval green">{len(long_list)}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="mbox"><div class="mlabel">Short (son {gun_filtre} İG)</div>'
                    f'<div class="mval red">{len(short_list)}</div></div>', unsafe_allow_html=True)
    with m3:
        bugun = df_tara[df_tara["Long İG"] == 0]
        st.markdown(f'<div class="mbox"><div class="mlabel">Bugün Long</div>'
                    f'<div class="mval purple">{len(bugun)}</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="mbox"><div class="mlabel">Taranan</div>'
                    f'<div class="mval blue">{len(df_tara)}</div></div>', unsafe_allow_html=True)

    st.divider()

    def rl(val):
        if pd.isna(val): return ""
        if val == 0:  return "background:#0d4429;color:#3fb950;font-weight:700"
        if val <= 3:  return "background:#0d2818;color:#3fb950"
        if val <= 7:  return "background:#0a1f14;color:#56d364"
        return ""

    def rs(val):
        if pd.isna(val): return ""
        if val == 0:  return "background:#4d1212;color:#f85149;font-weight:700"
        if val <= 3:  return "background:#2d1117;color:#f85149"
        if val <= 7:  return "background:#1f0d0d;color:#ffa198"
        return ""

    col_l, col_s = st.columns(2)
    with col_l:
        st.markdown("### 🟢 Long Sinyaller")
        if long_list.empty:
            st.info("Bu aralıkta long sinyal yok.")
        else:
            d = long_list[["Hisse","Son Long","Long İG"]].rename(columns={"Long İG":"İG Önce"})
            st.dataframe(d.style.applymap(rl, subset=["İG Önce"]),
                         use_container_width=True, hide_index=True,
                         height=min(600, 35*len(d)+38))

    with col_s:
        st.markdown("### 🔴 Short Sinyaller")
        if short_list.empty:
            st.info("Bu aralıkta short sinyal yok.")
        else:
            d = short_list[["Hisse","Son Short","Short İG"]].rename(columns={"Short İG":"İG Önce"})
            st.dataframe(d.style.applymap(rs, subset=["İG Önce"]),
                         use_container_width=True, hide_index=True,
                         height=min(600, 35*len(d)+38))

    with st.expander("📋 Tüm Hisseler"):
        st.dataframe(df_tara, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# SAYFA 2 — HİSSE DETAYI
# ════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Hisse Detayı":
    st.title("🔍 Hisse Detayı")

    c1, c2 = st.columns([3, 2])
    with c1:
        secilen = st.selectbox("Hisse", ALL_STOCKS,
                               index=ALL_STOCKS.index("THYAO") if "THYAO" in ALL_STOCKS else 0)
    with c2:
        dm = {"3 Ay":90,"6 Ay":180,"1 Yıl":365,"2 Yıl":730}
        gun = dm[st.selectbox("Dönem", list(dm.keys()), index=2)]

    with st.spinner(f"{secilen} yükleniyor..."):
        ham = fetch_ohlc(secilen, gun=gun)

    if ham.empty:
        st.error(f"**{secilen}** için veri alınamadı.")
        st.stop()

    sig  = compute_signals(ham)
    info = get_signals_info(sig)

    son   = float(sig["Close"].iloc[-1])
    prev  = float(sig["Close"].iloc[-2]) if len(sig) > 1 else son
    degis = (son - prev) / prev * 100
    frenk = "green" if degis >= 0 else "red"

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="mbox"><div class="mlabel">Son Kapanış</div>'
                    f'<div class="mval {frenk}">{son:.2f} ₺</div>'
                    f'<div style="color:#8b949e;font-size:13px">{"+" if degis>=0 else ""}{degis:.2f}%</div></div>',
                    unsafe_allow_html=True)
    with m2:
        ld  = info["last_long_days"]
        ldt = pd.Timestamp(info["last_long"]).strftime("%d.%m.%Y") if info["last_long"] else "—"
        st.markdown(f'<div class="mbox"><div class="mlabel">Son Long</div>'
                    f'<div class="mval green" style="font-size:20px">{"—" if ld is None else f"{ld} İG önce"}</div>'
                    f'<div style="color:#8b949e;font-size:12px">{ldt}</div></div>', unsafe_allow_html=True)
    with m3:
        sd  = info["last_short_days"]
        sdt = pd.Timestamp(info["last_short"]).strftime("%d.%m.%Y") if info["last_short"] else "—"
        st.markdown(f'<div class="mbox"><div class="mlabel">Son Short</div>'
                    f'<div class="mval red" style="font-size:20px">{"—" if sd is None else f"{sd} İG önce"}</div>'
                    f'<div style="color:#8b949e;font-size:12px">{sdt}</div></div>', unsafe_allow_html=True)
    with m4:
        sha = sig["sha_color"].iloc[-1]
        st.markdown(f'<div class="mbox"><div class="mlabel">SHA Durumu</div>'
                    f'<div class="mval {"green" if sha==1 else "red"}" style="font-size:22px">'
                    f'{"🟢 LONG" if sha==1 else "🔴 SHORT"}</div></div>', unsafe_allow_html=True)

    st.divider()
    grafik_ciz(sig, secilen)

    with st.expander("📅 Tüm Sinyal Tarihleri"):
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**🟢 Long**")
            for t in reversed(sig.index[sig["long_signal"]].strftime("%d.%m.%Y").tolist()):
                st.markdown(f'<span class="tag-l">▲ {t}</span>', unsafe_allow_html=True)
        with g2:
            st.markdown("**🔴 Short**")
            for t in reversed(sig.index[sig["short_signal"]].strftime("%d.%m.%Y").tolist()):
                st.markdown(f'<span class="tag-s">▼ {t}</span>', unsafe_allow_html=True)
