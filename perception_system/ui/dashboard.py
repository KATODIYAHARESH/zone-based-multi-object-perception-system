"""Streamlit mission-control dashboard (dark; amber / green / red / white accents only).

Run:  python main.py dashboard      (or: streamlit run ui/dashboard.py -- --config config.yaml)

The perception engine runs in-process on a background thread - one instance per
Streamlit server, held by ``st.cache_resource``. Live widgets refresh through
``st.fragment(run_every=...)`` so only they re-render. ``st.info`` is never used
(its default styling is blue); warning / success / error map to amber / green / red.
"""
from __future__ import annotations

import argparse
import dataclasses
import html
import json
import sys
import time
from pathlib import Path
from string import Template
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from core.config import load_config  # noqa: E402
from core.logging_setup import setup_logging  # noqa: E402
from core.types import EMOTIONS, Category  # noqa: E402
from ui.overlay import draw_zone  # noqa: E402
from ui.theme import AMBER, CATEGORY_HEX, EMOTION_HEX, HEX  # noqa: E402
from zones.zone import ZONE_COLORS, Zone  # noqa: E402

st.set_page_config(page_title="ZBMOPS // Mission Control", page_icon="🟠", layout="wide")

CSS = Template("""
<style>
:root { --amber:${amber}; --green:${green}; --red:${red}; --white:${white}; --grey:${grey}; --panel:${panel}; --bg:${bg}; }
html, body, [class*="css"], .stMarkdown, .stText { font-family: ui-monospace, 'JetBrains Mono', Menlo, Consolas, monospace; }
a, a:visited { color: var(--amber) !important; }
::selection { background: var(--amber); color: #000; }
.block-container { padding-top: 1.1rem; }
.hud-header { display:flex; justify-content:space-between; align-items:center; border:1px solid #2a2922; background:var(--panel); padding:.6rem 1rem; border-radius:6px; margin-bottom:.6rem; }
.hud-title { color: var(--amber); font-weight:700; letter-spacing:.2em; }
.hud-sub { color: var(--grey); font-size:.72rem; letter-spacing:.12em; }
.mon { display:flex; align-items:center; gap:.5rem; color: var(--white); border:1px solid var(--red); padding:.25rem .65rem; border-radius:4px; font-size:.78rem; letter-spacing:.14em; }
.mon .dot { width:10px; height:10px; border-radius:50%; background: var(--red); box-shadow:0 0 8px var(--red); animation: blink 1s steps(2, start) infinite; }
@keyframes blink { to { visibility: hidden; } }
.card { border:1px solid #2a2922; background: var(--panel); border-radius:6px; padding:.6rem .8rem; margin-bottom:.6rem; }
.card h4 { margin:0 0 .45rem 0; color: var(--amber); font-size:.74rem; letter-spacing:.18em; }
.kpi { display:grid; grid-template-columns: repeat(6, 1fr); gap:.5rem; margin-top:.4rem; }
.kpi div { border:1px solid #2a2922; background: var(--panel); border-radius:6px; padding:.35rem .6rem; }
.kpi b { display:block; font-size:1.1rem; color: var(--white); }
.kpi span { font-size:.62rem; color: var(--grey); letter-spacing:.16em; }
.bar { height:7px; background:#1d1e19; border-radius:4px; overflow:hidden; margin-bottom:.3rem; }
.bar > i { display:block; height:100%; }
.row { display:flex; justify-content:space-between; font-size:.7rem; color: var(--white); }
.log { font-size:.7rem; line-height:1.4; max-height: 340px; overflow-y:auto; }
.sev-info { color: var(--white); } .sev-warning { color: var(--amber); } .sev-critical { color: var(--red); } .ev { color: var(--grey); }
table.counts { width:100%; font-size:.7rem; border-collapse:collapse; }
table.counts th { color: var(--grey); font-weight:400; text-align:right; padding:2px 4px; }
table.counts td { text-align:right; padding:2px 4px; border-top:1px solid #23241e; color: var(--white); }
table.counts td:first-child, table.counts th:first-child { text-align:left; }
</style>
""").substitute(HEX)


def _config_path() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args.config


@st.cache_resource(show_spinner="Bringing the perception engine online...")
def get_engine(config_path: str) -> Any:
    from pipeline.engine import PerceptionEngine

    cfg = load_config(config_path)
    setup_logging(cfg.system.log_level, cfg.path(cfg.system.log_dir))
    cfg.ui.mode = "headless"
    return PerceptionEngine(cfg).start()


def _fmt_time(ts: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts))


def header(engine: Any) -> None:
    cfg = engine.cfg
    st.markdown(
        f'<div class="hud-header"><div><div class="hud-title">ZBMOPS // MISSION CONTROL</div>'
        f'<div class="hud-sub">{html.escape(cfg.system.name)} | {html.escape(cfg.system.site)} | '
        f'{html.escape(engine.device_label)} | model {html.escape(cfg.detection.model)}</div></div>'
        f'<div class="mon"><span class="dot"></span>{html.escape(cfg.privacy.indicator_text)}</div></div>',
        unsafe_allow_html=True)


@st.fragment(run_every=0.1)
def live_feed(engine: Any) -> None:
    frame = engine.latest_frame()
    if frame is None:
        st.markdown('<div class="card"><h4>FEED</h4>AWAITING SIGNAL...</div>', unsafe_allow_html=True)
    else:
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
    s = engine.state()
    fps, target = float(s.get("fps", 0.0)), engine.cfg.performance.target_fps
    fps_col = HEX["green"] if fps >= target else (HEX["amber"] if fps >= 0.6 * target else HEX["red"])
    kpis = [("FPS", f"<b style='color:{fps_col}'>{fps:.1f}</b>"),
            ("LATENCY", f"<b>{float(s.get('latency_ms', 0.0)):.0f} ms</b>"),
            ("FRAME SKIP", f"<b>{s.get('skip', '-')}</b>"), ("TRACKS", f"<b>{s.get('tracks', 0)}</b>"),
            ("SOURCE", f"<b>{html.escape(str(s.get('source', '-')).upper())}</b>"),
            ("FACE LAT", f"<b>{float(s.get('face_latency_ms') or 0):.0f} ms</b>")]
    st.markdown('<div class="kpi">' + "".join(f"<div><span>{k}</span>{v}</div>" for k, v in kpis) + "</div>",
                unsafe_allow_html=True)


@st.fragment(run_every=0.5)
def side_panel(engine: Any) -> None:
    s = engine.state()
    names = {z.id: z.name for z in engine.zones.list()}
    cats = [c.value for c in Category]
    head = "".join(f"<th style='color:{CATEGORY_HEX[c]}'>{c[:3].upper()}</th>" for c in cats)
    rows = "".join(f"<tr><td>{html.escape(names.get(zid, zid))}</td>" + "".join(f"<td>{per.get(c, 0)}</td>" for c in cats)
                   + "</tr>" for zid, per in s.get("counts", {}).items())
    st.markdown(f'<div class="card"><h4>LIVE ZONE COUNTS</h4><table class="counts"><tr><th>ZONE</th>{head}</tr>'
                f'{rows or "<tr><td colspan=7>no active zones</td></tr>"}</table></div>', unsafe_allow_html=True)

    emotions: Dict[str, int] = s.get("emotions", {})
    total = sum(emotions.values()) or 1
    bars = "".join(f"<div class='row'><span>{e.upper()}</span><span>{emotions.get(e, 0)}</span></div>"
                   f"<div class='bar'><i style='width:{100 * emotions.get(e, 0) / total:.0f}%;background:{EMOTION_HEX[e]}'></i></div>"
                   for e in EMOTIONS)
    st.markdown(f'<div class="card"><h4>EXPRESSION METER (VISIBLE FACES)</h4>{bars}</div>', unsafe_allow_html=True)

    items: List[tuple] = []
    for a in s.get("alerts", [])[:15]:
        items.append((a["timestamp"], f"<div class='sev-{a['severity']}'>{_fmt_time(a['timestamp'])} "
                                      f"{a['severity'].upper():8s} {html.escape(a['message'])}</div>"))
    for e in s.get("events", [])[:15]:
        items.append((e["timestamp"], f"<div class='ev'>{_fmt_time(e['timestamp'])} {e['type'].upper():5s} "
                                      f"{html.escape(e['zone_name'])} - {html.escape(e['label'])} #{e['track_id']}"
                                      + (f" ({e['dwell_s']:.0f}s)" if e["type"] == "exit" else "") + "</div>"))
    log_html = "".join(h for _, h in sorted(items, key=lambda x: x[0], reverse=True)) or "<div class='ev'>quiet</div>"
    st.markdown(f'<div class="card"><h4>EVENT LOG</h4><div class="log">{log_html}</div></div>', unsafe_allow_html=True)


def tab_zones(engine: Any) -> None:
    ss = st.session_state
    ss.setdefault("zone_pts", [])
    ss.setdefault("last_click", None)
    if ss.pop("zone_flash", None):
        st.success("Zone saved to zones.yaml and active in the running engine.")
    W, H = engine.cfg.source.width, engine.cfg.source.height
    frame = engine.latest_clean_frame()
    canvas = frame.copy() if frame is not None else np.full((H, W, 3), 12, np.uint8)
    for z in engine.zones.list():
        draw_zone(canvas, z, W, H)
    pts_px = [(int(x * W), int(y * H)) for x, y in ss.zone_pts]
    if pts_px:
        cv2.polylines(canvas, [np.array(pts_px, np.int32)], len(pts_px) > 2, AMBER, 2, cv2.LINE_AA)
        for p in pts_px:
            cv2.circle(canvas, p, 6, AMBER, -1, cv2.LINE_AA)
    disp = cv2.resize(canvas, (960, int(H * 960 / W)))
    rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
    left, right = st.columns([3, 1.35], gap="medium")
    with left:
        st.caption("Click the frame to add polygon vertices. Membership is tested on the bottom-centre of each box.")
        try:
            from streamlit_image_coordinates import streamlit_image_coordinates

            click = streamlit_image_coordinates(rgb, key="zone_canvas")
            if click and (click["x"], click["y"]) != ss.last_click:
                ss.last_click = (click["x"], click["y"])
                ss.zone_pts.append((round(click["x"] / disp.shape[1], 4), round(click["y"] / disp.shape[0], 4)))
                st.rerun()
        except ImportError:
            st.image(rgb, width="stretch")
            st.warning("streamlit-image-coordinates not installed - enter normalised 'x,y' vertices, one per line.")
            text = st.text_area("Vertices", value="\n".join(f"{x},{y}" for x, y in ss.zone_pts))
            if st.button("Use these vertices"):
                try:
                    ss.zone_pts = [tuple(float(v) for v in line.split(",")) for line in text.splitlines() if line.strip()]
                    st.rerun()
                except ValueError:
                    st.error("Each line must be 'x,y' with numbers in 0..1.")
    with right:
        st.markdown(f"**Vertices:** {len(ss.zone_pts)}")
        c1, c2 = st.columns(2)
        if c1.button("Undo vertex", disabled=not ss.zone_pts, width="stretch"):
            ss.zone_pts.pop()
            st.rerun()
        if c2.button("Clear", disabled=not ss.zone_pts, width="stretch"):
            ss.zone_pts = []
            st.rerun()
        with st.form("zone_form"):
            name = st.text_input("Name", value=f"Zone {len(engine.zones.list()) + 1}")
            mode = st.radio("Mode", ["restricted", "allowed"], horizontal=True,
                            help="allowed: categories are a whitelist - restricted: categories are a blacklist")
            cats = st.multiselect("Categories", [c.value for c in Category], default=["vehicle"])
            max_occ = st.number_input("Max occupancy (0 = none)", min_value=0, value=0)
            dwell = st.number_input("Dwell limit in s (0 = none)", min_value=0, value=0)
            color = st.selectbox("Colour", list(ZONE_COLORS))
            submitted = st.form_submit_button("Create zone", type="primary")
        if submitted:
            try:
                zone = Zone(id=engine.zones.unique_id(name), name=name, polygon=list(ss.zone_pts), mode=mode,
                            categories=cats, max_occupancy=int(max_occ) or None, dwell_limit_s=float(dwell) or None,
                            color=color)
                engine.zones.add(zone)
                engine.save_zones()
                ss.zone_pts, ss.zone_flash = [], True
                st.rerun()
            except ValueError as exc:
                st.error(f"Invalid zone: {exc}")
    st.markdown("##### Active zones")
    for z in engine.zones.list():
        c = st.columns([3, 1.6, 3, 1.2, 1.2])
        c[0].markdown(f"**{html.escape(z.name)}** `{z.id}`")
        c[1].write(z.mode.value)
        c[2].write(", ".join(z.categories) or "(none)")
        enabled = c[3].toggle("enabled", value=z.enabled, key=f"en_{z.id}")
        if enabled != z.enabled:
            engine.zones.add(dataclasses.replace(z, enabled=enabled), replace=True)
            engine.save_zones()
            st.rerun()
        if c[4].button("Delete", key=f"del_{z.id}"):
            engine.zones.remove(z.id)
            engine.save_zones()
            st.rerun()


def tab_analytics(engine: Any) -> None:
    from analytics.exporter import fetch_rows, rows_to_csv

    hours = st.select_slider("Window", options=[1, 6, 24, 72, 168], value=24, format_func=lambda h: f"{h} h")
    since = time.time() - hours * 3600
    ts = pd.DataFrame(engine.store.counts_timeseries(since, max(60, int(hours * 3600 / 120))))
    st.markdown("##### Occupancy timeline (objects inside zones, by category)")
    if ts.empty:
        st.warning("No count samples yet (written every analytics.counts_interval_s).")
    else:
        ts["time"] = pd.to_datetime(ts["bucket"], unit="s")
        pivot = ts.groupby(["time", "category"])["avg_count"].sum().unstack(fill_value=0)
        st.line_chart(pivot, color=[CATEGORY_HEX.get(c, HEX["white"]) for c in pivot.columns])
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("##### Zone traffic")
        zs = pd.DataFrame(engine.store.zone_summary(since))
        st.dataframe(zs, hide_index=True, width="stretch") if not zs.empty else st.caption("No zone events yet.")
        st.markdown("##### Alerts by severity")
        al = pd.DataFrame(engine.store.alert_summary(since))
        if not al.empty:
            st.bar_chart(al.groupby("severity")["n"].sum(), color=HEX["amber"])
        else:
            st.caption("No alerts in window.")
    with c2:
        st.markdown("##### Cumulative heatmap (ground-contact points)")
        if engine.heatmap is not None:
            st.image(cv2.cvtColor(engine.heatmap.image(640, 360), cv2.COLOR_BGR2RGB), width="stretch")
        st.markdown("##### Expressions")
        em = pd.DataFrame(engine.store.emotion_summary(since))
        if not em.empty:
            st.bar_chart(em.set_index("emotion")["n"], color=HEX["green"])
        else:
            st.caption("No expression samples in window.")
    st.markdown("##### Export")
    cols = st.columns(4)
    for i, table in enumerate(("events", "alerts", "counts", "emotions")):
        rows = fetch_rows(engine.store, table, since)
        cols[i].download_button(f"{table}.csv", rows_to_csv(rows), file_name=f"{table}.csv", mime="text/csv",
                                key=f"csv_{table}", width="stretch")
        cols[i].download_button(f"{table}.json", json.dumps(rows, default=str, indent=2), file_name=f"{table}.json",
                                mime="application/json", key=f"json_{table}", width="stretch")


def tab_settings(engine: Any) -> None:
    import yaml

    cfg = engine.cfg
    with st.form("runtime"):
        c1, c2 = st.columns(2)
        conf = c1.slider("Detection confidence", 0.05, 0.95, float(cfg.detection.confidence), 0.01)
        iou = c1.slider("NMS IoU", 0.1, 0.9, float(cfg.detection.iou), 0.05)
        skip = c1.number_input("Base frame skip", 1, 30, int(cfg.performance.detect_every_n_frames))
        adaptive = c2.toggle("Adaptive frame skipping", value=cfg.performance.adaptive_skip)
        face_every = c2.number_input("Face refresh every N frames", 1, 60, int(cfg.face.every_n_frames))
        heat = c2.toggle("Heatmap overlay on live feed", value=engine.show_heatmap)
        if st.form_submit_button("Apply to running engine", type="primary"):
            try:
                engine.update_runtime(confidence=conf, iou=iou, detect_every_n_frames=int(skip), adaptive_skip=adaptive,
                                      face_every_n_frames=int(face_every), show_heatmap=heat)
                st.success("Applied.")
            except Exception as exc:
                st.error(f"Rejected: {exc}")
    st.markdown("##### Alert rules (YAML) - fully validated before hot-reload")
    text = st.text_area("rules", value=yaml.safe_dump(cfg.alerts.rules, sort_keys=False), height=260,
                        label_visibility="collapsed")
    if st.button("Validate & reload rules"):
        try:
            specs = yaml.safe_load(text) or []
            engine.reload_rules(specs)
            st.success(f"{len(specs)} rule(s) active.")
        except Exception as exc:
            st.error(f"Rules rejected: {exc}")
    st.markdown("##### Stage latency (ms, rolling window)")
    stages = engine.state().get("stages", {})
    if stages:
        st.dataframe(pd.DataFrame(stages).T, width="stretch")


def tab_privacy(engine: Any) -> None:
    cfg = engine.cfg
    st.markdown(
        "<div class='card'><h4>DATA HANDLING</h4>"
        "<div class='row'><span>Face recognition</span><span>enrolled + consenting people only</span></div>"
        "<div class='row'><span>Everyone else</span><span>shown as Person #ID - never identified</span></div>"
        "<div class='row'><span>Biometric storage</span><span>ArcFace embeddings only, no face images</span></div>"
        f"<div class='row'><span>Retention</span><span>{cfg.privacy.retention_days} days, automatic purge</span></div>"
        "<div class='row'><span>Monitoring indicator</span><span>always on (cannot be disabled)</span></div></div>",
        unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        blur = st.toggle("Pixelate faces of non-enrolled people", value=cfg.privacy.blur_faces)
        if blur != cfg.privacy.blur_faces:
            engine.update_runtime(blur_faces=blur)
            st.rerun()
        days = st.number_input("Data retention (days)", 1, 3650, int(cfg.privacy.retention_days))
        if int(days) != cfg.privacy.retention_days:
            engine.update_runtime(retention_days=int(days))
            st.success(f"Retention set to {int(days)} days (applies at next purge).")
        if st.button("Purge expired data now"):
            st.json(engine.purge_now())
    with c2:
        st.markdown("##### Enrolled identities (consent records - no biometric data shown)")
        if engine.gallery is None:
            st.warning("Recognition disabled or no embedding-capable face backend available.")
            return
        identities = engine.gallery.list_identities()
        if not identities:
            st.caption("Nobody is enrolled - everyone is displayed as Person #ID.")
        for ident in identities:
            c = st.columns([2.5, 3.5, 1.5])
            c[0].markdown(f"**{html.escape(ident['name'])}**" + (" (revoked)" if ident["revoked"] else ""))
            exp = time.strftime("%Y-%m-%d", time.localtime(ident["expires_at"])) if ident["expires_at"] else "never"
            c[1].caption(f"consent {ident['consent_reference']} | expires {exp} | {ident['samples']} vectors")
            if not ident["revoked"] and c[2].button("Revoke", key=f"rv_{ident['name']}"):
                engine.gallery.revoke(ident["name"], operator="dashboard")
                st.rerun()


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    try:
        engine = get_engine(_config_path())
    except Exception as exc:
        st.error(f"Engine failed to start: {exc}")
        st.stop()
    header(engine)
    live, zones, analytics, settings, privacy = st.tabs(["LIVE", "ZONES", "ANALYTICS", "SETTINGS", "PRIVACY"])
    with live:
        c1, c2 = st.columns([3, 1.25], gap="medium")
        with c1:
            live_feed(engine)
        with c2:
            side_panel(engine)
    with zones:
        tab_zones(engine)
    with analytics:
        tab_analytics(engine)
    with settings:
        tab_settings(engine)
    with privacy:
        tab_privacy(engine)


main()
