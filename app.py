from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from skin_app import CaseAnalysis, SkinCancerPredictor


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config" / "inference_config.json"
MODEL_FIG_DIR = ROOT_DIR / "outputs" / "figures" / "modeling_2"
APP_BUILD_ID = "2026-05-05-rich-demo-v2"

ZONE_STYLES = {
    "negative": {
        "accent": "#1d4ed8",
        "soft": "#dbeafe",
        "surface": "linear-gradient(145deg, rgba(219,234,254,0.92), rgba(255,255,255,0.98))",
    },
    "review": {
        "accent": "#b45309",
        "soft": "#fef3c7",
        "surface": "linear-gradient(145deg, rgba(254,243,199,0.92), rgba(255,255,255,0.98))",
    },
    "positive": {
        "accent": "#b91c1c",
        "soft": "#fee2e2",
        "surface": "linear-gradient(145deg, rgba(254,226,226,0.95), rgba(255,255,255,0.98))",
    },
}


@st.cache_resource
def load_predictor(build_id: str) -> SkinCancerPredictor:
    return SkinCancerPredictor(config_path=CONFIG_PATH)


@st.cache_data
def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def ensure_state() -> None:
    if "analysis_cache" not in st.session_state:
        st.session_state.analysis_cache = {}
    if "analysis_order" not in st.session_state:
        st.session_state.analysis_order = []


def add_analysis_to_state(analysis: CaseAnalysis) -> None:
    st.session_state.analysis_cache[analysis.case_id] = analysis
    if analysis.case_id not in st.session_state.analysis_order:
        st.session_state.analysis_order.append(analysis.case_id)


def get_session_analyses() -> list[CaseAnalysis]:
    return [
        st.session_state.analysis_cache[case_id]
        for case_id in st.session_state.analysis_order
        if case_id in st.session_state.analysis_cache
    ]


def render_styles() -> None:
    st.markdown(
        """
        <style>
          .stApp {
            background:
              radial-gradient(circle at 0% 0%, rgba(191,219,254,0.35), transparent 24rem),
              radial-gradient(circle at 100% 0%, rgba(253,224,71,0.22), transparent 22rem),
              linear-gradient(180deg, #fffdf7 0%, #f7fafc 55%, #f8fafc 100%);
          }
          .block-container {
            max-width: 1280px;
            padding-top: 2.2rem;
            padding-bottom: 2.5rem;
          }
          .hero-shell {
            padding: 1.2rem 1.3rem 1.0rem 1.3rem;
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(30,41,59,0.92));
            color: #f8fafc;
            box-shadow: 0 24px 70px rgba(15,23,42,0.18);
            margin-bottom: 1.2rem;
          }
          .hero-shell h1 {
            margin: 0;
            font-size: 2.2rem;
            line-height: 1.05;
          }
          .hero-shell p {
            margin: 0.55rem 0 0 0;
            max-width: 52rem;
            font-size: 1rem;
            color: rgba(248,250,252,0.82);
          }
          .metric-card {
            border-radius: 22px;
            padding: 1rem 1.05rem;
            background: rgba(255,255,255,0.82);
            border: 1px solid rgba(15,23,42,0.07);
            box-shadow: 0 14px 32px rgba(15,23,42,0.06);
          }
          .metric-label {
            font-size: 0.85rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #64748b;
          }
          .metric-value {
            font-size: 1.7rem;
            font-weight: 800;
            color: #0f172a;
            margin-top: 0.2rem;
          }
          .metric-caption {
            font-size: 0.92rem;
            color: #475569;
            margin-top: 0.25rem;
          }
          .panel-card {
            border-radius: 24px;
            padding: 1.15rem;
            border: 1px solid rgba(15,23,42,0.08);
            background: rgba(255,255,255,0.88);
            box-shadow: 0 16px 36px rgba(15,23,42,0.06);
          }
                    .panel-card,
                    .panel-card p,
                    .panel-card span,
                    .panel-card div {
                        color: #0f172a;
                    }
          .result-card {
            border-radius: 26px;
            padding: 1.2rem;
            border: 1px solid rgba(15,23,42,0.08);
            box-shadow: 0 22px 56px rgba(15,23,42,0.08);
          }
          .pill {
            display: inline-block;
            padding: 0.36rem 0.75rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.92rem;
          }
          .section-kicker {
            color: #92400e;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            font-size: 0.80rem;
          }
          .risk-shell {
            border-radius: 18px;
            padding: 0.95rem 1rem 1rem 1rem;
            background: rgba(255,255,255,0.8);
            border: 1px solid rgba(15,23,42,0.07);
          }
          .small-note {
            color: #475569;
            font-size: 0.94rem;
          }
                    .stCaption,
                    .stCaption p {
                        color: #475569;
                    }
                    div[data-testid="stMetricLabel"] p {
                        color: #334155;
                        font-weight: 700;
                    }
                    div[data-testid="stMetricValue"],
                    div[data-testid="stMetricDelta"] {
                        color: #0f172a;
                    }
          .heuristic-card {
            border-radius: 18px;
            padding: 0.95rem;
            background: rgba(255,255,255,0.88);
            border: 1px solid rgba(15,23,42,0.08);
            box-shadow: 0 12px 28px rgba(15,23,42,0.05);
            height: 100%;
          }
          .heuristic-chip {
            display: inline-block;
            padding: 0.22rem 0.5rem;
            border-radius: 999px;
            background: #fef3c7;
            color: #92400e;
            font-weight: 700;
            font-size: 0.78rem;
          }
          div[data-testid="stTabs"] button[role="tab"] p {
            color: #334155 !important;
            font-weight: 600;
          }
          div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p {
            color: #0f172a !important;
            font-weight: 700;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_percent(value: float | None, decimals: int = 1) -> str:
    if value is None:
        return "n/d"
    return f"{value * 100:.{decimals}f}%"


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/d"
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.2f} s"


def compute_percentile(value: float, values: list[float]) -> float | None:
    if not values:
        return None
    rank = sum(1 for v in values if v <= value)
    return (rank / len(values)) * 100


def render_top_metrics(config: dict) -> None:
    cols = st.columns(6)
    metrics = config["metrics"]
    with cols[0]:
        render_metric_card("AUC", f'{metrics["auc"]:.4f}', "EfficientNet-B0 calibrado")
    with cols[1]:
        render_metric_card("Sensibilidade", f'{metrics["sensitivity"] * 100:.1f}%', "Meta clinica priorizando melanoma")
    with cols[2]:
        render_metric_card("Especificidade", f'{metrics["specificity"] * 100:.1f}%', "Controle de falso positivo")
    with cols[3]:
        render_metric_card("F1", f'{metrics["f1"]:.4f}', "Equilibrio entre precisao e recall")
    with cols[4]:
        render_metric_card("Precisao", f'{metrics["precision"] * 100:.1f}%', "Proporcao de alertas corretos")
    with cols[5]:
        render_metric_card("T_HIGH", f'{config["thresholds"]["t_high"]:.4f}', "Threshold operacional de alerta")


def render_risk_bar(case: CaseAnalysis) -> None:
    result = case.prediction
    probability = max(0.0, min(result.probability, 1.0))
    t_low = result.threshold_low
    t_high = result.threshold_high
    st.markdown(
        f"""
        <div class="risk-shell">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-weight:700;color:#0f172a;">Probabilidade prevista de melanoma</span>
            <span style="font-weight:800;color:#0f172a;">{probability * 100:.2f}%</span>
          </div>
          <div style="position:relative;margin-top:0.9rem;height:20px;border-radius:999px;overflow:hidden;background:#e2e8f0;">
            <div style="position:absolute;left:0;top:0;height:100%;width:{t_low * 100:.2f}%;background:#60a5fa;"></div>
            <div style="position:absolute;left:{t_low * 100:.2f}%;top:0;height:100%;width:{(t_high - t_low) * 100:.2f}%;background:#fbbf24;"></div>
            <div style="position:absolute;left:{t_high * 100:.2f}%;top:0;height:100%;width:{(1 - t_high) * 100:.2f}%;background:#ef4444;"></div>
            <div style="position:absolute;left:calc({probability * 100:.2f}% - 9px);top:1px;width:18px;height:18px;border-radius:50%;
                        background:#0f172a;border:3px solid #fff;box-shadow:0 8px 18px rgba(15,23,42,0.24);"></div>
          </div>
          <div style="display:flex;justify-content:space-between;margin-top:0.65rem;font-size:0.88rem;color:#334155;">
            <span>Baixo risco</span>
            <span>Revisao</span>
            <span>Alerta</span>
          </div>
          <div style="display:flex;justify-content:space-between;margin-top:0.5rem;font-size:0.86rem;color:#475569;">
            <span>T_LOW = {t_low:.6f}</span>
            <span>T_HIGH = {t_high:.6f}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


_PIPELINE_STEPS = [
    ("1 — Original", "Imagem bruta carregada"),
    ("2 — Mascara U-Net", "Pixels de lesao (branco) identificados pelo U-Net"),
    ("3 — Overlay", "Mascara vermelha sobreposta na imagem original"),
    ("4 — Bounding box", "Amarelo = bbox; verde = bbox expandida com margem"),
    ("5 — Crop + Pad", "Recorte centrado na lesao, padded para quadrado"),
    ("6 — Entrada 224x224", "Imagem final vista pelo EfficientNet-B0"),
]


def collect_with_live_preview(uploaded_files: list, predictor: SkinCancerPredictor) -> list[CaseAnalysis]:
    current_case_ids: list[str] = []
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        case_id = hashlib.sha1(
            f"{uploaded_file.name}:{len(file_bytes)}".encode("utf-8") + file_bytes
        ).hexdigest()[:12]

        if case_id in st.session_state.analysis_cache:
            current_case_ids.append(case_id)
            continue

        with st.status(f"Processando {uploaded_file.name}...", expanded=True) as status:
            st.markdown("**Pipeline de pre-processamento ao vivo**")
            cols = st.columns(len(_PIPELINE_STEPS), gap="small")
            placeholders = [col.empty() for col in cols]
            progress_msg = st.empty()

            def make_on_step(ph_list: list, msg_ph):
                def on_step(idx: int, image) -> None:
                    if idx >= len(_PIPELINE_STEPS):
                        return
                    label, caption = _PIPELINE_STEPS[idx]
                    msg_ph.markdown(f"Etapa: **{label}**")
                    if image is not None:
                        ph_list[idx].image(image, caption=caption, use_container_width=True)
                    else:
                        ph_list[idx].markdown(
                            f'<div style="text-align:center;padding:0.8rem;color:#94a3b8;'
                            f'font-size:0.82rem;border-radius:10px;background:#f1f5f9;">'
                            f'{label}<br>indisponivel</div>',
                            unsafe_allow_html=True,
                        )
                return on_step

            analysis = predictor.analyze_upload(
                file_bytes,
                uploaded_file.name,
                on_step=make_on_step(placeholders, progress_msg),
            )
            progress_msg.empty()
            status.update(
                label=f"{uploaded_file.name} — {analysis.prediction.zone_label} ({analysis.prediction.probability * 100:.1f}%)",
                state="complete",
                expanded=False,
            )

        add_analysis_to_state(analysis)
        current_case_ids.append(analysis.case_id)

    return [st.session_state.analysis_cache[cid] for cid in current_case_ids]


def collect_uploaded_analyses(uploaded_files: list, predictor: SkinCancerPredictor) -> list[CaseAnalysis]:
    current_case_ids: list[str] = []
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        case_id = hashlib.sha1(
            f"{uploaded_file.name}:{len(file_bytes)}".encode("utf-8") + file_bytes
        ).hexdigest()[:12]
        if case_id in st.session_state.analysis_cache:
            analysis = st.session_state.analysis_cache[case_id]
        else:
            analysis = predictor.analyze_upload(file_bytes, uploaded_file.name)
        add_analysis_to_state(analysis)
        current_case_ids.append(analysis.case_id)
    return [st.session_state.analysis_cache[case_id] for case_id in current_case_ids]


def render_case_header(case: CaseAnalysis) -> None:
    style = ZONE_STYLES[case.prediction.zone_key]
    dataset_hint = case.dataset_match.diagnosis_label or "Imagem externa"
    st.markdown(
        f"""
        <div class="result-card" style="background:{style["surface"]};">
          <div style="display:flex;justify-content:space-between;gap:1rem;align-items:flex-start;">
            <div>
              <div class="pill" style="background:{style["soft"]};color:{style["accent"]};">{case.prediction.zone_label}</div>
              <h2 style="margin:0.85rem 0 0.25rem 0;color:{style["accent"]};font-size:2.2rem;">{case.prediction.headline}</h2>
              <p style="margin:0;color:#1f2937;font-size:1.03rem;">{case.prediction.recommended_action}</p>
            </div>
            <div style="text-align:right;">
              <div style="font-size:0.84rem;color:#475569;text-transform:uppercase;letter-spacing:0.05em;">Caso ativo</div>
              <div style="font-weight:800;color:#0f172a;font-size:1.02rem;">{case.display_name}</div>
              <div style="margin-top:0.3rem;font-size:0.94rem;color:#334155;">Rotulo do dataset: {dataset_hint}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hotspot_gallery(title: str, hotspots: list) -> None:
    st.markdown(f"**{title}**")
    if not hotspots:
        st.info("Nenhum hotspot relevante foi extraido.")
        return
    cols = st.columns(len(hotspots))
    for col, hotspot in zip(cols, hotspots):
        with col:
            st.image(hotspot, use_container_width=True)


def render_abcde_composite(heuristics: list[dict]) -> None:
    valid = [h for h in heuristics if h["score"] is not None]
    if not valid:
        return
    composite = sum(h["score"] for h in valid) / len(valid)
    if composite >= 0.66:
        bar_color, label = "#ef4444", "Alto risco"
    elif composite >= 0.33:
        bar_color, label = "#fbbf24", "Moderado"
    else:
        bar_color, label = "#60a5fa", "Baixo risco"
    st.markdown(
        f"""
        <div style="border-radius:14px;padding:0.9rem 1rem;background:rgba(255,255,255,0.88);
                    border:1px solid rgba(15,23,42,0.08);margin-top:0.8rem;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-weight:700;color:#0f172a;">Score composto ABCDE</span>
            <span style="font-weight:800;color:{bar_color};">{composite * 100:.0f}/100 &mdash; {label}</span>
          </div>
          <div style="margin-top:0.7rem;height:12px;border-radius:999px;overflow:hidden;background:#e2e8f0;">
            <div style="height:100%;width:{composite * 100:.1f}%;background:{bar_color};border-radius:999px;"></div>
          </div>
          <div style="margin-top:0.5rem;font-size:0.85rem;color:#475569;">
            Baseado em {len(valid)} de 5 metricas disponiveis
            (A, B, C, D &mdash; E exige multiplas sessoes temporais)
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_heuristics(case: CaseAnalysis) -> None:
    st.markdown("**Leitura heuristica inspirada em ABCDE**")
    heuristics = case.lesion_heuristics
    if not heuristics:
        st.info("As heuristicas nao puderam ser calculadas para este caso.")
        return
    cols = st.columns(len(heuristics))
    for col, metric in zip(cols, heuristics):
        with col:
            score_text = "n/d" if metric["score"] is None else f'{metric["score"] * 100:.0f}/100'
            st.markdown(
                f"""
                <div class="heuristic-card">
                  <div class="heuristic-chip">{metric['code']}</div>
                  <h4 style="margin:0.7rem 0 0.2rem 0;">{metric['title']}</h4>
                  <div style="font-size:1.4rem;font-weight:800;color:#0f172a;">{score_text}</div>
                  <div style="margin-top:0.2rem;font-size:0.86rem;color:#92400e;text-transform:uppercase;font-weight:700;">
                    {metric['severity']}
                  </div>
                  <p style="margin:0.55rem 0 0 0;font-size:0.92rem;color:#475569;">{metric['description']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    render_abcde_composite(heuristics)


def render_similar_cases(case: CaseAnalysis) -> None:
    st.markdown("**Casos visualmente parecidos no dataset**")
    if not case.similar_cases:
        st.info("Nao foi possivel montar a busca de similares neste ambiente.")
        return
    cols = st.columns(min(4, len(case.similar_cases)))
    for col, similar in zip(cols, case.similar_cases):
        with col:
            st.image(similar.preview_image, use_container_width=True)
            st.caption(
                f"{similar.image_id} | {similar.diagnosis_label} | similaridade {similar.similarity:.3f}"
            )


def render_result_tab(case: CaseAnalysis, session_analyses: list[CaseAnalysis]) -> None:
    style = ZONE_STYLES[case.prediction.zone_key]
    top_left, top_right = st.columns([1.08, 1.0], gap="large")

    with top_left:
        st.markdown('<div class="section-kicker">Imagem original</div>', unsafe_allow_html=True)
        st.image(case.original_image, use_container_width=True)

        st.markdown('<div class="section-kicker" style="margin-top:0.8rem;">Transformacoes do pipeline</div>', unsafe_allow_html=True)
        pipe_cols = st.columns(3, gap="small")
        with pipe_cols[0]:
            st.image(case.uploaded_image, caption="Upload", use_container_width=True)
        with pipe_cols[1]:
            st.image(case.classifier_source_image, caption="Crop + Pad", use_container_width=True)
        with pipe_cols[2]:
            st.image(case.classifier_input_image, caption="224 x 224 (entrada do modelo)", use_container_width=True)

    with top_right:
        render_case_header(case)
        st.markdown("")
        render_risk_bar(case)
        st.markdown("")

        coverage_text = format_percent(case.segmentation_coverage, 1)
        clf_coverage_text = format_percent(case.classifier_segmentation_coverage, 1)
        scores = [h["score"] for h in case.lesion_heuristics if h.get("score") is not None]
        score_text = f"{(sum(scores) / len(scores)) * 100:.0f}/100" if scores else "n/d"
        st.markdown(
            f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;">
              <div style="border-radius:14px;padding:0.75rem 0.9rem;background:#f8fafc;border:1px solid #e2e8f0;">
                <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.05em;color:#64748b;font-weight:600;">Zona final</div>
                <div style="font-size:1.1rem;font-weight:800;color:{style['accent']};margin-top:0.2rem;">{case.prediction.zone_label}</div>
              </div>
              <div style="border-radius:14px;padding:0.75rem 0.9rem;background:#f8fafc;border:1px solid #e2e8f0;">
                <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.05em;color:#64748b;font-weight:600;">Score ABCD</div>
                <div style="font-size:1.1rem;font-weight:800;color:#0f172a;margin-top:0.2rem;">{score_text}</div>
              </div>
              <div style="border-radius:14px;padding:0.75rem 0.9rem;background:#f8fafc;border:1px solid #e2e8f0;">
                <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.05em;color:#64748b;font-weight:600;">Cobertura da lesao</div>
                <div style="font-size:1.1rem;font-weight:800;color:#0f172a;margin-top:0.2rem;">{coverage_text}</div>
                <div style="font-size:0.76rem;color:#94a3b8;margin-top:0.1rem;">% pixels de lesao na imagem original</div>
              </div>
              <div style="border-radius:14px;padding:0.75rem 0.9rem;background:#f8fafc;border:1px solid #e2e8f0;">
                <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.05em;color:#64748b;font-weight:600;">Cobertura no frame</div>
                <div style="font-size:1.1rem;font-weight:800;color:#0f172a;margin-top:0.2rem;">{clf_coverage_text}</div>
                <div style="font-size:0.76rem;color:#94a3b8;margin-top:0.1rem;">% pixels de lesao na entrada do modelo</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )



def render_outcome_indicator(case: CaseAnalysis) -> None:
    if not case.dataset_match.found or case.dataset_match.binary_label is None:
        return
    zone = case.prediction.zone_key
    binary = case.dataset_match.binary_label

    if zone == "review":
        outcome = "ZONA DE REVISAO"
        bg, color = "#fef3c7", "#92400e"
        desc = "Zona intermediaria — o modelo nao emitiu veredicto definitivo para este caso."
    elif zone == "positive" and binary == 1:
        outcome = "VERDADEIRO POSITIVO (TP)"
        bg, color = "#dcfce7", "#166534"
        desc = "Acerto: o modelo alertou para melanoma e o rotulo real do dataset confirma."
    elif zone == "negative" and binary == 0:
        outcome = "VERDADEIRO NEGATIVO (TN)"
        bg, color = "#dbeafe", "#1e40af"
        desc = "Acerto: o modelo classificou como benigno e o rotulo real do dataset confirma."
    elif zone == "positive" and binary == 0:
        outcome = "FALSO POSITIVO (FP)"
        bg, color = "#fff7ed", "#c2410c"
        desc = "Alarme falso: o modelo alertou para melanoma, mas o caso real e benigno."
    else:
        outcome = "FALSO NEGATIVO (FN) — CRITICO"
        bg, color = "#fee2e2", "#b91c1c"
        desc = "Erro critico: o modelo classificou como benigno, mas o caso real e melanoma."

    st.markdown(
        f"""
        <div style="border-radius:16px;padding:1rem 1.1rem;background:{bg};
                    border:2px solid {color}44;margin-bottom:0.9rem;">
          <div style="font-size:0.76rem;font-weight:700;text-transform:uppercase;
                      letter-spacing:0.07em;color:{color};margin-bottom:0.3rem;">
            Resultado da classificacao vs. ground truth
          </div>
          <div style="font-size:1.25rem;font-weight:800;color:{color};">{outcome}</div>
          <div style="margin-top:0.4rem;font-size:0.93rem;color:#374151;">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dataset_tab(case: CaseAnalysis) -> None:
    render_outcome_indicator(case)
    left, right = st.columns([0.95, 1.05], gap="large")
    with left:
        st.markdown("**Identificacao do dataset**")
        if case.dataset_match.found:
            st.write(f"ID reconhecido: `{case.dataset_match.image_id}`")
            st.write(f"Rotulo real: **{case.dataset_match.diagnosis_label or 'desconhecido'}**")
            if case.dataset_match.binary_label is not None:
                st.write(
                    f"Classe binaria real: {'melanoma' if case.dataset_match.binary_label == 1 else 'nao melanoma'}"
                )
            if case.truth_note:
                st.write(f"Leitura do resultado: {case.truth_note}")
        else:
            st.info("Este arquivo nao foi reconhecido no dataset local pelo nome.")

        st.markdown("**Origem do pipeline**")
        st.write(f"Modo de preparo: `{case.prediction.preprocessing_source}`")
        st.write(case.preprocessing_note)

        st.markdown("**Prevalencia no dataset**")
        stats = case.dataset_stats or {}
        st.write(f"Melanoma no dataset: {stats.get('melanoma_rate', 0.0) * 100:.2f}%")
        if stats.get("label_name"):
            st.write(
                f"Diagnostico {stats.get('label_name')}: {stats.get('label_rate', 0.0) * 100:.2f}%"
            )
        st.write(
            f"Classe prevista ({stats.get('predicted_label', 'n/d')}): "
            f"{stats.get('predicted_rate', 0.0) * 100:.2f}%"
        )

        if case.similar_cases:
            st.markdown("**Top similares (resumo)**")
            for similar in case.similar_cases[:3]:
                st.write(
                    f"{similar.image_id} | {similar.diagnosis_label} | similaridade {similar.similarity:.3f}"
                )

    with right:
        st.markdown("**Baixar relatorio deste caso**")
        report_json = json.dumps(case.report_dict(), indent=2, ensure_ascii=False)
        st.download_button(
            "Baixar JSON do caso",
            data=report_json,
            file_name=f"{case.display_name}_report.json",
            mime="application/json",
            use_container_width=True,
        )
        st.markdown("**Snapshot textual**")
        st.code(report_json, language="json")


def render_history_tab(analyses: list[CaseAnalysis]) -> None:
    if not analyses:
        st.info("Nenhum caso processado ainda.")
        return

    df = pd.DataFrame([case.summary_row() for case in analyses])

    # --- Cards de resumo da sessao ---
    st.markdown("**Resumo da sessao**")
    sum_cols = st.columns(4)
    sum_cols[0].metric("Total de casos", len(analyses))
    sum_cols[1].metric("Media de probabilidade", f"{df['prob_melanoma'].mean() * 100:.1f}%")
    alerts = int((df["zona"] == analyses[0].prediction.zone_label if False else df["zona"].str.contains("Melanoma") & ~df["zona"].str.contains("Possivel")).sum())
    sum_cols[2].metric("Casos em alerta", alerts)
    reviews = int(df["zona"].str.contains("Possivel").sum())
    sum_cols[3].metric("Casos em revisao", reviews)

    if len(analyses) >= 2:
        st.markdown("")
        st.markdown("**Distribuicao dos casos nesta sessao**")
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        fig.patch.set_facecolor("#f8fafc")

        zone_counts = df["zona"].value_counts()
        zone_color_map = {z: ("#ef4444" if "Possivel" not in z and "Melanoma" in z else "#fbbf24" if "Possivel" in z else "#60a5fa") for z in zone_counts.index}
        bar_colors = [zone_color_map.get(z, "#94a3b8") for z in zone_counts.index]
        axes[0].bar(zone_counts.index, zone_counts.values, color=bar_colors, width=0.55, edgecolor="white", linewidth=1.5)
        axes[0].set_title("Distribuicao de zonas", fontsize=11, pad=10, color="#0f172a")
        axes[0].set_ylabel("Casos", fontsize=9)
        axes[0].set_facecolor("#f8fafc")
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)
        axes[0].tick_params(axis="x", labelsize=9)
        axes[0].tick_params(axis="y", labelsize=9)

        t_low = analyses[0].prediction.threshold_low * 100
        t_high = analyses[0].prediction.threshold_high * 100
        prob_vals = df["prob_melanoma"].values * 100
        bins = max(5, min(len(analyses), 12))
        axes[1].hist(prob_vals, bins=bins, color="#60a5fa", edgecolor="white", rwidth=0.85)
        axes[1].axvline(x=t_low, color="#fbbf24", linestyle="--", linewidth=1.8, label=f"T_LOW ({t_low:.1f}%)")
        axes[1].axvline(x=t_high, color="#ef4444", linestyle="--", linewidth=1.8, label=f"T_HIGH ({t_high:.1f}%)")
        axes[1].set_title("Distribuicao de probabilidade de melanoma", fontsize=11, pad=10, color="#0f172a")
        axes[1].set_xlabel("Probabilidade (%)", fontsize=9)
        axes[1].set_ylabel("Casos", fontsize=9)
        axes[1].legend(fontsize=8)
        axes[1].set_facecolor("#f8fafc")
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        axes[1].tick_params(axis="both", labelsize=9)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("")
    st.markdown("**Tabela completa dos casos**")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "Baixar resumo em CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="session_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_model_tab(config: dict) -> None:
    st.markdown("**Resumo do experimento implantado**")
    st.write(
        "A aplicacao usa o checkpoint calibrado do EfficientNet-B0 com thresholds clinicos em tres zonas, "
        "somado a um U-Net de segmentacao para enriquecer o contexto visual e ajudar no recorte da lesao."
    )
    details_cols = st.columns(3)
    details_cols[0].write(f"Checkpoint: {config['checkpoint_path']}")
    details_cols[1].write(f"T_LOW: {config['thresholds']['t_low']:.6f}")
    details_cols[2].write(f"T_HIGH: {config['thresholds']['t_high']:.6f}")

    figure_specs = [
        ("Curvas ROC e PR", MODEL_FIG_DIR / "roc_pr_curves.png"),
        ("Trade-off por threshold", MODEL_FIG_DIR / "threshold_tradeoff.png"),
        ("Curvas de treino", MODEL_FIG_DIR / "training_curves.png"),
        ("Matriz de confusao", MODEL_FIG_DIR / "confusion_matrix.png"),
    ]

    available = [(title, path) for title, path in figure_specs if path.exists()]
    if not available:
        st.info("As figuras do experimento nao foram encontradas em `outputs/figures/modeling_2`.")
        return

    for idx in range(0, len(available), 2):
        row = st.columns(2, gap="large")
        for col, spec in zip(row, available[idx:idx + 2]):
            title, path = spec
            with col:
                st.markdown(f"**{title}**")
                st.image(str(path), use_container_width=True)


def render_pipeline_tab(case: CaseAnalysis) -> None:
    st.markdown(
        """
        <div class="panel-card" style="margin-bottom:1.1rem;">
          <div class="section-kicker">Como a imagem chega ao modelo</div>
          <p class="small-note" style="margin-top:0.3rem;">
            Cada etapa abaixo transforma a imagem antes que o EfficientNet-B0 a veja.
            O pipeline e determinado automaticamente dependendo se o arquivo foi reconhecido
            no dataset local ou se precisou de recorte via segmentacao.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    steps: list[tuple[str, object, str]] = [
        (
            "1 — Original",
            case.original_image,
            "Imagem bruta carregada (do dataset ou upload direto)",
        ),
        (
            "2 — Mascara U-Net",
            case.segmentation_mask_image,
            "Saida binaria do U-Net: branco = lesao, preto = fundo",
        ),
        (
            "3 — Overlay da segmentacao",
            case.segmentation_overlay,
            "Mascara em vermelho sobreposta na imagem original",
        ),
        (
            "4 — Bounding box",
            case.lesion_bbox_image,
            "Amarelo = bbox ajustada; verde = bbox expandida com margem",
        ),
        (
            "5 — Crop + Pad quadrado",
            case.classifier_source_image,
            "Recorte centrado na lesao, padded para proporcao quadrada",
        ),
        (
            "6 — Entrada do modelo (224x224)",
            case.classifier_input_image,
            "Versao final vista pelo EfficientNet-B0",
        ),
    ]

    valid = [(label, img, caption) for label, img, caption in steps if img is not None]

    n = len(valid)
    cols = st.columns(n, gap="small")
    for col, (label, img, caption) in zip(cols, valid):
        with col:
            st.markdown(
                f'<div style="font-size:0.78rem;font-weight:700;color:#0f172a;margin-bottom:0.4rem;">{label}</div>',
                unsafe_allow_html=True,
            )
            st.image(img, use_container_width=True)
            st.caption(caption)

    st.markdown("")
    st.markdown(
        f'<div class="panel-card"><b>Pipeline usado:</b> <code>{case.prediction.preprocessing_source}</code>'
        f"&nbsp;&mdash;&nbsp;{case.preprocessing_note}</div>",
        unsafe_allow_html=True,
    )


def _make_pipeline_figure(config: dict) -> plt.Figure:
    t_low = config["thresholds"]["t_low"]
    t_high = config["thresholds"]["t_high"]

    fig, ax = plt.subplots(figsize=(15, 4.8))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    ax.text(7.5, 4.55, "Pipeline de Inferência — Visão Geral", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#0f172a")

    # --- Caixas do pipeline principal ---
    PIPELINE = [
        (1.3,  3.2, "Imagem\nDermatoscópica", "#dbeafe", "#1e40af", "Dataset HAM10000\nou upload externo"),
        (3.6,  3.2, "U-Net\nSegmentação",     "#e0f2fe", "#0369a1", "Entrada 64×64\nSaída: máscara binária"),
        (6.0,  3.2, "BBox + Crop\n+ Pad Quadrado", "#dcfce7", "#15803d", "Margem 15%\nPad tipo 'edge'"),
        (8.5,  3.2, "EfficientNet-B0\nClassificador", "#faf5ff", "#7e22ce", "Entrada 224×224\nPré-treinado ImageNet"),
        (11.0, 3.2, "Calibração\nde Temperatura", "#fff7ed", "#c2410c", "Platt Scaling\nP(melanoma) ∈ [0,1]"),
    ]

    BW, BH = 1.85, 1.05
    for x, y, label, bg, color, sub in PIPELINE:
        rect = mpatches.FancyBboxPatch(
            (x - BW / 2, y - BH / 2), BW, BH,
            boxstyle="round,pad=0.09", facecolor=bg, edgecolor=color, linewidth=2.0, zorder=2,
        )
        ax.add_patch(rect)
        ax.text(x, y + 0.13, label, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=color, zorder=3)
        ax.text(x, y - 0.36, sub, ha="center", va="center",
                fontsize=7.0, color="#475569", zorder=3, style="italic")

    # Setas entre caixas
    for i in range(len(PIPELINE) - 1):
        x1 = PIPELINE[i][0] + BW / 2
        x2 = PIPELINE[i + 1][0] - BW / 2
        y_mid = PIPELINE[i][1]
        ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid),
                    arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.6, mutation_scale=14))

    # Seta para baixo da calibração → zonas
    x_cal = PIPELINE[-1][0]
    ax.annotate("", xy=(x_cal, 2.2), xytext=(x_cal, PIPELINE[-1][1] - BH / 2),
                arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.6, mutation_scale=14))
    ax.text(x_cal + 0.12, 2.35, "Sistema de 3 Zonas",
            ha="center", va="center", fontsize=8, fontweight="bold", color="#475569")

    # --- 3 Zonas ---
    ZONES = [
        (9.3,  1.35, f"Zona Negativa\np < {t_low:.4f}", "#dbeafe", "#1d4ed8"),
        (11.0, 1.35, f"Revisão\n{t_low:.4f} ≤ p < {t_high:.4f}", "#fef3c7", "#b45309"),
        (12.7, 1.35, f"Zona Positiva\np ≥ {t_high:.4f}", "#fee2e2", "#b91c1c"),
    ]
    ZW, ZH = 1.5, 0.8
    for x, y, label, bg, color in ZONES:
        rect = mpatches.FancyBboxPatch(
            (x - ZW / 2, y - ZH / 2), ZW, ZH,
            boxstyle="round,pad=0.07", facecolor=bg, edgecolor=color, linewidth=1.8, zorder=2,
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color=color, zorder=3)

    # Linha horizontal ligando as 3 zonas
    ax.plot([ZONES[0][0] - ZW / 2, ZONES[-1][0] + ZW / 2], [1.75, 1.75],
            color="#94a3b8", lw=1.2, ls="--", zorder=1)

    # Linha vertical caindo sobre a barra de zonas
    ax.plot([x_cal, x_cal], [2.18, 1.75], color="#94a3b8", lw=1.2, zorder=1)
    ax.plot([x_cal, ZONES[0][0]], [1.75, 1.75], color="#94a3b8", lw=1.2, zorder=1)

    # Legenda de módulos
    legend_items = [
        mpatches.Patch(facecolor="#dbeafe", edgecolor="#1e40af", label="Entrada"),
        mpatches.Patch(facecolor="#e0f2fe", edgecolor="#0369a1", label="U-Net (Segmentação)"),
        mpatches.Patch(facecolor="#faf5ff", edgecolor="#7e22ce", label="EfficientNet-B0 (Classificação)"),
        mpatches.Patch(facecolor="#fff7ed", edgecolor="#c2410c", label="Calibração de Temperatura"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=7.5,
              framealpha=0.9, edgecolor="#e2e8f0", ncol=4)

    plt.tight_layout(pad=0.3)
    return fig


def _section(title: str, body: str) -> str:
    return f"""
    <div style="border-radius:16px;padding:1.1rem 1.2rem;background:#ffffff;
                border:1px solid #e2e8f0;box-shadow:0 2px 12px rgba(15,23,42,0.05);
                margin-bottom:0.7rem;">
      <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.07em;
                  color:#64748b;font-weight:700;margin-bottom:0.55rem;">{title}</div>
      <div style="font-size:0.93rem;color:#1e293b;line-height:1.65;">{body}</div>
    </div>"""


def render_architecture_tab(config: dict) -> None:
    metrics = config["metrics"]
    t_low  = config["thresholds"]["t_low"]
    t_high = config["thresholds"]["t_high"]

    # Diagrama
    st.markdown("**Diagrama do pipeline de inferência**")
    fig = _make_pipeline_figure(config)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("")

    # Seções de texto: 3 colunas
    col_a, col_b, col_c = st.columns(3, gap="large")

    with col_a:
        st.markdown(_section(
            "Problema e Dataset",
            """O melanoma é o tipo de câncer de pele com maior letalidade. Detectado precocemente,
            a taxa de sobrevivência em 5 anos supera 98%; diagnosticado tardiamente, cai para menos de 25%.
            <br><br>
            O modelo foi treinado no <strong>HAM10000</strong> (<em>Human Against Machine with 10000 training images</em>),
            benchmark público do ISIC Archive com 10&nbsp;015 imagens dermatoscópicas de 7 classes de lesões cutâneas.
            O problema foi reformulado como <strong>classificação binária</strong>:
            melanoma (<em>MEL</em>) vs. não-melanoma (demais classes).
            <br><br>
            O dataset é fortemente desbalanceado — apenas ~11% das amostras são melanoma —
            o que motivou estratégias específicas de amostragem e calibração.""",
        ), unsafe_allow_html=True)

        st.markdown(_section(
            "Segmentação — U-Net",
            """A segmentação da lesão é realizada por uma <strong>U-Net</strong> treinada do zero,
            com arquitetura encoder–decoder e <em>skip connections</em> em 4 níveis de resolução
            (64 → 32 → 16 → 8 → 4 pixels).
            <br><br>
            A entrada é redimensionada para <strong>64×64</strong> para equilibrar custo computacional
            e qualidade da máscara. A saída é uma máscara binária do mesmo tamanho, depois
            upscalada bilinearmente para a resolução original.
            <br><br>
            A máscara é usada para calcular um <em>bounding box</em> expandido com margem de 15%,
            a partir do qual a imagem é recortada e padded para proporcão quadrada antes de
            ser enviada ao classificador.""",
        ), unsafe_allow_html=True)

    with col_b:
        st.markdown(_section(
            "Classificação — EfficientNet-B0",
            """O classificador é um <strong>EfficientNet-B0</strong> com pesos pré-treinados no
            ImageNet, fine-tuned no HAM10000 com a formulação binária.
            <br><br>
            O EfficientNet escala profundidade, largura e resolução da rede de forma conjunta
            (<em>compound scaling</em>), obtendo alta acurácia com poucos parâmetros (~5,3M).
            A entrada é <strong>224×224 pixels</strong> normalizada com a média e desvio-padrão
            calculados sobre o próprio dataset.
            <br><br>
            A saída da rede é um logit escalar convertido em probabilidade via sigmoide.
            O treinamento usou <em>Binary Cross-Entropy</em> sem <em>pos_weight</em>,
            complementado por augmentação geométrica e de cor.""",
        ), unsafe_allow_html=True)

        st.markdown(_section(
            "Explicabilidade",
            """Duas técnicas de explicabilidade são aplicadas sobre o classificador:
            <br><br>
            <strong>Grad-CAM</strong> — calcula o gradiente do score de saída em relação aos
            mapas de ativação da última camada convolucional (<code>conv_head</code>),
            produzindo um mapa de calor que destaca as regiões mais relevantes para a decisão.
            <br><br>
            <strong>Oclusão sistemática</strong> — percorre a imagem com uma janela de 40×40 px
            (passo 28 px) substituindo cada região pela cor média local e mede a queda
            de probabilidade. Regiões cuja oclusão causa maior queda são consideradas
            as mais informativas para o modelo.""",
        ), unsafe_allow_html=True)

    with col_c:
        st.markdown(_section(
            "Calibração e Sistema de 3 Zonas",
            f"""O score bruto do EfficientNet-B0 tende a ser mal-calibrado em datasets
            desbalanceados. A probabilidade final é ajustada via <strong>Platt Scaling</strong>
            (regressão logística sobre os logits do conjunto de validação).
            <br><br>
            O sistema de decisão usa <strong>dois thresholds</strong> em vez de um único ponto
            de corte, criando três zonas operacionais:
            <br><br>
            <span style="color:#1d4ed8;font-weight:700;">Negativa</span> — p &lt; {t_low:.4f}:
            triagem de baixo risco; improvável ser melanoma segundo o modelo.<br>
            <span style="color:#b45309;font-weight:700;">Revisão</span> — {t_low:.4f} ≤ p &lt; {t_high:.4f}:
            caso incerto; encaminhar para avaliação dermatológica.<br>
            <span style="color:#b91c1c;font-weight:700;">Positiva</span> — p ≥ {t_high:.4f}:
            alerta clínico; priorizar revisão especializada.
            <br><br>
            Os thresholds foram otimizados para atingir sensibilidade ≥ 85% para melanoma,
            aceitando especificidade reduzida para minimizar falsos negativos.""",
        ), unsafe_allow_html=True)

    # Linha de métricas
    st.markdown("")
    st.markdown("**Desempenho do modelo no conjunto de teste (HAM10000)**")
    m_cols = st.columns(5)
    for col, (label, val) in zip(m_cols, [
        ("AUC",          f'{metrics["auc"]:.4f}'),
        ("Sensibilidade",f'{metrics["sensitivity"]*100:.1f}%'),
        ("Especificidade",f'{metrics["specificity"]*100:.1f}%'),
        ("F1",           f'{metrics["f1"]:.4f}'),
        ("Precisão",     f'{metrics["precision"]*100:.1f}%'),
    ]):
        col.markdown(
            f'<div style="border-radius:12px;padding:0.7rem 0.8rem;background:#f8fafc;'
            f'border:1px solid #e2e8f0;text-align:center;">'
            f'<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.05em;'
            f'color:#64748b;font-weight:600;">{label}</div>'
            f'<div style="font-size:1.4rem;font-weight:800;color:#0f172a;margin-top:0.2rem;">{val}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.caption(
        "Ferramenta desenvolvida para fins acadêmicos (Insper — AI in Medicine). "
        "Não substitui avaliação dermatológica clínica."
    )


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="panel-card">
          <div class="section-kicker">O que esta versao faz</div>
          <h3 style="margin:0.35rem 0 0.4rem 0;">Demo completa do pipeline</h3>
          <p class="small-note">
            Envie uma ou mais imagens para obter classificacao em tres zonas, segmentacao da lesao,
            comparacao com o dataset local quando houver correspondencia por ID e um resumo exportavel
            da sessao.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Dermoscopy Triage Studio",
        page_icon=":microscope:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    ensure_state()
    render_styles()
    config = load_config()
    predictor = load_predictor(APP_BUILD_ID)
    if not hasattr(predictor, "analyze_upload"):
        load_predictor.clear()
        predictor = load_predictor(APP_BUILD_ID)

    st.markdown(
        """
        <div class="hero-shell">
          <div class="section-kicker" style="color:#fbbf24;">AI in Medicine Demo</div>
          <h1>Triagem dermatoscopica com classificacao e segmentacao assistida</h1>
          <p>
            Esta interface combina o EfficientNet-B0 final do projeto com segmentacao visual da lesao,
            lookup no dataset local e um resumo em lote para apresentacao.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_top_metrics(config)

    with st.sidebar:
        st.subheader("Sessao")
        st.caption("Casos processados ficam salvos durante a sessao atual.")
        if st.button("Limpar historico", use_container_width=True):
            st.session_state.analysis_cache = {}
            st.session_state.analysis_order = []
            st.rerun()

    st.markdown("")
    uploaded_files = st.file_uploader(
        "Envie uma ou mais imagens dermatoscopicas",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Se o nome corresponder a um ID ISIC do dataset local, a app tambem compara com o rotulo real.",
    )

    current_analyses = collect_with_live_preview(uploaded_files, predictor) if uploaded_files else []
    session_analyses = get_session_analyses()

    if not session_analyses:
        render_empty_state()
        st.markdown("---")
        render_architecture_tab(config)
        return

    options = [case.case_id for case in session_analyses]
    current_default = current_analyses[-1].case_id if current_analyses else options[-1]
    active_case_id = st.selectbox(
        "Caso ativo",
        options=options,
        index=options.index(current_default),
        format_func=lambda case_id: f"{st.session_state.analysis_cache[case_id].display_name}  |  {st.session_state.analysis_cache[case_id].prediction.zone_label}",
    )
    active_case = st.session_state.analysis_cache[active_case_id]

    result_tab, pipeline_tab, dataset_tab, history_tab, arch_tab, model_tab = st.tabs(
        ["Resultado", "Pipeline", "Dataset", "Lote e historico", "Arquitetura", "Modelo"]
    )

    with result_tab:
        render_result_tab(active_case, session_analyses)
    with pipeline_tab:
        render_pipeline_tab(active_case)
    with dataset_tab:
        render_dataset_tab(active_case)
    with history_tab:
        render_history_tab(session_analyses)
    with arch_tab:
        render_architecture_tab(config)
    with model_tab:
        render_model_tab(config)


if __name__ == "__main__":
    main()
