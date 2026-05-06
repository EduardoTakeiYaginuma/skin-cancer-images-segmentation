from __future__ import annotations

import hashlib
import json
from pathlib import Path

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
    del build_id
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


def render_top_metrics(config: dict) -> None:
    cols = st.columns(4)
    metrics = config["metrics"]
    with cols[0]:
        render_metric_card("AUC", f'{metrics["auc"]:.4f}', "Checkpoint final EfficientNet-B0")
    with cols[1]:
        render_metric_card("Sensibilidade", f'{metrics["sensitivity"] * 100:.1f}%', "Meta clinica priorizando melanoma")
    with cols[2]:
        render_metric_card("Especificidade", f'{metrics["specificity"] * 100:.1f}%', "Controle de falso positivo")
    with cols[3]:
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


def render_result_tab(case: CaseAnalysis) -> None:
    style = ZONE_STYLES[case.prediction.zone_key]
    top_left, top_right = st.columns([1.08, 1.0], gap="large")

    with top_left:
        st.markdown('<div class="section-kicker">Imagem base</div>', unsafe_allow_html=True)
        st.image(case.original_image, caption="Imagem original usada para contextualizacao", use_container_width=True)
        cols = st.columns(2)
        with cols[0]:
            st.markdown('<div class="section-kicker">Entrada enviada</div>', unsafe_allow_html=True)
            st.image(case.uploaded_image, caption="Arquivo enviado na sessao", use_container_width=True)
        with cols[1]:
            st.markdown('<div class="section-kicker">Pipeline de classificacao</div>', unsafe_allow_html=True)
            st.image(case.classifier_source_image, caption="Imagem fonte usada pelo classificador", use_container_width=True)

    with top_right:
        render_case_header(case)
        st.markdown("")
        render_risk_bar(case)
        st.markdown("")
        facts = st.columns(3)
        facts[0].metric("Zona final", case.prediction.zone_label)
        facts[1].metric("Device", case.prediction.device)
        coverage_text = f"{case.segmentation_coverage * 100:.1f}%" if case.segmentation_coverage is not None else "n/d"
        facts[2].metric("Cobertura da lesao", coverage_text)

        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("**Interpretacao operacional**")
        st.write(case.preprocessing_note)
        if case.truth_note:
            st.write(case.truth_note)
        st.caption("Ferramenta academica. O resultado nao substitui avaliacao dermatologica.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("")
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("**Relatorio rapido**")
        st.write(
            f"O caso entrou em **{case.prediction.zone_label}** com probabilidade prevista de "
            f"**{case.prediction.probability * 100:.2f}%**."
        )
        st.write(
            f"O pipeline usado para classificar foi **{case.prediction.preprocessing_source}**, "
            f"o que ajuda a explicar como a imagem chegou ao modelo final."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    bottom_cols = st.columns(2, gap="large")
    with bottom_cols[0]:
        st.markdown('<div class="section-kicker">Entrada do modelo</div>', unsafe_allow_html=True)
        st.image(case.classifier_input_image, caption="Versao redimensionada para 224x224", use_container_width=True)
    with bottom_cols[1]:
        st.markdown('<div class="section-kicker">Heatmap Grad-CAM</div>', unsafe_allow_html=True)
        if case.gradcam_overlay:
            st.image(case.gradcam_overlay, caption="Ativacao visual do EfficientNet-B0", use_container_width=True)
        else:
            st.info("Grad-CAM indisponivel para este caso.")


def render_explainability_tab(case: CaseAnalysis) -> None:
    top_row = st.columns(3, gap="large")
    with top_row[0]:
        st.markdown("**Segmentacao no contexto original**")
        if case.segmentation_overlay:
            st.image(
                case.segmentation_overlay,
                caption="Mascara prevista sobre a imagem original",
                use_container_width=True,
            )
        else:
            st.info("Segmentacao nao disponivel neste ambiente.")
    with top_row[1]:
        st.markdown("**Grad-CAM do classificador**")
        if case.gradcam_overlay:
            st.image(
                case.gradcam_overlay,
                caption="Ativacao visual na entrada 224x224",
                use_container_width=True,
            )
        else:
            st.info("Heatmap Grad-CAM indisponivel.")
    with top_row[2]:
        st.markdown("**Overlay combinado**")
        if case.combined_explainability_overlay:
            st.image(
                case.combined_explainability_overlay,
                caption="Grad-CAM combinado com segmentacao no frame do classificador",
                use_container_width=True,
            )
        else:
            st.info("Overlay combinado indisponivel.")

    second_row = st.columns(3, gap="large")
    with second_row[0]:
        st.markdown("**Segmentacao no frame do modelo**")
        if case.classifier_segmentation_overlay:
            st.image(
                case.classifier_segmentation_overlay,
                caption="Mascara prevista na imagem usada para classificar",
                use_container_width=True,
            )
        else:
            st.info("Segmentacao alinhada ao frame do modelo indisponivel.")
    with second_row[1]:
        st.markdown("**Sensibilidade por oclusao**")
        if case.occlusion_overlay:
            st.image(
                case.occlusion_overlay,
                caption="Queda de score quando partes da imagem sao ocultadas",
                use_container_width=True,
            )
        else:
            st.info("Mapa de oclusao indisponivel.")
    with second_row[2]:
        st.markdown("**Entrada final do classificador**")
        st.image(
            case.classifier_input_image,
            caption="Visao efetiva do EfficientNet-B0",
            use_container_width=True,
        )

    hotspot_left, hotspot_right = st.columns(2, gap="large")
    with hotspot_left:
        render_hotspot_gallery("Hotspots por Grad-CAM", case.gradcam_hotspots)
    with hotspot_right:
        render_hotspot_gallery("Hotspots por oclusao", case.occlusion_hotspots)

    render_heuristics(case)
    st.markdown("")
    render_similar_cases(case)

    if case.ground_truth_overlay:
        st.markdown("**Comparacao com a mascara real do dataset**")
        st.image(
            case.ground_truth_overlay,
            caption="Mascara real conhecida para este caso",
            use_container_width=True,
        )


def render_dataset_tab(case: CaseAnalysis) -> None:
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


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="panel-card">
          <div class="section-kicker">O que esta versao faz</div>
          <h3 style="margin:0.35rem 0 0.4rem 0;">Demo completa do pipeline</h3>
          <p class="small-note">
            Envie uma ou mais imagens para obter classificacao em tres zonas, segmentacao da lesao,
            Grad-CAM, comparacao com o dataset local quando houver correspondencia por ID e um resumo
            exportavel da sessao.
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
          <h1>Triagem dermatoscopica com classificacao, segmentacao e explicabilidade</h1>
          <p>
            Esta interface combina o EfficientNet-B0 final do projeto com segmentacao visual da lesao,
            Grad-CAM, lookup no dataset local e um resumo em lote para apresentacao.
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

    current_analyses = collect_uploaded_analyses(uploaded_files, predictor) if uploaded_files else []
    session_analyses = get_session_analyses()

    if not session_analyses:
        render_empty_state()
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

    result_tab, explain_tab, dataset_tab, history_tab, model_tab = st.tabs(
        ["Resultado", "Explicabilidade", "Dataset", "Lote e historico", "Modelo"]
    )

    with result_tab:
        render_result_tab(active_case)
    with explain_tab:
        render_explainability_tab(active_case)
    with dataset_tab:
        render_dataset_tab(active_case)
    with history_tab:
        render_history_tab(session_analyses)
    with model_tab:
        render_model_tab(config)


if __name__ == "__main__":
    main()
