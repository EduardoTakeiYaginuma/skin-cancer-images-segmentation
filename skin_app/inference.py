from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


DIAGNOSIS_LABELS = {
    "MEL": "Melanoma",
    "NV": "Nevo melanocitico",
    "BCC": "Carcinoma basocelular",
    "AKIEC": "Queratoses actinicas / Bowen",
    "BKL": "Queratose benigna",
    "DF": "Dermatofibroma",
    "VASC": "Lesao vascular",
}


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def rgb_array_to_pil(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array.astype(np.uint8), mode="RGB")


def load_pil_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def normalize_minmax(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value - min_value < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_value) / (max_value - min_value)


def compute_bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def expand_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    margin_ratio: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    box_w = max(x1 - x0, 1)
    box_h = max(y1 - y0, 1)
    margin_x = max(1, int(round(box_w * margin_ratio)))
    margin_y = max(1, int(round(box_h * margin_ratio)))
    return (
        max(0, x0 - margin_x),
        max(0, y0 - margin_y),
        min(width, x1 + margin_x),
        min(height, y1 + margin_y),
    )


def crop_array(array: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return array[y0:y1, x0:x1]


def pad_to_square(array: np.ndarray) -> np.ndarray:
    height, width = array.shape[:2]
    side = max(height, width)
    pad_top = (side - height) // 2
    pad_bottom = side - height - pad_top
    pad_left = (side - width) // 2
    pad_right = side - width - pad_left

    if array.ndim == 3:
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    else:
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
    return np.pad(array, pad_width=pad_width, mode="edge")


def overlay_mask_on_image(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color_rgb: tuple[int, int, int],
    alpha: float = 0.42,
) -> Image.Image:
    image = image_rgb.astype(np.float32).copy()
    overlay = np.zeros_like(image)
    overlay[..., 0] = color_rgb[0]
    overlay[..., 1] = color_rgb[1]
    overlay[..., 2] = color_rgb[2]
    mask_3d = np.repeat((mask > 0)[..., None], 3, axis=2)
    image[mask_3d] = image[mask_3d] * (1.0 - alpha) + overlay[mask_3d] * alpha
    return rgb_array_to_pil(np.clip(image, 0, 255).astype(np.uint8))


def render_binary_mask(mask: np.ndarray) -> Image.Image:
    mask_uint8 = ((mask > 0).astype(np.uint8) * 255)
    mask_rgb = np.repeat(mask_uint8[..., None], 3, axis=2)
    return rgb_array_to_pil(mask_rgb)


def render_heatmap_overlay(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.38) -> Image.Image:
    heatmap_uint8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    base = image_rgb.astype(np.float32)
    blended = np.clip(base * (1.0 - alpha) + heatmap_rgb * alpha, 0, 255)
    return rgb_array_to_pil(blended.astype(np.uint8))


def render_combined_explainability(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    mask: np.ndarray | None,
) -> Image.Image:
    combined = np.asarray(render_heatmap_overlay(image_rgb, heatmap, alpha=0.34)).astype(np.float32)
    if mask is not None:
        edge = cv2.Canny((mask > 0).astype(np.uint8) * 255, 40, 120) > 0
        combined[mask > 0] = combined[mask > 0] * 0.78 + np.array([255.0, 236.0, 179.0]) * 0.22
        combined[edge] = np.array([12.0, 74.0, 110.0])
    return rgb_array_to_pil(np.clip(combined, 0, 255).astype(np.uint8))


def extract_hotspot_crops(
    image: Image.Image,
    score_map: np.ndarray,
    top_k: int = 3,
    crop_fraction: float = 0.34,
) -> list[Image.Image]:
    score = normalize_minmax(score_map).copy()
    image_rgb = pil_to_rgb_array(image)
    height, width = score.shape
    crop_side = max(48, int(round(min(height, width) * crop_fraction)))
    suppression_radius = max(12, crop_side // 2)
    crops: list[Image.Image] = []

    for _ in range(top_k):
        peak_idx = int(np.argmax(score))
        peak_value = float(score.flat[peak_idx])
        if peak_value <= 1e-6:
            break
        cy, cx = divmod(peak_idx, width)
        x0 = max(0, cx - crop_side // 2)
        y0 = max(0, cy - crop_side // 2)
        x1 = min(width, x0 + crop_side)
        y1 = min(height, y0 + crop_side)
        x0 = max(0, x1 - crop_side)
        y0 = max(0, y1 - crop_side)
        crops.append(rgb_array_to_pil(image_rgb[y0:y1, x0:x1]))

        sx0 = max(0, cx - suppression_radius)
        sy0 = max(0, cy - suppression_radius)
        sx1 = min(width, cx + suppression_radius)
        sy1 = min(height, cy + suppression_radius)
        score[sy0:sy1, sx0:sx1] = 0.0

    return crops


def resize_binary_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return (cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)


def format_severity(score: float) -> str:
    if score < 0.33:
        return "baixo"
    if score < 0.66:
        return "moderado"
    return "alto"


@dataclass(frozen=True)
class DatasetMatch:
    found: bool
    image_id: str | None = None
    diagnosis_code: str | None = None
    diagnosis_label: str | None = None
    binary_label: int | None = None
    raw_path: str | None = None
    treated_path: str | None = None
    mask_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SimilarCase:
    image_id: str
    diagnosis_label: str
    similarity: float
    preview_image: Image.Image

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "diagnosis_label": self.diagnosis_label,
            "similarity": self.similarity,
        }


@dataclass(frozen=True)
class PredictionResult:
    probability: float
    zone_key: str
    zone_label: str
    headline: str
    recommended_action: str
    threshold_low: float
    threshold_high: float
    device: str
    predicted_binary: int
    preprocessing_source: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CaseAnalysis:
    case_id: str
    file_name: str
    display_name: str
    prediction: PredictionResult
    dataset_match: DatasetMatch
    original_image: Image.Image
    uploaded_image: Image.Image
    classifier_source_image: Image.Image
    classifier_input_image: Image.Image
    segmentation_overlay: Image.Image | None
    segmentation_mask_image: Image.Image | None
    classifier_segmentation_overlay: Image.Image | None
    combined_explainability_overlay: Image.Image | None
    ground_truth_overlay: Image.Image | None
    gradcam_overlay: Image.Image | None
    occlusion_overlay: Image.Image | None
    gradcam_hotspots: list[Image.Image]
    occlusion_hotspots: list[Image.Image]
    similar_cases: list[SimilarCase]
    lesion_heuristics: list[dict]
    segmentation_coverage: float | None
    classifier_segmentation_coverage: float | None
    preprocessing_note: str
    truth_note: str | None

    def summary_row(self) -> dict:
        return {
            "arquivo": self.file_name,
            "caso": self.display_name,
            "prob_melanoma": round(self.prediction.probability, 4),
            "zona": self.prediction.zone_label,
            "pipeline": self.prediction.preprocessing_source,
            "rotulo_real": self.dataset_match.diagnosis_label or "desconhecido",
            "avaliacao": self.truth_note or "sem verdade-terreno",
        }

    def report_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "file_name": self.file_name,
            "display_name": self.display_name,
            "prediction": self.prediction.to_dict(),
            "dataset_match": self.dataset_match.to_dict(),
            "segmentation_coverage": self.segmentation_coverage,
            "classifier_segmentation_coverage": self.classifier_segmentation_coverage,
            "similar_cases": [case.to_dict() for case in self.similar_cases],
            "lesion_heuristics": self.lesion_heuristics,
            "preprocessing_note": self.preprocessing_note,
            "truth_note": self.truth_note,
        }


class SegmentationUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(0.3)

        self.enc1 = self._double_conv(3, 64)
        self.enc2 = self._double_conv(64, 128)
        self.enc3 = self._double_conv(128, 256)
        self.enc4 = self._double_conv(256, 512)
        self.bridge = self._double_conv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._double_conv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._double_conv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._double_conv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._double_conv(128, 64)
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    @staticmethod
    def _double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(self.drop(self.pool(s1)))
        s3 = self.enc3(self.drop(self.pool(s2)))
        s4 = self.enc4(self.drop(self.pool(s3)))
        bridge = self.bridge(self.drop(self.pool(s4)))

        x = self.dec4(torch.cat([self.up4(bridge), s4], dim=1))
        x = self.dec3(torch.cat([self.up3(x), s3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), s2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), s1], dim=1))
        return self.out_conv(x)


class SkinCancerPredictor:
    def __init__(self, config_path: str | Path | None = None, device: str | None = None) -> None:
        self.repo_root = resolve_repo_root()
        self.config_path = Path(config_path) if config_path else self.repo_root / "config" / "inference_config.json"
        self.config = self._load_config(self.config_path)

        self.image_size = int(self.config["image_size"])
        self.mean = torch.tensor(self.config["normalization"]["mean"], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(self.config["normalization"]["std"], dtype=torch.float32).view(3, 1, 1)
        self.threshold_low = float(self.config["thresholds"]["t_low"])
        self.threshold_high = float(self.config["thresholds"]["t_high"])

        segmentation_cfg = self.config.get("segmentation", {})
        self.seg_image_size = int(segmentation_cfg.get("image_size", 64))
        self.seg_threshold = float(segmentation_cfg.get("threshold", 0.5))
        self.crop_margin_ratio = float(segmentation_cfg.get("crop_margin_ratio", 0.15))
        self.seg_checkpoint_rel = segmentation_cfg.get("checkpoint_path")

        self.device = torch.device(device) if device else self._select_device()
        self.model = self._load_classifier()
        self.segmentation_model = self._load_segmentation_model()

        self.metadata_df = pd.read_csv(self.repo_root / "data" / "metadata.csv")
        self.metadata_by_id = self.metadata_df.set_index("image")

        manifest_path = self.repo_root / "data" / "processed" / "treated_manifest.csv"
        self.treated_manifest_df = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
        self.treated_by_id = (
            self.treated_manifest_df.set_index("image_id") if not self.treated_manifest_df.empty else pd.DataFrame()
        )
        self._similarity_vectors: np.ndarray | None = None
        self._similarity_records: list[tuple[str, str, str]] = []

    @staticmethod
    def _load_config(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)

    @staticmethod
    def _select_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_classifier(self) -> nn.Module:
        model = timm.create_model(self.config["model_name"], pretrained=False, num_classes=1)
        checkpoint_path = self.repo_root / self.config["checkpoint_path"]
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        return model

    def _load_segmentation_model(self) -> SegmentationUNet | None:
        if not self.seg_checkpoint_rel:
            return None
        checkpoint_path = self.repo_root / self.seg_checkpoint_rel
        if not checkpoint_path.exists():
            return None
        model = SegmentationUNet()
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        return model

    def _lookup_dataset_match(self, image_id: str) -> DatasetMatch:
        raw_path = self.repo_root / "data" / "images" / f"{image_id}.jpg"
        mask_path = self.repo_root / "data" / "masks" / f"{image_id}.png"

        diagnosis_code = None
        diagnosis_label = None
        binary_label = None
        if image_id in self.metadata_by_id.index:
            row = self.metadata_by_id.loc[image_id]
            diagnosis_candidates = [code for code in DIAGNOSIS_LABELS if float(row.get(code, 0.0)) == 1.0]
            if diagnosis_candidates:
                diagnosis_code = diagnosis_candidates[0]
                diagnosis_label = DIAGNOSIS_LABELS[diagnosis_code]
                binary_label = 1 if diagnosis_code == "MEL" else 0

        treated_path = None
        if not self.treated_manifest_df.empty and image_id in self.treated_by_id.index:
            treated_path = str(self.treated_by_id.loc[image_id, "export_path"])

        found = any(
            [
                diagnosis_code is not None,
                raw_path.exists(),
                mask_path.exists(),
                treated_path is not None,
            ]
        )
        return DatasetMatch(
            found=found,
            image_id=image_id if found else None,
            diagnosis_code=diagnosis_code,
            diagnosis_label=diagnosis_label,
            binary_label=binary_label,
            raw_path=str(raw_path) if raw_path.exists() else None,
            treated_path=treated_path,
            mask_path=str(mask_path) if mask_path.exists() else None,
        )

    def _classifier_tensor(self, image: Image.Image) -> tuple[torch.Tensor, np.ndarray]:
        resized = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        image_np = np.asarray(resized).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
        tensor = (tensor - self.mean) / self.std
        return tensor.unsqueeze(0), np.asarray(resized)

    def _predict_probability(self, image: Image.Image) -> float:
        batch, _ = self._classifier_tensor(image)
        batch = batch.to(self.device)
        with torch.no_grad():
            logits = self.model(batch).squeeze()
            probability = float(torch.sigmoid(logits).item())
        return probability

    def _predict_segmentation_mask(self, image: Image.Image) -> tuple[np.ndarray | None, float | None]:
        if self.segmentation_model is None:
            return None, None

        resized = image.convert("RGB").resize((self.seg_image_size, self.seg_image_size), Image.Resampling.BILINEAR)
        image_np = np.asarray(resized).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.segmentation_model(tensor)
            probs = torch.sigmoid(logits).squeeze().detach().cpu().numpy()

        original_rgb = pil_to_rgb_array(image)
        prob_resized = cv2.resize(
            probs.astype(np.float32),
            (original_rgb.shape[1], original_rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        mask = (prob_resized >= self.seg_threshold).astype(np.uint8)
        coverage = float(mask.mean())
        return mask, coverage

    def _crop_for_classifier(
        self,
        image: Image.Image,
        segmentation_mask: np.ndarray | None,
    ) -> tuple[Image.Image, str, str]:
        image_rgb = pil_to_rgb_array(image)

        if segmentation_mask is None:
            square = pad_to_square(image_rgb)
            note = "Segmentacao indisponivel. A classificacao usou apenas o enquadramento quadrado da imagem enviada."
            return rgb_array_to_pil(square), "fallback_square", note

        bbox = compute_bbox_from_mask(segmentation_mask)
        if bbox is None:
            square = pad_to_square(image_rgb)
            note = "A mascara prevista ficou vazia. A classificacao usou o quadro completo da imagem enviada."
            return rgb_array_to_pil(square), "fallback_square", note

        expanded = expand_bbox(bbox, width=image_rgb.shape[1], height=image_rgb.shape[0], margin_ratio=self.crop_margin_ratio)
        cropped = crop_array(image_rgb, expanded)
        square = pad_to_square(cropped)
        note = "A classificacao usou um recorte centrado na lesao a partir da mascara prevista pelo U-Net."
        return rgb_array_to_pil(square), "predicted_lesion_crop", note

    def _predict_with_gradcam(self, image: Image.Image) -> tuple[float, Image.Image, np.ndarray]:
        batch, display_np = self._classifier_tensor(image)
        batch = batch.to(self.device)

        activations: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []

        def forward_hook(_module, _inputs, output):
            activations.append(output)

        def backward_hook(_module, grad_input, grad_output):
            del grad_input
            gradients.append(grad_output[0])

        handle_forward = self.model.conv_head.register_forward_hook(forward_hook)
        handle_backward = self.model.conv_head.register_full_backward_hook(backward_hook)

        self.model.zero_grad(set_to_none=True)
        logits = self.model(batch).squeeze()
        probability = float(torch.sigmoid(logits).item())
        logits.backward()

        handle_forward.remove()
        handle_backward.remove()

        if not activations or not gradients:
            return probability, rgb_array_to_pil(display_np), np.zeros((self.image_size, self.image_size), dtype=np.float32)

        activation = activations[-1]
        gradient = gradients[-1]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        heatmap = cam.squeeze().detach().cpu().numpy()
        heatmap = normalize_minmax(heatmap)

        overlay = render_heatmap_overlay(display_np, heatmap)
        return probability, overlay, heatmap

    def _build_occlusion_overlay(
        self,
        image: Image.Image,
        baseline_probability: float,
        patch_size: int = 40,
        stride: int = 28,
    ) -> tuple[Image.Image, np.ndarray]:
        display_image = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        image_np = np.asarray(display_image).astype(np.uint8)
        fill_color = image_np.reshape(-1, 3).mean(axis=0).astype(np.uint8)
        score_sum = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        score_count = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        for y in range(0, self.image_size - patch_size + 1, stride):
            for x in range(0, self.image_size - patch_size + 1, stride):
                occluded = image_np.copy()
                occluded[y:y + patch_size, x:x + patch_size] = fill_color
                probability = self._predict_probability(rgb_array_to_pil(occluded))
                drop = max(baseline_probability - probability, 0.0)
                score_sum[y:y + patch_size, x:x + patch_size] += drop
                score_count[y:y + patch_size, x:x + patch_size] += 1.0

        score_map = np.divide(score_sum, np.maximum(score_count, 1.0))
        score_map = normalize_minmax(score_map)
        overlay = render_heatmap_overlay(image_np, score_map, alpha=0.42)
        return overlay, score_map

    @staticmethod
    def _make_similarity_vector(image: Image.Image, size: int = 16) -> np.ndarray:
        arr = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        vec = arr.reshape(-1)
        vec = vec - vec.mean()
        norm = np.linalg.norm(vec)
        return vec / max(norm, 1e-8)

    def _ensure_similarity_bank(self) -> None:
        if self._similarity_vectors is not None:
            return
        if self.treated_manifest_df.empty:
            self._similarity_vectors = np.empty((0, 16 * 16 * 3), dtype=np.float32)
            self._similarity_records = []
            return

        vectors: list[np.ndarray] = []
        records: list[tuple[str, str, str]] = []
        for row in self.treated_manifest_df.itertuples():
            export_path = Path(str(row.export_path))
            if not export_path.exists():
                continue
            try:
                preview = load_pil_image(export_path)
            except Exception:
                continue
            vectors.append(self._make_similarity_vector(preview))
            diagnosis_label = DIAGNOSIS_LABELS.get(str(row.label), str(row.label))
            records.append((str(row.image_id), diagnosis_label, str(export_path)))

        self._similarity_vectors = np.vstack(vectors).astype(np.float32) if vectors else np.empty((0, 16 * 16 * 3), dtype=np.float32)
        self._similarity_records = records

    def _find_similar_cases(
        self,
        image: Image.Image,
        exclude_image_id: str | None,
        top_k: int = 4,
    ) -> list[SimilarCase]:
        self._ensure_similarity_bank()
        if self._similarity_vectors is None or len(self._similarity_records) == 0:
            return []

        query = self._make_similarity_vector(image)
        similarities = self._similarity_vectors @ query
        ranked = np.argsort(-similarities)
        results: list[SimilarCase] = []

        for idx in ranked:
            image_id, diagnosis_label, export_path = self._similarity_records[int(idx)]
            if exclude_image_id and image_id == exclude_image_id:
                continue
            try:
                preview = load_pil_image(export_path)
            except Exception:
                continue
            results.append(
                SimilarCase(
                    image_id=image_id,
                    diagnosis_label=diagnosis_label,
                    similarity=float(similarities[int(idx)]),
                    preview_image=preview,
                )
            )
            if len(results) >= top_k:
                break
        return results

    def _compute_lesion_heuristics(
        self,
        image: Image.Image,
        mask: np.ndarray | None,
    ) -> list[dict]:
        heuristics: list[dict] = []
        if mask is None or float(mask.sum()) <= 0.0:
            return [
                {
                    "code": "A",
                    "title": "Assimetria",
                    "score": None,
                    "severity": "indisponivel",
                    "description": "Nao foi possivel estimar sem uma mascara valida.",
                },
                {
                    "code": "B",
                    "title": "Borda",
                    "score": None,
                    "severity": "indisponivel",
                    "description": "Nao foi possivel estimar sem uma mascara valida.",
                },
                {
                    "code": "C",
                    "title": "Cor",
                    "score": None,
                    "severity": "indisponivel",
                    "description": "Nao foi possivel estimar sem uma mascara valida.",
                },
                {
                    "code": "D",
                    "title": "Diametro relativo",
                    "score": None,
                    "severity": "indisponivel",
                    "description": "Sem escala fisica, a app so consegue estimar um proxy em pixels.",
                },
                {
                    "code": "E",
                    "title": "Evolucao",
                    "score": None,
                    "severity": "indisponivel",
                    "description": "A evolucao temporal exige multiplas imagens do mesmo paciente.",
                },
            ]

        mask_uint8 = (mask > 0).astype(np.uint8)
        bbox = compute_bbox_from_mask(mask_uint8)
        if bbox is None:
            return []

        cropped_mask = crop_array(mask_uint8, bbox)
        square_mask = pad_to_square(cropped_mask)
        horizontal_iou = ((square_mask > 0) & (np.fliplr(square_mask) > 0)).sum() / max(((square_mask > 0) | (np.fliplr(square_mask) > 0)).sum(), 1)
        vertical_iou = ((square_mask > 0) & (np.flipud(square_mask) > 0)).sum() / max(((square_mask > 0) | (np.flipud(square_mask) > 0)).sum(), 1)
        asymmetry_score = float(np.clip(1.0 - ((horizontal_iou + vertical_iou) / 2.0), 0.0, 1.0))

        contours, _ = cv2.findContours(mask_uint8 * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = float(mask_uint8.sum())
        perimeter = float(sum(cv2.arcLength(cnt, True) for cnt in contours)) if contours else 0.0
        circularity = (perimeter ** 2) / max(4.0 * np.pi * area, 1e-8)
        border_score = float(np.clip((circularity - 1.0) / 3.0, 0.0, 1.0))

        image_np = np.asarray(image.convert("RGB"))
        lesion_pixels = image_np[mask_uint8 > 0]
        color_std = float(lesion_pixels.std(axis=0).mean()) if len(lesion_pixels) else 0.0
        color_score = float(np.clip(color_std / 64.0, 0.0, 1.0))

        x0, y0, x1, y1 = bbox
        lesion_diameter = float(max(x1 - x0, y1 - y0))
        image_diagonal = float(np.hypot(mask_uint8.shape[0], mask_uint8.shape[1]))
        diameter_score = float(np.clip(lesion_diameter / max(image_diagonal * 0.65, 1e-8), 0.0, 1.0))

        heuristics.extend(
            [
                {
                    "code": "A",
                    "title": "Assimetria",
                    "score": asymmetry_score,
                    "severity": format_severity(asymmetry_score),
                    "description": "Estimativa baseada na perda de simetria horizontal e vertical da lesao segmentada.",
                },
                {
                    "code": "B",
                    "title": "Borda",
                    "score": border_score,
                    "severity": format_severity(border_score),
                    "description": "Proxy de irregularidade de borda usando circularidade da mascara.",
                },
                {
                    "code": "C",
                    "title": "Cor",
                    "score": color_score,
                    "severity": format_severity(color_score),
                    "description": "Variacao cromatica media dentro da area segmentada.",
                },
                {
                    "code": "D",
                    "title": "Diametro relativo",
                    "score": diameter_score,
                    "severity": format_severity(diameter_score),
                    "description": "Proxy em pixels; nao substitui medida clinica real em milimetros.",
                },
                {
                    "code": "E",
                    "title": "Evolucao",
                    "score": None,
                    "severity": "indisponivel",
                    "description": "A app nao observa mudanca temporal porque recebe apenas uma imagem por vez.",
                },
            ]
        )
        return heuristics

    def classify_probability(self, probability: float) -> tuple[str, dict]:
        zones = self.config["zones"]
        if probability >= self.threshold_high:
            return "positive", zones["positive"]
        if probability < self.threshold_low:
            return "negative", zones["negative"]
        return "review", zones["review"]

    def _truth_note(self, prediction: PredictionResult, dataset_match: DatasetMatch) -> str | None:
        if not dataset_match.found or dataset_match.binary_label is None:
            return None

        if prediction.zone_key == "positive":
            return "Acerto: melanoma detectado." if dataset_match.binary_label == 1 else "Falso positivo: o caso real do dataset nao e melanoma."
        if prediction.zone_key == "negative":
            return "Acerto: caso benigno reconhecido." if dataset_match.binary_label == 0 else "Falso negativo: o caso real do dataset e melanoma."
        return "Zona intermediaria: o caso foi encaminhado para revisao manual."

    def analyze_upload(self, file_bytes: bytes, file_name: str) -> CaseAnalysis:
        uploaded_image = Image.open(BytesIO(file_bytes)).convert("RGB")
        image_id = Path(file_name).stem
        dataset_match = self._lookup_dataset_match(image_id)

        if dataset_match.raw_path:
            original_image = load_pil_image(dataset_match.raw_path)
        else:
            original_image = uploaded_image.copy()

        segmentation_mask, segmentation_coverage = self._predict_segmentation_mask(original_image)
        segmentation_overlay = None
        segmentation_mask_image = None
        if segmentation_mask is not None:
            segmentation_overlay = overlay_mask_on_image(
                pil_to_rgb_array(original_image),
                segmentation_mask,
                color_rgb=(239, 68, 68),
                alpha=0.40,
            )
            segmentation_mask_image = render_binary_mask(segmentation_mask)

        ground_truth_overlay = None
        if dataset_match.mask_path:
            gt_mask = np.asarray(Image.open(dataset_match.mask_path).convert("L"))
            gt_mask = (gt_mask > 0).astype(np.uint8)
            ground_truth_overlay = overlay_mask_on_image(
                pil_to_rgb_array(original_image),
                gt_mask,
                color_rgb=(37, 99, 235),
                alpha=0.36,
            )

        if dataset_match.treated_path:
            classifier_source_image = load_pil_image(dataset_match.treated_path)
            preprocessing_source = "treated_manifest"
            preprocessing_note = "O nome do arquivo foi reconhecido no dataset, entao a classificacao usou a imagem tratada exportada pelo pipeline oficial."
        else:
            classifier_source_image, preprocessing_source, preprocessing_note = self._crop_for_classifier(
                original_image, segmentation_mask
            )

        classifier_mask, classifier_mask_coverage = self._predict_segmentation_mask(classifier_source_image)
        classifier_segmentation_overlay = None
        if classifier_mask is not None:
            classifier_segmentation_overlay = overlay_mask_on_image(
                pil_to_rgb_array(classifier_source_image),
                classifier_mask,
                color_rgb=(245, 158, 11),
                alpha=0.36,
            )

        probability, gradcam_overlay, gradcam_heatmap = self._predict_with_gradcam(classifier_source_image)
        combined_explainability_overlay = render_combined_explainability(
            np.asarray(classifier_source_image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)),
            gradcam_heatmap,
            resize_binary_mask(classifier_mask, (self.image_size, self.image_size)) if classifier_mask is not None else None,
        )
        occlusion_overlay, occlusion_map = self._build_occlusion_overlay(classifier_source_image, probability)
        gradcam_hotspots = extract_hotspot_crops(
            classifier_source_image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR),
            gradcam_heatmap,
            top_k=3,
        )
        occlusion_hotspots = extract_hotspot_crops(
            classifier_source_image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR),
            occlusion_map,
            top_k=3,
        )
        similar_cases = self._find_similar_cases(classifier_source_image, dataset_match.image_id)
        lesion_heuristics = self._compute_lesion_heuristics(classifier_source_image, classifier_mask)

        zone_key, zone_config = self.classify_probability(probability)
        prediction = PredictionResult(
            probability=probability,
            zone_key=zone_key,
            zone_label=zone_config["label"],
            headline=zone_config["headline"],
            recommended_action=zone_config["action"],
            threshold_low=self.threshold_low,
            threshold_high=self.threshold_high,
            device=str(self.device),
            predicted_binary=1 if probability >= self.threshold_high else 0,
            preprocessing_source=preprocessing_source,
        )
        truth_note = self._truth_note(prediction, dataset_match)

        classifier_input_image = classifier_source_image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR,
        )
        case_id = hashlib.sha1(f"{file_name}:{len(file_bytes)}".encode("utf-8") + file_bytes).hexdigest()[:12]
        display_name = dataset_match.image_id or Path(file_name).name

        return CaseAnalysis(
            case_id=case_id,
            file_name=file_name,
            display_name=display_name,
            prediction=prediction,
            dataset_match=dataset_match,
            original_image=original_image,
            uploaded_image=uploaded_image,
            classifier_source_image=classifier_source_image,
            classifier_input_image=classifier_input_image,
            segmentation_overlay=segmentation_overlay,
            segmentation_mask_image=segmentation_mask_image,
            classifier_segmentation_overlay=classifier_segmentation_overlay,
            combined_explainability_overlay=combined_explainability_overlay,
            ground_truth_overlay=ground_truth_overlay,
            gradcam_overlay=gradcam_overlay,
            occlusion_overlay=occlusion_overlay,
            gradcam_hotspots=gradcam_hotspots,
            occlusion_hotspots=occlusion_hotspots,
            similar_cases=similar_cases,
            lesion_heuristics=lesion_heuristics,
            segmentation_coverage=segmentation_coverage,
            classifier_segmentation_coverage=classifier_mask_coverage,
            preprocessing_note=preprocessing_note,
            truth_note=truth_note,
        )
