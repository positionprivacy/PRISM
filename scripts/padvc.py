import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
import warnings
from pathlib import Path

import cv2
import jieba.posseg as pseg
import numpy as np

# 环境配置
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def build_ocr(
    ocr_backend="paddle",
    rapidocr_package_path=None,
    rapid_det_limit_side_len=736,
):
    if ocr_backend == "rapidocr":
        if rapidocr_package_path and rapidocr_package_path not in sys.path:
            sys.path.append(rapidocr_package_path)
        from rapidocr_onnxruntime import RapidOCR

        return RapidOCR(
            det_limit_side_len=int(rapid_det_limit_side_len),
            print_verbose=False,
        )

    from paddleocr import PaddleOCR

    try:
        return PaddleOCR(
            lang="ch",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
    except (TypeError, ValueError):
        try:
            return PaddleOCR(
                use_angle_cls=False,
                lang="ch",
                use_gpu=False,
                enable_mkldnn=False,
                show_log=False,
            )
        except (TypeError, ValueError):
            try:
                return PaddleOCR(use_angle_cls=False, lang="ch", use_gpu=False, show_log=False)
            except (TypeError, ValueError):
                return PaddleOCR(lang="ch")

class PADVCCalculator:
    @staticmethod
    def _resolve_hf_snapshot(repo_id):
        cache_root = os.environ.get("PADVC_HF_CACHE", str(Path(__file__).resolve().parents[1] / ".cache" / "hf" / "hub"))
        repo_dir = os.path.join(cache_root, "models--" + repo_id.replace("/", "--"))
        snapshots_dir = os.path.join(repo_dir, "snapshots")
        if os.path.isdir(snapshots_dir):
            snapshots = sorted(
                [os.path.join(snapshots_dir, name) for name in os.listdir(snapshots_dir)],
                reverse=True,
            )
            if snapshots:
                return snapshots[0]
        return None

    @staticmethod
    def _normalize_zh_token(token):
        return re.sub(r"\s+", "", token.strip())

    @staticmethod
    def _normalize_en_token(token):
        token = token.lower().strip("-'")
        if not token:
            return token
        irregular = {
            "led": "lead",
            "caused": "cause",
            "causes": "cause",
            "resulting": "result",
            "results": "result",
            "generated": "generate",
            "generates": "generate",
            "generating": "generate",
            "produced": "produce",
            "produces": "produce",
            "composed": "compose",
            "composes": "compose",
            "contained": "contain",
            "contains": "contain",
            "included": "include",
            "includes": "include",
            "consists": "consist",
            "belonged": "belong",
            "belongs": "belong",
            "compared": "compare",
            "compares": "compare",
            "comparing": "compare",
            "contrasted": "contrast",
            "contrasts": "contrast",
            "corresponds": "correspond",
            "mapped": "map",
            "maps": "map",
            "differences": "difference",
            "components": "component",
            "structures": "structure",
            "transforms": "transform",
            "transformed": "transform",
            "converts": "convert",
            "converted": "convert",
            "derives": "derive",
            "derived": "derive",
            "evolves": "evolve",
            "evolved": "evolve",
        }
        if token in irregular:
            return irregular[token]
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("ing") and len(token) > 5:
            return token[:-3]
        if token.endswith("ed") and len(token) > 4:
            return token[:-2]
        if token.endswith("es") and len(token) > 4:
            return token[:-2]
        if token.endswith("s") and len(token) > 3:
            return token[:-1]
        return token

    def _normalize_en_phrase(self, phrase):
        tokens = [self._normalize_en_token(token) for token in phrase.lower().split()]
        return " ".join(token for token in tokens if token)

    def __init__(
        self,
        p=0.7,
        device='cpu',
        event_threshold_mode='absolute',
        event_threshold_abs=50000.0,
        event_threshold_ratio=0.08,
        delta_mode='positive',
        text_dilate=7,
        ocr_cache_dir=None,
        ocr_backend="paddle",
        rapidocr_package_path=None,
        rapid_use_cls=False,
        rapid_use_rec=True,
        rapid_text_score=0.5,
        rapid_box_thresh=0.5,
        rapid_unclip_ratio=1.6,
        rapid_det_limit_side_len=736,
        sticky_peak_rescue=False,
        sticky_primary_above_ratio=0.95,
        sticky_secondary_above_ratio=0.98,
        sticky_primary_event_max=2,
        sticky_secondary_event_max=3,
        sticky_peak_smooth_window=3,
        sticky_peak_quantile=0.75,
        sticky_peak_min_rel_height=0.14,
        sticky_peak_merge_gap=4,
        score_norm_method="none",
        score_output="raw",
        norm_mu=None,
        norm_sigma=None,
        norm_log_space=True,
        norm_eps=1e-8,
    ):
        self.debug = os.environ.get("PADVC_DEBUG", "").lower() in {"1", "true", "yes", "on"}
        if self.debug:
            print("    [DEBUG] PADVCCalculator.__init__ start", flush=True)
        self.p = p
        self.event_threshold_mode = event_threshold_mode
        self.event_threshold_abs = float(event_threshold_abs)
        self.event_threshold_ratio = float(event_threshold_ratio)
        self.delta_mode = str(delta_mode)
        self.text_kernel = np.ones((max(1, int(text_dilate)), max(1, int(text_dilate))), dtype=np.uint8)
        if ocr_cache_dir is None:
            ocr_cache_dir = os.environ.get("PADVC_OCR_CACHE_DIR") or os.environ.get("DIFFICULTY_OCR_CACHE_DIR")
        self.ocr_cache_dir = Path(ocr_cache_dir) if ocr_cache_dir else None
        if self.ocr_cache_dir is not None:
            self.ocr_cache_dir.mkdir(parents=True, exist_ok=True)
        self._video_ocr_cache = {}
        self._video_ocr_cache_dirty = {}
        self.ocr_backend = str(ocr_backend)
        self.rapid_use_cls = bool(rapid_use_cls)
        self.rapid_use_rec = bool(rapid_use_rec)
        self.rapid_text_score = float(rapid_text_score)
        self.rapid_box_thresh = float(rapid_box_thresh)
        self.rapid_unclip_ratio = float(rapid_unclip_ratio)
        self.rapid_det_limit_side_len = int(rapid_det_limit_side_len)
        self.rapidocr_package_path = rapidocr_package_path
        self.sticky_peak_rescue = bool(sticky_peak_rescue)
        self.sticky_primary_above_ratio = float(sticky_primary_above_ratio)
        self.sticky_secondary_above_ratio = float(sticky_secondary_above_ratio)
        self.sticky_primary_event_max = int(sticky_primary_event_max)
        self.sticky_secondary_event_max = int(sticky_secondary_event_max)
        self.sticky_peak_smooth_window = max(1, int(sticky_peak_smooth_window))
        self.sticky_peak_quantile = float(sticky_peak_quantile)
        self.sticky_peak_min_rel_height = float(sticky_peak_min_rel_height)
        self.sticky_peak_merge_gap = max(0, int(sticky_peak_merge_gap))
        self.score_norm_method = str(score_norm_method or "none").lower()
        self.score_output = str(score_output or "raw").lower()
        self.norm_mu = float(norm_mu) if norm_mu is not None else None
        self.norm_sigma = float(norm_sigma) if norm_sigma is not None else None
        self.norm_log_space = bool(norm_log_space)
        self.norm_eps = float(norm_eps)

        if self.debug:
            print(f"    [DEBUG] Initializing OCR backend: {self.ocr_backend}", flush=True)
        self.ocr = build_ocr(
            ocr_backend=self.ocr_backend,
            rapidocr_package_path=self.rapidocr_package_path,
            rapid_det_limit_side_len=self.rapid_det_limit_side_len,
        )

        if self.debug:
            print("    [DEBUG] Importing vector similarity backends...", flush=True)
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.cosine_similarity = cosine_similarity

        zh_model_path = (
            os.environ.get("PADVC_ZH_MODEL")
            or self._resolve_hf_snapshot("shibing624/text2vec-base-chinese")
            or "shibing624/text2vec-base-chinese"
        )
        en_model_path = (
            os.environ.get("PADVC_EN_MODEL")
            or self._resolve_hf_snapshot("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        )
        if self.debug:
            print(f"    [DEBUG] Loading zh embedder from: {zh_model_path}", flush=True)
        self.zh_embedder = SentenceTransformer(zh_model_path, device=device)
        self.en_embedder = None
        self.en_vectorizer = None
        if en_model_path:
            if self.debug:
                print(f"    [DEBUG] Loading en embedder from: {en_model_path}", flush=True)
            self.en_embedder = SentenceTransformer(en_model_path, device=device)
        else:
            if self.debug:
                print("    [DEBUG] No local multilingual model found, fallback to TF-IDF english similarity", flush=True)
            self.en_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self.zh_similarity_threshold = 0.80
        self.en_similarity_threshold = 0.55
        
        # 中英对称锚点词库：精确命中 + 向量相似度共同决定语义项
        self.anchor_keywords = {
            '指向/因果': {
                'zh': ['演化', '导致', '转化', '转换', '衍生', '生成', '引起', '产生'],
                'en': [
                    'evolve', 'evolution', 'lead', 'lead to', 'cause', 'result', 'result in',
                    'convert', 'conversion', 'transform', 'transformation', 'derive',
                    'generate', 'produce',
                ],
            },
            '包含/结构': {
                'zh': ['包含', '分为', '组成', '属于', '囊括', '构成', '包括'],
                'en': [
                    'contain', 'include', 'consist', 'consist of', 'compose', 'be composed of',
                    'belong', 'belong to', 'form', 'structure', 'component',
                ],
            },
            '对比/平行': {
                'zh': ['区别', '相比', '对应', '对比', '映射', '不同于', '平行'],
                'en': [
                    'difference', 'different', 'compare', 'compare with', 'contrast',
                    'correspond', 'correspond to', 'mapping', 'map', 'parallel',
                    'versus', 'different from',
                ],
            },
        }
        self.zh_anchor_lexicon = {
            self._normalize_zh_token(word)
            for group in self.anchor_keywords.values()
            for word in group['zh']
        }
        self.en_anchor_lexicon = {
            self._normalize_en_phrase(word)
            for group in self.anchor_keywords.values()
            for word in group['en']
        }
        self.anchor_names = list(self.anchor_keywords.keys())
        self.zh_anchor_embs = np.array([
            np.mean(
                self.zh_embedder.encode(group['zh'], show_progress_bar=False),
                axis=0,
            )
            for group in self.anchor_keywords.values()
        ])
        self.en_anchor_embs = np.array([
            np.mean(
                self.en_embedder.encode(group['en'], show_progress_bar=False),
                axis=0,
            )
            for group in self.anchor_keywords.values()
        ]) if self.en_embedder is not None else None
        self.en_anchor_texts = [" ".join(group['en']) for group in self.anchor_keywords.values()]
        self.en_anchor_tfidf = (
            self.en_vectorizer.fit_transform(self.en_anchor_texts)
            if self.en_vectorizer is not None
            else None
        )
        if self.debug:
            print("    [DEBUG] PADVCCalculator.__init__ done", flush=True)

    @staticmethod
    def _true_runs(flags):
        runs = []
        in_run = False
        start = 0
        for index, flag in enumerate(flags):
            if flag and not in_run:
                start = index
                in_run = True
            elif not flag and in_run:
                runs.append((start, index - 1))
                in_run = False
        if in_run:
            runs.append((start, len(flags) - 1))
        return runs

    @staticmethod
    def _merge_runs(runs, gap_max):
        if not runs:
            return []
        merged = [runs[0]]
        for start, end in runs[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end - 1 <= gap_max:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def _smooth_signal(values, window):
        if window <= 1 or len(values) <= 2:
            return np.asarray(values, dtype=np.float64)
        kernel = np.ones(int(window), dtype=np.float64) / float(window)
        return np.convolve(np.asarray(values, dtype=np.float64), kernel, mode="same")

    @staticmethod
    def _above_ratio_from_events(events, diff_count):
        if diff_count <= 0:
            return 0.0
        above_count = sum(max(0, int(end) - int(start)) for start, end in events)
        return float(above_count / diff_count)

    @staticmethod
    def _local_peaks(signal):
        peaks = []
        for index in range(1, len(signal) - 1):
            if signal[index] >= signal[index - 1] and signal[index] > signal[index + 1]:
                peaks.append(index)
        return peaks

    def _split_event_by_peaks(self, diffs, start, end):
        run_values = np.asarray(diffs[start:end], dtype=np.float64)
        min_segment_len = max(8, self.sticky_peak_smooth_window * 2)
        if run_values.size < min_segment_len * 2:
            return [(start, end)], None

        smooth = self._smooth_signal(run_values, self.sticky_peak_smooth_window)
        global_max = float(smooth.max()) if smooth.size else 0.0
        peak_details = []

        def split_recursive(seg_start, seg_end, depth=0):
            if depth >= 8 or seg_end - seg_start < min_segment_len * 2:
                return [(seg_start, seg_end)]

            segment = smooth[seg_start:seg_end]
            peak_indices = [seg_start + index for index in self._local_peaks(segment)]
            if len(peak_indices) < 2:
                return [(seg_start, seg_end)]

            seg_quantile = float(np.quantile(segment, self.sticky_peak_quantile))
            seg_max = float(segment.max()) if segment.size else 0.0
            best = None
            min_gap = max(4, self.sticky_peak_merge_gap)
            min_prom_rel = max(0.14, self.sticky_peak_min_rel_height)
            max_valley_rel = 0.86
            min_depth_abs = max(seg_quantile * 0.08, seg_max * 0.035, global_max * 0.02)

            for left_peak, right_peak in zip(peak_indices, peak_indices[1:]):
                if right_peak - left_peak < min_gap:
                    continue
                valley_index = left_peak + int(np.argmin(smooth[left_peak : right_peak + 1]))
                left_value = float(smooth[left_peak])
                right_value = float(smooth[right_peak])
                valley_value = float(smooth[valley_index])
                min_peak_value = min(left_value, right_value)
                if min_peak_value <= 0:
                    continue
                depth_abs = min_peak_value - valley_value
                if depth_abs <= 0:
                    continue
                prom_rel = depth_abs / min_peak_value
                valley_rel = valley_value / min_peak_value
                if prom_rel < min_prom_rel or valley_rel > max_valley_rel or depth_abs < min_depth_abs:
                    continue
                candidate = {
                    "seg_start": int(seg_start),
                    "seg_end": int(seg_end),
                    "left_peak": int(left_peak + start),
                    "right_peak": int(right_peak + start),
                    "valley_index": int(valley_index + start),
                    "left_value": left_value,
                    "right_value": right_value,
                    "valley_value": valley_value,
                    "depth_abs": float(depth_abs),
                    "prom_rel": float(prom_rel),
                    "valley_rel": float(valley_rel),
                    "score": float(depth_abs * (right_peak - left_peak)),
                }
                if best is None or candidate["score"] > best["score"]:
                    best = candidate

            if best is None:
                return [(seg_start, seg_end)]

            split_index = int(best["valley_index"] - start)
            if split_index - seg_start < min_segment_len or seg_end - split_index < min_segment_len:
                return [(seg_start, seg_end)]

            peak_details.append(best)
            return split_recursive(seg_start, split_index, depth + 1) + split_recursive(split_index, seg_end, depth + 1)

        split_events_local = split_recursive(0, len(run_values))
        split_events = [(int(left + start), int(right + start)) for left, right in split_events_local if right > left]
        if len(split_events) < 2:
            return [(start, end)], None

        return split_events, {
            "original_event": [int(start), int(end)],
            "rescued_events": [[int(a), int(b)] for a, b in split_events],
            "smooth_window": int(self.sticky_peak_smooth_window),
            "segment_count": int(len(split_events)),
            "peak_splits": peak_details,
        }

    def _frame_diff_value(self, gray_start, gray_end):
        if self.delta_mode == "positive":
            return float(
                np.maximum(gray_end.astype(np.int32) - gray_start.astype(np.int32), 0).sum()
            )
        return float(np.sum(cv2.absdiff(gray_end, gray_start)))

    def _detect_events(self, gray_frames):
        if len(gray_frames) < 2:
            return [], [], 0.0, {"applied": False}

        diffs = [
            self._frame_diff_value(gray_frames[index - 1], gray_frames[index])
            for index in range(1, len(gray_frames))
        ]
        if self.event_threshold_mode == 'absolute':
            thresh = self.event_threshold_abs
        else:
            thresh = np.max(diffs) * self.event_threshold_ratio if diffs else 0.0

        events = []
        in_ev = False
        start = 0
        for i, d in enumerate(diffs):
            if d > thresh and not in_ev:
                start = i
                in_ev = True
            elif d <= thresh and in_ev:
                events.append((start, max(start + 1, i)))
                in_ev = False
        if in_ev:
            events.append((start, len(gray_frames) - 1))

        rescue_info = {
            "applied": False,
            "original_event_count": int(len(events)),
            "original_above_ratio": self._above_ratio_from_events(events, len(diffs)),
        }
        should_rescue = self.sticky_peak_rescue and self.delta_mode == "positive" and self.event_threshold_mode == "ratio"
        if should_rescue:
            above_ratio = rescue_info["original_above_ratio"]
            event_count = len(events)
            sticky = (
                event_count <= self.sticky_primary_event_max and above_ratio >= self.sticky_primary_above_ratio
            ) or (
                event_count <= self.sticky_secondary_event_max and above_ratio >= self.sticky_secondary_above_ratio
            )
            if sticky:
                rescued_events = []
                rescue_details = []
                for event_start, event_end in events:
                    split_events, detail = self._split_event_by_peaks(diffs, event_start, event_end)
                    rescued_events.extend(split_events)
                    if detail is not None:
                        rescue_details.append(detail)
                if len(rescued_events) > len(events):
                    events = rescued_events
                    rescue_info = {
                        "applied": True,
                        "original_event_count": int(event_count),
                        "rescued_event_count": int(len(events)),
                        "original_above_ratio": float(above_ratio),
                        "rescued_above_ratio": self._above_ratio_from_events(events, len(diffs)),
                        "details": rescue_details,
                    }

        return events, diffs, thresh, rescue_info

    def _ocr_cache_path(self, video_path):
        if self.ocr_cache_dir is None:
            return None
        stem = Path(video_path).stem[:80]
        backend_sig = self.ocr_backend
        if self.ocr_backend == "rapidocr":
            backend_sig += (
                f"_cls{int(self.rapid_use_cls)}"
                f"_rec{int(self.rapid_use_rec)}"
                f"_ts{self.rapid_text_score:.3f}"
                f"_bt{self.rapid_box_thresh:.3f}"
                f"_ur{self.rapid_unclip_ratio:.3f}"
                f"_ls{self.rapid_det_limit_side_len}"
            )
        digest = hashlib.sha1(f"{video_path}::{backend_sig}".encode("utf-8")).hexdigest()[:16]
        return self.ocr_cache_dir / f"{stem}__{digest}.json"

    def _load_video_ocr_cache(self, video_path):
        if self.ocr_cache_dir is None:
            return None
        if video_path in self._video_ocr_cache:
            return self._video_ocr_cache[video_path]

        cache_path = self._ocr_cache_path(video_path)
        payload = {"video_path": str(video_path), "frames": {}}
        if cache_path is not None and cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    payload = {"video_path": str(video_path), "frames": {}}
                payload.setdefault("video_path", str(video_path))
                payload.setdefault("frames", {})
            except Exception:
                payload = {"video_path": str(video_path), "frames": {}}

        self._video_ocr_cache[video_path] = payload
        self._video_ocr_cache_dirty[video_path] = False
        return payload

    def _flush_video_ocr_cache(self, video_path):
        if self.ocr_cache_dir is None:
            return
        payload = self._video_ocr_cache.get(video_path)
        if payload is None or not self._video_ocr_cache_dirty.get(video_path):
            return

        cache_path = self._ocr_cache_path(video_path)
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(tmp_path, cache_path)
        self._video_ocr_cache_dirty[video_path] = False

    @staticmethod
    def _render_text_mask(frame_shape, polys):
        text_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        for pts in polys:
            pts = np.array(pts, np.int32)
            if pts.size == 0:
                continue
            cv2.fillPoly(text_mask, [pts], 255)
        return text_mask

    @staticmethod
    def _extract_ocr_polys(result):
        def looks_like_line(item):
            try:
                first = item[0]
            except Exception:
                return False
            return (
                isinstance(first, (list, tuple, np.ndarray))
                and len(first) >= 4
                and isinstance(first[0], (list, tuple, np.ndarray))
                and len(first[0]) >= 2
            )

        def looks_like_poly(item):
            return (
                isinstance(item, (list, tuple, np.ndarray))
                and len(item) >= 4
                and isinstance(item[0], (list, tuple, np.ndarray))
                and len(item[0]) >= 2
                and not isinstance(item[0][0], (list, tuple, np.ndarray))
            )

        polys = []
        if isinstance(result, tuple) and len(result) >= 1:
            result = result[0]
        if result is None:
            return polys
        if isinstance(result, list) and result:
            if all(looks_like_poly(item) for item in result):
                for pts in result:
                    polys.append(np.array(pts).astype(np.int32).tolist())
                return polys
            first = result[0]
            lines = result
            if not all(looks_like_line(item) for item in result):
                if isinstance(first, list) and all(looks_like_line(item) for item in first):
                    lines = first
            if all(looks_like_line(item) for item in lines):
                for line in lines:
                    pts = np.array(line[0]).astype(np.int32).tolist()
                    polys.append(pts)
            elif isinstance(first, dict):
                if "rec_polys" in first and first["rec_polys"] is not None:
                    for pts in first["rec_polys"]:
                        polys.append(np.array(pts).astype(np.int32).tolist())
                elif "dt_polys" in first and first["dt_polys"] is not None:
                    for pts in first["dt_polys"]:
                        polys.append(np.array(pts).astype(np.int32).tolist())
        return polys

    def _build_text_mask(self, frame_bgr, video_path=None, frame_index=None, dilate=False):
        frame_cache = None
        cache_hit = False
        if video_path is not None and frame_index is not None:
            video_cache = self._load_video_ocr_cache(video_path)
            if video_cache is not None:
                frame_cache = video_cache["frames"].get(str(frame_index))

        if frame_cache is not None:
            polys = frame_cache.get("polys", [])
            box_count = int(frame_cache.get("box_count", len(polys)))
            text_mask = self._render_text_mask(frame_bgr.shape, polys)
            cache_hit = True
        else:
            if self.ocr_backend == "rapidocr":
                result = self.ocr(
                    frame_bgr,
                    use_cls=self.rapid_use_cls,
                    use_rec=self.rapid_use_rec,
                    box_thresh=self.rapid_box_thresh,
                    unclip_ratio=self.rapid_unclip_ratio,
                    text_score=self.rapid_text_score,
                )
            elif hasattr(self.ocr, "predict"):
                result = self.ocr.predict(frame_bgr)
            else:
                try:
                    result = self.ocr.ocr(frame_bgr, cls=False)
                except TypeError:
                    result = self.ocr.ocr(frame_bgr)
            polys = self._extract_ocr_polys(result)
            box_count = len(polys)
            text_mask = self._render_text_mask(frame_bgr.shape, polys)
            if video_path is not None and frame_index is not None:
                video_cache = self._load_video_ocr_cache(video_path)
                if video_cache is not None:
                    video_cache["frames"][str(frame_index)] = {
                        "shape": [int(frame_bgr.shape[0]), int(frame_bgr.shape[1])],
                        "box_count": int(box_count),
                        "polys": polys,
                    }
                    self._video_ocr_cache_dirty[video_path] = True

        if dilate and np.any(text_mask):
            text_mask = cv2.dilate(text_mask, self.text_kernel)
        return text_mask, box_count, cache_hit

    def _normalize_score(self, raw_score):
        raw_score = float(raw_score)
        if self.score_norm_method == "none":
            return None
        if self.norm_mu is None or self.norm_sigma is None or self.norm_sigma <= 0:
            raise ValueError("score normalization requires valid norm_mu and norm_sigma")

        base_value = math.log(raw_score + self.norm_eps) if self.norm_log_space else raw_score
        z = (base_value - self.norm_mu) / max(self.norm_sigma, 1e-12)
        if self.score_norm_method == "cdf":
            return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        raise ValueError(f"Unsupported score_norm_method: {self.score_norm_method}")

    @staticmethod
    def _strip_context_blocks(md_text):
        patterns = [
            r'<!--\s*CONTEXT:BEGIN\s*-->.*?<!--\s*CONTEXT:END\s*-->\s*',
            r'<context_begin>.*?<context_end>\s*',
        ]
        clean_text = md_text
        for pattern in patterns:
            clean_text = re.sub(pattern, '', clean_text, flags=re.DOTALL | re.IGNORECASE)
        return clean_text

    @staticmethod
    def _count_structural_markers(clean_md):
        patterns = {
            '标题': r'^#+',
            '列表': r'\n\s*[-*]',
            '引用': r'\n\s*>',
            '有序': r'\n\s*\d+\.',
        }
        counts = {}
        for label, pattern in patterns.items():
            matches = re.findall(pattern, clean_md, flags=re.MULTILINE)
            if matches:
                counts[label] = len(matches)
        return counts

    @staticmethod
    def _extract_english_tokens(text):
        return [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z\-]{1,}", text)]

    def _extract_semantic_tokens(self, pure_text):
        chinese_words = [
            self._normalize_zh_token(word)
            for word, flag in pseg.cut(pure_text)
            if (flag.startswith('v') or flag.startswith('p') or flag.startswith('c')) and len(word) > 1
        ]
        english_words = [
            self._normalize_en_token(word)
            for word in self._extract_english_tokens(pure_text)
        ]
        return chinese_words, english_words

    def _extract_english_phrases(self, english_words):
        phrases = set()
        for n in range(2, 5):
            for index in range(len(english_words) - n + 1):
                phrase = " ".join(token for token in english_words[index:index + n] if token)
                if phrase:
                    phrases.add(phrase)
        return sorted(phrases)

    def _calculate_action_bonus(self, pure_text, verbose=False):
        chinese_words, english_words = self._extract_semantic_tokens(pure_text)
        english_phrases = self._extract_english_phrases(english_words)
        zh_token_hits = {word for word in chinese_words if word in self.zh_anchor_lexicon}
        en_token_hits = {word for word in english_words if word in self.en_anchor_lexicon}

        zh_phrase_hits = {
            keyword for keyword in self.zh_anchor_lexicon
            if keyword not in zh_token_hits and len(keyword) > 1 and keyword in pure_text
        }

        en_phrase_hits = {phrase for phrase in english_phrases if phrase in self.en_anchor_lexicon}

        covered_en_tokens = {part for phrase in en_phrase_hits for part in phrase.split()}
        en_token_hits = {word for word in en_token_hits if word not in covered_en_tokens}

        zh_exact_hits = zh_token_hits | zh_phrase_hits
        en_exact_hits = en_token_hits | en_phrase_hits

        zh_semantic_hits = {}
        zh_candidates = [word for word in chinese_words if word and word not in zh_exact_hits]
        if zh_candidates:
            zh_embs = self.zh_embedder.encode(zh_candidates, show_progress_bar=False)
            zh_sims = self.cosine_similarity(zh_embs, self.zh_anchor_embs)
            for index, word in enumerate(zh_candidates):
                best_idx = int(np.argmax(zh_sims[index]))
                best_sim = float(zh_sims[index][best_idx])
                if best_sim >= self.zh_similarity_threshold:
                    zh_semantic_hits[word] = (self.anchor_names[best_idx], best_sim)

        en_semantic_hits = {}
        en_candidates = []
        seen = set()
        for item in english_words:
            if not item or item in en_exact_hits or item in seen:
                continue
            seen.add(item)
            en_candidates.append(item)
        if en_candidates:
            if self.en_embedder is not None:
                en_embs = self.en_embedder.encode(en_candidates, show_progress_bar=False)
                en_sims = self.cosine_similarity(en_embs, self.en_anchor_embs)
            else:
                en_matrix = self.en_vectorizer.transform(en_candidates)
                en_sims = self.cosine_similarity(en_matrix, self.en_anchor_tfidf)
            for index, phrase in enumerate(en_candidates):
                best_idx = int(np.argmax(en_sims[index]))
                best_sim = float(en_sims[index][best_idx])
                if best_sim >= self.en_similarity_threshold:
                    en_semantic_hits[phrase] = (self.anchor_names[best_idx], best_sim)

        matched_words = sorted(
            zh_exact_hits
            | en_exact_hits
            | set(zh_semantic_hits.keys())
            | set(en_semantic_hits.keys())
        )
        action_bonus = len(matched_words)

        if verbose:
            exact_hits = sorted(zh_exact_hits | en_exact_hits)
            if exact_hits:
                print(f"    - 锚点词精确命中 ({len(exact_hits)}个): {exact_hits}", flush=True)
            semantic_debug = []
            for word, (anchor_name, sim) in zh_semantic_hits.items():
                semantic_debug.append(f"{word}->{anchor_name}({sim:.3f})")
            for word, (anchor_name, sim) in en_semantic_hits.items():
                semantic_debug.append(f"{word}->{anchor_name}({sim:.3f})")
            if semantic_debug:
                print(f"    - 向量相似触发 ({len(semantic_debug)}个): {sorted(semantic_debug)}", flush=True)
            if matched_words:
                print(f"    - 动作关键短语总计 ({len(matched_words)}个): {matched_words}", flush=True)
        return action_bonus, matched_words

    def _get_reconstructed_text_energy(self, frame_bgr, video_path=None, frame_index=None):
        """等效色块法：计算文字框边界产生的拉普拉斯响应"""
        reconstructed, box_count, cache_hit = self._build_text_mask(
            frame_bgr,
            video_path=video_path,
            frame_index=frame_index,
            dilate=False,
        )
        # 计算色块边缘梯度能
        laplacian = np.absolute(cv2.Laplacian(reconstructed.astype(np.float64), cv2.CV_64F))
        total_reconstructed_e = np.sum(laplacian) / 1e5
        return total_reconstructed_e, {
            "box_count": int(box_count),
            "cache_hit": bool(cache_hit),
        }

    def _calculate_pvd(self, md_text):
        """增强版 PVD：严格剔除 Context 块，精准统计教学内容密度"""
        print("  [PVD 分析中...]", flush=True)
        
        clean_md = self._strip_context_blocks(md_text)
        clean_md = re.sub(r'```.*?```', '', clean_md, flags=re.DOTALL)
        structure_counts = self._count_structural_markers(clean_md)
        md_symbols_count = sum(structure_counts.values())
        all_found = [f"{label}({count})" for label, count in structure_counts.items()]
        
        if all_found:
            print(f"    - 识别到结构分布: {', '.join(all_found)}", flush=True)

        pure_text = re.sub(r'<.*?>', '', clean_md, flags=re.DOTALL)
        action_bonus, _ = self._calculate_action_bonus(pure_text, verbose=True)

        pvd_score = md_symbols_count + action_bonus
        print(f"    - PVD 修正总分: {pvd_score} (结构 {md_symbols_count} + 动作 {action_bonus})", flush=True)
        return max(pvd_score, 1)

    def _count_animation_segments(self, frames, events, video_path=None, source_fps=0.0):
        segment_text_energy_threshold = 0.005
        animation_segments = 1
        boundary_details = []
        ocr_cache_hits = 0
        ocr_cache_misses = 0

        if len(events) > 1:
            for index in range(1, len(events)):
                prev_end = int(events[index - 1][1])
                curr_start = int(events[index][0])
                mid_idx = int((prev_end + curr_start) // 2)
                mid_raw_text_e, meta = self._get_reconstructed_text_energy(
                    frames[mid_idx],
                    video_path=video_path,
                    frame_index=mid_idx,
                )
                cache_hit = bool(meta.get("cache_hit"))
                ocr_cache_hits += int(cache_hit)
                ocr_cache_misses += int(not cache_hit)
                is_new_segment = bool(mid_raw_text_e < segment_text_energy_threshold)
                if is_new_segment:
                    animation_segments += 1
                boundary_details.append({
                    "boundary_index": int(index - 1),
                    "prev_event_end_frame": int(prev_end),
                    "next_event_start_frame": int(curr_start),
                    "mid_frame_index": int(mid_idx),
                    "mid_time_sec": float(mid_idx / source_fps) if source_fps > 0 else None,
                    "mid_raw_text_energy": float(mid_raw_text_e),
                    "text_mask_cache_hit": bool(cache_hit),
                    "new_segment": bool(is_new_segment),
                })

        return (
            int(animation_segments),
            float(segment_text_energy_threshold),
            boundary_details,
            int(ocr_cache_hits),
            int(ocr_cache_misses),
        )

    def evaluate_single(self, md_content_or_path, video_path, return_details=False):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n" + "-" * 40 + f"\n[🔍 评估开始] {video_name}", flush=True)

        if os.path.exists(md_content_or_path) and os.path.isfile(md_content_or_path):
            with open(md_content_or_path, 'r', encoding='utf-8') as f:
                md_text = f.read()
        else:
            md_text = md_content_or_path

        if not os.path.exists(video_path):
            print(f"    [ERROR] 找不到视频文件: {video_path}")
            return 0.0

        cap = cv2.VideoCapture(video_path)
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            return 0.0

        video_frame_count = int(len(frames))
        video_duration_sec = float(video_frame_count / source_fps) if source_fps > 0 else None
        text_sample_stride = 15
        sampled_text_energy_series = []

        print("\n[Step 1: 文字背景负荷分析 (分母)]", flush=True)
        max_raw_text_e = 0.0
        peak_idx = 0
        ocr_cache_hits = 0
        ocr_cache_misses = 0
        for frame_index in range(0, video_frame_count, text_sample_stride):
            e_text, meta = self._get_reconstructed_text_energy(
                frames[frame_index],
                video_path=video_path,
                frame_index=frame_index,
            )
            cache_hit = bool(meta.get("cache_hit"))
            box_count = int(meta.get("box_count", 0))
            ocr_cache_hits += int(cache_hit)
            ocr_cache_misses += int(not cache_hit)
            sampled_text_energy_series.append({
                "frame_index": int(frame_index),
                "time_sec": float(frame_index / source_fps) if source_fps > 0 else None,
                "raw_text_energy": float(e_text),
                "box_count": int(box_count),
                "cache_hit": bool(cache_hit),
            })
            if e_text > max_raw_text_e:
                max_raw_text_e = e_text
                peak_idx = frame_index

        max_lp_text_burden = math.pow(max_raw_text_e, self.p) if max_raw_text_e > 0 else 0
        print(f"  >>> 文字背景边界能峰值: {max_raw_text_e:.6f} (Frame {peak_idx})", flush=True)
        print(f"  >>> 分母项贡献 (ΣE)^p: {max_lp_text_burden:.6f}", flush=True)

        print("\n[Step 2: 正向几何演化分析 (分子)]", flush=True)
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
        events, diffs, thresh, rescue_info = self._detect_events(gray_frames)
        frame_diff_series = [float(value) for value in diffs]
        print(
            f"  >>> 事件阈值模式: {self.event_threshold_mode} | 阈值: {thresh:.2f} | 事件数: {len(events)}",
            flush=True,
        )
        if rescue_info.get("applied"):
            print(
                f"  >>> sticky_peak_rescue: {rescue_info.get('original_event_count')} -> {rescue_info.get('rescued_event_count')}",
                flush=True,
            )

        total_lp_geo_evolution = 0.0
        event_energy_details = []
        for idx, (start_frame, end_frame) in enumerate(events):
            gray_start = gray_frames[start_frame].astype(np.int16)
            gray_end = gray_frames[end_frame].astype(np.int16)
            positive_delta = np.maximum(gray_end - gray_start, 0).astype(np.uint8)

            text_mask, text_box_count, cache_hit = self._build_text_mask(
                frames[end_frame],
                video_path=video_path,
                frame_index=end_frame,
                dilate=True,
            )
            ocr_cache_hits += int(cache_hit)
            ocr_cache_misses += int(not cache_hit)
            geo_mask = cv2.bitwise_not(text_mask)

            laplacian = np.absolute(cv2.Laplacian(positive_delta, cv2.CV_64F))
            raw_geo_e = np.sum(cv2.bitwise_and(laplacian, laplacian, mask=geo_mask)) / 1e5
            lp_geo_e = math.pow(raw_geo_e, self.p) if raw_geo_e > 0 else 0
            total_lp_geo_evolution += lp_geo_e
            event_energy_details.append({
                "event_index": int(idx),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "start_time_sec": float(start_frame / source_fps) if source_fps > 0 else None,
                "end_time_sec": float(end_frame / source_fps) if source_fps > 0 else None,
                "duration_frames": int(max(0, end_frame - start_frame)),
                "text_box_count_end_frame": int(text_box_count),
                "text_mask_cache_hit": bool(cache_hit),
                "raw_geo_energy": float(raw_geo_e),
                "geo_energy_p": float(lp_geo_e),
            })
            print(
                f"  {idx+1:02d} | Range: {start_frame:4d}-{end_frame:4d} | Pos_Raw_E: {raw_geo_e:8.4f} | Lp_Contrib: {lp_geo_e:8.4f}",
                flush=True,
            )

        print("\n[Step 3: 最终得分结算]", flush=True)
        pvd_raw = self._calculate_pvd(md_text)
        pvd_factor = math.log(pvd_raw + math.e)
        (
            animation_segments,
            segment_text_energy_threshold,
            segment_boundary_details,
            segment_cache_hits,
            segment_cache_misses,
        ) = self._count_animation_segments(
            frames,
            events,
            video_path=video_path,
            source_fps=source_fps,
        )
        ocr_cache_hits += segment_cache_hits
        ocr_cache_misses += segment_cache_misses
        normalized_geo_evolution = total_lp_geo_evolution / max(animation_segments, 1)
        final_score_raw = normalized_geo_evolution / (pvd_factor * (1 + max_lp_text_burden))
        final_score_norm = self._normalize_score(final_score_raw)
        final_score = final_score_norm if self.score_output == "norm" else final_score_raw
        if self.score_output == "norm" and final_score_norm is None:
            raise ValueError("score_output='norm' requires score_norm_method != 'none'")

        print(f"  [ΣE_geo^p] (分子和) : {total_lp_geo_evolution:.6f}", flush=True)
        print(f"  [Segments] (段数归一化): {animation_segments}", flush=True)
        print(f"  [ΣE_geo^p / Segments] : {normalized_geo_evolution:.6f}", flush=True)
        print(f"  [1 + E_txt^p] (分母项): {1 + max_lp_text_burden:.6f}", flush=True)
        print(f"  [PVD Factor] (语义项): {pvd_factor:.4f}", flush=True)
        print(
            f"  🎯 PADVC_raw = ({total_lp_geo_evolution:.4f} / {animation_segments}) / "
            f"({pvd_factor:.4f} * {1 + max_lp_text_burden:.4f}) = {final_score_raw:.8f}",
            flush=True,
        )
        if final_score_norm is not None:
            print(f"  🎯 PADVC_norm ({self.score_norm_method}) = {final_score_norm:.8f}", flush=True)
        print(f"  >>> OCR cache hits/misses: {ocr_cache_hits}/{ocr_cache_misses}", flush=True)

        self._flush_video_ocr_cache(video_path)
        if return_details:
            return {
                "detail_version": 4,
                "padvc_raw": float(final_score_raw),
                "padvc_norm": float(final_score_norm) if final_score_norm is not None else None,
                "padvc_selected": float(final_score),
                "score_output": self.score_output,
                "score_norm_method": self.score_norm_method,
                "norm_mu": self.norm_mu,
                "norm_sigma": self.norm_sigma,
                "norm_log_space": bool(self.norm_log_space),
                "p_exponent": float(self.p),
                "video_fps": float(source_fps),
                "video_frame_count": int(video_frame_count),
                "video_source_frame_count": int(source_frame_count),
                "video_duration_sec": video_duration_sec,
                "text_sample_stride": int(text_sample_stride),
                "text_peak_frame_index": int(peak_idx),
                "text_peak_time_sec": float(peak_idx / source_fps) if source_fps > 0 else None,
                "sampled_text_energy_series": sampled_text_energy_series,
                "frame_diff_series": frame_diff_series,
                "frame_diff_count": int(len(frame_diff_series)),
                "pvd_raw": int(pvd_raw),
                "pvd_factor": float(pvd_factor),
                "max_raw_text_energy": float(max_raw_text_e),
                "max_text_burden_p": float(max_lp_text_burden),
                "geo_evolution_sum_p": float(total_lp_geo_evolution),
                "geo_evolution_avg_p": float(normalized_geo_evolution),
                "animation_segment_count": int(animation_segments),
                "segment_text_energy_threshold": float(segment_text_energy_threshold),
                "segment_boundary_details": segment_boundary_details,
                "event_count": int(len(events)),
                "event_threshold_mode": self.event_threshold_mode,
                "event_threshold_abs": float(self.event_threshold_abs),
                "event_threshold_ratio": float(self.event_threshold_ratio),
                "event_threshold": float(thresh),
                "delta_mode": self.delta_mode,
                "event_rescue_info": rescue_info,
                "events": [[int(s), int(e)] for s, e in events],
                "event_energy_details": event_energy_details,
                "ocr_cache_hits": int(ocr_cache_hits),
                "ocr_cache_misses": int(ocr_cache_misses),
                "ocr_cache_path": str(self._ocr_cache_path(video_path)) if self._ocr_cache_path(video_path) is not None else None,
            }
        return final_score

def batch_process(md_folder, video_folder, calculator=None):
    calc = calculator or PADVCCalculator(p=0.7)
    md_paths = sorted(glob.glob(os.path.join(md_folder, "*.md")))
    results_list = []
    for md_path in md_paths:
        base = os.path.splitext(os.path.basename(md_path))[0]
        video_path = os.path.join(video_folder, f"{base}.mp4")
        if os.path.exists(video_path):
            try:
                score = calc.evaluate_single(md_path, video_path)
                results_list.append((base, score))
            except Exception as e:
                print(f"处理失败 {base}: {e}")

    print("\n" + "="*80)
    print(f"{'Video Name':<55} | {'Score':<15}")
    print("-" * 80)
    for name, s in sorted(results_list, key=lambda x: x[1], reverse=True):
        print(f"{name:<55} | {s:<15.8f}")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video")
    parser.add_argument("--text")
    parser.add_argument("--md-path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--p", type=float, default=0.7)
    parser.add_argument("--event-threshold-mode", default="absolute")
    parser.add_argument("--event-threshold-abs", type=float, default=50000.0)
    parser.add_argument("--event-threshold-ratio", type=float, default=0.08)
    parser.add_argument("--delta-mode", choices=["absolute", "positive"], default="positive")
    parser.add_argument("--text-dilate", type=int, default=7)
    parser.add_argument("--ocr-cache-dir")
    parser.add_argument("--ocr-backend", choices=["paddle", "rapidocr"], default="rapidocr")
    parser.add_argument("--rapidocr-package-path")
    parser.add_argument("--rapid-use-cls", action="store_true")
    parser.add_argument("--rapid-no-rec", action="store_true")
    parser.add_argument("--rapid-text-score", type=float, default=0.5)
    parser.add_argument("--rapid-box-thresh", type=float, default=0.5)
    parser.add_argument("--rapid-unclip-ratio", type=float, default=1.6)
    parser.add_argument("--rapid-det-limit-side-len", type=int, default=736)
    parser.add_argument("--sticky-peak-rescue", action="store_true")
    parser.add_argument("--sticky-primary-above-ratio", type=float, default=0.95)
    parser.add_argument("--sticky-secondary-above-ratio", type=float, default=0.98)
    parser.add_argument("--sticky-primary-event-max", type=int, default=2)
    parser.add_argument("--sticky-secondary-event-max", type=int, default=3)
    parser.add_argument("--sticky-peak-smooth-window", type=int, default=3)
    parser.add_argument("--sticky-peak-quantile", type=float, default=0.75)
    parser.add_argument("--sticky-peak-min-rel-height", type=float, default=0.14)
    parser.add_argument("--sticky-peak-merge-gap", type=int, default=4)
    parser.add_argument("--score-norm-method", default="none", choices=["none", "cdf"])
    parser.add_argument("--score-output", default="raw", choices=["raw", "norm"])
    parser.add_argument("--norm-mu", type=float)
    parser.add_argument("--norm-sigma", type=float)
    parser.add_argument("--batch-md-dir")
    parser.add_argument("--batch-video-dir")
    args = parser.parse_args()

    calc = PADVCCalculator(
        p=args.p,
        device=args.device,
        event_threshold_mode=args.event_threshold_mode,
        event_threshold_abs=args.event_threshold_abs,
        event_threshold_ratio=args.event_threshold_ratio,
        delta_mode=args.delta_mode,
        text_dilate=args.text_dilate,
        ocr_cache_dir=args.ocr_cache_dir,
        ocr_backend=args.ocr_backend,
        rapidocr_package_path=args.rapidocr_package_path,
        rapid_use_cls=args.rapid_use_cls,
        rapid_use_rec=not args.rapid_no_rec,
        rapid_text_score=args.rapid_text_score,
        rapid_box_thresh=args.rapid_box_thresh,
        rapid_unclip_ratio=args.rapid_unclip_ratio,
        rapid_det_limit_side_len=args.rapid_det_limit_side_len,
        sticky_peak_rescue=args.sticky_peak_rescue,
        sticky_primary_above_ratio=args.sticky_primary_above_ratio,
        sticky_secondary_above_ratio=args.sticky_secondary_above_ratio,
        sticky_primary_event_max=args.sticky_primary_event_max,
        sticky_secondary_event_max=args.sticky_secondary_event_max,
        sticky_peak_smooth_window=args.sticky_peak_smooth_window,
        sticky_peak_quantile=args.sticky_peak_quantile,
        sticky_peak_min_rel_height=args.sticky_peak_min_rel_height,
        sticky_peak_merge_gap=args.sticky_peak_merge_gap,
        score_norm_method=args.score_norm_method,
        score_output=args.score_output,
        norm_mu=args.norm_mu,
        norm_sigma=args.norm_sigma,
    )

    if args.video and (args.text or args.md_path):
        md_input = args.md_path or args.text
        payload = calc.evaluate_single(md_input, args.video, return_details=True)
        print(f"Final Score: {payload['padvc_selected']:.8f}", flush=True)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    else:
        md_dir = args.batch_md_dir or "./md"
        video_dir = args.batch_video_dir or "./video"
        if os.path.exists(md_dir) and os.path.exists(video_dir):
            batch_process(md_dir, video_dir, calculator=calc)
