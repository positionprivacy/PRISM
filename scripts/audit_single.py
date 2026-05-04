import ast
import argparse
import json
import sys
import os
import tempfile
import numpy as np
import warnings
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from manim import *

from repo_config import get_tmp_subdir
from matplotlib.path import Path as MPath

# ==========================================
# 1. 环境配置
# ==========================================
import matplotlib
matplotlib.use('Agg')
logging.getLogger("manim").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
AUDIT_TMP_ROOT = get_tmp_subdir("auditor")


def install_compat_shims():
    import manim

    compat_globals = {
        "CENTER": ORIGIN,
        "UP_LEFT": UL,
        "UP_RIGHT": UR,
        "DOWN_LEFT": DL,
        "DOWN_RIGHT": DR,
        "LIGHT_BLUE": BLUE_A,
        "CYAN": TEAL_A,
    }
    for name, value in compat_globals.items():
        globals().setdefault(name, value)
        setattr(manim, name, getattr(manim, name, value))

    if "HGroup" not in globals():
        class HGroup(VGroup):
            pass
        globals()["HGroup"] = HGroup
        setattr(manim, "HGroup", HGroup)

    if "BulletList" not in globals():
        class BulletList(VGroup):
            def __init__(self, *items, buff=0.3, font_size=24, dot_scale_factor=1.0, **kwargs):
                line_mobs = []
                for item in items:
                    if isinstance(item, Mobject):
                        row = item
                    else:
                        bullet = Text("•", font_size=font_size).scale(dot_scale_factor)
                        body = Text(str(item), font_size=font_size)
                        row = VGroup(bullet, body).arrange(RIGHT, buff=0.15, aligned_edge=UP)
                    line_mobs.append(row)
                super().__init__(*line_mobs)
                if len(self.submobjects) > 1:
                    self.arrange(DOWN, aligned_edge=LEFT, buff=buff)
                for key in ["name", "z_index"]:
                    if key in kwargs:
                        try:
                            setattr(self, key, kwargs[key])
                        except Exception:
                            pass
        globals()["BulletList"] = BulletList
        setattr(manim, "BulletList", BulletList)

    try:
        from manim.utils import color as manim_color
        if not hasattr(manim_color, "Colors"):
            class _CompatColors(SimpleNamespace):
                def __getattr__(self, item):
                    upper = item.upper()
                    if upper in globals():
                        return globals()[upper]
                    return WHITE
            manim_color.Colors = _CompatColors()
    except Exception:
        pass

    original_align_to = Mobject.align_to

    def compat_align_to(self, *args, **kwargs):
        kwargs.pop("buff", None)
        return original_align_to(self, *args, **kwargs)

    Mobject.align_to = compat_align_to

    original_mobject_init = Mobject.__init__

    def compat_mobject_init(self, *args, **kwargs):
        if "font_name" in kwargs and "font" not in kwargs:
            kwargs["font"] = kwargs.pop("font_name")
        else:
            kwargs.pop("font_name", None)
        if "font_weight" in kwargs and "weight" not in kwargs:
            kwargs["weight"] = kwargs.pop("font_weight")
        else:
            kwargs.pop("font_weight", None)
        if "bold" in kwargs and "weight" not in kwargs:
            if kwargs.pop("bold"):
                kwargs["weight"] = BOLD
        else:
            kwargs.pop("bold", None)
        for key in [
            "line_height",
            "line_spacing_height",
            "line_spacing_ratio",
            "alignment",
            "height",
            "corner_radius",
        ]:
            kwargs.pop(key, None)
        return original_mobject_init(self, *args, **kwargs)

    Mobject.__init__ = compat_mobject_init

    if hasattr(manim, "Code"):
        original_code_init = manim.Code.__init__

        def compat_code_init(self, *args, **kwargs):
            if "code" in kwargs and "code_string" not in kwargs and not args:
                kwargs["code_string"] = kwargs.pop("code")
            return original_code_init(self, *args, **kwargs)

        manim.Code.__init__ = compat_code_init
        globals()["Code"] = manim.Code


install_compat_shims()

# ==========================================
# 2. 核心几何审计算法
# ==========================================
class GeometryAuditor:
    def __init__(self, frame_width=14.22, frame_height=8.0):
        self.limit_x, self.limit_y = frame_width / 2, frame_height / 2
        self.parent_overlap_threshold = 0.95
        self.allowed_container_leak_ratio = 0.1

    def get_oob_score(self, mobject):
        pts = mobject.get_all_points()
        if len(pts) == 0: return 0
        mins, maxs = np.min(pts, axis=0), np.max(pts, axis=0)
        overflows = [max(0, -self.limit_x - mins[0]), max(0, maxs[0] - self.limit_x),
                     max(0, -self.limit_y - mins[1]), max(0, maxs[1] - self.limit_y)]
        return round(float(max(overflows)), 3)

    def get_overlap_score(self, mob1, mob2):
        pts1, pts2 = mob1.get_all_points(), mob2.get_all_points()
        if len(pts1) < 2 or len(pts2) < 3: return 0
        min1, max1 = np.min(pts1, axis=0), np.max(pts1, axis=0)
        min2, max2 = np.min(pts2, axis=0), np.max(pts2, axis=0)
        if (max1[0] < min2[0] or min1[0] > max2[0] or max1[1] < min2[1] or min1[1] > max2[1]):
            return 0
        path2 = MPath(pts2[:, :2])
        is_in_2 = path2.contains_points(pts1[:, :2])
        return round(float(np.sum(is_in_2) / len(pts1)), 3)

    def get_leak_ratio(self, content, container):
        pts_in = content.get_all_points()
        pts_out = container.get_all_points()
        if len(pts_in) == 0 or len(pts_out) < 3: return 0
        poly_path = MPath(pts_out[:, :2])
        outside_points = ~poly_path.contains_points(pts_in[:, :2])
        return round(float(np.sum(outside_points) / len(pts_in)), 3)

    @staticmethod
    def leaf_id(full_id):
        base = (full_id or "").split("#", 1)[0]
        return base.rsplit("/", 1)[-1]

    @staticmethod
    def normalized_path_parts(full_id):
        return [re.sub(r"#\d+$", "", part) for part in (full_id or "").split("/") if part]

    @staticmethod
    def parent_id(full_id):
        base = (full_id or "").split("#", 1)[0]
        if "/" not in base:
            return ""
        return base.rsplit("/", 1)[0]

    @staticmethod
    def sibling_family_name(full_id):
        leaf = GeometryAuditor.leaf_id(full_id)
        if "_" not in leaf:
            return leaf
        head, tail = leaf.rsplit("_", 1)
        if tail.isdigit() or len(tail) <= 3:
            return head
        return leaf

    @staticmethod
    def bbox_size(mobject):
        pts = mobject.get_all_points()
        if len(pts) == 0:
            return np.array([0.0, 0.0])
        mins, maxs = np.min(pts[:, :2], axis=0), np.max(pts[:, :2], axis=0)
        return np.maximum(maxs - mins, 1e-6)

    @staticmethod
    def bbox_bounds(mobject):
        pts = mobject.get_all_points()
        if len(pts) == 0:
            return None
        mins = np.min(pts[:, :2], axis=0)
        maxs = np.max(pts[:, :2], axis=0)
        return mins, maxs

    @staticmethod
    def is_text_like(mobject):
        return isinstance(mobject, (Text, Paragraph, MathTex, Tex))

    @staticmethod
    def is_outline_container(mobject):
        if not isinstance(mobject, (RoundedRectangle, Rectangle, Circle, Ellipse)):
            return False
        fill_opacity = 0.0
        stroke_opacity = 0.0
        stroke_width = 0.0
        try:
            fill_opacity = float(mobject.get_fill_opacity())
        except Exception:
            pass
        try:
            stroke_opacity = float(mobject.get_stroke_opacity())
        except Exception:
            pass
        try:
            stroke_width = float(mobject.get_stroke_width())
        except Exception:
            pass
        return fill_opacity <= 0.05 and stroke_opacity > 0 and stroke_width > 0

    def is_partial_highlight(self, item, cont, leak_ratio):
        if leak_ratio <= self.allowed_container_leak_ratio:
            return False
        if not self.is_text_like(item) or not self.is_outline_container(cont):
            return False

        children = [child for child in getattr(item, "submobjects", []) if len(child.get_all_points()) > 0]
        if len(children) < 2:
            return False

        inside_children = 0
        for child in children:
            child_leak = self.get_leak_ratio(child, cont)
            child_cover = self.get_overlap_score(child, cont)
            if child_leak <= 0.2 or child_cover >= 0.75:
                inside_children += 1

        min_inside = 1 if len(children) <= 4 else max(2, int(np.ceil(len(children) * 0.2)))
        return min_inside <= inside_children < len(children)

    def is_text_outline_highlight(self, item, cont):
        if not self.is_text_like(item) or not self.is_outline_container(cont):
            return False
        item_bounds = self.bbox_bounds(item)
        cont_bounds = self.bbox_bounds(cont)
        if item_bounds is None or cont_bounds is None:
            return False

        item_mins, item_maxs = item_bounds
        cont_mins, cont_maxs = cont_bounds
        item_center = (item_mins + item_maxs) / 2
        cont_size = np.maximum(cont_maxs - cont_mins, 1e-6)
        item_size = np.maximum(item_maxs - item_mins, 1e-6)

        center_inside = np.all(item_center >= cont_mins - 0.05) and np.all(item_center <= cont_maxs + 0.05)
        size_ratio = cont_size / item_size
        size_reasonable = np.all(size_ratio >= 1.0) and np.all(size_ratio <= 4.0)
        return center_inside and size_reasonable

    def get_text_directional_leak_ratios(self, item, cont):
        if not self.is_text_like(item):
            return None
        item_bounds = self.bbox_bounds(item)
        cont_bounds = self.bbox_bounds(cont)
        if item_bounds is None or cont_bounds is None:
            return None

        item_mins, item_maxs = item_bounds
        cont_mins, cont_maxs = cont_bounds
        item_size = np.maximum(item_maxs - item_mins, 1e-6)

        overflow_left = max(0.0, float(cont_mins[0] - item_mins[0]))
        overflow_right = max(0.0, float(item_maxs[0] - cont_maxs[0]))
        overflow_bottom = max(0.0, float(cont_mins[1] - item_mins[1]))
        overflow_top = max(0.0, float(item_maxs[1] - cont_maxs[1]))

        horizontal_ratio = (overflow_left + overflow_right) / float(item_size[0])
        vertical_ratio = (overflow_bottom + overflow_top) / float(item_size[1])
        return horizontal_ratio, vertical_ratio

    def text_leak_within_directional_tolerance(self, item, cont):
        directional = self.get_text_directional_leak_ratios(item, cont)
        if directional is None:
            return False
        horizontal_ratio, vertical_ratio = directional
        return horizontal_ratio <= 0.1 and vertical_ratio <= 0.2

    def is_local_focus_box(self, mobject):
        name = getattr(mobject, "_full_id", "").lower()
        keywords = ("focus", "highlight", "emphasis", "callout", "marker", "frame")
        if not any(keyword in name for keyword in keywords):
            return False
        if not self.is_box_like(mobject) or not self.is_outline_container(mobject):
            return False
        bounds = self.bbox_bounds(mobject)
        if bounds is None:
            return False
        mins, maxs = bounds
        size = np.maximum(maxs - mins, 1e-6)
        frame_area = max((self.limit_x * 2.0) * (self.limit_y * 2.0), 1e-6)
        area_ratio = float(size[0] * size[1]) / frame_area
        return area_ratio < 0.2

    def is_local_focus_label_pair(self, mob1, mob2, overlap_ratio):
        if overlap_ratio >= 0.5:
            return False
        if self.is_local_focus_box(mob1) and self.is_text_like(mob2):
            return True
        if self.is_local_focus_box(mob2) and self.is_text_like(mob1):
            return True
        return False

    def has_outline_sibling(self, mobject, parent_atoms):
        parent_id = self.parent_id(getattr(mobject, "_full_id", ""))
        if not parent_id:
            return False
        for sibling in parent_atoms.get(parent_id, []):
            if sibling is mobject:
                continue
            if self.is_outline_container(sibling):
                return True
        return False

    @staticmethod
    def has_visible_fill(mobject):
        try:
            return float(mobject.get_fill_opacity()) > 0.05
        except Exception:
            return False

    def is_diagram_boundary(self, mobject, parent_atoms):
        if not self.is_outline_container(mobject):
            return False
        parent_id = self.parent_id(getattr(mobject, "_full_id", ""))
        if not parent_id:
            return False

        siblings = [s for s in parent_atoms.get(parent_id, []) if s is not mobject]
        same_type_outline_count = sum(
            1 for s in siblings
            if s.__class__ is mobject.__class__ and self.is_outline_container(s)
        )
        has_filled_sibling = any(self.has_visible_fill(s) for s in siblings)
        return same_type_outline_count >= 1 and has_filled_sibling

    def is_repeated_structure_member(self, mobject, parent_atoms):
        parent_id = self.parent_id(getattr(mobject, "_full_id", ""))
        family = self.sibling_family_name(getattr(mobject, "_full_id", ""))
        siblings = [
            sibling for sibling in parent_atoms.get(parent_id, [])
            if sibling.__class__ is mobject.__class__
            and self.sibling_family_name(getattr(sibling, "_full_id", "")) == family
        ]
        return len(siblings) >= 4

    @staticmethod
    def is_annotation_text(mobject):
        if not GeometryAuditor.is_text_like(mobject):
            return False
        full_id = getattr(mobject, "_full_id", "").lower()
        keywords = ("label", "text", "tag", "note", "caption", "nomatch", "match", "hint", "desc")
        return any(keyword in full_id for keyword in keywords)

    def is_separator_like(self, mobject):
        full_id = getattr(mobject, "_full_id", "").lower()
        parent_id = self.parent_id(full_id)
        keywords = ("boundary", "divider", "separator", "dashedline", "split")
        if any(keyword in full_id for keyword in keywords):
            return True
        if any(keyword in parent_id for keyword in keywords):
            return True
        return isinstance(mobject, DashedLine)

    @staticmethod
    def is_box_like(mobject):
        if isinstance(mobject, (Rectangle, RoundedRectangle, Square)):
            return True
        if not isinstance(mobject, VMobject):
            return False
        pts = mobject.get_all_points()
        if len(pts) < 4:
            return False
        try:
            fill = float(mobject.get_fill_opacity())
        except Exception:
            fill = 0.0
        try:
            stroke = float(mobject.get_stroke_width())
        except Exception:
            stroke = 0.0
        bounds = GeometryAuditor.bbox_bounds(mobject)
        if bounds is None:
            return False
        mins, maxs = bounds
        size = np.maximum(maxs - mins, 1e-6)
        aspect = max(size[0], size[1]) / max(min(size[0], size[1]), 1e-6)
        return stroke > 0 and (fill >= 0 or stroke > 0) and aspect <= 8.0

    def is_edge_touching_boxes(self, mob1, mob2):
        if not self.is_box_like(mob1) or not self.is_box_like(mob2):
            return False
        bounds1 = self.bbox_bounds(mob1)
        bounds2 = self.bbox_bounds(mob2)
        if bounds1 is None or bounds2 is None:
            return False
        mins1, maxs1 = bounds1
        mins2, maxs2 = bounds2

        overlap_x = min(maxs1[0], maxs2[0]) - max(mins1[0], mins2[0])
        overlap_y = min(maxs1[1], maxs2[1]) - max(mins1[1], mins2[1])
        tol = 0.03

        touch_x = abs(maxs1[0] - mins2[0]) <= tol or abs(maxs2[0] - mins1[0]) <= tol
        touch_y = abs(maxs1[1] - mins2[1]) <= tol or abs(maxs2[1] - mins1[1]) <= tol
        size1 = np.maximum(maxs1 - mins1, 1e-6)
        size2 = np.maximum(maxs2 - mins2, 1e-6)

        span_y_close = abs(size1[1] - size2[1]) / max(size1[1], size2[1], 1e-6) <= 0.15
        span_x_close = abs(size1[0] - size2[0]) / max(size1[0], size2[0], 1e-6) <= 0.15

        horizontal_join = touch_x and overlap_y > tol and overlap_x <= tol and span_y_close
        vertical_join = touch_y and overlap_x > tol and overlap_y <= tol and span_x_close
        return horizontal_join or vertical_join

    def is_overlay_highlight_pair(self, mob1, mob2):
        if mob1.__class__ is not mob2.__class__:
            return False
        if not isinstance(mob1, (Square, Rectangle, Circle)):
            return False

        bounds1 = self.bbox_bounds(mob1)
        bounds2 = self.bbox_bounds(mob2)
        if bounds1 is None or bounds2 is None:
            return False
        mins1, maxs1 = bounds1
        mins2, maxs2 = bounds2
        center1 = (mins1 + maxs1) / 2
        center2 = (mins2 + maxs2) / 2
        size1 = np.maximum(maxs1 - mins1, 1e-6)
        size2 = np.maximum(maxs2 - mins2, 1e-6)

        center_close = np.linalg.norm(center1 - center2) <= 0.08
        rel_size_diff = np.max(np.abs(size1 - size2) / np.maximum(np.maximum(size1, size2), 1e-6))
        if not center_close or rel_size_diff > 0.08:
            return False

        def style_score(m):
            fill = 0.0
            stroke = 0.0
            try:
                fill = float(m.get_fill_opacity())
            except Exception:
                pass
            try:
                stroke = float(m.get_stroke_width())
            except Exception:
                pass
            return fill, stroke

        fill1, stroke1 = style_score(mob1)
        fill2, stroke2 = style_score(mob2)
        fill_contrast = abs(fill1 - fill2) >= 0.15
        stroke_contrast = abs(stroke1 - stroke2) >= 0.5
        different_parent = self.parent_id(getattr(mob1, "_full_id", "")) != self.parent_id(getattr(mob2, "_full_id", ""))
        return different_parent and (fill_contrast or stroke_contrast)

    def is_overlay_neighbor_residual(self, mob1, mob2, overlap_ratio, parent_atoms):
        if overlap_ratio > 0.125:
            return False
        if mob1.__class__ is not mob2.__class__ or not isinstance(mob1, Square):
            return False

        repeated1 = self.is_repeated_structure_member(mob1, parent_atoms)
        repeated2 = self.is_repeated_structure_member(mob2, parent_atoms)
        if not (repeated1 and repeated2):
            return False

        bounds1 = self.bbox_bounds(mob1)
        bounds2 = self.bbox_bounds(mob2)
        if bounds1 is None or bounds2 is None:
            return False
        mins1, maxs1 = bounds1
        mins2, maxs2 = bounds2
        center1 = (mins1 + maxs1) / 2
        center2 = (mins2 + maxs2) / 2
        size1 = np.maximum(maxs1 - mins1, 1e-6)
        size2 = np.maximum(maxs2 - mins2, 1e-6)
        rel_size_diff = np.max(np.abs(size1 - size2) / np.maximum(np.maximum(size1, size2), 1e-6))
        if rel_size_diff > 0.08:
            return False

        delta = np.abs(center1 - center2)
        side = float(max(size1[0], size1[1], size2[0], size2[1]))
        near_one_step = (
            (0.8 * side <= delta[0] <= 1.2 * side and delta[1] <= 0.2 * side)
            or (0.8 * side <= delta[1] <= 1.2 * side and delta[0] <= 0.2 * side)
        )
        return near_one_step

    @staticmethod
    def name_has_any(mobject, keywords):
        full_id = getattr(mobject, "_full_id", "").lower()
        return any(keyword in full_id for keyword in keywords)

    def is_grid_local_overlay_pair(self, mob1, mob2, overlap_ratio, parent_atoms):
        if overlap_ratio > 0.13:
            return False
        if mob1.__class__ is not mob2.__class__:
            return False
        if not isinstance(mob1, (Square, Rectangle)):
            return False

        bounds1 = self.bbox_bounds(mob1)
        bounds2 = self.bbox_bounds(mob2)
        if bounds1 is None or bounds2 is None:
            return False
        mins1, maxs1 = bounds1
        mins2, maxs2 = bounds2
        size1 = np.maximum(maxs1 - mins1, 1e-6)
        size2 = np.maximum(maxs2 - mins2, 1e-6)
        center1 = (mins1 + maxs1) / 2
        center2 = (mins2 + maxs2) / 2

        rel_size_diff = np.max(np.abs(size1 - size2) / np.maximum(np.maximum(size1, size2), 1e-6))
        if rel_size_diff > 0.12:
            return False

        repeated1 = self.is_repeated_structure_member(mob1, parent_atoms)
        repeated2 = self.is_repeated_structure_member(mob2, parent_atoms)
        if not (repeated1 or repeated2):
            return False

        parent1 = self.parent_id(getattr(mob1, "_full_id", ""))
        parent2 = self.parent_id(getattr(mob2, "_full_id", ""))
        family1 = self.sibling_family_name(getattr(mob1, "_full_id", ""))
        family2 = self.sibling_family_name(getattr(mob2, "_full_id", ""))
        if repeated1 and repeated2 and parent1 == parent2 and family1 == family2:
            return True

        grid_keywords = ("grid", "sq", "cell", "window", "kernel", "patch", "response", "obj_pos", "input", "output")
        if not (self.name_has_any(mob1, grid_keywords) or self.name_has_any(mob2, grid_keywords)):
            return False

        side = float(max(size1[0], size1[1], size2[0], size2[1]))
        delta = np.abs(center1 - center2)
        close_or_neighbor = (
            np.linalg.norm(center1 - center2) <= 0.12
            or (delta[0] <= 1.2 * side and delta[1] <= 1.2 * side)
        )
        return close_or_neighbor

    def is_table_cell_rect_pair(self, mob1, mob2, overlap_ratio):
        if overlap_ratio > 0.13:
            return False
        if mob1.__class__ is not mob2.__class__:
            return False
        if not isinstance(mob1, (Square, Rectangle)):
            return False

        parts1 = self.normalized_path_parts(getattr(mob1, "_full_id", "").lower())
        parts2 = self.normalized_path_parts(getattr(mob2, "_full_id", "").lower())
        if len(parts1) < 3 or len(parts2) < 3:
            return False
        if parts1[-2:] != ["cell", "rect"] or parts2[-2:] != ["cell", "rect"]:
            return False

        def has_table_signal(parts):
            for token in parts[:-2]:
                if (
                    "row" in token
                    or "header" in token
                    or "table" in token
                    or "grid" in token
                    or token == "r"
                    or token.startswith("hrow")
                ):
                    return True
            return False

        if not (has_table_signal(parts1) and has_table_signal(parts2)):
            return False

        bounds1 = self.bbox_bounds(mob1)
        bounds2 = self.bbox_bounds(mob2)
        if bounds1 is None or bounds2 is None:
            return False
        mins1, maxs1 = bounds1
        mins2, maxs2 = bounds2
        overlap_x = min(maxs1[0], maxs2[0]) - max(mins1[0], mins2[0])
        overlap_y = min(maxs1[1], maxs2[1]) - max(mins1[1], mins2[1])
        tol = 0.03

        # 表格中异宽/异高单元格常在角点或边界处“接触”，
        # point-in-polygon 会把这类接触误计成约 0.125 的 overlap。
        return overlap_x <= tol or overlap_y <= tol

    def has_named_outline_context(self, mobject, parent_atoms, keywords):
        parent_id = self.parent_id(getattr(mobject, "_full_id", ""))
        siblings = parent_atoms.get(parent_id, [])
        for sibling in siblings:
            if sibling is mobject:
                continue
            if self.is_outline_container(sibling) and self.name_has_any(sibling, keywords):
                return True
        return False

    def is_window_annotation_pair(self, mob1, mob2, overlap_ratio, parent_atoms):
        if overlap_ratio >= 0.4:
            return False

        pair = None
        if self.is_text_like(mob1) and self.is_box_like(mob2):
            pair = (mob1, mob2)
        elif self.is_text_like(mob2) and self.is_box_like(mob1):
            pair = (mob2, mob1)
        if pair is None:
            return False

        label, grid_box = pair
        if "window" not in getattr(label, "_full_id", "").lower():
            return False
        if not self.is_repeated_structure_member(grid_box, parent_atoms):
            return False
        return True

    def is_composite_region_pair(self, mob1, mob2):
        if not self.is_box_like(mob1) or not self.is_box_like(mob2):
            return False

        bounds1 = self.bbox_bounds(mob1)
        bounds2 = self.bbox_bounds(mob2)
        if bounds1 is None or bounds2 is None:
            return False
        mins1, maxs1 = bounds1
        mins2, maxs2 = bounds2
        size1 = np.maximum(maxs1 - mins1, 1e-6)
        size2 = np.maximum(maxs2 - mins2, 1e-6)
        center1 = (mins1 + maxs1) / 2
        center2 = (mins2 + maxs2) / 2

        area1 = float(size1[0] * size1[1])
        area2 = float(size2[0] * size2[1])
        larger, smaller = (mob1, mob2) if area1 >= area2 else (mob2, mob1)
        larger_bounds = self.bbox_bounds(larger)
        smaller_bounds = self.bbox_bounds(smaller)
        lmins, lmaxs = larger_bounds
        smins, smaxs = smaller_bounds
        lsize = np.maximum(lmaxs - lmins, 1e-6)
        ssize = np.maximum(smaxs - smins, 1e-6)
        lcenter = (lmins + lmaxs) / 2
        scenter = (smins + smaxs) / 2

        if (lsize[0] * lsize[1]) / max(ssize[0] * ssize[1], 1e-6) < 1.5:
            return False

        horizontal_partition = (
            abs(lsize[0] - ssize[0]) / max(lsize[0], ssize[0], 1e-6) <= 0.08
            and abs(lcenter[0] - scenter[0]) <= 0.05
            and ssize[1] < 0.8 * lsize[1]
            and min(lmaxs[1], smaxs[1]) - max(lmins[1], smins[1]) > 0.8 * ssize[1]
        )
        vertical_partition = (
            abs(lsize[1] - ssize[1]) / max(lsize[1], ssize[1], 1e-6) <= 0.08
            and abs(lcenter[1] - scenter[1]) <= 0.05
            and ssize[0] < 0.8 * lsize[0]
            and min(lmaxs[0], smaxs[0]) - max(lmins[0], smins[0]) > 0.8 * ssize[0]
        )

        name1 = getattr(mob1, "_full_id", "").lower()
        name2 = getattr(mob2, "_full_id", "").lower()
        name_signal = any(k in name1 or k in name2 for k in ("zone", "segment", "region", "bar", "part", "block"))
        same_parent = self.parent_id(name1) == self.parent_id(name2)
        return (horizontal_partition or vertical_partition) and (same_parent or name_signal)

    def is_embedded_band_pair(self, mob1, mob2):
        if not self.is_box_like(mob1) or not self.is_box_like(mob2):
            return False
        if not (self.has_visible_fill(mob1) and self.has_visible_fill(mob2)):
            return False

        name1 = getattr(mob1, "_full_id", "").lower()
        name2 = getattr(mob2, "_full_id", "").lower()
        bounds1 = self.bbox_bounds(mob1)
        bounds2 = self.bbox_bounds(mob2)
        if bounds1 is None or bounds2 is None:
            return False
        mins1, maxs1 = bounds1
        mins2, maxs2 = bounds2
        size1 = np.maximum(maxs1 - mins1, 1e-6)
        size2 = np.maximum(maxs2 - mins2, 1e-6)
        area1 = float(size1[0] * size1[1])
        area2 = float(size2[0] * size2[1])
        larger, smaller = (mob1, mob2) if area1 >= area2 else (mob2, mob1)
        lname = getattr(larger, "_full_id", "").lower()
        sname = getattr(smaller, "_full_id", "").lower()
        if not any(k in lname for k in ("body", "base")):
            return False
        if not any(k in sname for k in ("band", "stripe", "ring")):
            return False

        larger_bounds = self.bbox_bounds(larger)
        smaller_bounds = self.bbox_bounds(smaller)
        lmins, lmaxs = larger_bounds
        smins, smaxs = smaller_bounds
        lsize = np.maximum(lmaxs - lmins, 1e-6)
        ssize = np.maximum(smaxs - smins, 1e-6)
        scenter = (smins + smaxs) / 2

        center_inside = np.all(scenter >= lmins - 0.05) and np.all(scenter <= lmaxs + 0.05)
        long_side_match = (
            abs(lsize[0] - ssize[0]) / max(lsize[0], ssize[0], 1e-6) <= 0.12
            or abs(lsize[1] - ssize[1]) / max(lsize[1], ssize[1], 1e-6) <= 0.12
        )
        short_side_narrow = min(ssize[0] / max(lsize[0], 1e-6), ssize[1] / max(lsize[1], 1e-6)) <= 0.35
        area_gap = (lsize[0] * lsize[1]) / max(ssize[0] * ssize[1], 1e-6) >= 2.0
        return center_inside and long_side_match and short_side_narrow and area_gap

    def is_highlight_label_pair(self, mob1, mob2, overlap_ratio, parent_atoms):
        if overlap_ratio >= 0.6:
            return False

        def is_label(obj):
            return self.is_text_like(obj) and self.has_outline_sibling(obj, parent_atoms)

        pair = None
        if is_label(mob1) and self.is_repeated_structure_member(mob2, parent_atoms):
            pair = (mob1, mob2)
        elif is_label(mob2) and self.is_repeated_structure_member(mob1, parent_atoms):
            pair = (mob2, mob1)

        if pair is None:
            return False

        label, repeated = pair
        label_parent = self.parent_id(getattr(label, "_full_id", ""))
        repeated_parent = self.parent_id(getattr(repeated, "_full_id", ""))
        return label_parent != repeated_parent

    def is_structured_sibling_pair(self, mob1, mob2, overlap_ratio):
        parent1 = self.parent_id(getattr(mob1, "_full_id", ""))
        parent2 = self.parent_id(getattr(mob2, "_full_id", ""))
        if not parent1 or parent1 != parent2:
            return False
        if overlap_ratio >= 0.6:
            return False
        if self.is_line_like(mob1) or self.is_line_like(mob2):
            return False

        family1 = self.sibling_family_name(getattr(mob1, "_full_id", ""))
        family2 = self.sibling_family_name(getattr(mob2, "_full_id", ""))
        if family1 != family2:
            return False

        if mob1.__class__ is not mob2.__class__:
            return False

        size1 = self.bbox_size(mob1)
        size2 = self.bbox_size(mob2)
        rel_diff = np.max(np.abs(size1 - size2) / np.maximum(np.maximum(size1, size2), 1e-6))
        if rel_diff > 0.2:
            return False

        return True

    @staticmethod
    def is_line_like(mobject):
        if isinstance(mobject, (Line, Arrow, Vector, DoubleArrow, ParametricFunction, FunctionGraph, CubicBezier, Arc)):
            return True
        name = getattr(mobject, "_full_id", "").lower()
        if isinstance(mobject, VMobject):
            fill_opacity = 0.0
            stroke_opacity = 0.0
            stroke_width = 0.0
            try:
                fill_opacity = float(mobject.get_fill_opacity())
            except Exception:
                pass
            try:
                stroke_opacity = float(mobject.get_stroke_opacity())
            except Exception:
                pass
            try:
                stroke_width = float(mobject.get_stroke_width())
            except Exception:
                pass
            line_keywords = ("curve", "graph", "plot", "path", "line", "grid")
            if fill_opacity <= 1e-3 and stroke_width > 0 and stroke_opacity > 0:
                return True
            if any(keyword in name for keyword in line_keywords) and fill_opacity <= 1e-3:
                return True
        return False

    @staticmethod
    def is_arrow_tip_like(mobject):
        full_id = getattr(mobject, "_full_id", "").lower()
        type_name = mobject.__class__.__name__
        tip_names = {"ArrowTriangleFilledTip", "ArrowTip", "ArrowTriangleTip"}
        if type_name in tip_names:
            return True
        return any(keyword in full_id for keyword in ("arrowtrianglefilledtip", "arrowtip", "arrowtriangletip"))

# ==========================================
# 3. 增强审计场景类
# ==========================================
class AuditScene(Scene):
    def ensure_audit_vars(self):
        """确保审计变量存在，防止子类覆盖 __init__"""
        if not hasattr(self, "auditor"):
            frame_width = float(getattr(config, "frame_width", 14.22))
            frame_height = float(getattr(config, "frame_height", 8.0))
            self.auditor = GeometryAuditor(frame_width=frame_width, frame_height=frame_height)
        if not hasattr(self, "audit_log"): self.audit_log = []
        if not hasattr(self, "segment_snapshots"): self.segment_snapshots = []
        if not hasattr(self, "_current_seg_locked"): self._current_seg_locked = False
        if not hasattr(self, "_save_audit_images"):
            self._save_audit_images = os.environ.get("AUDIT_SAVE_IMAGES", "1") != "0"

    def remove(self, *mobjects):
        """
        拦截动作：只要有物体被 FadeOut 或 remove，就锁定当前这一段的快照。
        """
        self.ensure_audit_vars()
        # 如果移除的不是空物体，且当前 log 有内容，且本段还没锁定
        if len(mobjects) > 0 and self.audit_log and not self._current_seg_locked:
            # 锁定当前 log 里的最后时刻（即 FadeOut 发生前的最全时刻）
            last_best = self.audit_log[-1]
            if not self.segment_snapshots or last_best["section"] != self.segment_snapshots[-1]["section"]:
                self.segment_snapshots.append(last_best)
                self._current_seg_locked = True
                sys.stderr.write(f"\n[动作拦截] 检测到 FadeOut/Remove，锁定 Segment_{len(self.segment_snapshots)-1}\n")
        super().remove(*mobjects)

    def play(self, *args, **kwargs):
        self.ensure_audit_vars()
        self._current_seg_locked = False # 每次 play 开始时解锁，允许新的移除拦截
        super().play(*args, **kwargs)
        # 实时采样
        if self.mobjects:
            self._do_audit_snapshot(f"Step_{len(self.audit_log)}")

    def render(self, preview=False):
        self.ensure_audit_vars()
        super().render(preview)
        # 兜底：如果脚本结尾没有 FadeOut，把最后的画面存为一段
        if self.audit_log:
            last_snap = self.audit_log[-1]
            if not self.segment_snapshots or self.segment_snapshots[-1]["section"] != last_snap["section"]:
                self.segment_snapshots.append(last_snap)
        self.process_final_results()

    def process_final_results(self):
        if not self.segment_snapshots and self.audit_log:
            fallback = None
            for snap in reversed(self.audit_log):
                if any(len(snap.get(k, [])) > 0 for k in ["out_of_bounds", "overlaps", "leaks"]):
                    fallback = snap
                    break
            if fallback is None:
                fallback = self.audit_log[-1]
            self.segment_snapshots.append(fallback)
            sys.stderr.write("\n[兜底分段] segment_snapshots 为空，回退生成 Segment_0\n")

        full_report = {}
        
        for i, snap in enumerate(self.segment_snapshots):
            label = f"Segment_{i}"
            has_error = any(len(snap.get(k, [])) > 0 for k in ["out_of_bounds", "overlaps", "leaks"])
            
            # 命名逻辑：报错的叫 error，没报错的样张叫 sample
            filename = f"audit_{'error' if has_error else 'sample'}_{i}.png"
            if self._save_audit_images:
                self._save_snapshot_frame(snap, filename=filename)
            
            full_report[label] = {
                "out_of_bounds": [{"mobject": x["id"], "dist": x["dist"]} for x in snap.get("out_of_bounds", [])],
                "overlaps": [{"mobjects": x["ids"], "ratio": x["ratio"]} for x in snap.get("overlaps", [])],
                "leaks": [{"mobject": x["id"], "container": x.get("cont_id", "unknown"), "ratio": x["ratio"]} for x in snap.get("leaks", [])]
            }

        # 最终汇总 JSON 输出
        output = {self.__class__.__name__: full_report}
        self.final_report = output
        sys.stderr.write("\n" + json.dumps(output, indent=2, ensure_ascii=False) + "\n")

    def _save_snapshot_frame(self, snapshot, filename="audit_result_debug.png"):
        try:
            pixel_array = snapshot.get("frame_rgb")
            if pixel_array is None:
                sys.stderr.write(f"截图缺失 ({filename})\n")
                return
            from PIL import Image
            img = Image.fromarray(pixel_array)
            img.save(filename)
            sys.stderr.write(f"[分段出图] 已保存: {filename}\n")
        except Exception as e:
            sys.stderr.write(f"截图失败 ({filename}): {e}\n")

    def _do_audit_snapshot(self, label):
        self.ensure_audit_vars()
        if not self.mobjects: return None

        all_atoms = []
        ITEM_TYPES = (Text, Paragraph, MathTex, ImageMobject, Tex, Integer, DecimalNumber)
        CONTAINER_TYPES = (RoundedRectangle, Rectangle, Circle, Ellipse)
        LINE_TYPES = (Line, Arrow, Vector, DoubleArrow, ParametricFunction, FunctionGraph, CubicBezier, Arc)
        INTERNAL_LINE_NAMES = set()

        sibling_counts = {}

        def extract_atoms(mobs, parent_path=""):
            for m in mobs:
                m_type_name = m.__class__.__name__
                if "tick" in m_type_name.lower() or m_type_name in ["Dot", "TangentLine", "Elbow"]:
                    continue
                base_id = f"{parent_path}/{getattr(m, 'name', m_type_name)}" if parent_path else getattr(m, 'name', m_type_name)
                sibling_counts[base_id] = sibling_counts.get(base_id, 0) + 1
                suffix = sibling_counts[base_id] - 1
                m._full_id = base_id if suffix == 0 else f"{base_id}#{suffix}"
                
                if len(m.submobjects) > 0 and not isinstance(m, (ITEM_TYPES + CONTAINER_TYPES)):
                    extract_atoms(m.submobjects, m._full_id)
                else:
                    if len(m.get_all_points()) > 0:
                        all_atoms.append(m)
        
        extract_atoms(self.mobjects)
        if not all_atoms: return None

        oob_raw, overlap_raw, leak_raw = [], [], []
        parent_atoms = {}
        for atom in all_atoms:
            parent_atoms.setdefault(self.auditor.parent_id(atom._full_id), []).append(atom)
        parent_map = {atom._full_id: cont._full_id for atom in all_atoms if isinstance(atom, ITEM_TYPES)
                      for cont in all_atoms if isinstance(cont, CONTAINER_TYPES) and cont is not atom
                      if self.auditor.get_overlap_score(atom, cont) > self.auditor.parent_overlap_threshold}

        for i, m1 in enumerate(all_atoms):
            d = self.auditor.get_oob_score(m1)
            if d > 0.05:
                oob_raw.append({"obj": m1, "id": m1._full_id, "dist": d})

            for j in range(i + 1, len(all_atoms)):
                m2 = all_atoms[j]
                r = max(self.auditor.get_overlap_score(m1, m2), self.auditor.get_overlap_score(m2, m1))
                if r < 0.1: continue 

                t1, t2 = m1.__class__.__name__, m2.__class__.__name__
                id1, id2 = m1._full_id.lower(), m2._full_id.lower()

                # --- [新增/扩展] 语义化豁免逻辑 ---
                # 1. 图表组件豁免 (Axes, Graph, Label...)
                CHART_KEYWORDS = ["graph", "axes", "axis", "label", "tick", "plot"]
                
                # 2. 装饰/高亮组件豁免 (Highlight, Box, Rect, Frame...)
                # 这些物体通常就是为了包裹或压在别的物体上
                DECO_KEYWORDS = ["highlight"]

                is_m1_chart = any(k in id1 for k in CHART_KEYWORDS)
                is_m2_chart = any(k in id2 for k in CHART_KEYWORDS)
                
                is_m1_deco = any(k in id1 for k in DECO_KEYWORDS)
                is_m2_deco = any(k in id2 for k in DECO_KEYWORDS)

                # 判定 A: 如果两个都是图表内部组件，豁免
                if is_m1_chart and is_m2_chart:
                    if r < 0.85: continue
                
                # 判定 B: 如果其中一个是高亮/背景框，另一个是普通内容，豁免
                # 因为 highlight_rect 就是要盖在文字或格子上
                if is_m1_deco or is_m2_deco:
                    # 除非是两个高亮框几乎完全重叠(>0.95)，否则不报
                    if r < 0.95: continue
                # -----------------------

                if self.auditor.is_highlight_label_pair(m1, m2, r, parent_atoms):
                    continue
                if self.auditor.is_edge_touching_boxes(m1, m2):
                    continue
                if self.auditor.is_overlay_highlight_pair(m1, m2):
                    continue
                if self.auditor.is_overlay_neighbor_residual(m1, m2, r, parent_atoms):
                    continue
                if self.auditor.is_grid_local_overlay_pair(m1, m2, r, parent_atoms):
                    continue
                if self.auditor.is_table_cell_rect_pair(m1, m2, r):
                    continue
                if self.auditor.is_composite_region_pair(m1, m2):
                    continue
                if self.auditor.is_embedded_band_pair(m1, m2):
                    continue
                if self.auditor.is_local_focus_label_pair(m1, m2, r):
                    continue
                if self.auditor.is_window_annotation_pair(m1, m2, r, parent_atoms):
                    continue

                is_tip1 = self.auditor.is_arrow_tip_like(m1)
                is_tip2 = self.auditor.is_arrow_tip_like(m2)
                if (is_tip1 or is_tip2) and r < 0.8:
                    continue

                if (
                    (self.auditor.is_separator_like(m1) and self.auditor.is_text_like(m2))
                    or (self.auditor.is_separator_like(m2) and self.auditor.is_text_like(m1))
                ) and r < 0.8:
                    continue

                if (
                    (self.auditor.is_annotation_text(m1) and self.auditor.is_repeated_structure_member(m2, parent_atoms))
                    or (self.auditor.is_annotation_text(m2) and self.auditor.is_repeated_structure_member(m1, parent_atoms))
                ) and r < 0.25:
                    continue

                is_l1 = self.auditor.is_line_like(m1) or t1 in INTERNAL_LINE_NAMES
                is_l2 = self.auditor.is_line_like(m2) or t2 in INTERNAL_LINE_NAMES
                is_c1, is_c2 = isinstance(m1, CONTAINER_TYPES), isinstance(m2, CONTAINER_TYPES)
                is_i1, is_i2 = isinstance(m1, ITEM_TYPES), isinstance(m2, ITEM_TYPES)
                parent1 = self.auditor.parent_id(m1._full_id)
                parent2 = self.auditor.parent_id(m2._full_id)

                if is_l1 and is_l2: continue
                if (is_l1 and is_c2) or (is_l2 and is_c1): continue
                if (is_l1 and not is_i2) or (is_l2 and not is_i1): continue
                if parent1 and parent1 == parent2 and not (is_i1 or is_i2):
                    continue
                if self.auditor.is_structured_sibling_pair(m1, m2, r):
                    continue
                
                cont, item = (m1, m2) if (is_c1 and is_i2) else (m2, m1) if (is_c2 and is_i1) else (None, None)
                if cont and item:
                    if self.auditor.is_text_outline_highlight(item, cont):
                        continue
                    if self.auditor.is_diagram_boundary(cont, parent_atoms):
                        continue
                    if self.auditor.is_partial_highlight(item, cont, 1.0):
                        continue
                    lr = self.auditor.get_leak_ratio(item, cont)
                    if lr <= self.auditor.allowed_container_leak_ratio:
                        continue
                    if self.auditor.text_leak_within_directional_tolerance(item, cont):
                        continue
                    if self.auditor.is_partial_highlight(item, cont, lr):
                        continue
                    if self.auditor.is_window_annotation_pair(item, cont, lr, parent_atoms):
                        continue
                    if parent_map.get(item._full_id) != cont._full_id:
                        if 0.05 < lr <= 0.5:
                            leak_raw.append({"obj_mob": item, "cont_mob": cont, "ratio": lr, "id": item._full_id, "cont_id": cont._full_id})
                            continue 

                if r > 0.98: continue 
                overlap_raw.append({"mobs": [m1, m2], "ids": [m1._full_id, m2._full_id], "ratio": r})

        frame_rgb = None
        try:
            self.renderer.update_frame(self)
            frame_rgb = np.array(self.renderer.get_frame()).copy()
        except Exception as e:
            sys.stderr.write(f"实时截图失败 ({label}): {e}\n")

        snapshot = {
            "section": label,
            "out_of_bounds": oob_raw,
            "overlaps": overlap_raw,
            "leaks": leak_raw,
            "all_atoms": all_atoms,
            "frame_rgb": frame_rgb,
        }
        self.audit_log.append(snapshot)
        return snapshot

    def _visualize_log(self, snapshot):
        viz = VGroup()
        for oob in snapshot.get("out_of_bounds", []):
            rect = SurroundingRectangle(oob["obj"], color=RED, buff=0.05).set_stroke(width=10)
            viz.add(rect)
        for ov in snapshot.get("overlaps", []):
            m1, m2 = ov["mobs"]
            try:
                viz.add(Intersection(m1, m2, color=YELLOW, fill_opacity=0.8))
            except:
                viz.add(SurroundingRectangle(VGroup(m1, m2), color=YELLOW, fill_opacity=0.3))
        for lk in snapshot.get("leaks", []):
            viz.add(SurroundingRectangle(lk["obj_mob"], color=PURPLE).set_stroke(width=4))
        viz.set_z_index(100)
        self.add(viz)

    def get_final_report(self):
        """保持向前兼容，用于打印 JSON 汇总"""
        if not self.audit_log: return self._do_audit_snapshot("Last_Resort")
        for s in reversed(self.audit_log):
            if any(len(s.get(k, [])) > 0 for k in ["out_of_bounds", "overlaps", "leaks"]): return s
        return self.audit_log[-1]

# ==========================================
# 4. 执行逻辑
# ==========================================
class NameInjector(ast.NodeTransformer):
    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name_id = node.targets[0].id
            setter = ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id=name_id, ctx=ast.Load()), attr="set", ctx=ast.Load()), args=[], keywords=[ast.keyword(arg="name", value=ast.Constant(value=name_id))]))
            return [node, ast.Try(body=[setter], handlers=[ast.ExceptHandler(body=[ast.Pass()])], orelse=[], finalbody=[])]
        return node

def run_audit(file_path, json_out=None, output_dir=None):
    import manim
    manim.Scene = AuditScene
    config.dry_run = False 
    config.format = "png"
    config.verbosity = "ERROR"
    os.environ.setdefault("TMPDIR", str(AUDIT_TMP_ROOT))
    os.environ.setdefault("TEMP", str(AUDIT_TMP_ROOT))
    os.environ.setdefault("TMP", str(AUDIT_TMP_ROOT))
    temp_media_dir = tempfile.TemporaryDirectory(prefix="audit_media_", dir=str(AUDIT_TMP_ROOT))
    config.media_dir = temp_media_dir.name
    tree = NameInjector().visit(ast.parse(Path(file_path).read_text(encoding="utf-8")))
    ast.fix_missing_locations(tree)
    ctx = {"__name__": "__main__", "__file__": str(file_path)}
    ctx.update(manim.__dict__)
    reports = {}
    original_cwd = os.getcwd()
    current_scene = None
    try:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            os.chdir(output_dir)
        exec(compile(tree, filename=str(file_path), mode="exec"), ctx)
        for name in [n.name for n in tree.body if isinstance(n, ast.ClassDef)]:
            cls = ctx.get(name)
            if isinstance(cls, type) and issubclass(cls, Scene):
                inst = cls()
                current_scene = inst
                try:
                    inst.render()
                except Exception as scene_error:
                    sys.stderr.write(f"Audit Scene Error [{name}]: {str(scene_error)}\n")
                    if getattr(inst, "audit_log", None) or getattr(inst, "segment_snapshots", None):
                        try:
                            inst.process_final_results()
                            if getattr(inst, "final_report", None):
                                reports.update(inst.final_report)
                        except Exception as salvage_error:
                            sys.stderr.write(f"Audit Salvage Error [{name}]: {str(salvage_error)}\n")
                    continue
                if getattr(inst, "final_report", None):
                    reports.update(inst.final_report)
    except Exception as e:
        sys.stderr.write(f"Audit Runtime Error: {str(e)}\n")
        if current_scene is not None and (getattr(current_scene, "audit_log", None) or getattr(current_scene, "segment_snapshots", None)):
            try:
                current_scene.process_final_results()
                if getattr(current_scene, "final_report", None):
                    reports.update(current_scene.final_report)
            except Exception as salvage_error:
                sys.stderr.write(f"Audit Runtime Salvage Error: {str(salvage_error)}\n")
    finally:
        temp_media_dir.cleanup()
        if output_dir:
            os.chdir(original_cwd)

    if json_out:
        Path(json_out).write_text(
            json.dumps(reports, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return reports

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    run_audit(args.file_path, json_out=args.json_out, output_dir=args.output_dir)
