import torch
from typing import *

# --- slight redo

import math

# non overlapping?


# expect items 
def chk_within_bounds(canvas: Tuple, dims: List, placements: List):
    # dims = [(l,w)]
    # placements = [(x,y)]
    # take in canvas, tuple of (l,w), and an item assignment (x,y): check that no items are out of bounds

    assert len(dims) == len(placements)
    assert len(dims) % 2 == 0
    W, H = canvas
    for i in range(0, len(dims), 2):
        # turn tensors into plain numbers
        l = float(dims[i]) if isinstance(dims, torch.Tensor) else dims[i]
        w = float(dims[i+1]) if isinstance(dims, torch.Tensor) else dims[i+1]

        x = float(placements[i]) if isinstance(placements, torch.Tensor) else placements[i]
        y = float(placements[i+1]) if isinstance(placements, torch.Tensor) else placements[i+1]

        if x < 0 or y < 0 or x + l > W or y + w > H:
            return False

    return True

def quantify_overlap(canvas: Tuple, dims: List, placements: List) -> float:
    # dims = [(l,w)]
    # placements = [(x,y)]
    W, H = canvas
    assert len(dims) == len(placements)
    assert len(dims) % 2 == 0
    for i in range(0, len(dims), 2):
        l1, w1 = dims[i], dims[i+1]
        x1, y1 = placements[i], placements[i+1]
        r1 = x1 + l1
        b1 = y1 + w1
        for j in range(i+2, len(dims), 2):
            l2, w2 = dims[j], dims[j+1]
            x2, y2 = placements[j], placements[j+1]
            r2 = x2 + l2
            b2 = y2 + w2
            if r1 > x2 and r2 > x1 and b1 > y2 and b2 > y1:
                # overlap area
                overlap_w = min(r1, r2) - max(x1, x2)
                overlap_h = min(b1, b2) - max(y1, y2)
                return overlap_w * overlap_h
    return 0.0


# --- shawn's stuff

# # =========================
# # 1) Spatial Alignment & Structure
# # =========================

# def grid_alignment_score(items: List[Item], W: float, H: float, grid_cols: int = 12, grid_rows: Optional[int] = None, snap_px: float = 4.0) -> float:
#     """
#     Reward edges that fall near gridlines. Counts left/right (and top/bottom if rows given).
#     snap_px determines the tolerance to count as "on grid".
#     """
#     if not items or grid_cols <= 0: return 1.0
#     col_w = W / grid_cols
#     row_h = H / (grid_rows or grid_cols)

#     def near_grid(val: float, step: float) -> bool:
#         mod = val % step
#         return min(mod, step - mod) <= snap_px

#     edges = 0
#     hits = 0
#     for it in items:
#         # vertical lines
#         for x_edge in (it.x, _rect_right(it)):
#             edges += 1
#             if near_grid(x_edge, col_w): hits += 1
#         # horizontal lines if rows configured
#         if grid_rows:
#             for y_edge in (it.y, _rect_bottom(it)):
#                 edges += 1
#                 if near_grid(y_edge, row_h): hits += 1
#     return _clamp01(_safe_div(hits, max(edges, 1), 1.0))

# def left_alignment_consistency_score(items: List[Item], types_for_alignment: set = {ItemType.TEXT, ItemType.HEADING}, tolerance_px: float = 4.0) -> float:
#     """
#     Reward when most items of the chosen types share the same left x within tolerance.
#     """
#     xs = [it.x for it in items if it.t in types_for_alignment]
#     if len(xs) <= 1: return 1.0
#     # Find the densest "cluster" around some x using a simple vote
#     best_cluster = 0
#     for x in xs:
#         cluster = sum(1 for xx in xs if abs(xx - x) <= tolerance_px)
#         best_cluster = max(best_cluster, cluster)
#     return _clamp01(best_cluster / len(xs))

# def even_spacing_score(items: List[Item], axis: str = "y") -> float:
#     """
#     Reward uniform gaps along chosen axis ("y" for vertical rhythm, "x" for horizontal).
#     """
#     if len(items) <= 2: return 1.0
#     if axis == "y":
#         baselines = sorted([(it.y, it) for it in items], key=lambda t: t[0])
#         edges = [y for y, _ in baselines]
#     else:
#         baselines = sorted([(it.x, it) for it in items], key=lambda t: t[0])
#         edges = [x for x, _ in baselines]
#     gaps = _pairwise_gaps(edges)
#     if not gaps: return 1.0
#     v = _std(gaps)
#     # Map low std -> high score. Heuristic scale using median gap.
#     med = sorted(gaps)[len(gaps)//2]
#     norm = _safe_div(v, med if med > 1e-6 else 1.0, 0.0)
#     return _clamp01(1.0 - min(1.0, norm))  # smaller variance better

# def no_overlap_score(items: List[Item]) -> float:
#     """1.0 if no overlaps; drops to 0.0 with any overlap (can soften if desired)."""
#     n = len(items)
#     for i in range(n):
#         for j in range(i+1, n):
#             if _overlap(items[i], items[j]):
#                 return 0.0
#     return 1.0

# def margin_compliance_score(items: List[Item], W: float, H: float, min_margin_px: float = 16.0) -> float:
#     """
#     Reward all items keeping at least min_margin_px from canvas edges.
#     Partial credit if some violate slightly.
#     """
#     if not items: return 1.0
#     ok = 0
#     for it in items:
#         d = _min_dist_to_edge(it, W, H)
#         if d >= min_margin_px:
#             ok += 1
#         elif d > 0:
#             # linear scale if within [0, min_margin]
#             ok += d / min_margin_px
#     return _clamp01(ok / len(items))

# # =========================
# # 2) Balance & Composition
# # =========================

# def center_of_mass_balance_score(items: List[Item], W: float, H: float, type_weights: Optional[Dict[ItemType, float]] = None, target: str = "center") -> float:
#     """
#     Reward when weighted centroid is near center (or rule-of-thirds).
#     """
#     if not items: return 1.0
#     type_weights = type_weights or {ItemType.TITLE: 1.2, ItemType.HEADING: 1.0, ItemType.TEXT: 0.9}
#     total_w = 0.0
#     cx, cy = 0.0, 0.0
#     for it in items:
#         w = type_weights.get(it.t, 1.0) * _area(it)
#         x, y = _center(it)
#         total_w += w
#         cx += w * x
#         cy += w * y
#     if total_w == 0: return 1.0
#     cx /= total_w; cy /= total_w
#     if target == "thirds":
#         targets = [(W/3, H/3), (2*W/3, H/3), (W/3, 2*H/3), (2*W/3, 2*H/3)]
#         d = min(math.hypot(cx - tx, cy - ty) for tx, ty in targets)
#         max_d = math.hypot(W/3, H/3)
#     else:
#         d = math.hypot(cx - W/2, cy - H/2)
#         max_d = math.hypot(W/2, H/2)
#     return _clamp01(1.0 - d / max_d)

# def symmetry_score(items: List[Item], W: float, H: float) -> float:
#     """
#     Reward left/right and top/bottom mass balance.
#     """
#     if not items: return 1.0
#     left = sum(_area(it) for it in items if _center(it)[0] < W/2)
#     right = sum(_area(it) for it in items if _center(it)[0] >= W/2)
#     top = sum(_area(it) for it in items if _center(it)[1] < H/2)
#     bottom = sum(_area(it) for it in items if _center(it)[1] >= H/2)

#     lr = 1.0 - abs(left - right) / max(left + right, 1.0)
#     tb = 1.0 - abs(top - bottom) / max(top + bottom, 1.0)
#     return _clamp01(0.5 * (lr + tb))

# def whitespace_ratio_score(items: List[Item], W: float, H: float, ideal_range: Tuple[float, float] = (0.6, 0.85)) -> float:
#     """
#     Reward when whitespace ratio is within an ideal range.
#     """
#     fill = sum(_area(it) for it in items) / (W * H)
#     ws = 1.0 - fill
#     lo, hi = ideal_range
#     if ws < lo:
#         return _clamp01(ws / lo)
#     if ws > hi:
#         return _clamp01((1.0 - ws) / (1.0 - hi)) if hi < 1.0 else 0.0
#     return 1.0

# def visual_flow_score(items: List[Item]) -> float:
#     """
#     Reward TITLE above HEADING above TEXT (by top y).
#     """
#     tops: Dict[ItemType, float] = {}
#     for t in (ItemType.TITLE, ItemType.HEADING, ItemType.TEXT):
#         ys = [it.y for it in items if it.t == t]
#         if ys: tops[t] = min(ys)
#     # If a type is missing, don't penalize it.
#     score = 1.0
#     if ItemType.TITLE in tops and ItemType.HEADING in tops:
#         score *= 1.0 if tops[ItemType.TITLE] <= tops[ItemType.HEADING] else 0.0
#     if ItemType.HEADING in tops and ItemType.TEXT in tops:
#         score *= 1.0 if tops[ItemType.HEADING] <= tops[ItemType.TEXT] else 0.0
#     return float(score)

# # =========================
# # 3) Hierarchy & Readability
# # =========================

# def type_ordering_score(items: List[Item]) -> float:
#     """
#     Reward if minY(TITLE) <= minY(HEADING) <= minY(TEXT) AND
#     also maxY(TITLE) <= minY(HEADING) etc., preventing vertical crossovers.
#     """
#     def band(it: List[Item]) -> Tuple[float, float]:
#         return (min(x.y for x in it), max(_rect_bottom(x) for x in it))

#     groups = {t: [it for it in items if it.t == t] for t in (ItemType.TITLE, ItemType.HEADING, ItemType.TEXT)}
#     order = []
#     for t in (ItemType.TITLE, ItemType.HEADING, ItemType.TEXT):
#         if groups[t]:
#             order.append((t, *band(groups[t])))

#     # Check monotonic non-overlapping bands
#     for i in range(len(order)-1):
#         _, _, bottom = order[i]
#         _, top_next, _ = order[i+1]
#         if bottom > top_next:  # overlap in reading order bands
#             return 0.0
#     return 1.0

# def size_hierarchy_score(items: List[Item]) -> float:
#     """
#     Reward TITLE > HEADING > TEXT by average area.
#     """
#     def avg_area(t: ItemType) -> float:
#         arr = [_area(it) for it in items if it.t == t]
#         return sum(arr)/len(arr) if arr else 0.0
#     aT, aH, aB = avg_area(ItemType.TITLE), avg_area(ItemType.HEADING), avg_area(ItemType.TEXT)
#     checks = []
#     if aT and aH: checks.append(1.0 if aT > aH else 0.0)
#     if aH and aB: checks.append(1.0 if aH > aB else 0.0)
#     return sum(checks)/len(checks) if checks else 1.0

# def reading_flow_continuity_score(items: List[Item], kinds: Tuple[ItemType, ...] = (ItemType.TITLE, ItemType.HEADING, ItemType.TEXT), max_allowed_offset_px: float = 40.0) -> float:
#     """
#     Penalize large horizontal jumps between successive blocks in top-to-bottom order.
#     """
#     seq = sorted([it for it in items if it.t in kinds], key=lambda it: it.y)
#     if len(seq) <= 1: return 1.0
#     offsets = []
#     for i in range(len(seq)-1):
#         x1 = seq[i].x
#         x2 = seq[i+1].x
#         offsets.append(abs(x2 - x1))
#     avg = sum(offsets)/len(offsets)
#     return _clamp01(1.0 - min(1.0, avg / max_allowed_offset_px))

# # =========================
# # 4) Proportion & Scale
# # =========================

# def aspect_ratio_score(items: List[Item], ideal_range: Tuple[float, float] = (0.5, 5.0)) -> float:
#     """
#     Reward text boxes whose w/h stays within a sensible range.
#     Uses average clipped score across items (non-text ignored if desired).
#     """
#     lo, hi = ideal_range
#     scores = []
#     for it in items:
#         if it.h <= 0: continue
#         r = it.w / it.h
#         if r < lo:
#             scores.append(_clamp01(r / lo))
#         elif r > hi:
#             scores.append(_clamp01(hi / r))
#         else:
#             scores.append(1.0)
#     return sum(scores)/len(scores) if scores else 1.0

# def relative_scale_score(items: List[Item], W: float, H: float, max_fraction_per_item: float = 0.6, title_exempt_boost: float = 0.15) -> float:
#     """
#     Penalize items that individually dominate too much of the canvas.
#     Allows a small leniency for TITLEs.
#     """
#     A = W * H
#     worst = 0.0
#     for it in items:
#         frac = _area(it) / A
#         limit = max_fraction_per_item + (title_exempt_boost if it.t == ItemType.TITLE else 0.0)
#         worst = max(worst, max(0.0, frac - limit))
#     # Map "excess fraction" to penalty
#     return _clamp01(1.0 - min(1.0, worst / 0.4))  # 40% overage ~ zero

# def intra_type_size_consistency_score(items: List[Item]) -> float:
#     """
#     Reward consistent sizes within each type (low std/mean for areas).
#     """
#     scores = []
#     for t in (ItemType.TITLE, ItemType.HEADING, ItemType.TEXT):
#         areas = [_area(it) for it in items if it.t == t]
#         if len(areas) <= 1:
#             continue
#         m = sum(areas)/len(areas)
#         s = _std(areas)
#         scores.append(_clamp01(1.0 - _safe_div(s, m if m > 1e-6 else 1.0, 0.0)))
#     return sum(scores)/len(scores) if scores else 1.0

# # =========================
# # 5) Whitespace & Separation
# # =========================

# def minimum_gap_score(items: List[Item], min_gap_px: float = 8.0) -> float:
#     """
#     Reward when nearest edges between any two items are at least min_gap_px.
#     """
#     if len(items) <= 1: return 1.0
#     def edge_gap(a: Item, b: Item) -> float:
#         dx = max(0.0, max(a.x - _rect_right(b), b.x - _rect_right(a)))
#         dy = max(0.0, max(a.y - _rect_bottom(b), b.y - _rect_bottom(a)))
#         return max(dx, dy) if dx == 0 or dy == 0 else math.hypot(dx, dy)  # axis-prioritize
#     min_gap = float("inf")
#     for i in range(len(items)):
#         for j in range(i+1, len(items)):
#             if _overlap(items[i], items[j]):
#                 return 0.0
#             min_gap = min(min_gap, edge_gap(items[i], items[j]))
#     if min_gap == float("inf"): return 1.0
#     return _clamp01(min(1.0, min_gap / min_gap_px))

# def whitespace_evenness_score(items: List[Item], W: float, H: float, grid: Tuple[int, int] = (4, 4)) -> float:
#     """
#     Divide canvas into grid, compute occupancy per cell, reward low variance (uniform fill).
#     """
#     cols, rows = grid
#     if cols <= 0 or rows <= 0: return 1.0
#     cell_w, cell_h = W / cols, H / rows
#     occ = [0.0] * (cols * rows)

#     # crude occupancy: add fraction of item area to overlapped cells
#     for it in items:
#         c0 = int(it.x // cell_w); c1 = int((_rect_right(it) - 1) // cell_w)
#         r0 = int(it.y // cell_h); r1 = int((_rect_bottom(it) - 1) // cell_h)
#         for c in range(max(0, c0), min(cols-1, c1)+1):
#             for r in range(max(0, r0), min(rows-1, r1)+1):
#                 occ[r*cols + c] += _area(it) / (W*H)

#     v = _std(occ)
#     # Normalize by a rough expected scale (heuristic)
#     return _clamp01(1.0 - min(1.0, v / 0.1))

# # =========================
# # 6) Semantic & Structural
# # =========================

# def title_alignment_with_heading_score(items: List[Item], tolerance_px: float = 12.0) -> float:
#     """
#     Reward TITLE left edge near the dominant HEADING/TEXT left edge.
#     """
#     titles = [it for it in items if it.t == ItemType.TITLE]
#     texts = [it for it in items if it.t in {ItemType.HEADING, ItemType.TEXT}]
#     if not titles or not texts: return 1.0
#     # dominant left edge among texts:
#     xs = [it.x for it in texts]
#     dom = sorted(xs)[len(xs)//2]  # median
#     diffs = [abs(t.x - dom) for t in titles]
#     avg = sum(diffs)/len(diffs)
#     return _clamp01(1.0 - min(1.0, avg / tolerance_px))

# def logical_grouping_score(items: List[Item], same_type_max_dist: float = 80.0, diff_type_min_dist: float = 24.0) -> float:
#     """
#     Reward small distances among same-type items and separation across types.
#     """
#     if len(items) <= 1: return 1.0
#     def dist(a: Item, b: Item) -> float:
#         ax, ay = _center(a); bx, by = _center(b)
#         return math.hypot(ax - bx, ay - by)

#     same_good = []
#     diff_good = []
#     for i in range(len(items)):
#         for j in range(i+1, len(items)):
#             d = dist(items[i], items[j])
#             if items[i].t == items[j].t:
#                 same_good.append(1.0 if d <= same_type_max_dist else max(0.0, 1.0 - (d - same_type_max_dist)/same_type_max_dist))
#             else:
#                 diff_good.append(1.0 if d >= diff_type_min_dist else max(0.0, d / diff_type_min_dist))

#     parts = []
#     if same_good: parts.append(sum(same_good)/len(same_good))
#     if diff_good: parts.append(sum(diff_good)/len(diff_good))
#     return sum(parts)/len(parts) if parts else 1.0

# # =========================
# # Composite (example)
# # =========================

# DEFAULT_WEIGHTS = {
#     # Alignment & structure
#     "grid_alignment": 0.8,
#     "left_alignment": 1.0,
#     "even_spacing": 0.6,
#     "no_overlap": 1.5,
#     "margin": 0.8,
#     # Balance & composition
#     "center_of_mass": 0.6,
#     "symmetry": 0.5,
#     "whitespace_ratio": 0.7,
#     "visual_flow": 0.7,
#     # Hierarchy & readability
#     "type_order": 0.8,
#     "size_hierarchy": 0.6,
#     "reading_flow": 0.5,
#     # Proportion & scale
#     "aspect_ratio": 0.6,
#     "relative_scale": 0.7,
#     "intra_type_consistency": 0.5,
#     # Whitespace & separation
#     "minimum_gap": 0.8,
#     "whitespace_evenness": 0.5,
#     # Semantic & structural
#     "title_align_heading": 0.6,
#     "logical_grouping": 0.7,
# }

# def composite_aesthetic_score(
#     items: List[Item],
#     W: float,
#     H: float,
#     weights: Dict[str, float] = SLIDE_DIMS,
#     grid_cols: int = 12,
#     grid_rows: Optional[int] = None,
# ) -> float:
#     """
#     Weighted sum of all heuristics, normalized to [0,1] by dividing by total weights.
#     """
#     scores = {
#         "grid_alignment": grid_alignment_score(items, W, H, grid_cols, grid_rows),
#         "left_alignment": left_alignment_consistency_score(items),
#         "even_spacing": even_spacing_score(items, axis="y"),
#         "no_overlap": no_overlap_score(items),
#         "margin": margin_compliance_score(items, W, H),
#         "center_of_mass": center_of_mass_balance_score(items, W, H),
#         "symmetry": symmetry_score(items, W, H),
#         "whitespace_ratio": whitespace_ratio_score(items, W, H),
#         "visual_flow": visual_flow_score(items),
#         "type_order": type_ordering_score(items),
#         "size_hierarchy": size_hierarchy_score(items),
#         "reading_flow": reading_flow_continuity_score(items),
#         "aspect_ratio": aspect_ratio_score(items),
#         "relative_scale": relative_scale_score(items, W, H),
#         "intra_type_consistency": intra_type_size_consistency_score(items),
#         "minimum_gap": minimum_gap_score(items),
#         "whitespace_evenness": whitespace_evenness_score(items, W, H),
#         "title_align_heading": title_alignment_with_heading_score(items),
#         "logical_grouping": logical_grouping_score(items),
#     }

#     total_w = sum(weights.values())
#     if total_w <= 0: return sum(scores.values()) / len(scores)
#     weighted = sum(weights[k] * scores[k] for k in scores if k in weights)
#     return _clamp01(weighted / total_w)