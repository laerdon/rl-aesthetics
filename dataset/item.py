from enum import Enum
from typing import *
import math

class ItemType(Enum):
    TITLE = 1
    HEADING = 2
    TEXT = 3

class Item:
    def __init__(self, t, x, y, h, w):
        self.t = t
        self.x = x
        self.y = y
        self.h = h
        self.w = w

def _rect_right(it: Item) -> float: return it.x + it.w
def _rect_bottom(it: Item) -> float: return it.y + it.h
def _area(it: Item) -> float: return max(it.w, 0) * max(it.h, 0)
def _center(it: Item) -> Tuple[float, float]: return (it.x + it.w/2, it.y + it.h/2)

def _overlap(a: Item, b: Item) -> bool:
    return _rect_right(a) > b.x and _rect_right(b) > a.x and _rect_bottom(a) > b.y and _rect_bottom(b) > a.y

def _min_dist_to_edge(it: Item, W: float, H: float) -> float:
    return min(it.x, it.y, W - _rect_right(it), H - _rect_bottom(it))

def _clamp01(x: float) -> float:
    return 0.0 if math.isnan(x) else max(0.0, min(1.0, x))

def _variance(values: List[float]) -> float:
    if not values: return 0.0
    m = sum(values) / len(values)
    return sum((v - m) ** 2 for v in values) / len(values)

def _std(values: List[float]) -> float:
    return math.sqrt(_variance(values))

def _pairwise_gaps(sorted_edges: List[float]) -> List[float]:
    return [sorted_edges[i+1] - sorted_edges[i] for i in range(len(sorted_edges)-1)]

def _items_of(items: List[Item], kinds: set) -> List[Item]:
    return [it for it in items if it.t in kinds]

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return default if b == 0 else a / b