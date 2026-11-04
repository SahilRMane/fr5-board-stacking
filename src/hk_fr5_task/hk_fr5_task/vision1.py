"""Vision 1 /vision/boards JSON -> world_core detections (V3, decision-log TM-22). Pure, no ROS.

Schema = the agreed change request notes/vision1-changes.md (NOT the older vision-branch output):
    stamp (capture time, Unix s), calibrated, boards[] {variant, corners_xy, conf?, inside_calibration},
    zones[] {name, kind, x, y, yaw, age, size [long, short], top_z}.
Optional (TM-24): obstacles[] {label, corners_xy} - the LATER item of the change request; Vision 1 does not send it
yet, the fake world does (hands), so the obstacle stop works the same way once it arrives.
Ignored: occluded[], rejected[], destinations[], image_size, source, boards[].id / u / v / corners / z / yaw / zone,
zones[].frames / tag_id / age (zone memory = Vision 1's job, C8: a listed zone counts as seen).
"""
import math
from hk_fr5_task.world_core import rect_corners

ZONE_KINDS = ("drop_off", "processing")      # the fault zone is ours (config), never from Vision 1


def _finite(value):
    if isinstance(value,bool):
        raise ValueError('boolean is not a coordinate or timestamp')
    try:
        value = float(value)
    except (TypeError,ValueError,OverflowError) as error:
        raise ValueError('number cannot be converted to a finite float') from error
    if not math.isfinite(value):
        raise ValueError('number must be finite')
    return value


def _corners(points):
    cs = [[_finite(p[0]),_finite(p[1])] for p in points]
    if len(cs) != 4 or any(len(p) != 2 for p in points):
        raise ValueError('need four 2D corners')
    edges = [(cs[(i+1)%4][0]-p[0],cs[(i+1)%4][1]-p[1]) for i,p in enumerate(cs)]
    turns = [a[0]*b[1]-a[1]*b[0] for a,b in zip(edges,edges[1:]+edges[:1])]
    if not (all(t > 1e-12 for t in turns) or all(t < -1e-12 for t in turns)):
        raise ValueError('corners must form a nondegenerate convex quadrilateral')
    return cs


def parse(data):
    """data: the decoded JSON dict. Returns (stamp, dets, problems).
    dets: world_core.update() detections; problems: one text per skipped board / zone (bad or missing field).
    Raises ValueError for a frame that cannot be used at all (not calibrated, no stamp)."""
    if not isinstance(data, dict):
        raise ValueError("not a JSON object")
    if data.get("calibrated") is not True:
        raise ValueError("Vision 1 not calibrated (no base_link coordinates)")
    try:
        stamp = _finite(data["stamp"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("no valid 'stamp'")
    dets, problems = [], []
    for key in ('boards','zones','obstacles'):
        if data.get(key) is not None and not isinstance(data[key],list):
            raise ValueError(key+' must be a list')
    for k, b in enumerate(data.get("boards") or []):
        try:
            if not isinstance(b,dict):
                raise ValueError('board must be an object')
            cs = _corners(b['corners_xy'])
            conf = b.get("conf")
            score = 1.0 if conf is None else _finite(conf)
            inside = b.get('inside_calibration',True)
            if not 0 <= score <= 1 or type(inside) is not bool or not isinstance(b['variant'],str) or not b['variant']:
                raise ValueError('invalid confidence, variant or calibration flag')
            dets.append({"cls": f"board/{b['variant']}", "id": "", "score": score,
                         "corners": cs, "inside": inside})
        except (KeyError, TypeError, ValueError, IndexError) as e:
            problems.append(f"board {k}: {type(e).__name__} {e}")
    for k, z in enumerate(data.get("zones") or []):
        try:
            if not isinstance(z,dict):
                raise ValueError('zone must be an object')
            name, kind = z["name"], z["kind"]
            if not isinstance(name,str) or not name:
                raise ValueError("empty name")
            if kind not in ZONE_KINDS:
                raise ValueError(f"kind '{kind}' (expected {' / '.join(ZONE_KINDS)})")
            length, width = (_finite(v) for v in z["size"])
            if not length >= width > 0:
                raise ValueError('zone size must be positive [long, short]')
            dets.append({"cls": f"zone/{kind}", "id": name, "score": 1.0, "inside": True, "z": _finite(z["top_z"]),
                         "corners": rect_corners(_finite(z["x"]), _finite(z["y"]), _finite(z["yaw"]), length, width)})
        except (KeyError, TypeError, ValueError) as e:
            problems.append(f"zone {z.get('name', k) if isinstance(z, dict) else k}: {type(e).__name__} {e}")
    for k, o in enumerate(data.get("obstacles") or []):
        try:
            if not isinstance(o,dict):
                raise ValueError('obstacle must be an object')
            cs = _corners(o['corners_xy'])
            dets.append({"cls": f"obstacle/{o.get('label') or 'unknown'}", "id": "", "score": 1.0, "corners": cs})
        except (KeyError, TypeError, ValueError, IndexError) as e:
            problems.append(f"obstacle {k}: {type(e).__name__} {e}")
    return stamp, dets, problems
