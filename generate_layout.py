#!/usr/bin/env python3
"""
generate_layout.py

Inspect one or more .mcap files (from a video -> vipe -> dynhamr -> npz
pipeline) and emit a Foxglove Studio `<mcap_stem>.layout.json` sidecar file
next to each one, built from the topics actually present in that file.

Usage:
    python3 generate_layout.py file1.mcap file2.mcap ...
    python3 generate_layout.py /path/to/dir            # all *.mcap in dir
    python3 generate_layout.py /path/to/dir --recursive

Requires: pip install mcap
"""

import argparse
import json
import re
import sys
import uuid
from pathlib import Path

from mcap.reader import make_reader

# --------------------------------------------------------------------------
# Topic classification heuristics
# --------------------------------------------------------------------------
RE_DEPTH = re.compile(r"depth", re.I)
RE_ANNOTATIONS = re.compile(r"annotation", re.I)
RE_RGB = re.compile(r"(rgb|color|video)", re.I)
RE_IMU = re.compile(r"(imu|motion|gyro|accel)", re.I)
RE_SCENE = re.compile(r"^/scene/", re.I)
RE_LOG = re.compile(r"(caption|log|text|transcript)", re.I)

IMAGE_SCHEMAS = {
    "foxglove.CompressedImage",
    "foxglove.RawImage",
    "sensor_msgs/Image",
    "sensor_msgs/msg/Image",
    "sensor_msgs/CompressedImage",
    "sensor_msgs/msg/CompressedImage",
}
ANNOTATION_SCHEMAS = {"foxglove.ImageAnnotations", "foxglove_msgs/ImageAnnotations"}
SCENE_SCHEMAS = {"foxglove.SceneUpdate", "foxglove_msgs/SceneUpdate"}
LOG_SCHEMAS = {"foxglove.Log", "rcl_interfaces/Log", "rosgraph_msgs/Log"}


def panel_id(kind: str) -> str:
    """Foxglove panel ids look like 'Type!hexsuffix'."""
    return f"{kind}!{uuid.uuid4().hex[:7]}"


def read_topics(mcap_path: Path):
    """Return list of (topic, schema_name, message_encoding)."""
    topics = []
    try:
        with open(mcap_path, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()
            if summary is None:
                return topics
            for ch in summary.channels.values():
                schema = summary.schemas.get(ch.schema_id)
                topics.append((ch.topic, schema.name if schema else "", ch.message_encoding))
    except Exception as e:
        print(f"[!] Failed to read {mcap_path}: {e}", file=sys.stderr)
    return sorted(topics)


def classify(topics):
    """Bucket topics into image / depth / imu / scene / log / annotations / unmatched."""
    buckets = {
        "rgb": [],
        "depth": [],
        "imu": [],
        "scene": [],
        "log": [],
        "annotations": [],
        "unmatched": [],
    }

    for topic, schema, _enc in topics:
        if schema in ANNOTATION_SCHEMAS or RE_ANNOTATIONS.search(topic):
            buckets["annotations"].append(topic)
        elif RE_SCENE.match(topic) or schema in SCENE_SCHEMAS:
            buckets["scene"].append(topic)
        elif schema in LOG_SCHEMAS or RE_LOG.search(topic):
            buckets["log"].append(topic)
        elif RE_DEPTH.search(topic):
            buckets["depth"].append(topic)
        elif schema in IMAGE_SCHEMAS or RE_RGB.search(topic):
            buckets["rgb"].append(topic)
        elif RE_IMU.search(topic):
            buckets["imu"].append(topic)
        else:
            buckets["unmatched"].append(topic)

    return buckets


def match_annotations(image_topic: str, annotation_topics: list[str], used: set) -> str | None:
    """Find the annotation topic that belongs to a given image topic."""
    stem = image_topic.rstrip("/")
    candidates = {f"{stem}_annotations", f"{stem}/annotations", f"{stem}Annotations"}
    for ann in annotation_topics:
        if ann in used:
            continue
        if ann in candidates or ann.startswith(stem + "_") or ann.startswith(stem + "/"):
            return ann
    return None


# --------------------------------------------------------------------------
# Panel config builders (Optimized for side-by-side viewing)
# --------------------------------------------------------------------------

def make_3d_config(scene_topics):
    topics_cfg = {t: {"visible": True} for t in scene_topics}
    return {
        "cameraState": {
            "perspective": True,
            "distance": 3.0,
            "phi": 55,
            "thetaOffset": 30,
            "targetOffset": [0, 0, 0],
            "target": [0, 0, 0],
            "targetOrientation": [0, 0, 0, 1],
            "fovy": 45,
            "near": 0.01,
            "far": 50,
        },
        "followMode": "follow-frame",
        "followFrame": "/scene",
        "scene": {},
        "transforms": {},
        "topics": topics_cfg,
        "layers": {
            "grid": {
                "visible": True,
                "drawBehind": False,
                "label": "Grid",
                "instanceId": str(uuid.uuid4()),
                "layerId": "foxglove.Grid",
                "size": 5,
                "divisions": 10,
                "lineWidth": 1,
                "color": "#248eff",
                "position": [0, 0, 0],
                "rotation": [0, 0, 0],
            }
        },
        "publish": {
            "type": "point",
            "poseTopic": "/move_base_simple/goal",
            "pointTopic": "/clicked_point",
            "poseEstimateTopic": "/initialpose",
        },
        "imageMode": {},
        "foxglovePanelTitle": "3D Workspace View",
    }


def make_image_config(image_topic, annotation_topic, title):
    annotations_cfg = {}
    if annotation_topic:
        annotations_cfg[annotation_topic] = {"visible": True}
    return {
        "imageMode": {
            "imageTopic": image_topic,
            **({"annotations": annotations_cfg} if annotations_cfg else {}),
        },
        "cameraState": {
            "distance": 20,
            "perspective": True,
            "phi": 60,
            "target": [0, 0, 0],
            "targetOffset": [0, 0, 0],
            "targetOrientation": [0, 0, 0, 1],
            "thetaOffset": 60,
            "fovy": 45,
            "near": 0.5,
            "far": 5000,
        },
        "followMode": "follow-none",
        "scene": {},
        "transforms": {},
        "topics": {},
        "layers": {},
        "publish": {
            "type": "point",
            "poseTopic": "/move_base_simple/goal",
        },
        "foxglovePanelTitle": title,
    }


PLOT_COLORS = ["#4e98e2", "#f5774d", "#f7df71"]


def make_plot_config(base_topic, field_group, title):
    paths = [
        {
            "value": f"{base_topic}.{field_group}.{axis}",
            "enabled": True,
            "color": PLOT_COLORS[i],
        }
        for i, axis in enumerate(("x", "y", "z"))
    ]
    return {"paths": paths, "showLegend": True, "foxglovePanelTitle": title}


def make_rosout_config(topic, title):
    return {"topic": topic, "preload": True, "foxglovePanelTitle": title}


# --------------------------------------------------------------------------
# Mosaic (layout tree) builder
# --------------------------------------------------------------------------

def build_mosaic(panel_ids, direction="row"):
    """Recursively fold a flat list of panel ids into a balanced Foxglove
    mosaic tree, alternating split direction at each level."""
    if not panel_ids:
        return None
    if len(panel_ids) == 1:
        return panel_ids[0]
    mid = (len(panel_ids) + 1) // 2
    left = build_mosaic(panel_ids[:mid], "column" if direction == "row" else "row")
    right = build_mosaic(panel_ids[mid:], "column" if direction == "row" else "row")
    return {
        "direction": direction,
        "first": left,
        "second": right,
        "splitPercentage": 50,
    }


# --------------------------------------------------------------------------
# Main layout assembly
# --------------------------------------------------------------------------

def build_layout(topics):
    buckets = classify(topics)
    config_by_id = {}
    left_col = []   # 3D workspace panel 
    top_row = []    # Video/RGB camera panels
    mid_row = []    # IMU Plot panels
    bottom_row = [] # Text log/Caption panels

    # 1. Handle 3D Scene Updates (/scene/hands, /scene/camera)
    if buckets["scene"]:
        pid = panel_id("3D")
        config_by_id[pid] = make_3d_config(sorted(buckets["scene"]))
        left_col.append(pid)

    # 2. Process RGB Images and Depth Maps
    annotation_topics = buckets["annotations"]
    used_annotations = set()
    for img_topic in buckets["rgb"] + buckets["depth"]:
        ann = match_annotations(img_topic, annotation_topics, used_annotations)
        if ann:
            used_annotations.add(ann)
        label = img_topic.strip("/").split("/")[-1].replace("_", " ").title()
        pid = panel_id("Image")
        config_by_id[pid] = make_image_config(img_topic, ann, label)
        top_row.append(pid)

    leftover_annotations = [a for a in annotation_topics if a not in used_annotations]
    for extra in leftover_annotations:
        buckets["unmatched"].append(extra)

    # 3. Process IMU Topics
    for imu_topic in buckets["imu"]:
        base = imu_topic.strip("/").split("/")[-1].title()
        gpid = panel_id("Plot")
        config_by_id[gpid] = make_plot_config(imu_topic, "gyro", f"{base} Gyro (rad/s)")
        mid_row.append(gpid)
        apid = panel_id("Plot")
        config_by_id[apid] = make_plot_config(imu_topic, "accel", f"{base} Accel (m/s²)")
        mid_row.append(apid)

    # 4. Process Captions / Transcripts
    for log_topic in buckets["log"]:
        label = log_topic.strip("/").split("/")[-1].title()
        pid = panel_id("RosOut")
        config_by_id[pid] = make_rosout_config(log_topic, label)
        bottom_row.append(pid)

    # Compile the right pane content tree dynamically
    right_sections = []
    if top_row:
        right_sections.append(build_mosaic(top_row, "row"))
    if mid_row:
        right_sections.append(build_mosaic(mid_row, "row"))
    if bottom_row:
        right_sections.append(build_mosaic(bottom_row, "row"))

    right_tree = None
    for section in right_sections:
        if right_tree is None:
            right_tree = section
        else:
            right_tree = {
                "direction": "column",
                "first": right_tree,
                "second": section,
                "splitPercentage": 50,
            }

    left_tree = left_col[0] if left_col else None

    # Assemble the final parent layout tree
    if left_tree and right_tree:
        layout = {
            "direction": "row",
            "first": left_tree,
            "second": right_tree,
            "splitPercentage": 55, # 3D panel is given slightly more spacing weight
        }
    elif left_tree:
        layout = left_tree
    elif right_tree:
        layout = right_tree
    else:
        layout = None

    return config_by_id, layout, buckets["unmatched"]


def generate_layout_for_mcap(mcap_path: Path, out_path: Path = None, verbose=True):
    topics = read_topics(mcap_path)
    if not topics:
        print(f"[!] {mcap_path}: no channels found, skipping", file=sys.stderr)
        return None

    config_by_id, layout, unmatched = build_layout(topics)

    doc = {
        "configById": config_by_id,
        "globalVariables": {},
        "userNodes": {},
        "playbackConfig": {"speed": 1},
        "layout": layout,
    }

    out_path = out_path or mcap_path.with_suffix(".layout.json")
    out_path.write_text(json.dumps(doc, indent=2))

    if verbose:
        print(f"[+] {mcap_path.name} -> {out_path.name}  "
              f"({len(config_by_id)} panels)")
        if unmatched:
            print(f"    unmatched topics (not placed in a panel): {unmatched}")

    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help=".mcap files or directories")
    ap.add_argument("--recursive", action="store_true",
                    help="recurse into directories")
    args = ap.parse_args()

    mcap_files = []
    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            pattern = "**/*.mcap" if args.recursive else "*.mcap"
            mcap_files.extend(sorted(path.glob(pattern)))
        elif path.suffix == ".mcap":
            mcap_files.append(path)
        else:
            print(f"[!] skipping non-mcap path: {p}", file=sys.stderr)

    if not mcap_files:
        print("No .mcap files found.", file=sys.stderr)
        sys.exit(1)

    for mcap_path in mcap_files:
        generate_layout_for_mcap(mcap_path)


if __name__ == "__main__":
    main()