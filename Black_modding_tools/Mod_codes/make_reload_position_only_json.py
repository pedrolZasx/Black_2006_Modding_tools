#!/usr/bin/env python3
"""Create reload_edit_position_only.json from the baked baseline and Maya-edited JSON.

For the Pistol_Slide bone only, retain the original baked 3x3 world orientation
and the edited JSON's world translation. Other bones and clip metadata are copied
from the edited JSON. Matrix records are 3x4 row-major: translation indices are
3, 7, and 11.

Usage:
    python make_reload_position_only_json.py
    python make_reload_position_only_json.py --baseline reload_baked.json --edited reload_edit.json --output reload_edit_position_only.json
"""
import argparse
import copy
import json
from pathlib import Path

def find_slide_bone(names):
    """main slide bone of any weapon: <Weapon>_Slide, where <Weapon>_Root is the weapon root (Pistol_, AssualtRifle_, Heavy_, ...)"""
    for n in names:
        if n.endswith("_Root") and n != "Root" and n[:-4] + "Slide" in names:
            return n[:-4] + "Slide"
    return None

ROTATION_INDICES = (0, 1, 2, 4, 5, 6, 8, 9, 10)
TRANSLATION_INDICES = (3, 7, 11)


def load_json(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def bone_map(doc: dict, label: str) -> dict:
    bones = doc.get("bones")
    if not isinstance(bones, list):
        raise ValueError(f"{label} does not contain a valid 'bones' list")
    result = {}
    for bone in bones:
        name = bone.get("name")
        if not name or name in result:
            raise ValueError(f"{label} has a missing or duplicate bone name: {name!r}")
        result[name] = bone
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="reload_baked.json", help="Original baked JSON")
    parser.add_argument("--edited", default="reload_edit.json", help="JSON exported after Maya edits")
    parser.add_argument("--bone", default="auto", help="bone to keep original rotation for (default: auto-detect <Weapon>_Slide)")
    parser.add_argument("--output", default="reload_edit_position_only.json", help="Output JSON path")
    args = parser.parse_args()

    baseline = load_json(args.baseline)
    edited = load_json(args.edited)
    if baseline.get("frames") != edited.get("frames"):
        raise ValueError(
            f"Frame count mismatch: baseline={baseline.get('frames')} edited={edited.get('frames')}. "
            "This helper deliberately preserves the original timeline."
        )
    if baseline.get("clip") != edited.get("clip"):
        raise ValueError("The baseline and edited JSON refer to different clips.")

    base_bones = bone_map(baseline, "baseline")
    edit_bones = bone_map(edited, "edited JSON")
    if set(base_bones) != set(edit_bones):
        raise ValueError("Baseline and edited JSON bone-name sets do not match.")
    TARGET_BONE = find_slide_bone(set(base_bones)) if args.bone == "auto" else args.bone
    if TARGET_BONE is None or TARGET_BONE not in base_bones:
        print(f"No slide bone found ({args.bone!r}): copying the edited JSON unchanged.")
        Path(args.output).write_text(json.dumps(edited, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return

    out = copy.deepcopy(edited)
    out_bones = bone_map(out, "output")
    base_track = base_bones[TARGET_BONE].get("world", [])
    edit_track = edit_bones[TARGET_BONE].get("world", [])
    if len(base_track) != baseline["frames"] or len(edit_track) != edited["frames"]:
        raise ValueError("Target bone's world track length does not match the clip frame count.")

    new_track = []
    for frame_index, (base_matrix, edit_matrix) in enumerate(zip(base_track, edit_track)):
        if len(base_matrix) != 12 or len(edit_matrix) != 12:
            raise ValueError(f"Expected 12 numbers in 3x4 matrices at frame {frame_index}.")
        matrix = list(edit_matrix)
        # Keep original world orientation; keep edited world-space translation.
        for i in ROTATION_INDICES:
            matrix[i] = base_matrix[i]
        for i in TRANSLATION_INDICES:
            matrix[i] = edit_matrix[i]
        new_track.append(matrix)

    out_bones[TARGET_BONE]["world"] = new_track
    # Preserve the original bind/rest matrix for the target bone too.
    out_bones[TARGET_BONE]["rest_world"] = list(base_bones[TARGET_BONE]["rest_world"])

    output_path = Path(args.output)
    output_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote: {output_path.resolve()}")
    print(f"Clip: {out.get('clip')}")
    print(f"Frames preserved: {out.get('frames')}")
    print(f"Bones: {len(out_bones)}")
    print(f"Changed JSON fields: {TARGET_BONE}.world rotation entries and {TARGET_BONE}.rest_world")
    print("Other bones, events, FPS, frame count, and edited translation values are preserved.")


if __name__ == "__main__":
    main()
