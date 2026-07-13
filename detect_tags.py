import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

# Camera parameters (OpenCV)
IMG_W, IMG_H = 4000, 3000
K = np.array([
    [1741.5801,    0.0, 1994.2998],
    [   0.0, 1741.5801, 1516.07],
    [   0.0,    0.0,    1.0]
], dtype=np.float64)

# [k1, k2, p1, p2, k3]
D = np.array([0.04558643, 0.01975962, -0.01040291, 0.00137448], dtype=np.float64)

TARGETS_ID = {
    f"usaf_{i}": list(range(i * 4, i * 4 + 4))
    for i in range(72)
}

# USAF frequency element layout
#
# After undistortion and perspective warp, the cropped image contains
# frequency elements stacked vertically (top to bottom).
# Each element has:
#   - Left side: horizontal bars (scored along y)
#   - Right side: vertical bars (scored along x)
#
# The labels below define how many elements to expect and their
# spatial frequency names (top-to-bottom order). Detection of the
# actual row positions is fully automatic from image content.

FREQUENCY_LABELS = ["2.0", "1.0", "0.6"]


# Longest dimension (px) for visualization output images (detection overlays).
# Does NOT affect detection or scoring — those always run at full resolution.
MAX_SIZE = 1600

OUTPUT_FOLDER = "detections"   # Subfolder for detection overlay images
CROPS_FOLDER = "crops"         # Subfolder for undistorted USAF crop images

# Circle radius (in full-resolution pixels) centered on the image.
# Only targets whose 4 tags are entirely within this circle are processed.
# Purpose: reject targets near the edges where distortion/vignetting is worst.
CIRCLE_RADIUS = 1300

# Output size of the perspective-warped square (px).
# The tag-to-tag quad is warped to this size before inner cropping.
CROP_OUTPUT_SIZE = 800

# After warping, a margin is cropped from all sides to exclude the tags.
# Expressed as a fraction of the detected tag side length.
# e.g. 0.7 = trim 0.7 × tag_side from each edge inward.
CROP_INSET_FRACTION = 0.55


# Minimum MTF below which bars are considered unresolved.
# Any measurement below this floor is reported as 0.0.
MTF_FLOOR = 0.05


def compute_mtf(profile):
    """
    Compute MTF from a 1-D intensity profile of a 3-bar USAF element.

    Steps:
      1. Auto-detect the bar region by thresholding (dark bars on white).
      2. Crop the profile tightly around the bars (~3 full cycles).
      3. FFT on the cropped profile; evaluate near k=3.

    This makes the measurement robust to oversized ROIs where bars
    occupy only a small fraction of the profile.

    Returns dict {"mtf": float, "crop_start": int, "crop_end": int}
    or None on failure.
    """
    n = len(profile)
    if n < 6:
        return None

    y = profile.astype(np.float64)

    y_min, y_max = y.min(), y.max()
    if y_max - y_min < 1.0:
        return {"mtf": 0.0, "crop_start": 0, "crop_end": n}  # No contrast at all

    # Threshold at midpoint: pixels below are "bar"
    threshold = (y_min + y_max) / 2.0
    indices = np.where(y < threshold)[0]
    if len(indices) < 2:
        return {"mtf": 0.0, "crop_start": 0, "crop_end": n}  # No bars found

    bar_start = indices[0]
    bar_end = indices[-1]
    bar_span = bar_end - bar_start  # ~2.5 cycles (3 bars + 2 gaps = 5 half-periods)

    if bar_span < 4:
        return {"mtf": 0.0, "crop_start": 0, "crop_end": n}

    # Estimate one half-period (single bar or gap width)
    half_period = bar_span / 5.0

    # Extend to ~3 full cycles: need 0.5 half-period padding on each side
    pad = int(half_period * 0.5 + 0.5)
    crop_start = max(0, bar_start - pad)
    crop_end = min(n, bar_end + pad + 1)

    cropped = y[crop_start:crop_end]
    nc = len(cropped)
    if nc < 6:
        return {"mtf": 0.0, "crop_start": crop_start, "crop_end": crop_end}

    # --- FFT on tightly cropped profile ---
    spectrum = np.fft.rfft(cropped)
    magnitudes = np.abs(spectrum)

    # DC component
    dc = magnitudes[0] / nc
    if dc < 1e-6:
        return None

    # Fundamental expected at k=3, search in [2, 4] for robustness
    k_min = 2
    k_max = min(4, len(magnitudes) - 1)
    if k_max < k_min:
        return None

    k_peak = k_min + int(np.argmax(magnitudes[k_min:k_max + 1]))
    amplitude = 2.0 * magnitudes[k_peak] / nc

    # MTF = modulation = amplitude / DC
    mtf = amplitude / dc

    # Clamp to [0, 1]
    mtf = min(mtf, 1.0)

    # Below the floor → bars not resolved
    if mtf < MTF_FLOOR:
        mtf = 0.0

    return {"mtf": mtf, "crop_start": crop_start, "crop_end": crop_end}


def _refine_crop(cell_gray):
    """
    Given a grayscale cell containing bars on a white background,
    find the tight bounding box of the dark content (bars).

    An inset margin (10% from each edge) is applied before thresholding
    to avoid picking up dark content that bleeds from adjacent cells.

    Returns (x0, y0, x1, y1) in cell coordinates, or None.
    """
    h, w = cell_gray.shape
    if h < 4 or w < 4:
        return None

    # Inset by 10% from each edge to exclude adjacent-group bleed
    inset_x = max(1, w // 10)
    inset_y = max(1, h // 10)
    inner = cell_gray[inset_y:h - inset_y, inset_x:w - inset_x]
    if inner.size < 4:
        return None

    inner_u8 = inner if inner.dtype == np.uint8 else inner.astype(np.uint8)
    _, mask = cv2.threshold(inner_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ys, xs = np.where(mask > 0)
    if len(ys) < 4:
        return None

    # Map back to cell coordinates
    return (
        int(xs.min()) + inset_x,
        int(ys.min()) + inset_y,
        int(xs.max()) + 1 + inset_x,
        int(ys.max()) + 1 + inset_y,
    )


def score_usaf(cropped_img):
    """
    Score each USAF frequency element in the cropped image.

    Layout assumption (fixed):
      - The cropped square is divided into 2 columns × 3 rows = 6 cells
      - Left column: horizontal bars (profile along y)
      - Right column: vertical bars (profile along x)
      - Rows top-to-bottom: low freq, mid freq, high freq
        (matching FREQUENCY_LABELS order)

    For each cell:
      1. Refine crop to tight bounding box of dark content (Otsu).
      2. Sample 1-D profile through the central 50% of the bar group.
      3. Compute MTF from the profile.

    Returns list of dicts, one per frequency.
    """
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY) if len(cropped_img.shape) == 3 else cropped_img
    h, w = gray.shape

    num_rows = len(FREQUENCY_LABELS)
    row_h = h // num_rows
    col_w = w // 2

    results = []

    for row_idx in range(num_rows):
        freq_label = FREQUENCY_LABELS[row_idx]

        # Row vertical bounds
        ry0 = row_idx * row_h
        ry1 = (row_idx + 1) * row_h if row_idx < num_rows - 1 else h

        entry = {"frequency": freq_label, "roi": (0, ry0, w, ry1)}

        # --- Left cell: H-bars (horizontal lines, profile along y) ---
        h_cell = gray[ry0:ry1, 0:col_w]
        h_box = _refine_crop(h_cell)
        if h_box is not None:
            hx0, hy0, hx1, hy1 = h_box
            hb_w = hx1 - hx0
            # Sample profile along y, averaged over central 50% of bar width
            strip_x0 = hx0 + hb_w // 4
            strip_x1 = hx1 - hb_w // 4
            if strip_x1 <= strip_x0:
                strip_x0, strip_x1 = hx0, hx1
            strip = h_cell[hy0:hy1, strip_x0:strip_x1]
            profile_h = strip.mean(axis=1)
            result_h = compute_mtf(profile_h)
            if result_h is not None:
                entry["h_mtf"] = result_h["mtf"]
                cs, ce = result_h["crop_start"], result_h["crop_end"]
                entry["h_bar_roi"] = (hx0, ry0 + hy0, hx1, ry0 + hy1)
                entry["h_sample_roi"] = (strip_x0, ry0 + hy0, strip_x1, ry0 + hy1)
                entry["h_crop_roi"] = (strip_x0, ry0 + hy0 + cs, strip_x1, ry0 + hy0 + ce)
            else:
                entry["h_mtf"] = None
                entry["h_bar_roi"] = None
                entry["h_sample_roi"] = None
                entry["h_crop_roi"] = None
        else:
            entry["h_mtf"] = None
            entry["h_bar_roi"] = None
            entry["h_sample_roi"] = None
            entry["h_crop_roi"] = None

        # --- Right cell: V-bars (vertical lines, profile along x) ---
        v_cell = gray[ry0:ry1, col_w:w]
        v_box = _refine_crop(v_cell)
        if v_box is not None:
            vx0, vy0, vx1, vy1 = v_box
            vb_h = vy1 - vy0
            # Sample profile along x, averaged over central 50% of bar height
            strip_y0 = vy0 + vb_h // 4
            strip_y1 = vy1 - vb_h // 4
            if strip_y1 <= strip_y0:
                strip_y0, strip_y1 = vy0, vy1
            strip = v_cell[strip_y0:strip_y1, vx0:vx1]
            profile_v = strip.mean(axis=0)
            result_v = compute_mtf(profile_v)
            if result_v is not None:
                entry["v_mtf"] = result_v["mtf"]
                cs, ce = result_v["crop_start"], result_v["crop_end"]
                entry["v_bar_roi"] = (col_w + vx0, ry0 + vy0, col_w + vx1, ry0 + vy1)
                entry["v_sample_roi"] = (col_w + vx0, ry0 + strip_y0, col_w + vx1, ry0 + strip_y1)
                entry["v_crop_roi"] = (col_w + vx0 + cs, ry0 + strip_y0,
                                       col_w + vx0 + ce, ry0 + strip_y1)
            else:
                entry["v_mtf"] = None
                entry["v_bar_roi"] = None
                entry["v_sample_roi"] = None
                entry["v_crop_roi"] = None
        else:
            entry["v_mtf"] = None
            entry["v_bar_roi"] = None
            entry["v_sample_roi"] = None
            entry["v_crop_roi"] = None

        results.append(entry)

    return results

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_APRILTAG_36h11
)

parameters = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(dictionary, parameters)


def order_points(pts):
    """
    Order 4 points as: top-left, top-right, bottom-right, bottom-left.
    """
    # Sort by y first to get top pair and bottom pair
    sorted_by_y = pts[np.argsort(pts[:, 1])]
    top_pair = sorted_by_y[:2]
    bottom_pair = sorted_by_y[2:]
    # Among top pair, left has smaller x
    tl, tr = top_pair[np.argsort(top_pair[:, 0])]
    # Among bottom pair, left has smaller x
    bl, br = bottom_pair[np.argsort(bottom_pair[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def undistort_and_crop(img_full, tag_centers_full, tag_corners_full):
    """
    Given the full-resolution image, 4 tag centers and their corners (in full-res coords),
    undistort the image using Brown-Conrady model, then warp the full quad defined
    by the exact tag centers (best precision), and finally crop out the inner region
    excluding the tags.
    """
    h, w = img_full.shape[:2]

    # Compute optimal new camera matrix (alpha=1 keeps all pixels)
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=1.0)

    # Undistort full image
    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, None, new_K, (w, h), cv2.CV_16SC2
    )
    undistorted = cv2.remap(img_full, map1, map2, interpolation=cv2.INTER_LINEAR)

    # Undistort the tag center points
    pts = tag_centers_full.reshape(-1, 1, 2).astype(np.float64)
    undistorted_pts = cv2.undistortPoints(pts, K, D, P=new_K)
    undistorted_pts = undistorted_pts.reshape(-1, 2).astype(np.float32)

    # Undistort tag corner points and compute average tag side length
    # from the undistorted geometry (consistent with the warped output space)
    all_corners = tag_corners_full.reshape(-1, 1, 2).astype(np.float64)
    undistorted_corners = cv2.undistortPoints(all_corners, K, D, P=new_K)
    undistorted_corners = undistorted_corners.reshape(-1, 2).astype(np.float32)
    # 4 tags × 4 corners each = 16 points; group back into 4 tags of 4 corners
    tag_side_lengths = []
    for t in range(4):
        corners_t = undistorted_corners[t * 4:(t + 1) * 4]
        for i in range(4):
            side = np.linalg.norm(corners_t[(i + 1) % 4] - corners_t[i])
            tag_side_lengths.append(side)
    avg_tag_side_px = np.mean(tag_side_lengths)

    # Order points: TL, TR, BR, BL
    ordered = order_points(undistorted_pts)

    # Warp the full tag-to-tag quad using exact tag centers (most precise anchors)
    dst_size = CROP_OUTPUT_SIZE
    dst_pts = np.array([
        [0, 0],
        [dst_size - 1, 0],
        [dst_size - 1, dst_size - 1],
        [0, dst_size - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(ordered, dst_pts)
    warped = cv2.warpPerspective(undistorted, M, (dst_size, dst_size))

    # Now crop the inner region, excluding a margin proportional to tag side.
    # Compute margin in output pixels: tag side in source maps to some size in output.
    # Approximate: avg distance between adjacent tag centers in source vs output.
    src_side_top = np.linalg.norm(ordered[1] - ordered[0])
    output_scale = (dst_size - 1) / src_side_top  # px per source-px along top edge
    margin_px = int(avg_tag_side_px * CROP_INSET_FRACTION * output_scale)

    # Ensure margin doesn't eat the whole image
    margin_px = min(margin_px, dst_size // 4)

    cropped = warped[margin_px:dst_size - margin_px, margin_px:dst_size - margin_px]

    # Tag side length in output (warped/cropped) pixels
    tag_side_output_px = avg_tag_side_px * output_scale

    return cropped, tag_side_output_px, margin_px


def resize_keep_aspect(img, max_size=1600):
    h, w = img.shape[:2]

    scale = min(max_size / w, max_size / h)

    if scale >= 1:
        return img, 1.0

    new_size = (int(w * scale), int(h * scale))

    resized = cv2.resize(
        img,
        new_size,
        interpolation=cv2.INTER_AREA
    )

    return resized, scale


def process_image(filename, output_dir, crops_dir, all_scores):

    img = cv2.imread(str(filename))

    if img is None:
        print(f"Cannot read {filename}")
        return

    # Detect tags at full resolution for precision
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    # Full-resolution image center and circle check
    h_full, w_full = img.shape[:2]
    img_center_full = (w_full / 2.0, h_full / 2.0)

    # Downsampled visualization only
    img_small, vis_scale = resize_keep_aspect(img, MAX_SIZE)
    vis = img_small.copy()
    h_vis, w_vis = vis.shape[:2]
    vis_center = (w_vis // 2, h_vis // 2)
    vis_circle_radius = int(CIRCLE_RADIUS * vis_scale)
    cv2.circle(vis, vis_center, vis_circle_radius, (255, 0, 0), 2)

    if ids is not None:

        detected_ids = set(ids.flatten().tolist())

        # Build a lookup: tag_id -> index in corners/ids
        id_to_idx = {int(tid): i for i, tid in enumerate(ids.flatten())}

        # Determine which targets have all 4 tags inside the circle
        valid_tag_ids = set()
        for target_name, target_tag_ids in TARGETS_ID.items():
            if not all(tid in detected_ids for tid in target_tag_ids):
                continue
            # Check that all 4 tags of this target are inside the circle
            target_inside = True
            for tid in target_tag_ids:
                idx = id_to_idx[tid]
                pts = corners[idx].reshape(4, 2)
                for pt in pts:
                    dist = np.linalg.norm(pt - np.array(img_center_full, dtype=np.float32))
                    if dist > CIRCLE_RADIUS:
                        target_inside = False
                        break
                if not target_inside:
                    break
            if target_inside:
                valid_tag_ids.update(target_tag_ids)

        # Draw tag bounding boxes: green if part of a valid target, red otherwise
        for c, tag_id in zip(corners, ids.flatten()):
            bbox_color = (0, 255, 0) if int(tag_id) in valid_tag_ids else (0, 0, 255)

            # Draw bounding box on visualization (scaled down)
            pts_vis = (c.reshape(4, 2) * vis_scale).astype(int)

            for i in range(4):
                cv2.line(vis, tuple(pts_vis[i]), tuple(pts_vis[(i + 1) % 4]), bbox_color, 2)

            center_vis = pts_vis.mean(axis=0).astype(int)

            cv2.putText(
                vis,
                str(tag_id),
                tuple(center_vis),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                bbox_color,
                2,
                cv2.LINE_AA,
            )

        print(f"{filename.name}: detected {len(ids)} tags, {len(valid_tag_ids)//4} valid targets")

        # Process each USAF target that has all 4 tags visible and inside circle
        for target_name, target_tag_ids in TARGETS_ID.items():
            if not all(tid in valid_tag_ids for tid in target_tag_ids):
                continue

            # Tag centers and corners are already in full-resolution coordinates
            tag_centers = np.array([
                corners[id_to_idx[tid]].reshape(4, 2).mean(axis=0)
                for tid in target_tag_ids
            ], dtype=np.float32)
            tag_corners_full = np.array([
                corners[id_to_idx[tid]].reshape(4, 2)
                for tid in target_tag_ids
            ], dtype=np.float32)

            cropped, tag_side_output_px, margin_px = undistort_and_crop(img, tag_centers, tag_corners_full)

            # Score each USAF frequency element
            freq_scores = score_usaf(cropped)

            # Annotate crop image with MTF scores
            crop_vis = cropped.copy()
            if len(crop_vis.shape) == 2:
                crop_vis = cv2.cvtColor(crop_vis, cv2.COLOR_GRAY2BGR)
            ch, cw = crop_vis.shape[:2]
            col_w = cw // 2

            # Draw grid lines (2 cols × 3 rows)
            num_rows = len(FREQUENCY_LABELS)
            row_h_vis = ch // num_rows
            # Vertical split
            cv2.line(crop_vis, (col_w, 0), (col_w, ch), (128, 128, 128), 1)
            # Horizontal splits
            for ri in range(1, num_rows):
                y_line = ri * row_h_vis
                cv2.line(crop_vis, (0, y_line), (cw, y_line), (128, 128, 128), 1)

            for entry in freq_scores:
                if entry.get("roi"):
                    x0, y0, x1, y1 = entry["roi"]

                    # --- V-bars (right column) ---
                    if entry.get("v_bar_roi"):
                        vb = entry["v_bar_roi"]
                        cv2.rectangle(crop_vis, (vb[0], vb[1]), (vb[2], vb[3]), (255, 0, 0), 1)
                    if entry.get("v_sample_roi"):
                        vs = entry["v_sample_roi"]
                        cv2.rectangle(crop_vis, (vs[0], vs[1]), (vs[2], vs[3]), (255, 255, 0), 2)
                        vy_mid = (vs[1] + vs[3]) // 2
                        cv2.arrowedLine(crop_vis, (vs[0], vy_mid), (vs[2], vy_mid),
                                        (255, 255, 0), 1, cv2.LINE_AA, tipLength=0.05)

                    # --- H-bars (left column) ---
                    if entry.get("h_bar_roi"):
                        hb = entry["h_bar_roi"]
                        cv2.rectangle(crop_vis, (hb[0], hb[1]), (hb[2], hb[3]), (0, 0, 255), 1)
                    if entry.get("h_sample_roi"):
                        hs = entry["h_sample_roi"]
                        cv2.rectangle(crop_vis, (hs[0], hs[1]), (hs[2], hs[3]), (0, 255, 255), 2)
                        hx_mid = (hs[0] + hs[2]) // 2
                        cv2.arrowedLine(crop_vis, (hx_mid, hs[1]), (hx_mid, hs[3]),
                                        (0, 255, 255), 1, cv2.LINE_AA, tipLength=0.05)

                    # MTF labels
                    v_mtf = entry.get("v_mtf")
                    v_str = f"V:{v_mtf:.2f}" if v_mtf is not None else "V:--"
                    cv2.putText(crop_vis, v_str, (col_w + 4, y0 + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
                    h_mtf = entry.get("h_mtf")
                    h_str = f"H:{h_mtf:.2f}" if h_mtf is not None else "H:--"
                    cv2.putText(crop_vis, h_str, (4, y0 + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

            crop_path = crops_dir / f"{filename.stem}_{target_name}_crop.png"
            cv2.imwrite(str(crop_path), crop_vis)
            print(f"  -> saved crop: {crop_path.name}")

            for entry in freq_scores:
                entry.pop("roi", None)
                entry.pop("v_bar_roi", None)
                entry.pop("h_bar_roi", None)
                entry.pop("v_sample_roi", None)
                entry.pop("h_sample_roi", None)
                entry.pop("v_crop_roi", None)
                entry.pop("h_crop_roi", None)
                entry["image"] = filename.name
                entry["target"] = target_name
                all_scores.append(entry)

                v_mod = entry.get("v_mtf")
                h_mod = entry.get("h_mtf")
                v_str = f"{v_mod:.4f}" if v_mod is not None else "FAIL"
                h_str = f"{h_mod:.4f}" if h_mod is not None else "FAIL"
                print(f"     freq={entry['frequency']}  V_MTF={v_str}  H_MTF={h_str}")

    else:
        print(f"{filename.name}: no tags")

    out = output_dir / filename.name
    cv2.imwrite(str(out), vis)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "folder",
        help="Folder containing images"
    )

    args = parser.parse_args()

    folder = Path(args.folder)

    output_dir = folder / OUTPUT_FOLDER
    output_dir.mkdir(exist_ok=True)

    crops_dir = folder / CROPS_FOLDER
    crops_dir.mkdir(exist_ok=True)

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    }

    files = sorted(
        f for f in folder.iterdir()
        if f.suffix.lower() in extensions
    )

    print(f"Found {len(files)} images.")

    all_scores = []

    for f in files:
        process_image(f, output_dir, crops_dir, all_scores)

    # Write CSV
    if all_scores:
        csv_path = folder / "usaf_scores.csv"
        fieldnames = ["image", "target", "frequency", "v_mtf", "h_mtf"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_scores)
        print(f"\nScores written to: {csv_path}")

    print()
    print("Done.")
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()