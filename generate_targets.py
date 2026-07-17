"""
Generate USAF resolution target sheets for use with detect_tags.py.

Produces 12 A4 sheets, each with 6 USAF targets in a 2x3 grid.
Each target is bounded by 4 unique AprilTags (36h11 family).
Tag IDs are sequential across all pages — no duplicates.

Convention matches detect_tags.py:
  - Tag order per target: [TL, TR, BR, BL] (clockwise from top-left)
  - USAF content in the center, between the 4 tags
  - Within each frequency element: H-lines on the left half, V-lines on the right half
  - Frequencies stacked top-to-bottom, largest to smallest

Output: usaf_targets_page_XX.png and usaf_targets.pdf (multi-page) at the specified DPI.
Also prints the TARGETS_ID dict to paste into detect_tags.py.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import os

DPI = 300
A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297
MARGIN_MM = 10       # page margin
GAP_MM = 8           # gap between the 4 target quadrants

APRILTAG_DICT = cv2.aruco.DICT_APRILTAG_36h11

# Tag size as fraction of cell size (each target is a 3x3 grid of cells;
# tags occupy the 4 corner cells, content is in the center cell area)
TAG_SCALE = 0.70

# Line-pair frequencies in mm (period = one dark bar + one light gap).
# Bar width = period / 2.
# Ordered largest to smallest — drawn top to bottom in the target.
FREQUENCIES_MM = [2.0, 1.0, 0.6]

# Internal margin inside each frequency element row (mm).
# Keeps bars away from the row edges for cleaner profiles.
ELEMENT_MARGIN_MM = 1.0

# Number of pages to generate
NUM_PAGES = 12

# 6 targets per page (2x3 grid), 4 tags per target
TARGETS_PER_PAGE = 6
TAGS_PER_TARGET = 4

# 2x3 layout: (column, row) on the page
TARGET_POSITIONS = [
    (0, 0),  # usaf_0: top-left
    (1, 0),  # usaf_1: top-right
    (0, 1),  # usaf_2: middle-left
    (1, 1),  # usaf_3: middle-right
    (0, 2),  # usaf_4: bottom-left
    (1, 2),  # usaf_5: bottom-right
]

N_COLS = 2
N_ROWS = 3


def mm_to_px(mm):
    return int(round(mm * DPI / 25.4))


def px_to_mm(px):
    return px * 25.4 / DPI


def main():
    page_w = mm_to_px(A4_WIDTH_MM)
    page_h = mm_to_px(A4_HEIGHT_MM)
    margin = mm_to_px(MARGIN_MM)
    gap = mm_to_px(GAP_MM)
    elem_margin = mm_to_px(ELEMENT_MARGIN_MM)

    caption_font_size = mm_to_px(6)
    caption_h = mm_to_px(10)

    usable_w = page_w - 2 * margin
    usable_h = page_h - 2 * margin - caption_h

    avail_w = (usable_w - (N_COLS + 1) * gap) // N_COLS
    avail_h = (usable_h - (N_ROWS + 1) * gap) // N_ROWS

    grid_size = min(avail_w, avail_h)
    cell_px = grid_size // 3
    grid_size = cell_px * 3

    tag_size_px = int(cell_px * TAG_SCALE)
    if tag_size_px % 2 == 0:
        tag_size_px -= 1

    content_offset = cell_px // 2 + tag_size_px // 2
    content_end = 2 * cell_px + cell_px // 2 - tag_size_px // 2
    content_size = content_end - content_offset

    total_grids_w = N_COLS * grid_size
    remaining_w = usable_w - total_grids_w
    gap_w = remaining_w // (N_COLS + 1)

    total_grids_h = N_ROWS * grid_size
    remaining_h = usable_h - total_grids_h
    gap_h = remaining_h // (N_ROWS + 1)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", caption_font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_DICT)
    tag_grid = [(0, 0), (2, 0), (2, 2), (0, 2)]

    row_h = content_size // len(FREQUENCIES_MM)

    output_folder = Path("./targets")
    output_folder.mkdir(parents=True, exist_ok=True)
    
    all_targets = {}  # "usaf_XX": [id0, id1, id2, id3]
    page_images = []
    next_tag_id = 0

    for page_num in range(NUM_PAGES):
        img = Image.new("L", (page_w, page_h), 255)
        draw = ImageDraw.Draw(img)

        caption_text = f"THIS SIDE UP (Page {page_num + 1})"
        bbox = draw.textbbox((0, 0), caption_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = (page_w - text_w) // 2
        text_y = margin + (caption_h - text_h) // 2
        draw.text((text_x, text_y), caption_text, fill=0, font=font)

        # Arrows
        arrow_x = text_x - mm_to_px(8)
        arrow_top = text_y
        arrow_bottom = text_y + text_h
        arrow_mid = (arrow_top + arrow_bottom) // 2
        draw.line([(arrow_x, arrow_bottom), (arrow_x, arrow_top)], fill=0, width=3)
        draw.line([(arrow_x - mm_to_px(2), arrow_mid), (arrow_x, arrow_top)], fill=0, width=3)
        draw.line([(arrow_x + mm_to_px(2), arrow_mid), (arrow_x, arrow_top)], fill=0, width=3)

        arrow_x_r = text_x + text_w + mm_to_px(8)
        draw.line([(arrow_x_r, arrow_bottom), (arrow_x_r, arrow_top)], fill=0, width=3)
        draw.line([(arrow_x_r - mm_to_px(2), arrow_mid), (arrow_x_r, arrow_top)], fill=0, width=3)
        draw.line([(arrow_x_r + mm_to_px(2), arrow_mid), (arrow_x_r, arrow_top)], fill=0, width=3)

        area_top = margin + caption_h

        for target_idx, (col, row) in enumerate(TARGET_POSITIONS):
            global_target_idx = page_num * TARGETS_PER_PAGE + target_idx
            target_name = f"usaf_{global_target_idx}"
            tag_ids = list(range(next_tag_id, next_tag_id + TAGS_PER_TARGET))
            next_tag_id += TAGS_PER_TARGET
            all_targets[target_name] = tag_ids

            gx0 = margin + gap_w + col * (grid_size + gap_w)
            gy0 = area_top + gap_h + row * (grid_size + gap_h)

            # Place 4 AprilTags
            for tag_id, (gcol, grow) in zip(tag_ids, tag_grid):
                tag_marker = np.zeros((tag_size_px, tag_size_px), dtype=np.uint8)
                cv2.aruco.generateImageMarker(dictionary, tag_id, tag_size_px, tag_marker, 1)
                tag_img = Image.fromarray(tag_marker)
                tx = gx0 + gcol * cell_px + (cell_px - tag_size_px) // 2
                ty = gy0 + grow * cell_px + (cell_px - tag_size_px) // 2
                img.paste(tag_img, (tx, ty))

            # Draw USAF frequency elements
            cx0 = gx0 + content_offset
            cy0 = gy0 + content_offset

            for freq_idx, freq_mm in enumerate(FREQUENCIES_MM):
                ry0 = cy0 + freq_idx * row_h + elem_margin
                ry1 = cy0 + (freq_idx + 1) * row_h - elem_margin
                rx0 = cx0 + elem_margin
                rx1 = cx0 + content_size - elem_margin
                row_mid_x = (rx0 + rx1) // 2

                bar_w_px = mm_to_px(freq_mm / 2.0)
                gap_px_bar = bar_w_px
                group_extent = 3 * bar_w_px + 2 * gap_px_bar

                # V-lines (right half)
                v_region_cx = (row_mid_x + rx1) // 2
                v_start_x = v_region_cx - group_extent // 2
                for b in range(3):
                    bx = v_start_x + b * (bar_w_px + gap_px_bar)
                    draw.rectangle([bx, ry0 + 2, bx + bar_w_px - 1, ry1 - 2], fill=0)

                # H-lines (left half)
                h_region_cy = (ry0 + ry1) // 2
                h_start_y = h_region_cy - group_extent // 2
                for b in range(3):
                    by = h_start_y + b * (bar_w_px + gap_px_bar)
                    draw.rectangle([rx0 + 2, by, row_mid_x - 2, by + bar_w_px - 1], fill=0)
        
        png_path = f"usaf_targets_page_{page_num + 1:02d}.png"
        img.save(output_folder / png_path, dpi=(DPI, DPI))
        print(f"Saved: {png_path}")
        page_images.append(img)

    pdf_path = output_folder / "usaf_targets.pdf"
    page_images[0].save(
        pdf_path,
        dpi=(DPI, DPI),
        save_all=True,
        append_images=page_images[1:],
    )
    print(f"Saved: {pdf_path} ({NUM_PAGES} pages)")

    print()
    print("=" * 60)
    print("Paste this into detect_tags.py:")
    print("=" * 60)
    print()
    print("TARGETS_ID = {")
    for name, ids in all_targets.items():
        print(f'    "{name}" : {ids},')
    print("}")

    print()
    print(f"Total: {NUM_PAGES} pages, {len(all_targets)} targets, {next_tag_id} unique tag IDs")
    print(f"DPI: {DPI}")
    print(f"Cell: {px_to_mm(cell_px):.1f}mm ({cell_px}px)")
    print(f"Tag side: {px_to_mm(tag_size_px):.1f}mm ({tag_size_px}px)")
    print(f"Content area: {px_to_mm(content_size):.1f}mm ({content_size}px)")
    print(f"Bar widths: {', '.join(f'{f/2:.1f}mm' for f in FREQUENCIES_MM)}")


if __name__ == "__main__":
    main()
