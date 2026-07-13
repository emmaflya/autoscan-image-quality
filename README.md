# autoscan-image-quality

Tools to evaluate image quality for Autoscan by evaluating resolved GSD on sampled points.

## Scripts

### 1. Generate targets

Creates printable A4 sheets (PNG + PDF) with USAF resolution targets surrounded by AprilTags.
Targets at 0.3mm lines = 0.6mmLP, 0.5mm lines= 1.0LP and 1.0mm lines = 2.0LP. 

```
python generate_april_tag.py
```

Output goes to `targets/`.

### 2. Detect tags & score sharpness

Scans a folder of photos for AprilTags, crops each USAF target, and computes MTF (sharpness) scores per frequency. Writes results to `usaf_scores.csv`. 

```
python detect_tags.py <image_folder>
```

### 3. Plot results

Generates histogram and distribution plots from the scores CSV.

```
python plot_scores.py <path/to/usaf_scores.csv>
```

Optionally specify an output directory: `python plot_scores.py scores.csv -o plots/`

## Install

```
uv sync
```

Requires Python >= 3.10.