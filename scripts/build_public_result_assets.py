"""Build lightweight public assets without model inference or training."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def seed_metrics():
    from refresh_seed_statistics import DATASETS, MODELS, SEEDS

    destination = ROOT / "results/task1_point_forecasting/seed_metrics.json"
    if destination.exists():
        return
    metrics = {}
    for dataset in DATASETS:
        metrics[dataset] = {}
        for model in MODELS:
            metrics[dataset][model] = {}
            for seed in SEEDS:
                path = ROOT / "checkpoints" / model / f"{dataset}_seed{seed}" / "test_metrics.json"
                data = json.loads(path.read_text(encoding="utf-8"))
                metrics[dataset][model][str(seed)] = {key: data[key]
                    for key in ["overall", "horizon_3", "horizon_6", "horizon_12"]}
    write_json(destination, metrics)
    print("Exported 63 compact per-seed metric records")


def chickenpox_drift():
    directory = ROOT / "results/nontraffic_graph_sanity"
    with np.load(directory / "chickenpox_protocol_arrays.npz") as protocol:
        targets = protocol["test_targets"].transpose(0, 2, 1)[..., None]
    values = []
    for path in sorted((directory / "chickenpox_run_artifacts").glob("*.npz")):
        with np.load(path) as arrays:
            covered = np.abs(arrays["test_predictions"] - targets) <= arrays["coordinate_quantiles"]
            values.append([float(covered.mean())] + [float(part.mean())
                          for part in np.array_split(covered, 3, axis=0)])
    result = {"source": "21 public target-disjoint run arrays", "runs": len(values),
              "origin_groups": [33, 33, 33],
              "coverage": dict(zip(["all", "first_third", "middle_third", "final_third"],
                                   np.mean(values, axis=0).tolist()))}
    output = ROOT / "results/release/chickenpox_drift.json"
    write_json(output, result)
    print(result)


def banner():
    from PIL import Image, ImageDraw, ImageFont

    regular = "C:/Windows/Fonts/arial.ttf"
    bold = "C:/Windows/Fonts/arialbd.ttf"
    def font(size, heavy=False):
        try:
            return ImageFont.truetype(bold if heavy else regular, size)
        except OSError:
            return ImageFont.load_default()
    points = [(850, 72), (970, 50), (1070, 132), (1040, 235),
              (900, 252), (790, 181), (930, 152)]
    edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(6,0),(6,2),(6,4),(6,5)]
    frames = []
    colors = ["#1fc4bf", "#efbd41", "#f26d86"]
    for step in range(36):
        image = Image.new("RGB", (1200, 320), "#14252b")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 9, 320), fill=colors[0])
        draw.text((48, 37), "RELIABILITY-FIRST", font=font(17, True), fill=colors[0])
        draw.text((45, 82), "STGNN Benchmark", font=font(48, True), fill="#ffffff")
        draw.text((48, 155), "Traffic forecasting beyond point accuracy", font=font(22), fill="#d8e3e4")
        for number, (a, b) in enumerate(edges):
            draw.line([points[a], points[b]], fill="#496366", width=2)
            t = ((step + number * 3) % 36) / 36
            x = points[a][0] + t * (points[b][0] - points[a][0])
            y = points[a][1] + t * (points[b][1] - points[a][1])
            draw.ellipse((x-3,y-3,x+3,y+3), fill=colors[number % 3])
        for number, (x, y) in enumerate(points):
            draw.ellipse((x-10,y-10,x+10,y+10), fill=colors[number % 3])
        frames.append(image)
    output = ROOT / "figures/readme/benchmark.gif"
    frames[0].save(output, save_all=True, append_images=frames[1:],
                   duration=100, loop=0, optimize=True)
    print(f"Built {output.name}: {output.stat().st_size} bytes")


if __name__ == "__main__":
    seed_metrics()
    chickenpox_drift()
    banner()
