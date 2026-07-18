"""Tiny dependency-free SVG plots for the command-line experiment."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np


def save_line_plot(values: np.ndarray, output_path: str | Path, *, title: str) -> None:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    if array.ndim != 1 or not finite.any():
        raise ValueError("values must be a vector containing at least one finite number")
    width, height, padding = 760, 420, 55
    x_values = np.arange(len(array), dtype=np.float64)
    valid_x = x_values[finite]
    valid_y = array[finite]
    x_span = max(1.0, float(valid_x.max() - valid_x.min()))
    y_min = float(valid_y.min())
    y_span = max(1e-12, float(valid_y.max() - y_min))
    points = []
    for x_value, y_value in zip(valid_x, valid_y, strict=True):
        x_pixel = padding + (x_value - valid_x.min()) / x_span * (width - 2 * padding)
        y_pixel = height - padding - (y_value - y_min) / y_span * (height - 2 * padding)
        points.append(f"{x_pixel:.2f},{y_pixel:.2f}")
    markup = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">{html.escape(title)}</text>
<line x1="{padding}" y1="{height-padding}" x2="{width-padding}" y2="{height-padding}" stroke="#222"/>
<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height-padding}" stroke="#222"/>
<polyline points="{' '.join(points)}" fill="none" stroke="#2563eb" stroke-width="3"/>
<text x="{width / 2}" y="{height-12}" text-anchor="middle" font-family="sans-serif" font-size="14">Training step</text>
<text x="18" y="{height / 2}" text-anchor="middle" transform="rotate(-90 18 {height / 2})" font-family="sans-serif" font-size="14">Loss</text>
<text x="{padding}" y="{padding-8}" font-family="sans-serif" font-size="12">max={valid_y.max():.5g}</text>
<text x="{padding}" y="{height-padding+20}" font-family="sans-serif" font-size="12">min={valid_y.min():.5g}</text>
</svg>
"""
    Path(output_path).write_text(markup, encoding="utf-8")

