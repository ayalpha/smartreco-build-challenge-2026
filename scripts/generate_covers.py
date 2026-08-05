"""Procedural SVG cover generator.

Why this exists
---------------
31 course covers are first-party AI-generated raster art committed under
``app/static/img/courses``. The curated catalog is larger than that, and no two
courses may share an image — so the remaining covers are generated here as
deterministic SVGs in the *same* visual language: near-black canvas, fine grid,
emerald primary, violet secondary, a single geometric motif with generous
negative space on the left.

Every cover is seeded from its slug, so the output is stable across runs (the
same course always gets the same artwork) while every slug gets a visibly
different composition — a different motif, a different accent mix and different
geometry. SVG keeps them ~2 KB each and infinitely crisp.

Usage::

    python -m scripts.generate_covers          # write any missing covers
    python -m scripts.generate_covers --force  # rewrite all of them
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "app" / "static" / "img" / "courses"

WIDTH, HEIGHT = 800, 447

#: Brand palette — identical to the raster covers and the Tailwind theme.
BG_TOP = "#0A0E16"
BG_BOTTOM = "#05070C"
EMERALD = "#22C98A"
EMERALD_BRIGHT = "#4ADE9A"
IRIS = "#8B7CF6"
IRIS_BRIGHT = "#A99EF8"
STEEL = "#98A4B5"


class Seed:
    """Deterministic pseudo-random stream derived from a slug."""

    def __init__(self, slug: str) -> None:
        self._digest = hashlib.blake2b(slug.encode("utf-8"), digest_size=32).digest()
        self._i = 0

    def next(self) -> float:
        """Return the next value in ``[0, 1)``."""
        byte = self._digest[self._i % len(self._digest)]
        self._i += 1
        return byte / 256.0

    def between(self, low: float, high: float) -> float:
        """Return the next value scaled into ``[low, high)``."""
        return low + (high - low) * self.next()

    def pick(self, options: list) -> object:
        """Choose deterministically from ``options``."""
        return options[int(self.next() * len(options)) % len(options)]


def _header() -> str:
    """Shared defs: background gradient, grid pattern and glow filter."""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}" role="img">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="{BG_TOP}"/>
      <stop offset="100%" stop-color="{BG_BOTTOM}"/>
    </linearGradient>
    <pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse">
      <path d="M24 0H0V24" fill="none" stroke="#FFFFFF" stroke-opacity="0.028" stroke-width="1"/>
    </pattern>
    <radialGradient id="haze" cx="70%" cy="42%" r="52%">
      <stop offset="0%" stop-color="{EMERALD}" stop-opacity="0.13"/>
      <stop offset="100%" stop-color="{EMERALD}" stop-opacity="0"/>
    </radialGradient>
    <filter id="glow" x="-60%" y="-60%" width="220%" height="220%">
      <feGaussianBlur stdDeviation="7" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <rect width="{WIDTH}" height="{HEIGHT}" fill="url(#bg)"/>
  <rect width="{WIDTH}" height="{HEIGHT}" fill="url(#grid)"/>
  <rect width="{WIDTH}" height="{HEIGHT}" fill="url(#haze)"/>
"""


def motif_network(r: Seed) -> str:
    """Connected nodes — agents, graphs, coordination."""
    cx, cy = 530, HEIGHT / 2
    count = int(r.between(6, 9))
    pts = []
    for i in range(count):
        angle = (i / count) * math.tau + r.between(-0.2, 0.2)
        radius = r.between(70, 145)
        pts.append((cx + math.cos(angle) * radius * 1.25, cy + math.sin(angle) * radius * 0.85))

    edges = "".join(
        f'<line x1="{pts[i][0]:.1f}" y1="{pts[i][1]:.1f}" x2="{pts[j][0]:.1f}" y2="{pts[j][1]:.1f}" '
        f'stroke="{IRIS}" stroke-opacity="0.42" stroke-width="1.3"/>'
        for i in range(count) for j in range(i + 1, count) if (i + j) % 3 == 0
    )
    hub = f'<circle cx="{cx}" cy="{cy}" r="13" fill="{EMERALD}" filter="url(#glow)"/>'
    spokes = "".join(
        f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="{EMERALD}" '
        f'stroke-opacity="0.32" stroke-width="1.2"/>' for x, y in pts
    )
    nodes = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r.between(6, 10):.1f}" '
        f'fill="{EMERALD_BRIGHT if i % 3 else IRIS_BRIGHT}" filter="url(#glow)"/>'
        for i, (x, y) in enumerate(pts)
    )
    return edges + spokes + hub + nodes


def motif_layers(r: Seed) -> str:
    """Stacked strata — data layers, lakehouse, model layers."""
    out = []
    n = int(r.between(5, 8))
    for i in range(n):
        y = 108 + i * (230 / n)
        w = r.between(210, 330)
        x = 470 + r.between(-30, 30)
        active = i == int(r.between(0, n))
        colour = EMERALD if active else STEEL
        opacity = 0.9 if active else r.between(0.16, 0.3)
        out.append(
            f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="16" rx="4" '
            f'fill="{colour}" fill-opacity="{opacity:.2f}"'
            + (' filter="url(#glow)"' if active else "")
            + "/>"
        )
        if active:
            out.append(
                f'<line x1="{x - 34:.0f}" y1="{y + 8:.0f}" x2="{x - 6:.0f}" y2="{y + 8:.0f}" '
                f'stroke="{IRIS_BRIGHT}" stroke-width="2"/>'
            )
    return "".join(out)


def motif_flow(r: Seed) -> str:
    """Parallel lanes through a junction — pipelines, streaming, requests."""
    out = []
    lanes = int(r.between(4, 6))
    for i in range(lanes):
        y = 120 + i * (210 / (lanes - 1))
        out.append(
            f'<path d="M330 {y:.0f} H520 Q560 {y:.0f} 575 {HEIGHT/2:.0f} Q590 {y:.0f} 640 {y:.0f} H760" '
            f'fill="none" stroke="{IRIS if i % 2 else EMERALD}" stroke-opacity="0.5" stroke-width="1.6"/>'
        )
        out.append(
            f'<circle cx="{r.between(660, 750):.0f}" cy="{y:.0f}" r="4.5" fill="{EMERALD_BRIGHT}"/>'
        )
    out.append(
        f'<circle cx="575" cy="{HEIGHT/2:.0f}" r="17" fill="none" stroke="{EMERALD}" '
        f'stroke-width="2" filter="url(#glow)"/>'
    )
    return "".join(out)


def motif_orbit(r: Seed) -> str:
    """Concentric rings with travellers — loops, cycles, orchestration."""
    cx, cy = 545, HEIGHT / 2
    out = []
    for i in range(3):
        rx = 66 + i * 46
        out.append(
            f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{rx * 0.72:.0f}" fill="none" '
            f'stroke="{IRIS if i % 2 else EMERALD}" stroke-opacity="{0.5 - i * 0.11:.2f}" stroke-width="1.5"/>'
        )
        angle = r.between(0, math.tau)
        out.append(
            f'<circle cx="{cx + math.cos(angle) * rx:.1f}" cy="{cy + math.sin(angle) * rx * 0.72:.1f}" '
            f'r="{7 - i:.0f}" fill="{EMERALD_BRIGHT}" filter="url(#glow)"/>'
        )
    out.append(f'<circle cx="{cx}" cy="{cy}" r="11" fill="{IRIS_BRIGHT}" filter="url(#glow)"/>')
    return "".join(out)


def motif_field(r: Seed) -> str:
    """A field of cells with a highlighted region — tabular, grids, matrices."""
    out = []
    cols, rows = 9, 6
    hot_c, hot_r = int(r.between(3, 7)), int(r.between(1, 4))
    for c in range(cols):
        for row in range(rows):
            x = 420 + c * 38
            y = 100 + row * 42
            near = abs(c - hot_c) <= 1 and abs(row - hot_r) <= 1
            if near:
                fill, op = EMERALD, r.between(0.55, 0.95)
            else:
                fill, op = STEEL, r.between(0.05, 0.16)
            out.append(
                f'<rect x="{x}" y="{y}" width="26" height="28" rx="3" fill="{fill}" '
                f'fill-opacity="{op:.2f}"/>'
            )
    out.append(
        f'<rect x="{420 + (hot_c - 1) * 38 - 6}" y="{100 + (hot_r - 1) * 42 - 6}" '
        f'width="{3 * 38 + 4}" height="{3 * 42 + 4}" rx="8" fill="none" stroke="{IRIS_BRIGHT}" '
        f'stroke-width="1.6" stroke-opacity="0.75"/>'
    )
    return "".join(out)


def motif_wave(r: Seed) -> str:
    """Overlapping waveforms — distributions, signals, performance."""
    out = []
    for i in range(3):
        amp = r.between(34, 68)
        phase = r.between(0, math.tau)
        base = HEIGHT / 2 + (i - 1) * 34
        pts = " ".join(
            f"{x},{base + math.sin((x / 92) + phase) * amp * (1 - abs(x - 570) / 700):.1f}"
            for x in range(330, 790, 14)
        )
        out.append(
            f'<polyline points="{pts}" fill="none" stroke="{EMERALD if i == 1 else IRIS}" '
            f'stroke-opacity="{0.85 if i == 1 else 0.4:.2f}" stroke-width="{2.4 if i == 1 else 1.5}" '
            f'stroke-linecap="round"' + (' filter="url(#glow)"' if i == 1 else "") + "/>"
        )
    return "".join(out)


def motif_shield(r: Seed) -> str:
    """A guarded boundary — security, contracts, reliability."""
    cx, cy = 555, HEIGHT / 2
    out = [
        f'<path d="M{cx} {cy - 96} L{cx + 78} {cy - 58} V{cy + 18} Q{cx + 78} {cy + 74} {cx} {cy + 104} '
        f'Q{cx - 78} {cy + 74} {cx - 78} {cy + 18} V{cy - 58} Z" fill="none" stroke="{EMERALD}" '
        f'stroke-width="2.4" filter="url(#glow)"/>'
    ]
    for i in range(4):
        y = cy - 52 + i * 34
        out.append(
            f'<line x1="{cx - 46}" y1="{y}" x2="{cx + 46}" y2="{y}" stroke="{IRIS}" '
            f'stroke-opacity="{r.between(0.25, 0.6):.2f}" stroke-width="1.4"/>'
        )
    out.append(f'<circle cx="{cx}" cy="{cy}" r="9" fill="{EMERALD_BRIGHT}" filter="url(#glow)"/>')
    return "".join(out)


def motif_ascend(r: Seed) -> str:
    """Rising platforms — growth, career, seniority."""
    out = []
    steps = 5
    for i in range(steps):
        x = 400 + i * 74
        h = 30 + i * 30
        y = HEIGHT - 96 - h
        out.append(
            f'<rect x="{x}" y="{y:.0f}" width="52" height="{h:.0f}" rx="6" fill="{EMERALD}" '
            f'fill-opacity="{0.18 + i * 0.17:.2f}"'
            + (' filter="url(#glow)"' if i == steps - 1 else "")
            + "/>"
        )
        if i:
            out.append(
                f'<path d="M{x - 22} {y + 34:.0f} Q{x - 8} {y:.0f} {x + 4} {y:.0f}" fill="none" '
                f'stroke="{IRIS_BRIGHT}" stroke-opacity="0.5" stroke-width="1.5"/>'
            )
    return "".join(out)


def motif_blocks(r: Seed) -> str:
    """Modular units assembling — containers, infrastructure, platforms."""
    out = []
    for i in range(7):
        col, row = i % 3, i // 3
        x = 460 + col * 92 + r.between(-6, 6)
        y = 112 + row * 92
        lifted = i == int(r.between(0, 7))
        out.append(
            f'<rect x="{x:.0f}" y="{y - (14 if lifted else 0):.0f}" width="74" height="70" rx="8" '
            f'fill="none" stroke="{EMERALD if lifted else STEEL}" '
            f'stroke-opacity="{0.95 if lifted else 0.26:.2f}" stroke-width="{2 if lifted else 1.3}"'
            + (' filter="url(#glow)"' if lifted else "")
            + "/>"
        )
        out.append(
            f'<line x1="{x + 12:.0f}" y1="{y + 18 - (14 if lifted else 0):.0f}" '
            f'x2="{x + 62:.0f}" y2="{y + 18 - (14 if lifted else 0):.0f}" '
            f'stroke="{IRIS}" stroke-opacity="0.35" stroke-width="1.2"/>'
        )
    return "".join(out)


MOTIFS = [
    motif_network, motif_layers, motif_flow, motif_orbit,
    motif_field, motif_wave, motif_shield, motif_ascend, motif_blocks,
]

#: Slug → motif index. Chosen by topic so the artwork means something, rather
#: than being purely random.
MOTIF_FOR_SLUG: dict[str, int] = {
    "rag-production": 2,
    "gradient-boosting": 8,
    "distributed-training": 0,
    "airflow": 2,
    "duckdb": 4,
    "lakehouse": 1,
    "fastapi": 2,
    "testing-python": 6,
    "rust": 8,
    "typescript": 4,
    "react": 0,
    "frontend-performance": 5,
    "accessibility": 6,
    "cicd": 2,
    "incident-response": 5,
    "finops": 7,
    "event-driven": 0,
    "tech-lead": 7,
    "system-design": 8,
}


def render(slug: str) -> str:
    """Render one deterministic SVG cover for ``slug``."""
    r = Seed(slug)
    motif = MOTIFS[MOTIF_FOR_SLUG.get(slug, int(r.between(0, len(MOTIFS))))]
    return _header() + motif(r) + "\n</svg>\n"


def main(argv: list[str] | None = None) -> int:
    """Write every cover in :data:`MOTIF_FOR_SLUG`."""
    parser = argparse.ArgumentParser(description="Generate procedural SVG course covers.")
    parser.add_argument("--force", action="store_true", help="rewrite existing files")
    args = parser.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for slug in MOTIF_FOR_SLUG:
        path = OUT_DIR / f"{slug}.svg"
        if path.exists() and not args.force:
            continue
        path.write_text(render(slug), encoding="utf-8")
        written += 1
        print(f"  {path.name}  {path.stat().st_size // 1024 or 1}KB")

    print(f"generated {written} cover(s) in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
