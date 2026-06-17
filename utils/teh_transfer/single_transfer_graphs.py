"""Heatmaps for single-source transfer summary matrices."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402

from utils.teh_transfer.transfer_jobs import (
    SINGLE_TRANSFER_1ST_ITER_SUBDIR,
    SINGLE_TRANSFER_MATRIX_IMPROVE_PATH,
    SINGLE_TRANSFER_MATRIX_TEST_PATH,
)


def _load_matrix_csv(path: Path) -> Tuple[List[str], List[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or len(header) < 2:
            raise ValueError(f"Invalid transfer matrix CSV: {path}")
        col_keys = [str(k) for k in header[1:]]
        row_keys: List[str] = []
        rows: List[List[float]] = []
        for row in reader:
            if not row:
                continue
            row_keys.append(str(row[0]))
            values: List[float] = []
            for cell in row[1 : 1 + len(col_keys)]:
                if cell in ("", None):
                    values.append(float("nan"))
                else:
                    values.append(float(cell))
            rows.append(values)
    if not row_keys:
        raise ValueError(f"Empty transfer matrix CSV: {path}")
    return row_keys, col_keys, np.asarray(rows, dtype=float)


def _mask_self_transfer(
    row_keys: Sequence[str],
    col_keys: Sequence[str],
    data: np.ndarray,
) -> np.ma.MaskedArray:
    masked = np.ma.array(data, mask=np.zeros_like(data, dtype=bool))
    for row_idx, target in enumerate(row_keys):
        for col_idx, source in enumerate(col_keys):
            if target == source:
                masked.mask[row_idx, col_idx] = True
    return masked


def _heatmap_norm(
    data: np.ma.MaskedArray,
    *,
    improve_matrix: bool,
) -> Tuple[Normalize | TwoSlopeNorm, float]:
    valid = data.compressed()
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0), 0.0
    vmin = float(np.min(valid))
    vmax = float(np.max(valid))
    if improve_matrix:
        vcenter = 0.0
        if vmin >= vcenter:
            vmin = vcenter - max(abs(vmax - vcenter), 1e-6)
        if vmax <= vcenter:
            vmax = vcenter + max(abs(vcenter - vmin), 1e-6)
        return TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax), vcenter
    green_at = 0.0
    if vmin >= green_at:
        vmin = green_at - max(abs(vmax - green_at), 1e-6)
    return Normalize(vmin=vmin, vmax=green_at), (vmin + green_at) / 2.0


LEGACY_SINGLE_TRANSFER_MATRIX_TEST_PATH = "test_loglik.csv"
LEGACY_SINGLE_TRANSFER_MATRIX_IMPROVE_PATH = "improve_test_loglik.csv"


def _matrix_csv_has_values(path: Path) -> bool:
    if not path.is_file():
        return False
    _, _, data = _load_matrix_csv(path)
    return bool(np.isfinite(data).any())


def _resolve_matrix_csv(csv_dir: Path, csv_name: str) -> Path | None:
    primary = csv_dir / csv_name
    if _matrix_csv_has_values(primary):
        return primary
    legacy_name = {
        SINGLE_TRANSFER_MATRIX_TEST_PATH: LEGACY_SINGLE_TRANSFER_MATRIX_TEST_PATH,
        SINGLE_TRANSFER_MATRIX_IMPROVE_PATH: LEGACY_SINGLE_TRANSFER_MATRIX_IMPROVE_PATH,
    }.get(csv_name)
    if legacy_name is None:
        return primary if primary.is_file() else None
    legacy = csv_dir / legacy_name
    if _matrix_csv_has_values(legacy):
        return legacy
    return primary if primary.is_file() else None


def _short_label(name: str) -> str:
    return name.replace("mixed_gambles", "mixed").split("20", 1)[0]


def plot_single_transfer_matrix_heatmap(
    matrix_csv_path: Path,
    output_path: Path,
    *,
    title: str,
) -> Path:
    row_keys, col_keys, data = _load_matrix_csv(matrix_csv_path)
    masked = _mask_self_transfer(row_keys, col_keys, data)
    improve_matrix = "improve" in matrix_csv_path.name
    norm, text_threshold = _heatmap_norm(masked, improve_matrix=improve_matrix)

    n_rows, n_cols = masked.shape
    fig_w = max(8.0, 0.55 * n_cols + 2.5)
    fig_h = max(6.5, 0.55 * n_rows + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)

    im = ax.imshow(masked, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels([_short_label(k) for k in col_keys], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([_short_label(k) for k in row_keys], fontsize=8)
    ax.set_xlabel("Source dataset")
    ax.set_ylabel("Target dataset")
    ax.set_title(title, fontsize=11, pad=12)

    text_fs = 7 if max(n_rows, n_cols) <= 10 else 6
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            if masked.mask[row_idx, col_idx]:
                continue
            value = float(masked[row_idx, col_idx])
            text_color = "black" if value > text_threshold else "white"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=text_fs,
                color=text_color,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Improve vs global (test)" if improve_matrix else "Test loglik")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_single_transfer_matrix_heatmaps(
    run_dir: Path,
) -> List[Path]:
    """Write heatmaps for best and iteration-1 single-source transfer matrices."""
    summary_csv_root = run_dir / "summary_csv" / "single_transfer"
    summary_graph_root = run_dir / "summary_graph" / "single_transfer"
    written: List[Path] = []
    bundles = [
        (
            summary_csv_root,
            summary_graph_root,
            "Single-source transfer",
        ),
        (
            summary_csv_root / SINGLE_TRANSFER_1ST_ITER_SUBDIR,
            summary_graph_root / SINGLE_TRANSFER_1ST_ITER_SUBDIR,
            "Single-source transfer (iteration 1)",
        ),
    ]
    matrix_specs = [
        (
            SINGLE_TRANSFER_MATRIX_TEST_PATH,
            "matrix_test_loglik.png",
            "{prefix} test loglik",
        ),
        (
            SINGLE_TRANSFER_MATRIX_IMPROVE_PATH,
            "matrix_improve_test_loglik.png",
            "{prefix} improve vs global (test)",
        ),
    ]
    for csv_dir, graph_dir, prefix in bundles:
        for csv_name, png_name, title_tpl in matrix_specs:
            csv_path = _resolve_matrix_csv(csv_dir, csv_name)
            if csv_path is None:
                continue
            out_path = graph_dir / png_name
            plot_single_transfer_matrix_heatmap(
                csv_path,
                out_path,
                title=title_tpl.format(prefix=prefix),
            )
            written.append(out_path)
    return written
