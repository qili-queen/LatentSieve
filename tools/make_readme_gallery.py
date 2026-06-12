"""Build compact README gallery images from cached evaluation outputs.

This script is intended for preparing the GitHub release. It does not run
generation or attacks; it only collects existing images into contact sheets.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


RELEASE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = RELEASE_ROOT.parent
SOURCE_ROOT = WORKSPACE_ROOT / "MarkDiffusion" / "qi_eval_n500_sdp"
OUT_DIR = RELEASE_ROOT / "assets" / "gallery"
GALLERY_SOURCE = OUT_DIR / "source"
PAIRED_ORIGINAL_ROOT = OUT_DIR / "source" / "original"

SCHEMES = ["MIRAGE", "GS", "PRC", "T2SMark", "TR"]
COLUMNS = ["Original"] + SCHEMES

# One seed/message per prompt for a compact README view.
# Indices 0, 5, 10, 15, 20 correspond to prompt_idx 0..4 on the SDP slice.
SAMPLE_IDXS = [0, 5, 10, 15, 20]

CONDITIONS = {
    "clean_watermarks": {
        "title": "Clean watermarked outputs",
        "suffix": "wm",
    },
    "ctrlgen_attacked_watermarks": {
        "title": "After Ctrlgen removal attack",
        "suffix": "attacked_Ctrlgen",
    },
    "unmarker_s2_attacked_watermarks": {
        "title": "After UnMarker-S2 removal attack",
        "suffix": "attacked_s2",
    },
    "shift_attacked_watermarks": {
        "title": "After SHIFT trajectory deflection at lambda=0.7",
        "suffix": "source:shift_lambda_0.7",
    },
}

CELL = 150
HEADER_H = 30
PADDING = 8
BG = (255, 255, 255)
GRID = (220, 220, 220)
TEXT = (30, 30, 30)


def font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def image_path(column: str, idx: int, suffix: str) -> Path:
    name = f"img_{idx:04d}.png"
    if column == "Original":
        return PAIRED_ORIGINAL_ROOT / name
    if suffix.startswith("source:"):
        return GALLERY_SOURCE / suffix.split(":", 1)[1] / column / name
    return SOURCE_ROOT / f"{column}_{suffix}" / name


def load_tile(path: Path) -> Image.Image:
    if not path.exists():
        if PAIRED_ORIGINAL_ROOT in path.parents:
            raise FileNotFoundError(
                f"{path} is missing. Generate paired unwatermarked originals "
                f"with the same SDP prompt/seed indices under "
                f"{PAIRED_ORIGINAL_ROOT} before rebuilding galleries."
            )
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB").resize((CELL, CELL), Image.LANCZOS)


def draw_centered(draw: ImageDraw.ImageDraw, box, text: str, fnt, fill=TEXT) -> None:
    x0, y0, x1, y1 = box
    bbox = draw.textbbox((0, 0), text, font=fnt)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2), text, font=fnt, fill=fill)


def build_grid(name: str, suffix: str) -> Path:
    cols = len(COLUMNS)
    rows = len(SAMPLE_IDXS)
    width = cols * CELL + (cols + 1) * PADDING
    height = HEADER_H + rows * CELL + (rows + 1) * PADDING
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    header_font = font(15)
    row_font = font(13)

    for ci, column in enumerate(COLUMNS):
        x = PADDING + ci * (CELL + PADDING)
        draw_centered(draw, (x, 0, x + CELL, HEADER_H), column, header_font)

    for ri, idx in enumerate(SAMPLE_IDXS):
        y = HEADER_H + PADDING + ri * (CELL + PADDING)
        for ci, column in enumerate(COLUMNS):
            x = PADDING + ci * (CELL + PADDING)
            tile = load_tile(image_path(column, idx, suffix))
            canvas.paste(tile, (x, y))
            draw.rectangle((x, y, x + CELL - 1, y + CELL - 1), outline=GRID)
            if ci == 0:
                draw.rectangle((x + 4, y + 4, x + 30, y + 22), fill=(255, 255, 255), outline=GRID)
                draw.text((x + 8, y + 6), f"P{ri + 1}", font=row_font, fill=TEXT)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}.png"
    canvas.save(out, optimize=True)
    return out


def main() -> None:
    for name, cfg in CONDITIONS.items():
        out = build_grid(name, cfg["suffix"])
        print(out)


if __name__ == "__main__":
    main()
