"""Append a publication legend to an existing regional-map PDF.

The map page is translated upwards by 60 points without scaling, cropping,
or changing its drawing content. The colour ramp is copied from the
original shared-legend PDF, preserving its colour mapping.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import shutil
import subprocess

from pypdf import PdfReader, PdfWriter, PageObject, Transformation
from pypdf.generic import RectangleObject
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

MAP_WIDTH_PT = 237.6
MAP_HEIGHT_PT = 187.2
FOOTER_HEIGHT_PT = 60.0


def _page(source):
    """Read one PDF page from a path or PDF bytes."""
    reader = PdfReader(BytesIO(source) if isinstance(source, bytes) else source)
    if len(reader.pages) != 1:
        raise ValueError("A standalone, one-page PDF is required")
    return reader.pages[0]


def _font(font_path):
    path = Path(font_path).resolve(strict=True)
    name = "GeoPRMapLegend_" + sha256(str(path).encode("utf-8")).hexdigest()[:12]
    if name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(name, str(path)))
    return name


def _symbol(drawing, x, y, marker, color):
    drawing.setFillColor(color)
    drawing.setStrokeColorRGB(0, 0, 0)
    drawing.setLineWidth(.6)
    radius = 3.2
    if marker == "s":
        drawing.rect(x - radius, y - radius, 2 * radius, 2 * radius, stroke=1, fill=1)
        return
    vertices = {
        "^": [(0, 1), (-1, -1), (1, -1)],
        "h": [(0, 1), (-.866, .5), (-.866, -.5), (0, -1), (.866, -.5), (.866, .5)],
        "d": [(0, -1.414), (.848, 0), (0, 1.414), (-.848, 0)],
    }
    if marker not in vertices:
        raise ValueError(f"Unsupported publication occurrence marker: {marker}")
    points = vertices[marker]
    path = drawing.beginPath()
    path.moveTo(x + radius * points[0][0], y + radius * points[0][1])
    for px, py in points[1:]:
        path.lineTo(x + radius * px, y + radius * py)
    path.close()
    drawing.drawPath(path, stroke=1, fill=1)


def legend_footer(shared_legend_pdf, items, font_path):
    """Return a 60-point footer for (label, marker, hex-colour) legend items."""
    items = list(items)
    font = _font(font_path)
    original = deepcopy(_page(shared_legend_pdf))
    # These are the axes fractions in regional_maps.legend_figure().
    width, height = float(original.mediabox.width), float(original.mediabox.height)
    x0, y0 = .265 * width, .772 * height
    x1, y1 = (.265 + .47) * width, (.772 + .115) * height
    original.cropbox = RectangleObject((x0, y0, x1, y1))
    footer = PageObject.create_blank_page(width=MAP_WIDTH_PT, height=FOOTER_HEIGHT_PT)
    bar_x, bar_y, bar_width, bar_height = 22.0, 49.0, 193.6, 7.0
    sx, sy = bar_width / (x1 - x0), bar_height / (y1 - y0)
    footer.merge_transformed_page(
        original, Transformation((sx, 0, 0, sy, bar_x - x0 * sx, bar_y - y0 * sy)))

    stream = BytesIO()
    drawing = canvas.Canvas(stream, pagesize=(MAP_WIDTH_PT, FOOTER_HEIGHT_PT))
    drawing.setStrokeColorRGB(0, 0, 0)
    drawing.setFillColorRGB(0, 0, 0)
    drawing.setLineWidth(.45)
    drawing.rect(bar_x, bar_y, bar_width, bar_height, stroke=1, fill=0)
    drawing.setFont(font, 9.5)
    for index in range(6):
        x = bar_x + bar_width * index / 5
        drawing.line(x, bar_y, x, bar_y - 2.4)
        drawing.drawCentredString(x, 37.0, f"{index / 5:.1f}")
    drawing.setFont(font, 10.0)
    drawing.drawCentredString(MAP_WIDTH_PT / 2, 24.0, "Mean element-response percentile")
    display = [(label.replace("-", "\u2013"), marker, color) for label, marker, color in items]
    sizes = [10 + pdfmetrics.stringWidth(label, font, 10) for label, _, _ in display]
    gap = 13.0
    total_width = sum(sizes) + gap * max(0, len(sizes) - 1)
    if total_width > MAP_WIDTH_PT - 4:
        raise ValueError("Occurrence legend is too wide for a publication panel")
    x = (MAP_WIDTH_PT - total_width) / 2
    for (label, marker, color), width in zip(display, sizes):
        _symbol(drawing, x + 3.2, 10, marker, color)
        drawing.setFillColorRGB(0, 0, 0)
        drawing.drawString(x + 10, 6.7, label)
        x += width + gap
    drawing.save()
    footer.merge_page(_page(stream.getvalue()))
    return footer


def append_legend(map_pdf, footer, output):
    """Write a standalone map plus footer, preserving the original map scale."""
    original = _page(map_pdf)
    width, height = float(original.mediabox.width), float(original.mediabox.height)
    if abs(width - MAP_WIDTH_PT) > 1e-5 or abs(height - MAP_HEIGHT_PT) > 1e-5:
        raise ValueError("Regional map must be 237.6 by 187.2 points before legend assembly")
    final = PageObject.create_blank_page(width=width, height=height + FOOTER_HEIGHT_PT)
    final.merge_transformed_page(original, Transformation().translate(0, FOOTER_HEIGHT_PT))
    final.merge_page(deepcopy(footer))
    writer = PdfWriter()
    writer.add_page(final)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        writer.write(stream)


def png_preview(pdf_path, dpi=180):
    """Render the integrated PDF when optional Poppler is installed."""
    executable = shutil.which("pdftoppm")
    if executable is None:
        raise RuntimeError("Integrated --png previews require Poppler's pdftoppm on PATH")
    pdf_path = Path(pdf_path)
    subprocess.run([executable, "-singlefile", "-r", str(dpi), "-png",
                    str(pdf_path), str(pdf_path.with_suffix(""))], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
