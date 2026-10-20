"""Vector-only assembly of the manuscript's geological and fold panels.

Input PDFs are local artifacts supplied by the user, never bundled results.
The fixed component sizes are checked before applying publication coordinates.
"""
from copy import deepcopy
from io import BytesIO
from pathlib import Path

from pypdf import PageObject, PdfReader, PdfWriter
from pypdf.generic import (
    DecodedStreamObject, DictionaryObject, NameObject, NumberObject, RectangleObject,
)

GEOLOGY_SIZE = (547.2, 475.2)
FOLD_SIZE = (159.06873072916672, 142.83846247806228)
FOLD_SPINES = (36.42, 156.56873072916672)


def one_page(source, expected_size=None):
    reader = PdfReader(source)
    if len(reader.pages) != 1:
        raise ValueError("Each component must be a single-page PDF")
    page = reader.pages[0]
    if page.get("/Rotate", 0) or list(page.mediabox[:2]) != [0, 0]:
        raise ValueError("Components must have an unrotated, zero-origin page")
    if expected_size is not None:
        actual = (float(page.mediabox.width), float(page.mediabox.height))
        if any(abs(a - b) > .02 for a, b in zip(actual, expected_size)):
            raise ValueError(f"Component size {actual} differs from required {expected_size}; do not rescale blindly")
    return page


def compose(width, height, placements, output):
    """Reuse the complete source page streams/resources as clipped PDF Forms."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=width, height=height)
    forms, commands = DictionaryObject(), []
    for index, (source, scale, tx, ty) in enumerate(placements):
        form = DecodedStreamObject()
        form.set_data(source.get_contents().get_data())
        form.update({
            NameObject("/Type"): NameObject("/XObject"),
            NameObject("/Subtype"): NameObject("/Form"),
            NameObject("/FormType"): NumberObject(1),
            NameObject("/BBox"): source.cropbox,
            NameObject("/Resources"): source["/Resources"],
        })
        key = NameObject(f"/Panel{index}")
        forms[key] = writer._add_object(form.flate_encode().clone(writer))
        commands.append(f"q {scale:.9f} 0 0 {scale:.9f} {tx:.9f} {ty:.9f} cm {key} Do Q")
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/XObject"): forms})
    stream = DecodedStreamObject()
    stream.set_data(("\n".join(commands)).encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream.flate_encode())
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        writer.write(handle)


def assemble_geology(map_pdf, locator_pdf, occurrence_legend_pdf, output):
    """Combine validated original-size components into one Fig02.pdf.

    The main map must not already contain the locator or occurrence legend.
    No geometry, colors, symbol positions, text or font programs are redrawn.
    """
    main = one_page(map_pdf, GEOLOGY_SIZE)
    locator = one_page(locator_pdf)
    legend = one_page(occurrence_legend_pdf)
    width, height = GEOLOGY_SIZE
    if abs(float(legend.mediabox.width) - width) > .02:
        raise ValueError("Occurrence legend must retain the main map page width")
    scale = .16614 * width / float(locator.mediabox.width)
    compose(width, height, [
        (main, 1, 0, 0),
        (locator, scale, .75986 * width, .22303859 * width),
        (legend, 1, 0, 0),
    ], output)


def fold_legend_page(counts, font_path, size=FOLD_SIZE, spines=FOLD_SPINES):
    """Two-row/two-column key; counts come from the supplied split table.

    Fixed publication anchors preserve the manuscript's Times 7.4 pt layout.
    A longer right-column label shifts that entire column left by its extra
    advance width. A collision is rejected instead of shrinking the font.
    """
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.colors import HexColor

    if set(counts) != {"T", "C", "E", "X"} or any(int(v) != v or v < 0 for v in counts.values()):
        raise ValueError("Counts must contain nonnegative integers T, C, E and X")
    font = "GeoPRFoldLegend"
    pdfmetrics.registerFont(TTFont(font, str(font_path)))
    labels = [f"Train: {counts['T']:,}", f"Calibration: {counts['C']:,}",
              f"Test: {counts['E']:,}", f"Excluded: {counts['X']:,}"]
    reference = pdfmetrics.stringWidth("Calibration: 932", font, 7.4)
    right_width = max(pdfmetrics.stringWidth(labels[i], font, 7.4) for i in (1, 3))
    left = spines[0] - .16
    right = spines[1] - 55.62873072916672 - (right_width - reference)
    if max(left + 8 + pdfmetrics.stringWidth(labels[i], font, 7.4) for i in (0, 2)) >= right - 3:
        raise ValueError("Fold count labels do not fit the fixed publication legend")
    stream = BytesIO()
    c = canvas.Canvas(stream, pagesize=size)
    c.setFont(font, 7.4)
    positions = [(left, 14.4), (right, 14.4), (left, 3.4), (right, 3.4)]
    colors = ["#8297AF", "#E5BE32", "#FC8D62", "#BBBBBB"]
    for label, color, (x, y) in zip(labels, colors, positions):
        c.setFillColor(HexColor(color))
        c.setStrokeColorRGB(0, 0, 0)
        c.setLineWidth(.30)
        c.rect(x, y + .65, 4.9, 4.9, stroke=1, fill=1)
        c.setFillColorRGB(0, 0, 0)
        c.drawString(x + 8, y, label)
    c.showPage()
    c.save()
    stream.seek(0)
    return one_page(stream)


def assemble_fold(panel_pdf, counts, font_path, output, remove_old_footer=True):
    """Replace the original abbreviated count footer without touching the map."""
    original = one_page(panel_pdf, FOLD_SIZE)
    clipped = deepcopy(original)
    if remove_old_footer:
        clipped.cropbox = RectangleObject([0, 23, *FOLD_SIZE])
    footer = fold_legend_page(counts, font_path)
    page = PageObject.create_blank_page(width=FOLD_SIZE[0], height=FOLD_SIZE[1])
    page.merge_page(clipped)
    page.merge_page(footer)
    writer = PdfWriter()
    writer.add_page(page)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        writer.write(stream)
