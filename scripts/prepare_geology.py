"""Extract classified geological polygons and faults from supplied GeoPackages.

This does not download or bundle geological data. Supply a verified 1:200,000
source and its classification mapping. Expected converted source layers are
D01D_polygons_raw, ..., D05M_polygons_raw and D08D_lines. Source coordinates
are retained. The generated display geometry is separate from the source.
Mapping JSON schema: entries=[{source_layer, match_fields:{field:value},class_id}].
More specific attribute matches take priority; ambiguous matches are flagged
as unclassified. A preclassified source can instead use --class-field.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import fiona
import shapely
from shapely.geometry import shape, mapping
from pyproj import CRS

PALETTE = {
    "q": ("Quaternary", "#fff2a8"), "paleogene_neogene": ("Paleogene–Neogene", "#efc46e"),
    "k": ("Cretaceous", "#a4d583"), "j": ("Jurassic", "#84ccd7"), "t": ("Triassic", "#ad94ca"),
    "p": ("Permian", "#cd967d"), "c": ("Carboniferous", "#a8beb5"),
    "dc": ("Devonian-Carboniferous", "#c3b791"), "d": ("Devonian", "#d0af7d"),
    "s": ("Silurian", "#a4d2a0"), "o": ("Ordovician", "#78b79d"), "cambrian": ("Cambrian", "#a6ad76"),
    "sinian": ("Sinian", "#dbb1a7"), "proterozoic": ("Proterozoic", "#cbb7d1"),
    "granitoids": ("Granitoids", "#f0a1a1"), "other_intrusive": ("Other intrusive rocks", "#bd7397"),
    "volcanic": ("Volcanic rocks", "#caa66f"), "dyke_rocks": ("Dykes and veins", "#918576"),
    "migmatitic": ("Migmatitic rocks", "#d28daa"), "metamorphic_undiff": ("Metamorphic rocks", "#b1a5b6"),
    "unclassified": ("Unclassified unit", "#e0e0e0"), "unmapped": ("No mapped unit", "#ffffff")}

LAYERS = ["D01D", "D01B", "D02B", "D03D", "D07H", "D04N", "D04P", "D02H", "D05M"]
MERGED = {"paleogene": "paleogene_neogene", "neogene": "paleogene_neogene", "granite": "granitoids",
          "monzogranite": "granitoids", "other_granitic": "granitoids", "granodiorite": "granitoids",
          "diorite_monzonite": "other_intrusive", "mafic_ultramafic": "other_intrusive",
          "felsic_porphyry": "other_intrusive", "intermediate_porphyry": "other_intrusive",
          "intrusive_undiff": "other_intrusive", "pegmatite": "other_intrusive", "dyke_undiff": "dyke_rocks"}


def classifier(document):
    groups = defaultdict(dict)
    normal = lambda value: "" if value is None else str(value).strip()
    for entry in document["entries"]:
        fields = tuple(sorted(entry["match_fields"]))
        key = tuple(normal(entry["match_fields"][field]) for field in fields)
        groups[entry["source_layer"], fields].setdefault(key, set()).add(entry["class_id"])
    def classify(layer, properties):
        matches = []
        for (source, fields), lookup in groups.items():
            if source == layer:
                categories = lookup.get(tuple(normal(properties.get(field)) for field in fields))
                if categories:
                    matches.append((len(fields), categories))
        if not matches:
            return "unclassified"
        specific = max(length for length, _ in matches)
        categories = set().union(*(values for length, values in matches if length == specific))
        return next(iter(categories)) if len(categories) == 1 else "unclassified"
    return classify


def polygons(geometry):
    if geometry.geom_type == "Polygon":
        yield geometry
    elif hasattr(geometry, "geoms"):
        for member in geometry.geoms:
            yield from polygons(member)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, action="append", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mapping", type=Path)
    mode.add_argument("--class-field")
    parser.add_argument("--map-crs", default="EPSG:4214")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    classify = classifier(json.loads(args.mapping.read_text(encoding="utf-8-sig"))) if args.mapping else None
    expected = CRS.from_user_input(args.map_crs)
    geology, faults = [], []
    repairs = 0
    for code in LAYERS + ["D08D"]:
        suffix = "_lines" if code == "D08D" else "_polygons_raw"
        layer = code + suffix
        for package in args.package:
            if layer not in fiona.listlayers(package):
                continue
            with fiona.open(package, layer=layer) as source:
                if not source.crs or CRS.from_user_input(source.crs) != expected:
                    raise ValueError("Unexpected source CRS; explicit preprocessing is required: " + str(package))
                for feature in source:
                    properties = dict(feature["properties"])
                    if properties.get("SRC_ACTIVE", 1) == 0 or feature["geometry"] is None:
                        continue
                    geometry = shape(feature["geometry"])
                    if code == "D08D":
                        faults.append(dict(type="Feature", geometry=mapping(geometry), properties={"class_id": "fault"}))
                    else:
                        category = classify(code, properties) if classify else str(properties[args.class_field])
                        category = MERGED.get(category, category)
                        if category not in PALETTE:
                            raise ValueError("Supply a supported class mapping for: " + category)
                        if not geometry.is_valid:
                            geometry = shapely.make_valid(geometry)
                            repairs += 1
                        for polygon in polygons(geometry):
                            geology.append(dict(type="Feature", geometry=mapping(polygon), properties={"class_id": category}))
    if not geology or not faults:
        raise ValueError("No polygons or faults found in the supplied layers")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("geology", geology), ("faults", faults)):
        document = dict(type="FeatureCollection", name=name, features=rows,
                        crs={"type": "name", "properties": {"name": expected.to_string()}})
        (args.output / (name + ".geojson")).write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    (args.output / "legend.json").write_text(json.dumps(PALETTE, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(polygons=len(geology), faults=len(faults), invalid_source_polygons_made_valid_for_display=repairs,
                          source_coordinates_reprojected=False, source_files_modified=False)))


if __name__ == "__main__":
    main()
