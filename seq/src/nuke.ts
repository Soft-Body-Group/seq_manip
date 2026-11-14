// Nuke interchange: the `::elementSelection::` clipboard protocol.
//
// "Copy for Nuke" puts `::elementSelection::` + JSON on the clipboard; a
// small paste handler inside Nuke (see NUKE_HANDLER below — the user installs
// it once via "Copy Nuke handler") turns that into Read nodes with the right
// frame range, missing-frame handling and colorspace.

import type { EntityView, InputSpace } from "$lib/api";

export const ELEMENT_PREFIX = "::elementSelection::";

export interface NukeElement {
  name: string;
  /** printf-style path for sequences (`/x/shot.%04d.exr`), plain for others. */
  printf: string;
  kind: "sequence" | "still" | "movie";
  first?: number;
  last?: number;
  missing?: number[];
  colorspace?: string;
  fps?: number;
}

export interface NukePayload {
  version: 1;
  elements: NukeElement[];
}

const MOVIE_EXTS = new Set(["mov", "mp4", "m4v", "avi", "mkv", "webm", "mxf"]);

function joinPath(dir: string, name: string): string {
  const sep = dir.includes("\\") && !dir.includes("/") ? "\\" : "/";
  return dir.replace(/[\\/]+$/, "") + sep + name;
}

/** Nuke wants forward slashes on every platform. */
function nukePath(p: string): string {
  return p.replace(/\\/g, "/");
}

/** Colorspace label from the app's input-space setting (EXR is scene-linear;
 * everything else is display-referred). Nuke-side aliases map these onto the
 * comp's OCIO config. */
function colorspaceFor(ext: string, input: InputSpace): string {
  if (ext === "exr") return input === "acesCg" ? "ACEScg" : "linear";
  if (MOVIE_EXTS.has(ext)) return "rec709";
  return "sRGB";
}

/** Builds one protocol element from an entity, or null if it isn't media. */
export function elementFor(
  e: EntityView,
  input: InputSpace,
  fps = 24,
): NukeElement | null {
  if (e.kind === "sequence" && e.sequence) {
    const s = e.sequence;
    const delim = s.delimiter ?? "";
    const suffix = s.suffix ?? "";
    const ext = s.extension.toLowerCase();
    // %0Nd — Nuke's printf notation. Padding 0 (unpadded numbers) → %d.
    const pat = s.padding > 0 ? `%0${s.padding}d` : "%d";
    const filename = `${s.prefix}${delim}${pat}${suffix}.${s.extension}`;
    return {
      name: s.name,
      printf: nukePath(joinPath(e.dir, filename)),
      kind: "sequence",
      first: s.firstFrame,
      last: s.lastFrame,
      missing: s.missingFrames.length ? s.missingFrames : undefined,
      colorspace: colorspaceFor(ext, input),
      fps,
    };
  }
  if (e.kind === "file" && e.entry) {
    const ext = e.entry.name.split(".").pop()?.toLowerCase() ?? "";
    const isMovie = MOVIE_EXTS.has(ext);
    return {
      name: e.entry.name,
      printf: nukePath(e.entry.path),
      kind: isMovie ? "movie" : "still",
      colorspace: colorspaceFor(ext, input),
      ...(isMovie ? { fps } : {}),
    };
  }
  return null;
}

export function buildPayload(
  entities: EntityView[],
  input: InputSpace,
  fps = 24,
): NukePayload {
  return {
    version: 1,
    elements: entities
      .map((e) => elementFor(e, input, fps))
      .filter((x): x is NukeElement => x !== null),
  };
}

/** The clipboard string Nuke's handler consumes. */
export function clipboardText(payload: NukePayload): string {
  return ELEMENT_PREFIX + JSON.stringify(payload);
}

/**
 * The Nuke-side paste handler. The user runs this ONCE in Nuke's Script
 * Editor (or drops it in ~/.nuke and imports it from menu.py) and from then on
 * Ctrl+V with an ::elementSelection:: payload on the clipboard builds Read
 * nodes; any other clipboard content falls through to Nuke's normal paste.
 */
export const NUKE_HANDLER = String.raw`"""seq_manip -> Nuke paste handler for the ::elementSelection:: protocol.

Install (pick one):
  * Quick:  paste this whole file into Nuke's Script Editor and run it.
            Active until Nuke restarts.
  * Permanent: save as ~/.nuke/seqmanip_paste.py and add to ~/.nuke/menu.py:
            import seqmanip_paste
            seqmanip_paste.install()

Then in seq_manip: put things in the Bin (or select them) -> "Copy for Nuke",
switch to Nuke, Ctrl+V in the node graph. Anything else on the clipboard
pastes normally.
"""

import json

import nuke

PREFIX = "::elementSelection::"

# Map protocol colorspace names onto YOUR OCIO config's names. Keys are
# lowercased-alphanumeric. Edit to taste.
COLORSPACE_ALIASES = {
    "srgb": "Output - sRGB",
    "rec709": "Output - Rec.709",
    "acescg": "ACES - ACEScg",
    "linear": "linear",
}

_GRID_COLS = 6
_GRID_DX = 110
_GRID_DY = 120


def _clipboard_text():
    try:
        from PySide6.QtWidgets import QApplication  # Nuke 16+
    except ImportError:
        from PySide2.QtWidgets import QApplication  # Nuke 12-15
    app = QApplication.instance()
    return (app.clipboard().text() if app else "") or ""


def _alias_colorspace(colorspace):
    key = "".join(ch for ch in colorspace.lower() if ch.isalnum())
    return COLORSPACE_ALIASES.get(key, colorspace)


def _apply_colorspace(node, colorspace, name, warnings):
    if not colorspace:
        return
    knob = node.knob("colorspace")
    if knob is None:
        return
    target = _alias_colorspace(colorspace)
    try:
        knob.setValue(target)
        applied = knob.value()
    except Exception:
        applied = None
    if applied != target:
        note = repr(colorspace) if target == colorspace else "%r -> %r" % (colorspace, target)
        warnings.append(
            "%s: colorspace %s not in this comp's OCIO config -- left at %r."
            % (name, note, knob.value())
        )


def _create_read(element, warnings):
    path = element.get("printf") or element.get("path")
    name = element.get("name") or (path.rsplit("/", 1)[-1] if path else "element")
    if not path:
        warnings.append("%s: no path in payload -- skipped." % name)
        return None

    kind = element.get("kind") or ("sequence" if "%" in path else "still")
    first = element.get("first")
    last = element.get("last")
    missing = element.get("missing") or []

    node = nuke.createNode("Read", inpanel=False)

    if kind == "movie" and (first is None or last is None):
        node["file"].fromUserText(path)
    else:
        node["file"].setValue(path)
        if first is not None and last is not None:
            for knob, value in (("first", first), ("last", last),
                                ("origfirst", first), ("origlast", last)):
                node[knob].setValue(int(value))

    if missing:
        node["on_error"].setValue("checkerboard")

    _apply_colorspace(node, element.get("colorspace"), name, warnings)

    label = name
    if first is not None and last is not None and kind != "still":
        label += "\n%s-%s" % (first, last)
    if missing:
        label += " (%d missing)" % len(missing)
    node["label"].setValue(label)
    return node


def build_reads(payload):
    elements = payload.get("elements") or []
    if not elements:
        nuke.message("elementSelection: payload contains no elements")
        return []

    for node in nuke.selectedNodes():
        node.setSelected(False)

    created = []
    warnings = []
    undo = nuke.Undo()
    undo.begin("Paste %d element%s" % (len(elements), "s" if len(elements) != 1 else ""))
    try:
        anchor = None
        for element in elements:
            node = _create_read(element, warnings)
            if node is None:
                continue
            if anchor is None:
                anchor = (node.xpos(), node.ypos())
            else:
                col = len(created) % _GRID_COLS
                row = len(created) // _GRID_COLS
                node.setXYpos(anchor[0] + col * _GRID_DX, anchor[1] + row * _GRID_DY)
            created.append(node)
    finally:
        undo.end()

    for node in created:
        node.setSelected(True)
    if warnings:
        nuke.message("Created %d Read node(s) with warnings:\n\n%s"
                     % (len(created), "\n".join(warnings)))
    return created


def paste():
    """Ctrl+V: build Reads from an ::elementSelection:: payload, else the
    normal Nuke paste."""
    text = _clipboard_text()
    if text.startswith(PREFIX):
        try:
            payload = json.loads(text[len(PREFIX):])
        except ValueError as exc:
            nuke.message("elementSelection payload is not valid JSON:\n%s" % exc)
            return
        build_reads(payload)
    else:
        nuke.nodePaste("%clipboard%")


def install():
    """Bind Ctrl+V in the Node Graph to the smart paste. Idempotent."""
    edit = nuke.menu("Nuke").findItem("Edit")
    if edit is None:
        return
    existing = edit.findItem("Paste")
    if existing is not None:
        edit.removeItem("Paste")
    edit.addCommand("Paste", paste, "^v", index=6)
    print("[seq_manip] elementSelection paste handler installed on Ctrl+V")


install()
`;
