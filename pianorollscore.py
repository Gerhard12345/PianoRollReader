"""LilyPond score generation and layout extraction for PianoRollReader.

Each hand is written to its own LilyPond file (right hand: treble clef, left
hand: bass clef) and both are fused into one piano-staff score. One engraved
piece (note, chord or rest) is written per source line, so the point-and-click
links of the SVG backend tell us where every musical moment ends up on a page.
"""

from __future__ import annotations

import bisect
import glob
import os
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

SLOTS_PER_MEASURE = 16
PITCH_NAMES = ["c", "cis", "d", "dis", "e", "f", "fis", "g", "gis", "a", "ais", "b"]
DURATION_NAMES = {16: "1", 12: "2.", 8: "2", 6: "4.", 4: "4", 3: "8.", 2: "8", 1: "16"}
SVG_NS = "{http://www.w3.org/2000/svg}"
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"


def find_lilypond() -> str | None:
	configured = os.environ.get("LILYPOND_PATH")
	if configured and Path(configured).exists():
		return configured
	found = shutil.which("lilypond")
	if found:
		return found
	patterns = [
		r"C:\Program Files*\lilypond*\bin\lilypond.exe",
		r"C:\Program Files*\lilypond*\*\bin\lilypond.exe",
		r"C:\lilypond*\bin\lilypond.exe",
		r"C:\lilypond*\*\bin\lilypond.exe",
	]
	candidates: list[str] = []
	for pattern in patterns:
		candidates.extend(glob.glob(pattern))
	return sorted(candidates)[-1] if candidates else None


def lilypond_pitch(midi_note: int) -> str:
	marks = midi_note // 12 - 1 - 3
	suffix = "'" * marks if marks > 0 else "," * -marks
	return PITCH_NAMES[midi_note % 12] + suffix


def split_length(position: int, length: int, sounding: bool = False) -> list[int]:
	"""Split a span inside one measure into notatable lengths.

	Sounding notes may start off-beat (a quarter on an eighth position stays
	one quarter) as long as they do not cross the middle of the measure.
	Half and dotted-half notes may start on any beat (e.g. beat 2) so that
	held notes are not split into tied quarters. Rests keep strict beat
	alignment.
	"""
	pieces: list[int] = []
	half = SLOTS_PER_MEASURE // 2
	while length > 0:
		for candidate in (16, 12, 8, 6, 4, 3, 2, 1):
			if candidate > length:
				continue
			offset = position % SLOTS_PER_MEASURE
			alignment = 1 << (candidate.bit_length() - 1)
			if sounding and candidate == 3 and length != 3:
				continue
			allowed = offset % alignment == 0
			if sounding and not allowed:
				if candidate <= half and offset // half == (offset + candidate - 1) // half:
					allowed = True
				elif candidate in (8, 12) and offset % 4 == 0:
					allowed = True
			if allowed:
				pieces.append(candidate)
				position += candidate
				length -= candidate
				break
	return pieces


@dataclass
class Piece:
	slot: int
	length: int
	pitches: tuple[int, ...]
	tied: frozenset[int]


def hand_pieces(notes: list[tuple[int, int, int]], total_slots: int) -> list[Piece]:
	"""Turn (start, end, midi) notes into a gap-free sequence of pieces."""
	bounds = {0, total_slots}
	bounds.update(range(0, total_slots, SLOTS_PER_MEASURE))
	for start, end, _ in notes:
		bounds.add(start)
		bounds.add(end)
	ordered = sorted(bounds)
	pending = sorted(notes)
	next_note = 0
	active: dict[int, int] = {}
	pieces: list[Piece] = []
	for index, begin in enumerate(ordered[:-1]):
		finish = ordered[index + 1]
		while next_note < len(pending) and pending[next_note][0] <= begin:
			start, end, pitch = pending[next_note]
			active[pitch] = max(end, active.get(pitch, 0))
			next_note += 1
		active = {pitch: end for pitch, end in active.items() if end > begin}
		pitches = tuple(sorted(active))
		continuing = frozenset(pitch for pitch, end in active.items() if end > finish)
		parts = split_length(begin, finish - begin, sounding=bool(pitches))
		position = begin
		for part_index, part in enumerate(parts):
			is_last = part_index == len(parts) - 1
			pieces.append(Piece(position, part, pitches, continuing if is_last else frozenset(pitches)))
			position += part
	return pieces


def render_hand(
	variable: str, clef: str, pieces: list[Piece]
) -> tuple[str, dict[int, int], set[tuple[int, int]], dict[tuple[int, int], int]]:
	"""Return the LilyPond text, a {line: slot} map, tie columns and {(line, column): pitch}."""
	lines = [f"{variable} = {{", f"  \\clef {clef}"]
	line_slots: dict[int, int] = {}
	tie_columns: set[tuple[int, int]] = set()
	pitch_columns: dict[tuple[int, int], int] = {}
	for piece in pieces:
		duration = DURATION_NAMES[piece.length]
		offsets: list[tuple[int, int]] = []
		if not piece.pitches:
			text = f"r{duration}"
		elif len(piece.pitches) == 1:
			pitch = piece.pitches[0]
			text = lilypond_pitch(pitch) + duration + ("~" if pitch in piece.tied else "")
			offsets.append((2, pitch))
		else:
			tokens = []
			position = 3
			for pitch in piece.pitches:
				token = lilypond_pitch(pitch) + ("~" if pitch in piece.tied else "")
				tokens.append(token)
				offsets.append((position, pitch))
				position += len(token) + 1
			text = f"<{' '.join(tokens)}>{duration}"
		if (piece.slot + piece.length) % SLOTS_PER_MEASURE == 0:
			text += " |"
		lines.append(f"  {text}")
		line_slots[len(lines)] = piece.slot
		tie_columns.update(
			(len(lines), column) for column, char in enumerate(f"  {text}") if char == "~"
		)
		pitch_columns.update(((len(lines), column), pitch) for column, pitch in offsets)
	lines.append("  \\bar \"|.\"")
	lines.append("}")
	return "\n".join(lines) + "\n", line_slots, tie_columns, pitch_columns


@dataclass
class ScoreSource:
	"""Generated LilyPond sources plus the mapping needed for the playhead."""

	main_name: str
	files: dict[str, str]
	line_slots: dict[tuple[str, int], int]
	tie_columns: set[tuple[str, int, int]]
	time_origin: float
	grid_seconds: float
	total_slots: int
	measures: int
	pitch_columns: dict[tuple[str, int, int], int] = field(default_factory=dict)


def build_score(
	notes: list[dict[str, object]],
	bpm: int,
	origin_seconds: float,
	title: str,
	base: str = "score",
) -> ScoreSource:
	"""notes: dicts with hand ('left'/'right'), midi_note, start, end in seconds."""
	grid = 60.0 / bpm / 4
	converted: dict[str, list[tuple[int, int, int]]] = {"left": [], "right": []}
	for note in notes:
		start = round((float(note["start"]) - origin_seconds) / grid)
		end = round((float(note["end"]) - origin_seconds) / grid)
		end = max(end, start + 1)
		hand = "left" if note["hand"] == "left" else "right"
		converted[hand].append((start, end, int(note["midi_note"])))
	all_starts = [start for hand_notes in converted.values() for start, _, _ in hand_notes]
	shift = (min(all_starts) // SLOTS_PER_MEASURE) * SLOTS_PER_MEASURE if all_starts else 0
	for hand, hand_notes in converted.items():
		converted[hand] = [(s - shift, e - shift, m) for s, e, m in hand_notes]
	last_end = max((end for hand_notes in converted.values() for _, end, _ in hand_notes), default=0)
	measures = max(1, -(-last_end // SLOTS_PER_MEASURE))
	total_slots = measures * SLOTS_PER_MEASURE

	right_name, left_name = f"{base}_right.ly", f"{base}_left.ly"
	right_text, right_lines, right_ties, right_pitches = render_hand(
		"rightHand", "treble", hand_pieces(converted["right"], total_slots)
	)
	left_text, left_lines, left_ties, left_pitches = render_hand(
		"leftHand", "bass", hand_pieces(converted["left"], total_slots)
	)
	safe_title = title.replace("\\", "\\\\").replace('"', '\\"')
	main_text = (
		'\\version "2.24.0"\n'
		f'\\include "{right_name}"\n'
		f'\\include "{left_name}"\n'
		f'\\header {{ title = "{safe_title}" tagline = ##f }}\n'
		'\\paper { #(set-paper-size "a4") indent = 0 ragged-last-bottom = ##t page-breaking = #ly:minimal-breaking }\n'
		"\\score {\n"
		"  \\new PianoStaff <<\n"
		f'    \\new Staff = "right" {{ \\tempo 4 = {bpm} \\time 4/4 \\rightHand }}\n'
		'    \\new Staff = "left" { \\time 4/4 \\leftHand }\n'
		"  >>\n"
		"  \\layout {\n"
		"    \\context {\n"
		"      \\Score\n"
		"      barNumberVisibility = #all-bar-numbers-visible\n"
		"      \\override BarNumber.break-visibility = ##(#f #t #t)\n"
		"    }\n"
		"  }\n"
		"}\n"
	)
	line_slots = {(right_name, line): slot for line, slot in right_lines.items()}
	line_slots.update({(left_name, line): slot for line, slot in left_lines.items()})
	return ScoreSource(
		main_name=f"{base}.ly",
		files={f"{base}.ly": main_text, right_name: right_text, left_name: left_text},
		line_slots=line_slots,
		tie_columns={(right_name, *item) for item in right_ties}
		| {(left_name, *item) for item in left_ties},
		time_origin=origin_seconds + shift * grid,
		grid_seconds=grid,
		total_slots=total_slots,
		measures=measures,
		pitch_columns={(right_name, *key): pitch for key, pitch in right_pitches.items()}
		| {(left_name, *key): pitch for key, pitch in left_pitches.items()},
	)


def lilypond_excerpt(source: ScoreSource, hand: str, first_slot: int, last_slot: int) -> str:
	"""LilyPond lines of one hand whose slots lie in [first_slot, last_slot)."""
	name = f"{source.main_name.removesuffix('.ly')}_{hand}.ly"
	lines = source.files[name].splitlines()
	chosen = sorted(
		line for (file, line), slot in source.line_slots.items()
		if file == name and first_slot <= slot < last_slot
	)
	return "\n".join(lines[line - 1].strip() for line in chosen)


def write_score_files(source: ScoreSource, directory: Path) -> Path:
	directory.mkdir(parents=True, exist_ok=True)
	for name, text in source.files.items():
		(directory / name).write_text(text, encoding="utf-8")
	return directory / source.main_name


@dataclass
class SystemLayout:
	page: int
	top: float
	bottom: float
	left: float
	right: float
	anchors: list[tuple[int, float]] = field(default_factory=list)
	split: float | None = None
	# per hand: (slot, x, y, pitch or None for rests) of every note head and rest
	points: dict[str, list[tuple[int, float, float, int | None]]] = field(default_factory=dict)


def _find_targets(
	notes: list[dict[str, object]],
	hand: str,
	slot: int,
	origin: float,
	grid: float,
	pitch: int | None = None,
) -> tuple[dict[int, tuple[int, int]], list[int]]:
	spans: dict[int, tuple[int, int]] = {}
	for index, note in enumerate(notes):
		if ("left" if note["hand"] == "left" else "right") != hand:
			continue
		start = round((float(note["start"]) - origin) / grid)
		spans[index] = (start, max(round((float(note["end"]) - origin) / grid), start + 1))
	active = [
		i for i, (start, end) in spans.items()
		if start <= slot < end and (pitch is None or int(notes[i]["midi_note"]) == pitch)
	]
	targets = [i for i in active if spans[i][0] == slot] or active
	return spans, targets


def find_note(
	notes: list[dict[str, object]],
	hand: str,
	slot: int,
	origin: float,
	grid: float,
	pitch: int | None = None,
) -> dict[str, object] | None:
	"""The note that adjust_duration would change at this slot."""
	_, targets = _find_targets(notes, hand, slot, origin, grid, pitch)
	return notes[targets[0]] if targets else None


def insert_note(
	notes: list[dict[str, object]],
	hand: str,
	pitch: int,
	slot: int,
	length: int,
	push: bool,
	origin: float,
	grid: float,
	video_seconds: float | None = None,
) -> list[dict[str, object]]:
	"""Add a note; with push, later notes of the same hand move by its length."""
	result = [note.copy() for note in notes]
	if push:
		for note in result:
			if ("left" if note["hand"] == "left" else "right") != hand:
				continue
			if round((float(note["start"]) - origin) / grid) >= slot:
				note["start"] = float(note["start"]) + length * grid
				note["end"] = float(note["end"]) + length * grid
	start = origin + slot * grid
	added: dict[str, object] = {
		"hand": hand,
		"midi_note": pitch,
		"start": start,
		"end": start + length * grid,
	}
	if video_seconds is not None:
		added["video_seconds"] = video_seconds
	result.append(added)
	return result


def video_time_map(notes: list[dict[str, object]]):
	"""Map score seconds to video seconds by interpolating between note onsets."""
	pairs = sorted(
		(float(note["start"]), float(note["video_seconds"]))
		for note in notes
		if "video_seconds" in note
	)
	if not pairs:
		return None
	xs: list[float] = []
	ys: list[float] = []
	for x, y in pairs:
		if xs and x - xs[-1] < 1e-9:
			continue
		xs.append(x)
		ys.append(max(y, ys[-1]) if ys else y)

	def convert(seconds: float) -> float:
		if seconds <= xs[0]:
			return ys[0] + (seconds - xs[0])
		if seconds >= xs[-1]:
			return ys[-1] + (seconds - xs[-1])
		index = bisect.bisect_right(xs, seconds)
		fraction = (seconds - xs[index - 1]) / (xs[index] - xs[index - 1])
		return ys[index - 1] + fraction * (ys[index] - ys[index - 1])

	return convert


def current_length(
	notes: list[dict[str, object]],
	hand: str,
	slot: int,
	origin: float,
	grid: float,
	pitch: int | None = None,
) -> int | None:
	"""Length in slots of the note(s) that adjust_duration would change."""
	spans, targets = _find_targets(notes, hand, slot, origin, grid, pitch)
	return max((spans[i][1] - spans[i][0] for i in targets), default=None)


def adjust_duration(
	notes: list[dict[str, object]],
	hand: str,
	slot: int,
	new_length: int,
	add_pause: bool,
	origin: float,
	grid: float,
	pitch: int | None = None,
) -> list[dict[str, object]] | None:
	"""Change the length of one note (or, without pitch, of the chord) at a slot.

	Following notes of that hand move along unless a shortened note is
	replaced by a rest (add_pause). Lengthening always pushes them.
	"""
	def to_seconds(position: int) -> float:
		return origin + position * grid

	spans, targets = _find_targets(notes, hand, slot, origin, grid, pitch)
	if not targets:
		return None
	old_end = max(spans[i][1] for i in targets)
	new_end = max(spans[i][0] + new_length for i in targets)
	delta = new_end - old_end
	result = [note.copy() for note in notes]
	for i in targets:
		result[i]["end"] = to_seconds(spans[i][0] + new_length)
	if delta > 0 or (delta < 0 and not add_pause):
		for i, (start, _) in spans.items():
			if i not in targets and start >= old_end:
				result[i]["start"] = float(result[i]["start"]) + delta * grid
				result[i]["end"] = float(result[i]["end"]) + delta * grid
	return result


@dataclass
class ScoreLayout:
	page_sizes: list[tuple[float, float]]
	systems: list[SystemLayout]
	total_slots: int

	def _first_slots(self) -> list[int]:
		return [system.anchors[0][0] for system in self.systems]

	def locate(self, slot: float) -> tuple[int, float, float, float] | None:
		"""Return (page, x, top, bottom) as page fractions for a fractional slot."""
		if not self.systems:
			return None
		firsts = self._first_slots()
		system_index = max(0, bisect.bisect_right(firsts, slot) - 1)
		system = self.systems[system_index]
		slots = [anchor[0] for anchor in system.anchors]
		index = max(0, bisect.bisect_right(slots, slot) - 1)
		slot_a, x_a = system.anchors[index]
		if slot < slot_a:
			x = x_a
		else:
			if index + 1 < len(system.anchors):
				slot_b, x_b = system.anchors[index + 1]
			elif system_index + 1 < len(self.systems):
				slot_b, x_b = firsts[system_index + 1], system.right
			else:
				slot_b, x_b = self.total_slots, system.right
			fraction = min(1.0, max(0.0, (slot - slot_a) / max(1, slot_b - slot_a)))
			x = x_a + fraction * (x_b - x_a)
		width, height = self.page_sizes[system.page]
		return system.page, x / width, system.top / height, system.bottom / height

	def slot_at(self, page: int, fx: float, fy: float) -> int | None:
		"""Nearest slot for a click given as page fractions."""
		candidates = [system for system in self.systems if system.page == page]
		if not candidates:
			return None
		width, height = self.page_sizes[page]
		y = fy * height
		system = min(
			candidates,
			key=lambda item: 0.0 if item.top <= y <= item.bottom else min(abs(y - item.top), abs(y - item.bottom)),
		)
		x = fx * width
		anchor = min(system.anchors, key=lambda item: abs(item[1] - x))
		return anchor[0]

	def hit_test(self, page: int, fx: float, fy: float) -> tuple[str, int, int | None] | None:
		"""Return (hand, slot, pitch) of the note head or rest nearest to a click."""
		candidates = [system for system in self.systems if system.page == page]
		if not candidates:
			return None
		width, height = self.page_sizes[page]
		y = fy * height
		system = min(
			candidates,
			key=lambda item: 0.0 if item.top <= y <= item.bottom else min(abs(y - item.top), abs(y - item.bottom)),
		)
		if system.split is None:
			return None
		hand = "right" if y < system.split else "left"
		points = system.points.get(hand)
		if not points:
			return None
		x = fx * width
		slot, _, _, pitch = min(points, key=lambda item: (item[1] - x) ** 2 + (item[2] - y) ** 2)
		return hand, slot, pitch


def _translate(element: ET.Element) -> tuple[float, float]:
	match = re.fullmatch(r"\s*translate\(\s*([-\d.]+)[ ,]+([-\d.]+)\s*\)\s*", element.get("transform", ""))
	return (float(match.group(1)), float(match.group(2))) if match else (0.0, 0.0)


def _parse_page(
	svg_path: Path,
) -> tuple[tuple[float, float], list[tuple[float, float, float]], list[tuple[str, int, int, float, float]]]:
	root = ET.parse(svg_path).getroot()
	view_box = [float(value) for value in root.get("viewBox", "0 0 1 1").split()]
	size = (view_box[2], view_box[3])
	staff_lines: list[tuple[float, float, float]] = []
	anchors: list[tuple[str, int, int, float, float]] = []

	def visit(element: ET.Element, x: float, y: float, link: tuple[str, int, int] | None) -> None:
		for child in element:
			tag = child.tag.removeprefix(SVG_NS)
			if tag == "a":
				href = child.get(XLINK_HREF, "")
				match = re.search(r"([^/\\:]+):(\d+):(\d+):\d+$", href)
				visit(child, x, y, (match.group(1), int(match.group(2)), int(match.group(3))) if match else link)
			elif tag == "g":
				dx, dy = _translate(child)
				if link is not None:
					anchors.append((link[0], link[1], link[2], x + dx, y + dy))
					continue
				visit(child, x + dx, y + dy, link)
			elif tag == "line" and link is None:
				y1, y2 = float(child.get("y1", 0)), float(child.get("y2", 0))
				x1, x2 = float(child.get("x1", 0)), float(child.get("x2", 0))
				if y1 == y2 and x2 - x1 > 8:
					staff_lines.append((y + y1, x + x1, x + x2))

	visit(root, 0.0, 0.0, None)
	return size, staff_lines, anchors


def _group_staves(
	lines: list[tuple[float, float, float]]
) -> list[tuple[float, float, float, float]]:
	staves: list[tuple[float, float, float, float]] = []
	group: list[tuple[float, float, float]] = []
	for line in sorted(lines):
		if group and line[0] - group[-1][0] > 1.5:
			staves.append(_close_staff(group))
			group = []
		group.append(line)
	if group:
		staves.append(_close_staff(group))
	return [staff for staff in staves if staff[0] >= 0]


def _close_staff(group: list[tuple[float, float, float]]) -> tuple[float, float, float, float]:
	return (
		group[0][0],
		group[-1][0],
		min(item[1] for item in group),
		max(item[2] for item in group),
	)


def svg_page_files(directory: Path, base: str) -> list[Path]:
	pattern = re.compile(re.escape(base) + r"-(?:page)?(\d+)\.svg")
	paged = sorted(
		(path for path in directory.glob(f"{base}-*.svg") if pattern.fullmatch(path.name)),
		key=lambda path: int(pattern.fullmatch(path.name).group(1)),
	)
	single = directory / f"{base}.svg"
	return paged or ([single] if single.exists() else [])


def load_layout(directory: Path, source: ScoreSource) -> ScoreLayout:
	base = source.main_name.removesuffix(".ly")
	page_sizes: list[tuple[float, float]] = []
	systems: list[SystemLayout] = []
	for page_index, svg_path in enumerate(svg_page_files(directory, base)):
		size, lines, raw_anchors = _parse_page(svg_path)
		page_sizes.append(size)
		staves = _group_staves(lines)
		page_systems = [
			SystemLayout(
				page_index,
				min(pair[0][0], pair[-1][0]) - 3.0,
				max(pair[0][1], pair[-1][1]) + 3.0,
				min(item[2] for item in pair),
				max(item[3] for item in pair),
				split=(pair[0][1] + pair[1][0]) / 2 if len(pair) == 2 else None,
			)
			for pair in (staves[index:index + 2] for index in range(0, len(staves), 2))
		]
		by_slot: list[tuple[int, float, float]] = []
		hand_of_slot: list[str] = []
		pitch_of_slot: list[int | None] = []
		for name, line, column, x, y in raw_anchors:
			if (name, line, column) in source.tie_columns:
				continue
			slot = source.line_slots.get((name, line))
			if slot is not None:
				by_slot.append((slot, x, y))
				hand_of_slot.append("right" if name.endswith("_right.ly") else "left")
				pitch_of_slot.append(source.pitch_columns.get((name, line, column)))
		grouped: dict[tuple[int, int], list[float]] = {}
		for (slot, x, y), hand, pitch in zip(by_slot, hand_of_slot, pitch_of_slot):
			if not page_systems:
				break
			system_index = min(
				range(len(page_systems)),
				key=lambda i: 0.0 if page_systems[i].top <= y <= page_systems[i].bottom
				else min(abs(y - page_systems[i].top), abs(y - page_systems[i].bottom)),
			)
			grouped.setdefault((system_index, slot), []).append(x)
			page_systems[system_index].points.setdefault(hand, []).append((slot, x, y, pitch))
		for (system_index, slot), xs in grouped.items():
			xs.sort()
			page_systems[system_index].anchors.append((slot, xs[len(xs) // 2]))
		for system in page_systems:
			system.anchors.sort()
			if system.anchors:
				systems.append(system)
	return ScoreLayout(page_sizes, systems, source.total_slots)
