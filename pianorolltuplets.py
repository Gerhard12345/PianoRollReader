"""Parse the measure editor's rhythmic notation without rounding tuplets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

from pianorollscore import LilypondError, _parse_length, _parse_pitch, lilypond_pitch


@dataclass(frozen=True)
class TupletGroup:
	start: Fraction
	end: Fraction
	notes: int
	in_time_of: int


@dataclass(frozen=True)
class NotationEvent:
	start: Fraction
	duration: Fraction
	written_duration: str
	pitches: tuple[int, ...]
	ties: frozenset[int]
	group: int | None


@dataclass(frozen=True)
class ParsedNotation:
	events: tuple[NotationEvent, ...]
	groups: tuple[TupletGroup, ...]
	length: Fraction


def parse_notation(text: str) -> ParsedNotation:
	"""Parse notes, chords, rests and non-nested 3-, 5- and 7-tuplets.

	Positions and durations are exact fractions of a sixteenth note. Group
	boundaries are retained even when the group contains only rests.
	"""
	body = re.sub(r"%[^\n]*", " ", text)
	body = re.sub(r'\\bar\s*"[^"]*"', " ", body)
	body = re.sub(r"\\clef\s+\w+", " ", body).replace("|", " ")
	tokens = re.findall(r"<[^>]*>[\d.]*~?|\\[a-zA-Z]+|[{}<>]|[^\s{}<>]+", body)
	events: list[NotationEvent] = []
	groups: list[TupletGroup] = []
	position = Fraction(0)
	last_length = "4"
	index = 0

	def sequence(multiplier: Fraction, group_index: int | None, nested: bool) -> None:
		nonlocal index, position, last_length
		while index < len(tokens):
			token = tokens[index]
			index += 1
			if token == "}":
				if not nested:
					raise LilypondError("Unexpected closing brace.")
				return
			if token in ("\\tuplet", "\\times"):
				if nested:
					raise LilypondError("Nested tuplets are not supported.")
				if index + 1 >= len(tokens):
					raise LilypondError(f"Expected a ratio and '{{' after {token}.")
				ratio = re.fullmatch(r"(\d+)/(\d+)", tokens[index])
				if ratio is None or tokens[index + 1] != "{":
					raise LilypondError(f"Expected a ratio and '{{' after {token}.")
				first, second = map(int, ratio.groups())
				count, in_time_of = (first, second) if token == "\\tuplet" else (second, first)
				if count not in (3, 5, 7) or in_time_of not in (1, 2, 4, 8):
					raise LilypondError("Only non-nested 3-, 5- and 7-tuplets with power-of-two ratios are supported.")
				index += 2
				begin = position
				group = len(groups)
				groups.append(TupletGroup(begin, begin, count, in_time_of))
				sequence(Fraction(in_time_of, count), group, True)
				if position == begin:
					raise LilypondError("Empty tuplet.")
				groups[group] = TupletGroup(begin, position, count, in_time_of)
				continue
			if token == "{":
				raise LilypondError("Unexpected opening brace.")
			if token.startswith("<"):
				close = token.find(">")
				if close < 0:
					raise LilypondError("Unclosed chord.")
				pitch_tokens = token[1:close].split()
				if not pitch_tokens:
					raise LilypondError("Empty chord.")
				rest = token[close + 1:]
			else:
				match = re.fullmatch(r"([a-gRrs][a-z,']*?)(\d+\.*)?(~?)", token)
				if match is None:
					raise LilypondError(f"Cannot read '{token}'.")
				head, duration, tie = match.groups()
				pitch_tokens = [] if head in ("r", "R", "s") else [head + tie]
				rest = (duration or "") + tie
				if not pitch_tokens and tie:
					raise LilypondError("A rest cannot be tied.")
			length = rest.rstrip("~") or last_length
			trailing_tie = rest.endswith("~")
			last_length = length
			duration = Fraction(_parse_length(length)) * multiplier
			pitches: list[int] = []
			ties: set[int] = set()
			for pitch_token in pitch_tokens:
				pitch, tied = _parse_pitch(pitch_token)
				if not 0 <= pitch <= 127:
					raise LilypondError(f"Pitch '{pitch_token}' is outside the MIDI range.")
				if pitch in pitches:
					raise LilypondError(f"Duplicate pitch '{pitch_token}' in chord.")
				pitches.append(pitch)
				if tied or trailing_tie:
					ties.add(pitch)
			events.append(NotationEvent(position, duration, length, tuple(pitches), frozenset(ties), group_index))
			position += duration
		if nested:
			raise LilypondError("Unclosed tuplet.")

	sequence(Fraction(1), None, False)
	return ParsedNotation(tuple(events), tuple(groups), position)


def notation_notes(
	notation: ParsedNotation,
) -> tuple[list[tuple[Fraction, Fraction, int]], set[int]]:
	"""Return exact note spans and ties extending beyond the parsed range."""
	notes: list[tuple[Fraction, Fraction, int]] = []
	open_ties: dict[int, int] = {}
	for event in notation.events:
		missing = set(open_ties).difference(event.pitches)
		if missing:
			raise LilypondError("A tied note must continue at the next event.")
		next_open: dict[int, int] = {}
		for pitch in event.pitches:
			if pitch in open_ties:
				index = open_ties[pitch]
				start, _, _ = notes[index]
				notes[index] = (start, event.start + event.duration, pitch)
			else:
				index = len(notes)
				notes.append((event.start, event.start + event.duration, pitch))
			if pitch in event.ties:
				next_open[pitch] = index
		open_ties = next_open
	return notes, set(open_ties)


def format_notation(notation: ParsedNotation) -> str:
	"""Write canonical, re-parseable LilyPond while preserving group boundaries."""
	lines: list[str] = []
	active_group: int | None = None
	for event in notation.events:
		if event.group != active_group:
			if active_group is not None:
				lines.append("}")
			active_group = event.group
			if active_group is not None:
				group = notation.groups[active_group]
				lines.append(f"\\tuplet {group.notes}/{group.in_time_of} {{")
		if not event.pitches:
			text = "r" + event.written_duration
		elif len(event.pitches) == 1:
			pitch = event.pitches[0]
			text = lilypond_pitch(pitch) + event.written_duration
			if pitch in event.ties:
				text += "~"
		else:
			pitches = [
				lilypond_pitch(pitch) + ("~" if pitch in event.ties else "")
				for pitch in event.pitches
			]
			text = "<" + " ".join(pitches) + ">" + event.written_duration
		lines.append(("  " if active_group is not None else "") + text)
	if active_group is not None:
		lines.append("}")
	return "\n".join(lines)


def replace_measure(
	notes: list[dict[str, object]], hand: str, measure: int,
	notation: ParsedNotation, origin: float, grid: float,
) -> list[dict[str, object]]:
	"""Replace exactly one bar without quantizing fractional note positions."""
	if notation.length != 16:
		raise LilypondError("Tuplet editing requires exactly four beats in the bar.")
	parsed, dangling = notation_notes(notation)
	first, last = origin + measure * 16 * grid, origin + (measure + 1) * 16 * grid
	epsilon = grid * 1e-7
	result: list[dict[str, object]] = []
	heads: dict[int, dict[str, object]] = {}
	tails: dict[int, float] = {}
	reusable: dict[tuple[int, int], dict[str, object]] = {}
	for original in notes:
		note = original.copy()
		if note["hand"] != hand:
			result.append(note)
			continue
		start, end = float(note["start"]), float(note["end"])
		pitch = int(note["midi_note"])
		if end <= first + epsilon or start >= last - epsilon:
			result.append(note)
			continue
		if end > last + epsilon:
			tails[pitch] = end
		if start < first - epsilon:
			note["end"] = first
			heads[pitch] = note
			result.append(note)
		else:
			reusable[(pitch, round((start - first) / grid * 1000000))] = note
	if dangling.difference(tails):
		raise LilypondError("A tie at the end of the bar must continue an existing held note.")
	for start, end, pitch in parsed:
		end_seconds = first + float(end) * grid
		if end == 16 and pitch in dangling and pitch in tails:
			end_seconds = tails[pitch]
		if start == 0 and pitch in heads:
			heads[pitch]["end"] = end_seconds
			continue
		template = reusable.get((pitch, round(float(start) * 1000000)))
		note = template.copy() if template is not None else {"hand": hand, "midi_note": pitch}
		note["start"] = first + float(start) * grid
		note["end"] = end_seconds
		result.append(note)
	return result


def notation_matches_measure(
	notes: list[dict[str, object]], hand: str, measure: int,
	notation: ParsedNotation, origin: float, grid: float,
) -> bool:
	"""Check that stored notation and playable notes describe the same bar."""
	expected, _ = notation_notes(notation)
	expected_spans = sorted((round(float(start), 6), round(float(end), 6), pitch) for start, end, pitch in expected)
	first, last = origin + measure * 16 * grid, origin + (measure + 1) * 16 * grid
	epsilon = grid * 1e-7
	actual = []
	for note in notes:
		if note["hand"] != hand:
			continue
		start, end = float(note["start"]), float(note["end"])
		if end <= first + epsilon or start >= last - epsilon:
			continue
		actual.append((
			round((max(start, first) - first) / grid, 6),
			round((min(end, last) - first) / grid, 6),
			int(note["midi_note"]),
		))
	return sorted(actual) == expected_spans
