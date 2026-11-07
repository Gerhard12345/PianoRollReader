"""Waterfall (falling bar) onset tracking.

Bars fall towards the keyboard and are cut off at its upper edge. A bar counts
as sounding while it straddles a reference line a little above that edge: its
onset is the frame its lower end reaches the line, its offset the frame its
upper end passes it. Repeated notes are separated by a gap of only a pixel or
two that is visible for a single frame, so bars are tracked by their geometry
instead of by the mere presence of the note. Short dropouts (touching bars of
neighbouring keys merge into one blob that is skipped) are bridged.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

STRIKE_FRACTION = 0.75
LINE_ABOVE_STRIKE = 18
BAND_BELOW_STRIKE = 9
HAND_HUES = (("left", (85, 110)), ("right", (45, 80)))
# a following bar starts when the straddling segment jumps up by more than this
SWITCH_TOLERANCE_PIXELS = 2
MAX_DROPOUT_FRAMES = 2

Segment = tuple[str, str, int, int]


def strike_line(frame_height: int, keyboard_fraction: float | None = None, line_offset: int = LINE_ABOVE_STRIKE) -> int:
	fraction = STRIKE_FRACTION if keyboard_fraction is None else keyboard_fraction
	return round(frame_height * fraction) - line_offset


def waterfall_segments(
	frame: np.ndarray, keys: list[dict[str, object]], key_width: float,
	keyboard_fraction: float | None = None,
	hand_hues: tuple = HAND_HUES,
	line_offset: int = LINE_ABOVE_STRIKE,
	black_hand_hues: tuple | None = None,
) -> list[Segment]:
	"""(note, hand, top, bottom) of every bar piece near the keyboard, in frame rows."""
	frame_height = frame.shape[0]
	strike_y = round(frame_height * (STRIKE_FRACTION if keyboard_fraction is None else keyboard_fraction))
	search_height = round(key_width * 2.5)
	if line_offset != LINE_ABOVE_STRIKE:
		search_height = max(search_height, line_offset + 5)
	search_top = max(0, strike_y - search_height)
	search_bottom = min(frame_height, strike_y + BAND_BELOW_STRIKE)
	hsv = cv2.cvtColor(frame[search_top:search_bottom, :], cv2.COLOR_BGR2HSV)
	key_centers = [
		(str(key["note"]), float(key["bbox"][0]) + float(key["bbox"][2]) / 2, key["kind"])
		for key in keys
	]
	segments: list[Segment] = []
	black_by_hand = dict(black_hand_hues if black_hand_hues is not None else hand_hues)
	ranges = []
	for hand, white_range in hand_hues:
		black_range = black_by_hand[hand]
		if tuple(white_range) == tuple(black_range):
			ranges.append((hand, white_range, None))
		else:
			ranges.extend(((hand, white_range, "white"), (hand, black_range, "black")))
	for hand, (low, high), kind in ranges:
		mask = (
			(hsv[:, :, 1] > 120)
			& (hsv[:, :, 2] > 100)
			& (hsv[:, :, 0] >= low)
			& (hsv[:, :, 0] <= high)
		).astype(np.uint8)
		count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
		for component in range(1, count):
			x, y, width, height, area = (int(value) for value in stats[component])
			if not (0.55 * key_width <= width <= 1.6 * key_width and height >= 5 and area >= 35):
				continue
			center = x + width / 2
			note, key_center, key_kind = min(key_centers, key=lambda item: abs(item[1] - center))
			if (kind is None or kind == key_kind) and abs(key_center - center) <= key_width * 0.48:
				segments.append((note, hand, search_top + y, search_top + y + height - 1))
	return segments


@dataclass
class _Bar:
	hand: str
	top: int
	bottom: int
	frame: int = 0
	missing: int = 0


def _continues(previous: _Bar, top: int, bottom: int) -> bool:
	"""A falling bar's lower end never moves up; a following bar's lower end sits above the last bar.

	The upper end is ignored because the gap between touching bars is
	visible only intermittently, which merges them into one tall segment.
	"""
	return bottom >= previous.bottom - SWITCH_TOLERANCE_PIXELS


class WaterfallTracker:
	"""Turns per-frame bar segments into note on/off decisions.

	update() returns (action, note, hand, frame) tuples; the frame may lie one
	or two frames in the past because starts are confirmed by the next frame
	and ends by the absence of the bar in the following frames.
	"""

	def __init__(self, line: int) -> None:
		self.line = line
		self.active: dict[str, _Bar] = {}
		self.pending: dict[str, _Bar] = {}

	def update(self, frame: int, segments: list[Segment]) -> list[tuple[str, str, str, int]]:
		straddling = {
			note: (hand, top, bottom)
			for note, hand, top, bottom in segments
			if top < self.line <= bottom
		}
		actions: list[tuple[str, str, str, int]] = []
		for note in set(self.active) | set(self.pending) | set(straddling):
			current = straddling.get(note)
			active = self.active.get(note)
			pending = self.pending.get(note)
			if current is None:
				self.pending.pop(note, None)
				if active is not None:
					active.missing += 1
					if active.missing > MAX_DROPOUT_FRAMES:
						actions.append(("off", note, active.hand, frame - active.missing + 1))
						del self.active[note]
				continue
			hand, top, bottom = current
			if pending is not None:
				if _continues(pending, top, bottom):
					if active is not None:
						actions.append(("off", note, active.hand, pending.frame))
					actions.append(("on", note, hand, pending.frame))
					self.active[note] = _Bar(hand, top, bottom)
					del self.pending[note]
					continue
				del self.pending[note]
			if active is not None and _continues(active, top, bottom):
				active.top, active.bottom, active.missing = top, bottom, 0
			else:
				self.pending[note] = _Bar(hand, top, bottom, frame)
		actions.sort(key=lambda item: (item[3], item[0] != "off"))
		return actions

	def finish(self, frame: int) -> list[tuple[str, str, str, int]]:
		actions = [
			("off", note, bar.hand, frame - bar.missing) for note, bar in self.active.items()
		]
		self.active = {}
		self.pending = {}
		return actions
