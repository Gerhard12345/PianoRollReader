"""Tempo-adaptive beat tracking.

Performances drift in tempo, so snapping onsets to a rigid grid eventually
lands notes on the wrong beat. Here the grid follows the performance: the
slot of every onset is predicted from the tempo of a line fitted through the
most recent onsets, then all slots are re-decided from neighbours on both
sides. The result is a monotone warp from performance time to grid slots,
which can then be rendered at any fixed target tempo.
"""

from __future__ import annotations

import numpy as np

CLUSTER_GAP_SECONDS = 0.05
RATE_WINDOW = 16
RATE_MIN_SPAN_SECONDS = 1.5
MAX_RATE_DEVIATION = 0.2


def cluster_times(times: list[float], gap: float = CLUSTER_GAP_SECONDS) -> list[float]:
	"""Merge near-simultaneous onsets (chords) into one mean time."""
	ordered = sorted(times)
	if not ordered:
		return []
	clusters: list[float] = []
	current = [ordered[0]]
	for value in ordered[1:]:
		if value - current[-1] <= gap:
			current.append(value)
		else:
			clusters.append(sum(current) / len(current))
			current = [value]
	clusters.append(sum(current) / len(current))
	return clusters


def initial_slot_rate(clusters: list[float], hint_slots_per_second: float) -> float:
	"""Refine the tempo hint on the opening onsets (slots per second)."""
	opening = clusters[: min(len(clusters), 40)]
	if len(opening) < 3:
		return hint_slots_per_second
	offsets = np.array(opening) - opening[0]
	candidates = hint_slots_per_second * np.linspace(0.85, 1.15, 301)
	best_rate = hint_slots_per_second
	best_score = float("inf")
	for rate in candidates:
		position = offsets * rate
		score = float(np.mean(np.abs(position - np.round(position))))
		score += 0.02 * abs(rate / hint_slots_per_second - 1.0)
		if score < best_score:
			best_score = score
			best_rate = float(rate)
	return best_rate


class BeatWarp:
	"""Monotone map from performance seconds to (fractional) grid slots."""

	def __init__(self, anchors: list[tuple[float, int]], rate_start: float, rate_end: float) -> None:
		merged: dict[int, list[float]] = {}
		for seconds, slot in anchors:
			merged.setdefault(slot, []).append(seconds)
		slots = sorted(merged)
		self.slots = np.array(slots, dtype=float)
		self.seconds = np.array([sum(merged[slot]) / len(merged[slot]) for slot in slots])
		self.rate_start = rate_start
		self.rate_end = rate_end

	def __call__(self, seconds: float) -> float:
		if len(self.slots) == 1 or seconds <= self.seconds[0]:
			return float(self.slots[0] + (seconds - self.seconds[0]) * self.rate_start)
		if seconds >= self.seconds[-1]:
			return float(self.slots[-1] + (seconds - self.seconds[-1]) * self.rate_end)
		return float(np.interp(seconds, self.seconds, self.slots))


REFINE_HALF_WINDOW = 8
REFINE_ITERATIONS = 6


DEFAULT_AGILITY = 0.5


def agility_parameters(agility: float) -> tuple[int, float, int]:
	"""(rate window, max rate deviation, refine half window) for agility in 0..1.

	0 holds one fixed tempo, 1 follows the performance as fast as it can. The
	default 0.5 equals the tuned constants above.
	"""
	agility = min(1.0, max(0.0, agility))
	window = max(6, round(RATE_WINDOW * 2 * (1.0 - agility)))
	deviation = 2 * MAX_RATE_DEVIATION * agility
	half_window = max(2, window // 2)
	return window, deviation, half_window


def _neighbour_prediction(
	times: np.ndarray, slots: list[int], index: int, half_window: int
) -> float | None:
	"""Slot of one onset predicted by a line through its neighbours (excluding itself)."""
	count = len(slots)
	neighbours = [
		j
		for j in range(max(0, index - half_window), min(count, index + half_window + 1))
		if j != index
	]
	if len(neighbours) < 4:
		return None
	design = np.vstack([times[neighbours] - times[index], np.ones(len(neighbours))]).T
	(_, intercept), *_ = np.linalg.lstsq(
		design, np.array([slots[j] for j in neighbours], dtype=float), rcond=None
	)
	return float(intercept)


def refine_anchors(
	anchors: list[tuple[float, int]], half_window: int = REFINE_HALF_WINDOW
) -> list[tuple[float, int]]:
	"""Re-decide every slot from its neighbours on both sides.

	The forward pass can only look back, so after a pause or tempo change it
	may sit one slot off for a while. Here each onset is predicted by a
	line fitted through the surrounding onsets (excluding itself), which also
	sees the future and therefore corrects such lag.
	"""
	times = np.array([seconds for seconds, _ in anchors])
	slots = [slot for _, slot in anchors]
	count = len(slots)
	for _ in range(REFINE_ITERATIONS):
		updated = list(slots)
		for index in range(count):
			predicted = _neighbour_prediction(times, slots, index, half_window)
			if predicted is not None:
				updated[index] = int(round(predicted))
		for index in range(1, count):
			updated[index] = max(updated[index], updated[index - 1])
		if updated == slots:
			break
		slots = updated
	return list(zip(times.tolist(), slots))


def track_beats(
	onset_seconds: list[float],
	hint_bpm: float,
	slots_per_beat: float,
	agility: float = DEFAULT_AGILITY,
) -> tuple[BeatWarp, dict[str, object]]:
	"""Fit a tempo-following grid to the onsets.

	slots_per_beat: grid resolution (2 = eighth notes). The first onset is slot 0.
	agility: 0 keeps one fixed tempo, 1 follows tempo changes quickly.
	info["rates"] lists the tempo (slots per second) used at every onset cluster.
	"""
	window, max_deviation, half_window = agility_parameters(agility)
	clusters = cluster_times(onset_seconds)
	if not clusters:
		raise ValueError("No onsets to track")
	hint_rate = hint_bpm / 60.0 * slots_per_beat
	rate0 = initial_slot_rate(clusters, hint_rate)
	rate = rate0
	last_slot = 0
	previous = clusters[0]
	anchors = [(clusters[0], 0)]
	rates = [rate0]
	for seconds in clusters[1:]:
		recent = anchors[-window:]
		recent_times = np.array([moment for moment, _ in recent])
		if len(recent) >= 4 and recent_times[-1] - recent_times[0] > RATE_MIN_SPAN_SECONDS * (1.5 - agility):
			design = np.vstack(
				[recent_times - recent_times[-1], np.ones(len(recent))]
			).T
			(fitted, _), *_ = np.linalg.lstsq(
				design, np.array([slot for _, slot in recent], dtype=float), rcond=None
			)
			rate = min(rate0 * (1 + max_deviation), max(rate0 * (1 - max_deviation), fitted))
		else:
			rate = rate0
		slot = max(last_slot, int(round(last_slot + rate * (seconds - previous))))
		anchors.append((seconds, slot))
		rates.append(rate)
		last_slot = slot
		previous = seconds
	anchors = refine_anchors(anchors, half_window)
	times_array = np.array([seconds for seconds, _ in anchors])
	slots_list = [slot for _, slot in anchors]
	residuals = []
	for index in range(len(anchors)):
		predicted = _neighbour_prediction(times_array, slots_list, index, half_window)
		residuals.append(0.0 if predicted is None else abs(predicted - slots_list[index]))
	info: dict[str, object] = {
		"initial_bpm": rate0 * 60.0 / slots_per_beat,
		"final_bpm": rate * 60.0 / slots_per_beat,
		"onset_clusters": float(len(clusters)),
		"cluster_times": clusters,
		"rates": rates,
		"rate0": rate0,
		"max_deviation": max_deviation,
		"residuals": residuals,
	}
	return BeatWarp(anchors, rate0, rate), info
