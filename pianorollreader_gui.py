import argparse
import math
import threading
import time
from collections import Counter
from pathlib import Path

import cv2
import mido
import numpy as np
from music21 import note as music21_note
from music21 import stream as music21_stream
from music21 import note as music21_note
from music21 import stream as music21_stream
from matplotlib.backends.backend_qtagg import (
	FigureCanvasQTAgg,
	NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6.QtCore import QAbstractTableModel, QEvent, QModelIndex, QRectF, Qt, QTimer
from PySide6.QtGui import (
	QBrush,
	QColor,
	QFont,
	QFontMetrics,
	QImage,
	QIntValidator,
	QPainter,
	QPen,
	QPixmap,
)
from PySide6.QtWidgets import (
	QApplication,
	QCheckBox,
	QComboBox,
	QDialog,
	QDialogButtonBox,
	QFileDialog,
	QHBoxLayout,
	QHeaderView,
	QLabel,
	QLineEdit,
	QMainWindow,
	QMessageBox,
	QProgressBar,
	QPushButton,
	QScrollArea,
	QSizePolicy,
	QSlider,
	QSplitter,
	QTableView,
	QToolTip,
	QVBoxLayout,
	QWidget,
)

from pianorollreader import (
	DEFAULT_VIDEO,
	midi_note_name,
	read_video_frame,
	save_frame_zero_baseline,
)


NOTE_TO_MIDI = {midi_note_name(number): number for number in range(21, 109)}
TABLE_FLUSH_INTERVAL_MS = 100
TABLE_FLUSH_BATCH_SIZE = 40
SONG_REFERENCE_BPM = 154


def note_value_label(duration_seconds: float, bpm: int) -> str:
	duration_beats = duration_seconds * bpm / 60.0
	note_values = (
		(0.125, "1/32"),
		(0.25, "1/16"),
		(0.5, "1/8"),
		(1.0, "1/4"),
		(2.0, "1/2"),
		(4.0, "1"),
	)
	value, label = min(note_values, key=lambda pair: abs(pair[0] - duration_beats))
	return label if abs(value - duration_beats) <= value * 0.2 else f"~{duration_beats:.1f} beats"


def score_waterfall_notes(
	frame: np.ndarray,
	keys: list[dict[str, object]],
	key_width: float,
) -> dict[str, str]:
	frame_height, frame_width = frame.shape[:2]
	strike_y = round(frame_height * 0.75)
	search_top = max(0, strike_y - round(key_width * 2.5))
	search_bottom = min(frame_height, strike_y + 9)
	search_frame = frame[search_top:search_bottom, :]
	hsv = cv2.cvtColor(search_frame, cv2.COLOR_BGR2HSV)
	key_centers = [
		(
			str(key["note"]),
			float(key["bbox"][0]) + float(key["bbox"][2]) / 2,
		)
		for key in keys
	]
	waterfall_notes: dict[str, str] = {}
	for hand, hue_range in (("left", (85, 110)), ("right", (45, 80))):
		mask = (
			(hsv[:, :, 1] > 120)
			& (hsv[:, :, 2] > 100)
			& (hsv[:, :, 0] >= hue_range[0])
			& (hsv[:, :, 0] <= hue_range[1])
		).astype(np.uint8)
		component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
		for component in range(1, component_count):
			x, y, width, height, area = (int(value) for value in stats[component])
			if not (
				0.55 * key_width <= width <= 1.6 * key_width
				and height >= 5
				and area >= 35
				and y + height - 1 >= strike_y - 18 - search_top
				and y <= strike_y + 8 - search_top
			):
				continue
			center_x = x + width / 2
			note, key_center = min(
				key_centers, key=lambda item: abs(item[1] - center_x)
			)
			if abs(key_center - center_x) <= key_width * 0.48:
				waterfall_notes[note] = hand
	return waterfall_notes


def combine_detection_and_waterfall(
	events: list[dict[str, object]],
	waterfall_events: list[dict[str, object]],
	fps: float,
	match_window_seconds: float = 0.12,
) -> list[dict[str, object]]:
	combined = [event.copy() for event in events]
	waterfall_by_note: dict[str, list[dict[str, object]]] = {}
	for event in waterfall_events:
		waterfall_by_note.setdefault(str(event["note"]), []).append(event)
	used_waterfall_ids: set[int] = set()
	match_window_frames = round(match_window_seconds * fps)
	for event in combined:
		note = str(event["note"])
		detected_onset = int(event["onset_frame"])
		candidates = [
			candidate
			for candidate in waterfall_by_note.get(note, [])
			if int(candidate["event_id"]) not in used_waterfall_ids
		]
		if not candidates:
			continue
		closest = min(
			candidates,
			key=lambda candidate: abs(
				int(candidate["onset_frame"]) - detected_onset
			),
		)
		waterfall_onset = int(closest["onset_frame"])
		if abs(waterfall_onset - detected_onset) > match_window_frames:
			continue
		combined_onset = round((detected_onset + waterfall_onset) / 2)
		event["waterfall_onset_frame"] = waterfall_onset
		event["combined_onset_frame"] = combined_onset
		event["onset_frame"] = combined_onset
		event["video_onset_seconds"] = combined_onset / fps
		used_waterfall_ids.add(int(closest["event_id"]))

	for event in waterfall_events:
		if int(event["event_id"]) in used_waterfall_ids:
			continue
		combined_event = event.copy()
		combined_event["event_id"] = len(combined) + 1
		combined_event["combined_onset_frame"] = int(event["onset_frame"])
		combined_event["video_onset_seconds"] = int(event["onset_frame"]) / fps
		offset_frame = event["offset_frame"]
		if offset_frame is not None:
			combined_event["video_offset_seconds"] = int(offset_frame) / fps
			combined_event["duration_seconds"] = (
				int(offset_frame) - int(event["onset_frame"])
			) / fps
		combined.append(combined_event)
	return combined


def normalize_events_to_reference(
	events: list[dict[str, object]],
	waterfall_events: list[dict[str, object]],
	fps: float,
	bpm: int = SONG_REFERENCE_BPM,
	beat_length: float = 0.5,
	current_frame: int = 0,
	origin_frame: int | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
	refined_events = combine_detection_and_waterfall(
		events, waterfall_events, fps
	)

	quantized_events, settings, _ = quantize_events_manually(
		refined_events, fps, bpm, beat_length, current_frame, origin_frame
	)
	for event in quantized_events:
		event["normalized_onset_seconds"] = float(event["midi_onset_seconds"])
		event["normalized_offset_seconds"] = float(event["midi_offset_seconds"])
		event["normalized_onset_grid_index"] = int(event["midi_onset_grid_index"])
		event["normalized_offset_grid_index"] = int(event["midi_offset_grid_index"])
		event["normalized_grid_seconds"] = float(event["midi_grid_seconds"])
		event["normalized_origin_seconds"] = float(settings["origin_seconds"])
		for field in (
			"midi_onset_seconds",
			"midi_offset_seconds",
			"midi_onset_grid_index",
			"midi_offset_grid_index",
			"midi_grid_seconds",
		):
			event.pop(field, None)
	return quantized_events, settings


def estimate_bpm(events: list[dict[str, object]], fps: float) -> int | None:
	return estimate_bpm_pair(events, fps)[0]


def estimate_bpm_pair(
	events: list[dict[str, object]], fps: float
) -> tuple[int | None, int | None]:
	onsets = sorted({int(event["onset_frame"]) for event in events})
	if len(onsets) < 3 or fps <= 0:
		return None, None

	intervals = [
		(right - left) / fps
		for left, right in zip(onsets, onsets[1:])
		if right > left
	]
	if len(intervals) < 2:
		return None, None

	def score_tempo(bpm: int) -> float:
		subdivision = 60.0 / bpm / 4
		phase_errors = [
			abs(interval / subdivision - round(interval / subdivision))
			for interval in intervals
		]
		fit_error = sum(phase_errors) / len(phase_errors)
		tempo_prior = 0.015 * abs(np.log2(bpm / 120.0))
		return fit_error + tempo_prior

	best_bpm = min(range(40, 241), key=score_tempo)
	best_score = score_tempo(best_bpm)
	if best_bpm <= 100 and best_bpm * 2 <= 240:
		double_bpm = best_bpm * 2
		if score_tempo(double_bpm) <= best_score + 0.12:
			return double_bpm, best_bpm
	return best_bpm, None


def quantize_events_with_music21(
	events: list[dict[str, object]],
	fps: float,
	bpm: int,
	beat_length: float,
	current_frame: int,
	origin_frame: int | None = None,
) -> list[dict[str, object]]:
	if not events:
		return []

	quarter_lengths_per_second = bpm / 60.0
	first_onset_frame = (
		origin_frame
		if origin_frame is not None
		else min(int(event["onset_frame"]) for event in events)
	)
	origin_seconds = first_onset_frame / fps
	divisor = max(1, round(1.0 / beat_length))
	grid_seconds = 60.0 / bpm * beat_length
	score = music21_stream.Stream()
	for event in events:
		event_id = int(event["event_id"])
		note_id = f"piano_roll_event_{event_id}"
		onset_seconds = int(event["onset_frame"]) / fps
		offset_frame = event["offset_frame"]
		if offset_frame is None:
			offset_frame = max(current_frame + 1, int(event["onset_frame"]) + 1)
		offset_seconds = int(offset_frame) / fps
		onset_quarter_length = max(
			0.0, (onset_seconds - origin_seconds) * quarter_lengths_per_second
		)
		duration_quarter_length = max(
			(1.0 / fps) * quarter_lengths_per_second,
			(offset_seconds - onset_seconds) * quarter_lengths_per_second,
		)
		music21_item = music21_note.Note(
			int(event["midi_note"]), quarterLength=duration_quarter_length
		)
		music21_item.id = note_id
		score.insert(onset_quarter_length, music21_item)

	quantized_score = score.quantize(
		( divisor, ),
		processOffsets=True,
		processDurations=True,
		inPlace=False,
	)
	quantized_by_id = {
		music21_item.id: music21_item
		for music21_item in quantized_score.notes
	}
	result = [event.copy() for event in events]
	for event in result:
		music21_item = quantized_by_id[f"piano_roll_event_{event['event_id']}"]
		onset_grid_index = round(float(music21_item.offset) * divisor)
		onset_seconds = (
			origin_seconds
			+ float(music21_item.offset) / quarter_lengths_per_second
		)
		duration_quarter_length = max(
			1.0 / divisor,
			float(music21_item.duration.quarterLength),
		)
		event["music21_onset_seconds"] = onset_seconds
		event["music21_onset_grid_index"] = onset_grid_index
		event["music21_grid_seconds"] = grid_seconds
		event["music21_origin_seconds"] = origin_seconds
		event["music21_offset_seconds"] = (
			onset_seconds + duration_quarter_length / quarter_lengths_per_second
		)
		event["music21_offset_grid_index"] = onset_grid_index + round(
			duration_quarter_length * divisor
		)
	return result


def quantize_events_manually(
	events: list[dict[str, object]],
	fps: float,
	bpm: int,
	beat_length: float,
	current_frame: int,
	origin_frame: int | None = None,
) -> tuple[list[dict[str, object]], dict[str, object], int]:
	grid_seconds = 60.0 / bpm * beat_length
	first_onset_frame = (
		origin_frame
		if origin_frame is not None
		else min((int(event["onset_frame"]) for event in events), default=0)
	)
	origin_seconds = first_onset_frame / fps
	quantized_events = [event.copy() for event in events]
	grid_onset_sources: dict[int, set[int]] = {}
	for event in quantized_events:
		onset_seconds = int(event["onset_frame"]) / fps
		onset_grid = max(0, round((onset_seconds - origin_seconds) / grid_seconds))
		quantized_onset = origin_seconds + onset_grid * grid_seconds
		event["midi_onset_grid_index"] = onset_grid
		event["midi_onset_seconds"] = quantized_onset
		grid_onset_sources.setdefault(onset_grid, set()).add(int(event["onset_frame"]))
		offset_frame = event["offset_frame"]
		if offset_frame is None:
			offset_frame = max(current_frame + 1, int(event["onset_frame"]) + 1)
		offset_seconds = int(offset_frame) / fps
		offset_grid = round((offset_seconds - origin_seconds) / grid_seconds)
		if offset_grid <= onset_grid:
			offset_grid = onset_grid + 1
		event["midi_duration_grid_units"] = offset_grid - onset_grid
		event["midi_offset_grid_index"] = offset_grid
		event["midi_grid_seconds"] = grid_seconds
		event["midi_offset_seconds"] = origin_seconds + offset_grid * grid_seconds

	collapsed_onsets = sum(
		len(source_frames) - 1
		for source_frames in grid_onset_sources.values()
		if len(source_frames) > 1
	)
	settings = {
		"bpm": bpm,
		"beat_length": beat_length,
		"grid_seconds": grid_seconds,
		"origin_frame": first_onset_frame,
		"origin_seconds": origin_seconds,
	}
	return quantized_events, settings, collapsed_onsets


def quantize_events_music21(
	events: list[dict[str, object]],
	fps: float,
	bpm: int,
	beat_length: float,
	current_frame: int,
) -> list[dict[str, object]]:
	if not events:
		return []

	quarter_lengths_per_second = bpm / 60.0
	origin_seconds = min(int(event["onset_frame"]) for event in events) / fps
	divisor = max(1, round(1.0 / beat_length))
	score = music21_stream.Stream()
	for event in events:
		event_id = int(event["event_id"])
		music21_id = f"piano_roll_event_{event_id}"
		onset_seconds = int(event["onset_frame"]) / fps
		offset_frame = event["offset_frame"]
		if offset_frame is None:
			offset_frame = max(current_frame + 1, int(event["onset_frame"]) + 1)
		offset_seconds = int(offset_frame) / fps
		onset_quarter_length = max(
			0.0, (onset_seconds - origin_seconds) * quarter_lengths_per_second
		)
		duration_quarter_length = max(
			(1.0 / fps) * quarter_lengths_per_second,
			(offset_seconds - onset_seconds) * quarter_lengths_per_second,
		)
		music21_item = music21_note.Note(
			int(event["midi_note"]), quarterLength=duration_quarter_length
		)
		music21_item.id = music21_id
		score.insert(onset_quarter_length, music21_item)

	quantized_score = score.quantize(
		(divisor,),
		processOffsets=True,
		processDurations=True,
		inPlace=False,
	)
	quantized_by_id = {
		music21_item.id: music21_item
		for music21_item in quantized_score.notes
	}
	quantized_events = [event.copy() for event in events]
	for event in quantized_events:
		music21_id = f"piano_roll_event_{event['event_id']}"
		music21_item = quantized_by_id[music21_id]
		onset_seconds = (
			origin_seconds
			+ float(music21_item.offset) / quarter_lengths_per_second
		)
		offset_seconds = (
			onset_seconds
			+ float(music21_item.duration.quarterLength)
			/ quarter_lengths_per_second
		)
		event["midi21_onset_seconds"] = onset_seconds
		event["midi21_offset_seconds"] = offset_seconds
	return quantized_events


class KeyHitTableModel(QAbstractTableModel):
	HEADERS = (
		"Key",
		"Hand",
		"MIDI",
		"Onset frame",
		"Start (s)",
		"Offset frame",
		"End (s)",
		"Length (s)",
	)

	def __init__(self, parent=None) -> None:
		super().__init__(parent)
		self.rows: list[dict[str, object]] = []
		self.row_by_event_id: dict[int, int] = {}

	def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
		return 0 if parent.isValid() else len(self.rows)

	def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
		return 0 if parent.isValid() else len(self.HEADERS)

	def headerData(self, section: int, orientation, role=Qt.ItemDataRole.DisplayRole):
		if (
			orientation == Qt.Orientation.Horizontal
			and role == Qt.ItemDataRole.DisplayRole
			and 0 <= section < len(self.HEADERS)
		):
			return self.HEADERS[section]
		return None

	def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
		if not index.isValid():
			return None
		event = self.rows[index.row()]
		if role == Qt.ItemDataRole.ForegroundRole:
			color = "#123b75" if event["hand"] == "left" else "#761c24"
			return QBrush(QColor(color))
		if role == Qt.ItemDataRole.TextAlignmentRole:
			return Qt.AlignmentFlag.AlignCenter
		if role != Qt.ItemDataRole.DisplayRole:
			return None

		offset_frame = event["offset_frame"]
		offset_seconds = event["video_offset_seconds"]
		duration = event["duration_seconds"]
		values = (
			str(event["note"]),
			str(event["hand"]).title(),
			str(event["midi_note"]),
			str(event["onset_frame"]),
			f"{float(event['video_onset_seconds']):.3f}",
			"Active" if offset_frame is None else str(offset_frame),
			"-" if offset_seconds is None else f"{float(offset_seconds):.3f}",
			"-" if duration is None else f"{float(duration):.3f}",
		)
		return values[index.column()]

	def upsert_event(self, event: dict[str, object]) -> None:
		event_id = int(event["event_id"])
		row = self.row_by_event_id.get(event_id)
		if row is None:
			row = len(self.rows)
			self.beginInsertRows(QModelIndex(), row, row)
			self.rows.append(event)
			self.row_by_event_id[event_id] = row
			self.endInsertRows()
			return

		self.rows[row] = event
		self.dataChanged.emit(
			self.index(row, 0),
			self.index(row, len(self.HEADERS) - 1),
			[Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ForegroundRole],
		)

	def clear(self) -> None:
		self.beginResetModel()
		self.rows.clear()
		self.row_by_event_id.clear()
		self.endResetModel()


class PianoRollTimeline(QWidget):
	def __init__(self, parent=None) -> None:
		super().__init__(parent)
		self.setMinimumSize(640, 360)
		self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
		self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
		self.setMouseTracking(True)
		self.notes: list[str] = []
		self.raw_bars: list[dict[str, object]] = []
		self.corrected_bars: list[dict[str, object]] = []
		self.window = (0.0, 1.0)
		self.unit_scale = 1.0
		self.unit_name = "seconds"
		self.origin = 0.0
		self.bpm = 120
		self.detected_alpha = 0.35
		self.corrected_alpha = 0.75
		self.visible_sources: set[str] = set()
		self.visible_raw_bars: list[dict[str, object]] = []
		self.visible_corrected_bars: list[dict[str, object]] = []
		self._timeline_initialized = False
		self.reference_label = "Detection"
		self.playhead_visible = True
		self.preview: dict[str, object] | None = None
		self.playhead: float | None = None
		self.dragging = False
		self.panning = False
		self.pan_last_x = 0.0
		self.on_drag_start = None
		self.on_drag_move = None
		self.on_drag_end = None
		self.on_zoom = None
		self.on_horizontal_scroll = None
		self.on_pan = None
		self.on_add_note = None
		self.on_remove_note = None
		self.on_select_note = None
		self.selected_bar: tuple[str, int] | None = None
		self.hovered_bar: tuple[str, int] | None = None
		self.add_note_mode = False
		self.interaction_enabled = True

	def set_interaction_enabled(self, enabled: bool) -> None:
		self.interaction_enabled = enabled
		if not enabled:
			self.dragging = False
			self.panning = False
			self.add_note_mode = False
			self.setCursor(Qt.CursorShape.ArrowCursor)
			QToolTip.hideText()

	def set_timeline_data(
		self,
		notes: list[str],
		raw_bars: list[dict[str, object]],
		corrected_bars: list[dict[str, object]],
		window: tuple[float, float],
		unit_scale: float,
		unit_name: str,
		origin: float,
		bpm: int,
		detected_alpha: float,
		corrected_alpha: float,
		visible_sources: set[str],
		reference_label: str,
	) -> None:
		self.notes = notes
		self.setMinimumHeight(max(480, len(notes) * 28 + 76))
		self.raw_bars = raw_bars
		self.corrected_bars = corrected_bars
		self.window = window
		self.unit_scale = unit_scale
		self.unit_name = unit_name
		self.origin = origin
		self.bpm = bpm
		self.detected_alpha = detected_alpha
		self.corrected_alpha = corrected_alpha
		self.visible_sources = visible_sources
		self.reference_label = reference_label
		self.preview = None
		self.set_window(window)

	def set_window(self, window: tuple[float, float]) -> None:
		self.window = window
		minimum, maximum = window
		self.visible_raw_bars = [
			bar for bar in self.raw_bars
			if float(bar["end"]) >= minimum and float(bar["start"]) <= maximum
		]
		self.visible_corrected_bars = [
			bar for bar in self.corrected_bars
			if float(bar["end"]) >= minimum and float(bar["start"]) <= maximum
		]
		self.update()

	@property
	def chart_rect(self):
		return self.rect().adjusted(86, 38, -14, -38)

	def _x_pixel(self, value: float) -> float:
		chart = self.chart_rect
		minimum, maximum = self.window
		return chart.left() + (value - minimum) / max(1e-9, maximum - minimum) * chart.width()

	def _x_value(self, pixel: float) -> float:
		chart = self.chart_rect
		minimum, maximum = self.window
		return minimum + (pixel - chart.left()) / max(1.0, chart.width()) * (maximum - minimum)

	def _bar_rect(self, bar: dict[str, object], chart) -> object:
		row_height = chart.height() / max(1, len(self.notes))
		base_y = chart.bottom() - (float(bar["row"]) + 0.5) * row_height
		if bar.get("source") in ("detected", "normalized", "waterfall-reference", "combined-reference") and not self.visible_corrected_bars:
			center_y = base_y
			bar_height = max(4.0, min(18.0, row_height * 0.72))
		else:
			lane = -1 if bar.get("source") in ("detected", "normalized", "waterfall-reference", "combined-reference") else (0 if bar.get("source") == "manual" else 1)
			lane_spacing = min(row_height * 0.24, 12.0)
			center_y = base_y - lane * lane_spacing
			bar_height = max(8.0, min(12.0, row_height * 0.34))
		left = self._x_pixel(float(bar["start"]))
		right = self._x_pixel(float(bar["end"]))
		return QRectF(left, center_y - bar_height / 2, max(1.0, right - left), bar_height)

	def _bar_description(self, bar: dict[str, object]) -> str:
		note_name = self.notes[int(bar["row"])]
		duration_seconds = (float(bar["end"]) - float(bar["start"])) / self.unit_scale
		value_label = note_value_label(duration_seconds, self.bpm)
		return f"{note_name} - {value_label} note"

	def _draw_bar_label(self, painter: QPainter, bar: dict[str, object], rect) -> None:
		font = QFont(painter.font())
		font.setPointSizeF(5.0)
		font.setBold(True)
		metrics = QFontMetrics(font)
		label = f"{self.notes[int(bar['row'])]} {note_value_label((float(bar['end']) - float(bar['start'])) / self.unit_scale, self.bpm)}"
		if rect.height() < metrics.height() + 1 or rect.width() < metrics.horizontalAdvance(label) + 4:
			return
		painter.save()
		painter.setClipRect(rect)
		painter.setFont(font)
		painter.setPen(QColor("#0b1218"))
		painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)
		painter.restore()

	def _visible_bars(self, bars: list[dict[str, object]]) -> list[dict[str, object]]:
		if bars is self.raw_bars:
			return self.visible_raw_bars
		if bars is self.corrected_bars:
			return self.visible_corrected_bars
		return bars

	def paintEvent(self, event) -> None:
		painter = QPainter(self)
		dirty = QRectF(event.rect())
		painter.fillRect(event.rect(), QColor("#0b1218"))
		painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
		chart = self.chart_rect
		painter.fillRect(dirty.intersected(QRectF(chart)), QColor("#101a22"))
		painter.setPen(QPen(QColor("#334550"), 1))
		painter.drawRect(chart)
		if not self.notes:
			painter.setPen(QColor("#53616d"))
			painter.drawText(chart, Qt.AlignmentFlag.AlignCenter, "No key hits to display")
			return

		quarter = 60.0 / max(1, self.bpm) * self.unit_scale
		sixteenth = quarter / 4
		first_tick = math.floor((self.window[0] - self.origin) / sixteenth)
		last_tick = math.ceil((self.window[1] - self.origin) / sixteenth)
		font_metrics = QFontMetrics(painter.font())
		for tick in range(first_tick, last_tick + 1):
			value = self.origin + tick * sixteenth
			measure_line = tick % 16 == 0
			beat_line = tick % 4 == 0
			x = round(self._x_pixel(value))
			if measure_line or beat_line:
				if not dirty.left() - 48 <= x <= dirty.right() + 48:
					continue
				label = f"{value / self.unit_scale:.2f}" if self.unit_name == "seconds" else f"{round(value)}"
				painter.setPen(QColor("#ffe49a" if measure_line else "#c2d0d7"))
				painter.drawText(round(x - 24), round(chart.bottom() + 18), 48, font_metrics.height(), Qt.AlignmentFlag.AlignHCenter, label)

		row_height = chart.height() / max(1, len(self.notes))
		if row_height >= 8:
			for row in range(len(self.notes)):
				row_top = chart.bottom() - (row + 1) * row_height
				row_bottom = chart.bottom() - row * row_height
				if row_bottom < dirty.top() or row_top > dirty.bottom():
					continue
				if row % 2 == 0:
					painter.fillRect(QRectF(chart.left(), row_top, chart.width(), row_height), QColor("#14212a"))
				painter.setPen(QPen(QColor("#253640"), 0.6))
				painter.drawLine(round(chart.left()), round(chart.bottom() - row * row_height), round(chart.right()), round(chart.bottom() - row * row_height))
			painter.setPen(QColor("#c2d0d7"))
			for row, note in enumerate(self.notes):
				y = chart.bottom() - (row + 0.5) * row_height + font_metrics.ascent() / 2
				if y < dirty.top() or y > dirty.bottom():
					continue
				painter.drawText(4, round(y), 74, font_metrics.height(), Qt.AlignmentFlag.AlignRight, note)

		for tick in range(first_tick, last_tick + 1):
			value = self.origin + tick * sixteenth
			measure_line = tick % 16 == 0
			beat_line = tick % 4 == 0
			x = round(self._x_pixel(value))
			if x < dirty.left() - 2 or x > dirty.right() + 2:
				continue
			line_color = "#f3c75f" if measure_line else "#657e8c" if beat_line else "#24343e"
			line_width = 2.4 if measure_line else 1.25 if beat_line else 0.6
			painter.setPen(QPen(QColor(line_color), line_width))
			painter.drawLine(x, round(chart.top()), x, round(chart.bottom()))

		for bar in self._visible_bars(self.raw_bars):
			rect = self._bar_rect(bar, chart)
			if not rect.intersects(dirty):
				continue
			color = QColor(str(bar["color"]))
			color.setAlphaF(max(0.72, self.detected_alpha))
			painter.fillRect(rect, color)
			self._draw_bar_label(painter, bar, rect)
		for bar in self._visible_bars(self.corrected_bars):
			rect = self._bar_rect(bar, chart)
			if not rect.intersects(dirty):
				continue
			color = QColor(str(bar["color"]))
			color.setAlphaF(max(0.88, self.corrected_alpha))
			painter.fillRect(rect, color)
			painter.setPen(QPen(QColor("#e4edf1"), 0.7))
			painter.drawRect(rect)
			if self.selected_bar == (str(bar["source"]), int(bar["event_id"])):
				painter.setPen(QPen(QColor("#ffffff"), 2.0))
				painter.drawRect(rect.adjusted(-2, -2, 2, 2))
			painter.save()
			painter.setClipRect(rect)
			painter.setPen(QPen(QColor("#0b1218"), 0.75))
			start_x = rect.left() - rect.height()
			while start_x < rect.right() + rect.height():
				if bar.get("source") == "manual":
					painter.drawLine(round(start_x), round(rect.bottom()), round(start_x + rect.height()), round(rect.top()))
				else:
					painter.drawLine(round(start_x), round(rect.top()), round(start_x + rect.height()), round(rect.bottom()))
				start_x += 5
			painter.restore()
			self._draw_bar_label(painter, bar, rect)
		if self.preview is not None:
			rect = self._bar_rect(self.preview, chart)
			if rect.intersects(dirty):
				painter.setPen(QPen(QColor("#101820"), 1.2, Qt.PenStyle.DashLine))
				painter.setBrush(QColor(str(self.preview["color"])))
				painter.drawRect(rect)
		if self.playhead_visible and self.playhead is not None:
			x = round(self._x_pixel(self.playhead))
			if dirty.left() - 2 <= x <= dirty.right() + 2:
				painter.setPen(QPen(QColor("#ff4b45"), 2.0))
				painter.drawLine(x, round(chart.top()), x, round(chart.bottom()))
		if dirty.top() <= 32:
			legend_x = round(chart.left())
			legend_entries = [(self.reference_label, "#28a9bd")]
			legend_entries.extend((("Manual", "#28a9bd"), ("music21", "#42a45f")))
			for label, color in legend_entries:
				if label != "Detected" and label.lower() not in self.visible_sources:
					if label not in (self.reference_label, "Waterfall", "Crossings"):
						continue
				if label == self.reference_label and not self.visible_raw_bars:
					continue
				legend_width = QFontMetrics(painter.font()).horizontalAdvance(label) + 30
				painter.setPen(QPen(QColor("#52636d"), 0.8))
				painter.setBrush(QColor("#1b2a33"))
				painter.drawRoundedRect(
					QRectF(legend_x - 5, 5, legend_width, 24), 3, 3
				)
				painter.fillRect(legend_x, 12, 14, 10, QColor(color))
				painter.setPen(QColor("#f0f5f7"))
				painter.drawText(legend_x + 19, 21, label)
				legend_x += legend_width + 7
		if dirty.bottom() >= self.height() - 24:
			painter.setPen(QColor("#a9bac4"))
			painter.drawText(round(chart.left()), self.height() - 6, chart.width(), 18, Qt.AlignmentFlag.AlignCenter, f"Time ({self.unit_name})")

	def mousePressEvent(self, event) -> None:
		if not self.interaction_enabled:
			event.accept()
			return
		if event.button() != Qt.MouseButton.LeftButton:
			return
		self.setFocus(Qt.FocusReason.MouseFocusReason)
		position = event.position()
		if self.add_note_mode and self.on_add_note is not None:
			chart = self.chart_rect
			if QRectF(chart).contains(position) and self.notes:
				row_height = chart.height() / len(self.notes)
				row = int((chart.bottom() - position.y()) / row_height)
				row = max(0, min(len(self.notes) - 1, row))
				self.add_note_mode = False
				self.on_add_note(self.notes[row], self._x_value(position.x()))
				self.setCursor(Qt.CursorShape.ArrowCursor)
				event.accept()
				return
		candidate = None
		candidate_distance = 9.0
		for bar in self._visible_bars(self.corrected_bars):
			rect = self._bar_rect(bar, self.chart_rect)
			if not rect.top() - 3 <= position.y() <= rect.bottom() + 3:
				continue
			for side, edge in (("onset", rect.left()), ("offset", rect.right())):
				distance = abs(position.x() - edge)
				if distance < candidate_distance:
					candidate = (str(bar["source"]), int(bar["event_id"]), side)
					candidate_distance = distance
		if candidate is None:
			for bar in self._visible_bars(self.corrected_bars):
				if self._bar_rect(bar, self.chart_rect).contains(position):
					self.selected_bar = (str(bar["source"]), int(bar["event_id"]))
					if self.on_select_note is not None:
						self.on_select_note(*self.selected_bar)
					self.update()
					event.accept()
					return
		if candidate is not None and self.on_drag_start is not None:
			self.dragging = bool(self.on_drag_start(*candidate))
			if self.dragging:
				self.on_drag_move(self._x_value(position.x()))
				event.accept()
				return
		if QRectF(self.chart_rect).contains(position):
			self.panning = True
			self.pan_last_x = position.x()
			self.setCursor(Qt.CursorShape.ClosedHandCursor)
			event.accept()
			return
		event.ignore()

	def mouseMoveEvent(self, event) -> None:
		if not self.interaction_enabled:
			return
		if self.dragging and self.on_drag_move is not None:
			self.on_drag_move(self._x_value(event.position().x()))
			return
		if self.panning:
			position = event.position()
			delta_pixels = position.x() - self.pan_last_x
			self.pan_last_x = position.x()
			if self.on_pan is not None:
				window_span = self.window[1] - self.window[0]
				self.on_pan(-delta_pixels / max(1.0, self.chart_rect.width()) * window_span)
			return
		position = event.position()
		hovered = None
		for bar in reversed(self._visible_bars(self.corrected_bars) + self._visible_bars(self.raw_bars)):
			if self._bar_rect(bar, self.chart_rect).contains(position):
				hovered = bar
				break
		if hovered is not None:
			bar_key = (str(hovered["source"]), int(hovered["event_id"]))
			if bar_key != self.hovered_bar:
				QToolTip.showText(
					event.globalPosition().toPoint(),
					self._bar_description(hovered),
					self,
				)
				self.hovered_bar = bar_key
		else:
			self.hovered_bar = None
			QToolTip.hideText()
		near_edge = any(
			abs(position.x() - edge) < 9
			and rect.top() - 3 <= position.y() <= rect.bottom() + 3
			for bar in self._visible_bars(self.corrected_bars)
			for rect in (self._bar_rect(bar, self.chart_rect),)
			for edge in (rect.left(), rect.right())
		)
		self.setCursor(Qt.CursorShape.SizeHorCursor if near_edge else Qt.CursorShape.ArrowCursor)

	def mouseDoubleClickEvent(self, event) -> None:
		if not self.interaction_enabled:
			event.accept()
			return
		if event.button() != Qt.MouseButton.LeftButton or self.on_add_note is None:
			return
		position = event.position()
		chart = self.chart_rect
		if not QRectF(chart).contains(position) or not self.notes:
			return
		row_height = chart.height() / len(self.notes)
		row = int((chart.bottom() - position.y()) / row_height)
		row = max(0, min(len(self.notes) - 1, row))
		self.on_add_note(self.notes[row], self._x_value(position.x()))
		event.accept()

	def wheelEvent(self, event) -> None:
		if not self.interaction_enabled:
			event.accept()
			return
		delta = event.angleDelta().y()
		if not delta:
			event.ignore()
			return
		if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
			if self.on_zoom is not None:
				self.on_zoom(1 if delta > 0 else -1, self._x_value(event.position().x()))
		elif event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
			if self.on_horizontal_scroll is not None:
				self.on_horizontal_scroll(-1 if delta > 0 else 1)
		else:
			event.ignore()
			return
		event.accept()

	def keyPressEvent(self, event) -> None:
		if not self.interaction_enabled:
			event.accept()
			return
		if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
			if self.selected_bar is not None and self.on_remove_note is not None:
				self.on_remove_note(*self.selected_bar)
				event.accept()
				return
		if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right) and self.on_horizontal_scroll is not None:
			self.on_horizontal_scroll(-1 if event.key() == Qt.Key.Key_Left else 1)
			event.accept()
			return
		super().keyPressEvent(event)

	def mouseReleaseEvent(self, event) -> None:
		if not self.interaction_enabled:
			event.accept()
			return
		if self.panning and event.button() == Qt.MouseButton.LeftButton:
			self.panning = False
			self.setCursor(Qt.CursorShape.ArrowCursor)
			event.accept()
			return
		if self.dragging and event.button() == Qt.MouseButton.LeftButton:
			self.dragging = False
			if self.on_drag_end is not None:
				self.on_drag_end(self._x_value(event.position().x()))
			event.accept()

	def set_drag_preview(self, preview: dict[str, object] | None) -> None:
		self.preview = preview
		self.update()

	def set_playhead(self, value: float | None) -> None:
		old_value = self.playhead
		self.playhead = value
		chart = self.chart_rect
		for position in (old_value, value):
			if position is not None:
				x = round(self._x_pixel(position))
				self.update(x - 3, round(chart.top()), 7, round(chart.height()))

	def set_playhead_visible(self, visible: bool) -> None:
		if self.playhead_visible == visible:
			return
		self.playhead_visible = visible
		self.update(self.chart_rect)


class PlaybackScheduler(threading.Thread):
	def __init__(self, output, notes, wall_start: float, media_start: float) -> None:
		super().__init__(daemon=True)
		self.output = output
		self.notes = notes
		self.wall_start = wall_start
		self.media_start = media_start
		self.stop_event = threading.Event()
		self.completed_event = threading.Event()
		self.close_output_on_stop = True

	def run(self) -> None:
		active_notes: set[tuple[int, int]] = set()
		try:
			for event_time, is_on, channel, note_number in self.notes:
				deadline = self.wall_start + max(0.0, event_time - self.media_start)
				if self.stop_event.wait(max(0.0, deadline - time.perf_counter())):
					break
				self.output.send(mido.Message(
					"note_on" if is_on else "note_off",
					channel=channel,
					note=note_number,
					velocity=85 if is_on else 0,
				))
				key = (channel, note_number)
				if is_on:
					active_notes.add(key)
				else:
					active_notes.discard(key)
		finally:
			for channel, note_number in active_notes:
				self.output.send(mido.Message("note_off", channel=channel, note=note_number, velocity=0))
			if self.close_output_on_stop:
				self.output.close()
			self.completed_event.set()


class DurationAnalysisDialog(QDialog):
	def __init__(
		self,
		events: list[dict[str, object]],
		fps: float,
		parent=None,
		on_accept=None,
		on_music21_accept=None,
		current_frame: int = 0,
		view_mode: str = "timeline",
		waterfall_events: list[dict[str, object]] | None = None,
		on_normalized_accept=None,
	) -> None:
		super().__init__(parent)
		self.view_mode = view_mode
		self.setWindowTitle(
			"Key Hit Statistics" if view_mode == "statistics" else "Hit Timeline"
		)
		self.resize(1120, 760)
		self.setMinimumSize(1000, 620)
		self.events = events
		self.fps = fps
		self.current_frame = current_frame
		self.timeline_window_start_seconds: float | None = None
		self.timeline_max_start_seconds = 0.0
		self.estimated_bpm, self.alternative_bpm = estimate_bpm_pair(events, fps)
		self.on_accept = on_accept
		self.on_music21_accept = on_music21_accept
		self.on_normalized_accept = on_normalized_accept
		self.waterfall_events = waterfall_events or []
		self.normalized_detection_events, self.normalization_detection_settings = normalize_events_to_reference(
			events,
			[],
			fps,
			SONG_REFERENCE_BPM,
			0.5,
			current_frame,
		)
		waterfall_origin = min(
			(int(event["onset_frame"]) for event in self.waterfall_events),
			default=None,
		)
		self.normalized_waterfall_events, self.normalization_waterfall_settings = normalize_events_to_reference(
			self.waterfall_events,
			[],
			fps,
			SONG_REFERENCE_BPM,
			0.5,
			current_frame,
			waterfall_origin,
		)
		self.normalized_events, self.normalization_settings = normalize_events_to_reference(
			events,
			self.waterfall_events,
			fps,
			SONG_REFERENCE_BPM,
			0.5,
			current_frame,
		)
		self.quantized_events: list[dict[str, object]] | None = None
		self.editable_events: list[dict[str, object]] | None = None
		self.selected_editable_event_id: int | None = None
		self.detected_alpha = 0.82
		self.corrected_alpha = 0.90
		self.music21_quantized_events: list[dict[str, object]] | None = None
		self.quantization_settings: dict[str, object] | None = None
		self.music21_quantization_settings: dict[str, object] | None = None
		self.plot_quantized_preview = False
		self.timeline_drag: dict[str, object] | None = None
		self.playback_timer = QTimer(self)
		self.playback_timer.setInterval(20)
		self.playback_timer.timeout.connect(self._playback_tick)
		self.midi_output = None
		self.playback_notes: list[tuple[float, int, int, int]] = []
		self.playback_start_wall = 0.0
		self.playback_start_seconds = 0.0
		self.playback_current_seconds = 0.0
		self.playback_scheduler: PlaybackScheduler | None = None
		self.setStyleSheet(
			"QDialog { background: #eef1f4; color: #18212b; }"
			"QWidget { color: #18212b; font-family: 'Segoe UI'; font-size: 10pt; }"
			"QPushButton { color: #18212b; background: #ffffff; padding: 7px 13px; "
			"border: 1px solid #aeb8c4; border-radius: 3px; }"
			"QPushButton:hover { background: #e1edf5; }"
			"QComboBox, QLineEdit { color: #18212b; background: #ffffff; "
			"padding: 5px; border: 1px solid #aeb8c4; }"
		)

		layout = QVBoxLayout(self)
		layout.setContentsMargins(16, 14, 16, 12)
		layout.setSpacing(12)
		self.summary_label = QLabel()
		self.summary_label.setStyleSheet("font-size: 12pt; font-weight: 700;")
		layout.addWidget(self.summary_label)

		plot_options = QHBoxLayout()
		self.plot_type_label = QLabel("Plot")
		plot_options.addWidget(self.plot_type_label)
		self.plot_type_combo = QComboBox()
		self.plot_type_combo.addItem("Histogram", "histogram")
		self.plot_type_combo.addItem("Hit timeline", "timeline")
		self.plot_type_combo.currentIndexChanged.connect(self._plot_type_changed)
		plot_options.addWidget(self.plot_type_combo)
		self.correction_label = QLabel("Correction")
		plot_options.addWidget(self.correction_label)
		self.correction_combo = QComboBox()
		self.correction_combo.addItem("Manual", "manual")
		self.correction_combo.addItem("music21", "music21")
		self.correction_combo.addItem("Both", "both")
		self.correction_combo.setCurrentIndex(2)
		self.correction_combo.currentIndexChanged.connect(self._update_plot)
		plot_options.addWidget(self.correction_combo)
		self.measure_label = QLabel("Measure")
		plot_options.addWidget(self.measure_label)
		self.measure_combo = QComboBox()
		self.measure_combo.addItem("Duration", "duration")
		self.measure_combo.addItem("Onset", "onset")
		self.measure_combo.addItem("Offset", "offset")
		self.measure_combo.currentIndexChanged.connect(self._update_plot)
		plot_options.addWidget(self.measure_combo)
		plot_options.addWidget(QLabel("X axis units"))
		self.units_combo = QComboBox()
		self.units_combo.addItem("Seconds", "seconds")
		self.units_combo.addItem("Samples (frames)", "frames")
		self.units_combo.currentIndexChanged.connect(self._update_plot)
		plot_options.addWidget(self.units_combo)
		self.visible_measures_label = QLabel("Visible measures")
		plot_options.addWidget(self.visible_measures_label)
		self.visible_measures_combo = QComboBox()
		for measure_count in (1, 2, 4, 8, 16):
			self.visible_measures_combo.addItem(str(measure_count), measure_count)
		self.visible_measures_combo.setCurrentIndex(1)
		self.visible_measures_combo.currentIndexChanged.connect(self._update_plot)
		plot_options.addWidget(self.visible_measures_combo)
		self.zoom_out_button = QPushButton("Zoom -")
		self.zoom_out_button.clicked.connect(lambda: self._zoom_timeline(-1, self._timeline_center_position()))
		plot_options.addWidget(self.zoom_out_button)
		self.zoom_in_button = QPushButton("Zoom +")
		self.zoom_in_button.clicked.connect(lambda: self._zoom_timeline(1, self._timeline_center_position()))
		plot_options.addWidget(self.zoom_in_button)
		plot_options.addStretch(1)
		layout.addLayout(plot_options)

		self.figure = Figure(figsize=(8, 4.4), tight_layout=True)
		self.canvas = FigureCanvasQTAgg(self.figure)
		self.axes = self.figure.add_subplot(111)
		self.navigation_toolbar = NavigationToolbar2QT(self.canvas, self)
		layout.addWidget(self.navigation_toolbar)
		self.plot_scroll_area = QScrollArea()
		self.plot_scroll_area.setWidgetResizable(False)
		self.canvas.setMinimumSize(800, 440)
		self.plot_scroll_area.setWidget(self.canvas)
		layout.addWidget(self.plot_scroll_area, 1)
		self.timeline_panel = QWidget(self)
		timeline_panel_layout = QHBoxLayout(self.timeline_panel)
		timeline_panel_layout.setContentsMargins(0, 0, 0, 0)
		timeline_panel_layout.setSpacing(12)
		self.timeline_widget = PianoRollTimeline(self)
		self.timeline_widget.on_drag_start = self._timeline_drag_start
		self.timeline_widget.on_drag_move = self._update_drag_preview
		self.timeline_widget.on_drag_end = self._timeline_mouse_release
		self.timeline_widget.on_zoom = self._zoom_timeline
		self.timeline_widget.on_horizontal_scroll = self._scroll_timeline
		self.timeline_widget.on_pan = self._pan_timeline
		self.timeline_widget.on_add_note = self._add_editable_note
		self.timeline_widget.on_remove_note = self._remove_timeline_note
		self.timeline_widget.on_select_note = self._select_timeline_note
		self.timeline_chart_scroll = QScrollArea()
		self.timeline_chart_scroll.setWidgetResizable(True)
		self.timeline_chart_scroll.setWidget(self.timeline_widget)
		self.timeline_chart_scroll.installEventFilter(self)
		self.timeline_chart_scroll.viewport().installEventFilter(self)
		timeline_panel_layout.addWidget(self.timeline_chart_scroll, 1)
		self.timeline_controls_scroll = QScrollArea()
		self.timeline_controls_scroll.setWidgetResizable(True)
		self.timeline_controls_scroll.setFixedWidth(300)
		self.timeline_controls_scroll.setStyleSheet(
			"QScrollArea { border: 0; background: #e8edf1; }"
			"QScrollArea > QWidget > QWidget { background: #e8edf1; }"
		)
		self.timeline_controls_panel = QWidget()
		self.timeline_controls_panel.setStyleSheet(
			"QWidget { background: #e8edf1; color: #18212b; }"
			"QLabel { background: transparent; color: #263746; font-weight: 600; }"
			"QCheckBox { background: #f7f9fa; color: #18212b; padding: 7px; "
			"border: 1px solid #c2ccd3; border-radius: 4px; }"
			"QCheckBox::indicator { width: 16px; height: 16px; }"
			"QPushButton { background: #ffffff; color: #18212b; padding: 8px 10px; "
			"border: 1px solid #aebbc5; border-radius: 4px; font-weight: 600; }"
			"QPushButton:hover { background: #dce8ee; border-color: #6c8797; }"
			"QComboBox, QLineEdit { background: #ffffff; color: #18212b; "
			"border: 1px solid #aebbc5; border-radius: 4px; padding: 6px; }"
		)
		self.timeline_controls_layout = QVBoxLayout(self.timeline_controls_panel)
		self.timeline_controls_layout.setContentsMargins(8, 4, 8, 4)
		self.timeline_controls_layout.setSpacing(10)
		self.timeline_controls_scroll.setWidget(self.timeline_controls_panel)
		timeline_panel_layout.addWidget(self.timeline_controls_scroll)
		layout.addWidget(self.timeline_panel, 1)
		self.timeline_panel.hide()

		self.timeline_navigation = QWidget()
		timeline_navigation_layout = QVBoxLayout(self.timeline_navigation)
		timeline_navigation_layout.setContentsMargins(0, 0, 0, 0)
		timeline_navigation_layout.setSpacing(5)
		timeline_navigation_layout.addWidget(QLabel("Start measure"))
		self.timeline_start_slider = QSlider(Qt.Orientation.Horizontal)
		self.timeline_start_slider.setRange(0, 0)
		self.timeline_start_slider.valueChanged.connect(self._timeline_start_changed)
		timeline_navigation_layout.addWidget(self.timeline_start_slider, 1)
		self.timeline_navigation_label = QLabel("1 / 1")
		self.timeline_navigation_label.setAlignment(Qt.AlignmentFlag.AlignRight)
		timeline_navigation_layout.addWidget(self.timeline_navigation_label)
		reference_row = QHBoxLayout()
		reference_row.addWidget(QLabel("Reference"))
		self.reference_source_combo = QComboBox()
		self.reference_source_combo.addItem("Detection", "detected")
		self.reference_source_combo.addItem("Waterfall", "waterfall")
		self.reference_source_combo.addItem("Mean: Detection + Waterfall", "combined")
		self.reference_source_combo.addItem("Detection normalized", "normalized_detection")
		self.reference_source_combo.addItem("Waterfall normalized", "normalized_waterfall")
		self.reference_source_combo.addItem("Mean normalized", "normalized")
		self.reference_source_combo.currentIndexChanged.connect(
			self._reference_source_changed
		)
		reference_row.addWidget(self.reference_source_combo, 1)
		timeline_navigation_layout.addLayout(reference_row)
		self.reference_visible_checkbox = QCheckBox("Show reference")
		self.reference_visible_checkbox.setChecked(True)
		self.reference_visible_checkbox.setStyleSheet(
			"QCheckBox { color: #18212b; background-color: #e8edf1; "
			"padding: 4px; border: 1px solid #b8c3cc; }"
		)
		self.reference_visible_checkbox.toggled.connect(self._update_plot)
		timeline_navigation_layout.addWidget(self.reference_visible_checkbox)
		self.playhead_visible_checkbox = QCheckBox("Show playhead")
		self.playhead_visible_checkbox.setChecked(True)
		self.playhead_visible_checkbox.setStyleSheet(
			"QCheckBox { color: #18212b; background-color: #e8edf1; "
			"padding: 4px; border: 1px solid #b8c3cc; }"
		)
		self.playhead_visible_checkbox.toggled.connect(
			self.timeline_widget.set_playhead_visible
		)
		timeline_navigation_layout.addWidget(self.playhead_visible_checkbox)
		listen_row = QHBoxLayout()
		listen_row.addWidget(QLabel("Listen"))
		self.playback_source_combo = QComboBox()
		self.playback_source_combo.addItem("Detected", "detected")
		self.playback_source_combo.addItem("Waterfall", "waterfall")
		self.playback_source_combo.addItem("Mean: Detection + Waterfall", "combined")
		self.playback_source_combo.addItem("Detection normalized", "normalized_detection")
		self.playback_source_combo.addItem("Waterfall normalized", "normalized_waterfall")
		self.playback_source_combo.addItem("Mean normalized", "normalized")
		self.playback_source_combo.addItem("Manual", "manual")
		self.playback_source_combo.addItem("music21", "music21")
		listen_row.addWidget(self.playback_source_combo, 1)
		timeline_navigation_layout.addLayout(listen_row)
		playback_buttons = QHBoxLayout()
		self.play_button = QPushButton("Play")
		self.play_button.clicked.connect(self._toggle_playback)
		playback_buttons.addWidget(self.play_button)
		self.stop_button = QPushButton("Stop")
		self.stop_button.clicked.connect(self._stop_playback)
		playback_buttons.addWidget(self.stop_button)
		timeline_navigation_layout.addLayout(playback_buttons)
		self.autoscroll_checkbox = QCheckBox("Follow playback")
		self.autoscroll_checkbox.setChecked(True)
		self.autoscroll_checkbox.toggled.connect(self._update_timeline_interaction)
		timeline_navigation_layout.addWidget(self.autoscroll_checkbox)
		self.timeline_controls_layout.addWidget(self.timeline_navigation)
		self.timeline_navigation.setVisible(False)

		settings_panel = QWidget()
		settings = QVBoxLayout(settings_panel)
		settings.setContentsMargins(0, 0, 0, 0)
		grid_row = QHBoxLayout()
		grid_row.addWidget(QLabel("Grid note value"))
		self.beat_length_combo = QComboBox()
		for label, beat_value in (
			("Quarter note (1 beat)", 1.0),
			("Eighth note (1/2 beat)", 0.5),
			("16th note (1/4 beat)", 0.25),
			("32nd note (1/8 beat)", 0.125),
			("64th note (1/16 beat)", 0.0625),
			("128th note (1/32 beat)", 0.03125),
		):
			self.beat_length_combo.addItem(label, beat_value)
		self.beat_length_combo.setCurrentIndex(2)
		self.beat_length_combo.currentIndexChanged.connect(
			self._invalidate_quantization_preview
		)
		grid_row.addWidget(self.beat_length_combo, 1)
		settings.addLayout(grid_row)
		self.quantize_button = QPushButton("Quantize")
		self.quantize_button.clicked.connect(self._quantize_events)
		settings.addWidget(self.quantize_button)
		self.music21_quantize_button = QPushButton("Quantize with music21")
		self.music21_quantize_button.clicked.connect(self._quantize_events_with_music21)
		settings.addWidget(self.music21_quantize_button)
		self.editable_copy_button = QPushButton("Create editable copy")
		self.editable_copy_button.clicked.connect(self._create_editable_copy)
		settings.addWidget(self.editable_copy_button)
		self.snap_edit_checkbox = QCheckBox("Snap edits to 1/16 note")
		self.snap_edit_checkbox.setChecked(True)
		self.snap_edit_checkbox.setToolTip(
			"Snap only edits in the editable copy to a fixed sixteenth-note grid."
		)
		settings.addWidget(self.snap_edit_checkbox)
		self.add_note_button = QPushButton("Add note")
		self.add_note_button.clicked.connect(self._arm_add_note)
		settings.addWidget(self.add_note_button)
		self.remove_note_button = QPushButton("Remove selected note")
		self.remove_note_button.setEnabled(False)
		self.remove_note_button.clicked.connect(self._remove_selected_editable_note)
		settings.addWidget(self.remove_note_button)
		bpm_row = QHBoxLayout()
		bpm_row.addWidget(QLabel("BPM"))
		self.bpm_slider = QSlider(Qt.Orientation.Horizontal)
		self.bpm_slider.setRange(20, 300)
		initial_bpm = SONG_REFERENCE_BPM
		self.bpm_slider.setValue(initial_bpm)
		self.bpm_slider.setMinimumWidth(80)
		self.bpm_entry = QLineEdit(str(initial_bpm))
		self.bpm_entry.setValidator(QIntValidator(20, 300, self))
		self.bpm_entry.setFixedWidth(64)
		self.bpm_slider.valueChanged.connect(self._bpm_changed)
		self.bpm_entry.editingFinished.connect(self._sync_bpm_from_entry)
		self.bpm_slider.hide()
		bpm_row.addWidget(self.bpm_entry)
		settings.addLayout(bpm_row)
		if self.estimated_bpm is None:
			estimate_label = f"Song reference: {SONG_REFERENCE_BPM} BPM"
		elif self.alternative_bpm is not None:
			estimate_label = (
				f"Song reference: {SONG_REFERENCE_BPM} BPM; "
				f"onset estimate {self.estimated_bpm} BPM "
				f"({self.alternative_bpm} BPM half-time)"
			)
		else:
			estimate_label = (
				f"Song reference: {SONG_REFERENCE_BPM} BPM; "
				f"onset estimate {self.estimated_bpm} BPM"
			)
		self.estimated_bpm_label = QLabel(estimate_label)
		self.estimated_bpm_label.setStyleSheet("color: #53616d;")
		self.estimated_bpm_label.setToolTip(
			"Quantization defaults to the song reference tempo. The onset-based "
			"estimate is shown for comparison and does not change that tempo."
		)
		self.estimated_bpm_label.setWordWrap(True)
		settings.addWidget(self.estimated_bpm_label)
		self.timeline_controls_layout.addWidget(settings_panel)

		self.grid_phase_label = QLabel("Grid phase will be fitted when quantized.")
		self.grid_phase_label.setStyleSheet("color: #53616d;")
		self.grid_phase_label.setWordWrap(True)
		self.timeline_controls_layout.addWidget(self.grid_phase_label)

		self.placeholder_label = QLabel(
			"Run a quantizer to prepare a separate export copy."
		)
		self.placeholder_label.setStyleSheet("color: #53616d;")
		self.placeholder_label.setWordWrap(True)
		self.timeline_controls_layout.addWidget(self.placeholder_label)
		self.timeline_controls_layout.addStretch(1)

		buttons = QDialogButtonBox(
			QDialogButtonBox.StandardButton.Ok
			| QDialogButtonBox.StandardButton.Cancel
		)
		buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Accept")
		buttons.accepted.connect(self._stop_playback)
		buttons.rejected.connect(self._stop_playback)
		if self.view_mode == "timeline":
			buttons.accepted.connect(self._accept_quantization)
		else:
			buttons.accepted.connect(self.accept)
		buttons.rejected.connect(self.reject)
		layout.addWidget(buttons)
		self.plot_type_combo.setCurrentIndex(
			self.plot_type_combo.findData(
				"histogram" if self.view_mode == "statistics" else "timeline"
			)
		)
		self.plot_type_label.hide()
		self.plot_type_combo.hide()
		self._plot_type_changed(self.plot_type_combo.currentIndex())

	def _create_editable_copy(self) -> None:
		source_events = self._quantization_source_events()
		if not source_events:
			QMessageBox.information(self, "No notes", "There are no notes to copy.")
			return
		self.editable_events = [event.copy() for event in source_events]
		index = self.correction_combo.findData("editable")
		if index < 0:
			self.correction_combo.addItem("Editable copy", "editable")
		if self.playback_source_combo.findData("editable") < 0:
			self.playback_source_combo.addItem("Editable copy", "editable")
		self.correction_combo.setCurrentIndex(self.correction_combo.findData("editable"))
		self.placeholder_label.setText(
			f"Editable copy created with {len(self.editable_events)} notes. "
			"Changes do not modify the captured reference."
		)
		self._update_plot()

	def _arm_add_note(self) -> None:
		if self.editable_events is None:
			self._create_editable_copy()
		if self.editable_events is not None:
			self.timeline_widget.add_note_mode = True
			self.timeline_widget.setCursor(Qt.CursorShape.CrossCursor)
			self.timeline_widget.setFocus(Qt.FocusReason.OtherFocusReason)
			self.placeholder_label.setText("Click a key row and time position to add a note.")

	def _add_editable_note(self, note: str, timeline_position: float) -> None:
		if self.editable_events is None:
			return
		unit_scale = self.fps if self.units_combo.currentData() == "frames" else 1.0
		onset_seconds = float(timeline_position) / unit_scale
		if self.snap_edit_checkbox.isChecked():
			grid, origin = self._editable_snap_grid()
			onset_seconds = origin + round((onset_seconds - origin) / grid) * grid
		onset_frame = max(0, round(onset_seconds * self.fps))
		duration_frames = max(
			1,
			round(60.0 / self.bpm_slider.value() * 0.5 * self.fps),
		)
		event_id = max(
			(int(event["event_id"]) for event in self.editable_events),
			default=0,
		) + 1
		midi_note = NOTE_TO_MIDI[note]
		hand = next(
			(event["hand"] for event in self.editable_events if event["note"] == note),
			"left" if midi_note < 60 else "right",
		)
		event = {
			"event_id": event_id,
			"note": note,
			"midi_note": midi_note,
			"hand": hand,
			"onset_frame": onset_frame,
			"offset_frame": onset_frame + duration_frames,
			"video_onset_seconds": onset_frame / self.fps,
			"video_offset_seconds": (onset_frame + duration_frames) / self.fps,
			"duration_seconds": duration_frames / self.fps,
			"editable_only": True,
		}
		self.editable_events.append(event)
		self.timeline_widget.selected_bar = ("editable", event_id)
		self.selected_editable_event_id = event_id
		self.remove_note_button.setEnabled(True)
		self._update_plot()

	def _editable_snap_grid(self) -> tuple[float, float]:
		grid_seconds = 60.0 / self.bpm_slider.value() / 4
		origin_frame = self._selected_grid_origin_frame()
		if origin_frame is None:
			source_events = self._quantization_source_events()
			origin_frame = min(
				(int(event["onset_frame"]) for event in source_events),
				default=0,
			)
		return grid_seconds, origin_frame / self.fps

	def _select_timeline_note(self, source: str, event_id: int) -> None:
		self.selected_editable_event_id = event_id if source == "editable" else None
		self.remove_note_button.setEnabled(self.selected_editable_event_id is not None)

	def _remove_timeline_note(self, source: str, event_id: int) -> None:
		if source != "editable" or self.editable_events is None:
			return
		self.editable_events = [
			event for event in self.editable_events
			if int(event["event_id"]) != event_id
		]
		self.selected_editable_event_id = None
		self.timeline_widget.selected_bar = None
		self.remove_note_button.setEnabled(False)
		self._update_plot()

	def _remove_selected_editable_note(self) -> None:
		if self.selected_editable_event_id is not None:
			self._remove_timeline_note("editable", self.selected_editable_event_id)

	def _sync_bpm_from_entry(self) -> None:
		try:
			bpm = int(self.bpm_entry.text())
		except ValueError:
			bpm = self.bpm_slider.value()
		bpm = max(self.bpm_slider.minimum(), min(self.bpm_slider.maximum(), bpm))
		self.bpm_slider.setValue(bpm)
		self.bpm_entry.setText(str(bpm))

	def _bpm_changed(self, bpm: int) -> None:
		self.bpm_entry.setText(str(bpm))
		self._invalidate_quantization_preview()

	def _invalidate_quantization_preview(self, _index: int = 0) -> None:
		if self.quantized_events is None and self.music21_quantized_events is None:
			self._update_plot()
			return
		self.quantized_events = None
		self.music21_quantized_events = None
		self.quantization_settings = None
		self.music21_quantization_settings = None
		self.plot_quantized_preview = False
		self.correction_combo.setCurrentIndex(
			self.correction_combo.findData("manual")
		)
		self.placeholder_label.setText(
			"Grid settings changed. Click a quantizer to prepare an export copy."
		)
		self.grid_phase_label.setText("Grid phase will be fitted when quantized.")
		self._update_plot()

	def _quantization_source_events(self) -> list[dict[str, object]]:
		source = self.reference_source_combo.currentData()
		if source == "waterfall":
			return [event.copy() for event in self.waterfall_events]
		if source == "combined":
			return combine_detection_and_waterfall(
				self.events, self.waterfall_events, self.fps
			)
		normalized_source = {
			"normalized_detection": self.normalized_detection_events,
			"normalized_waterfall": self.normalized_waterfall_events,
			"normalized": self.normalized_events,
	}.get(source)
		if normalized_source is not None:
			quantization_events = [event.copy() for event in normalized_source]
			for event in quantization_events:
				onset = float(event["normalized_onset_seconds"])
				offset = float(event["normalized_offset_seconds"])
				event["onset_frame"] = round(onset * self.fps)
				event["offset_frame"] = round(offset * self.fps)
				event["video_onset_seconds"] = onset
				event["video_offset_seconds"] = offset
			return quantization_events
		return [event.copy() for event in self.events]

	def _quantize_events(self) -> None:
		bpm = self.bpm_slider.value()
		beat_length = float(self.beat_length_combo.currentData())
		source_events = self._quantization_source_events()
		self.quantized_events, self.quantization_settings, collapsed_onsets = (
			quantize_events_manually(
				source_events,
				self.fps,
				bpm,
				beat_length,
				self.current_frame,
				self._selected_grid_origin_frame(),
			)
		)
		self.plot_quantized_preview = True
		self.correction_combo.setCurrentIndex(
			self.correction_combo.findData("manual")
		)
		grid_seconds = float(self.quantization_settings["grid_seconds"])
		completed_count = sum(
			event["offset_frame"] is not None for event in source_events
		)
		self.grid_phase_label.setText(
			f"Grid interval: {grid_seconds:.5f} s "
			f"({self.beat_length_combo.currentText()} at {bpm} BPM); "
			f"origin anchored to first onset at frame "
			f"{self.quantization_settings['origin_frame']}. "
			"Onsets and offsets snap to grid lines; durations are integer "
			"multiples of the grid."
		)
		collision_warning = (
			f" Warning: {collapsed_onsets} distinct detected onset times merge "
			"on this grid; choose a finer note value to preserve them."
			if collapsed_onsets
			else ""
		)
		self.placeholder_label.setText(
			f"Prepared {completed_count} quantized hits at {bpm} BPM "
			f"({self.beat_length_combo.currentText()} grid). "
			f"Accept to enable export.{collision_warning}"
		)
		self._update_plot()

	def _quantize_events_with_music21(self) -> None:
		bpm = self.bpm_slider.value()
		beat_length = float(self.beat_length_combo.currentData())
		source_events = self._quantization_source_events()
		try:
			self.music21_quantized_events = quantize_events_with_music21(
				source_events,
				self.fps,
				bpm,
				beat_length,
				self.current_frame,
				self._selected_grid_origin_frame(),
			)
		except Exception as error:
			QMessageBox.critical(self, "music21 quantization failed", str(error))
			return

		self.music21_quantization_settings = {
			"bpm": bpm,
			"beat_length": beat_length,
		}
		self.correction_combo.setCurrentIndex(
			self.correction_combo.findData("music21")
		)
		self.placeholder_label.setText(
			f"music21 quantized {len(self.music21_quantized_events)} events at "
			f"{bpm} BPM ({self.beat_length_combo.currentText()})."
		)
		self._update_plot()

	def _accept_quantization(self) -> None:
		if self.quantized_events is None:
			self._quantize_events()
		if self.music21_quantized_events is None:
			self._quantize_events_with_music21()
		self.correction_combo.setCurrentIndex(
			self.correction_combo.findData("both")
		)
		if self.quantized_events is not None and self.on_accept is not None:
			self.on_accept(self.quantized_events, self.quantization_settings or {})
		if (
			self.music21_quantized_events is not None
			and self.on_music21_accept is not None
		):
			self.on_music21_accept(
				self.music21_quantized_events,
				self.music21_quantization_settings or {},
			)
		if self.on_normalized_accept is not None:
			self.on_normalized_accept(
				self.normalized_events,
				self.normalization_settings,
			)
		self.accept()

	def _selected_grid_origin_frame(self) -> int | None:
		reference_source = self.reference_source_combo.currentData()
		if reference_source == "waterfall" and self.waterfall_events:
			return min(int(event["onset_frame"]) for event in self.waterfall_events)
		if reference_source == "combined":
			combined_events = combine_detection_and_waterfall(
				self.events, self.waterfall_events, self.fps
			)
			if combined_events:
				return min(int(event["onset_frame"]) for event in combined_events)
		normalization_settings = {
			"normalized_detection": self.normalization_detection_settings,
			"normalized_waterfall": self.normalization_waterfall_settings,
			"normalized": self.normalization_settings,
		}.get(reference_source)
		if normalization_settings is not None:
			return int(normalization_settings["origin_frame"])
		return None

	def _reference_source_changed(self, _index: int = 0) -> None:
		origin_frame = self._selected_grid_origin_frame()
		bpm = self.bpm_slider.value()
		beat_length = float(self.beat_length_combo.currentData())
		self.normalized_detection_events, self.normalization_detection_settings = normalize_events_to_reference(
			self.events, [], self.fps, SONG_REFERENCE_BPM, 0.5, self.current_frame
		)
		waterfall_origin = min(
			(int(event["onset_frame"]) for event in self.waterfall_events),
			default=None,
		)
		self.normalized_waterfall_events, self.normalization_waterfall_settings = normalize_events_to_reference(
			self.waterfall_events,
			[],
			self.fps,
			SONG_REFERENCE_BPM,
			0.5,
			self.current_frame,
			waterfall_origin,
		)
		combined_events = combine_detection_and_waterfall(
			self.events, self.waterfall_events, self.fps
		)
		combined_origin = min(
			(int(event["onset_frame"]) for event in combined_events),
			default=None,
		)
		self.normalized_events, self.normalization_settings = normalize_events_to_reference(
			self.events,
			self.waterfall_events,
			self.fps,
			SONG_REFERENCE_BPM,
			0.5,
			self.current_frame,
			combined_origin,
		)
		source_events = self._quantization_source_events()
		if self.quantized_events is not None:
			self.quantized_events, self.quantization_settings, _ = quantize_events_manually(
				source_events,
				self.fps,
				bpm,
				beat_length,
				self.current_frame,
				origin_frame,
			)
		if self.music21_quantized_events is not None:
			self.music21_quantized_events = quantize_events_with_music21(
				source_events,
				self.fps,
				bpm,
				beat_length,
				self.current_frame,
				origin_frame,
			)
		self._update_plot()
		if self.playback_timer.isActive():
			self._refresh_playback_schedule()

	def _event_bounds_seconds(
		self,
		event: dict[str, object],
		corrected: bool,
		correction_source: str = "manual",
	) -> tuple[float, float]:
		raw_onset = int(event["onset_frame"]) / self.fps
		raw_end_frame = event["offset_frame"]
		raw_end = (
			int(raw_end_frame) / self.fps
			if raw_end_frame is not None
			else max(self.current_frame + 1, int(event["onset_frame"]) + 1) / self.fps
		)
		if not corrected:
			return raw_onset, raw_end
		if correction_source == "editable":
			return raw_onset, raw_end

		if correction_source == "normalized":
			onset = float(event.get("normalized_onset_seconds", raw_onset))
			offset = event.get("normalized_offset_seconds")
		elif correction_source == "music21":
			onset = float(event.get("music21_onset_seconds", raw_onset))
			offset = event.get("music21_offset_seconds")
		else:
			onset = float(event.get("midi_onset_seconds", raw_onset))
			offset = event.get("midi_offset_seconds")
		if offset is not None:
			return onset, float(offset)
		if correction_source == "manual" and self.quantization_settings is not None:
			grid = float(self.quantization_settings["grid_seconds"])
			origin = float(self.quantization_settings["origin_seconds"])
			offset_index = round((raw_end - origin) / grid)
			return onset, max(onset + grid, origin + offset_index * grid)
		return onset, raw_end

	def _add_histogram_grid(self, minimum: float, maximum: float) -> None:
		if minimum == maximum:
			return
		beat_seconds = 60.0 / self.bpm_slider.value()
		subdivision = beat_seconds / 4
		measure = self.measure_combo.currentData()
		if measure == "duration":
			origin = 0.0
		elif self.plot_quantized_preview and self.quantization_settings is not None:
			origin = float(self.quantization_settings["origin_seconds"])
		else:
			origin = min(
				int(event["onset_frame"]) / self.fps for event in self.events
			)
		if self.units_combo.currentData() == "frames":
			origin *= self.fps
			subdivision *= self.fps
		first_line = math.ceil((minimum - origin) / subdivision)
		last_line = math.floor((maximum - origin) / subdivision)
		positions = [
			origin + line * subdivision
			for line in range(first_line, last_line + 1)
		]
		if positions:
			self.axes.vlines(positions, 0, 1, transform=self.axes.get_xaxis_transform(),
							 color="#718096", linewidth=0.6, alpha=0.35, zorder=0)

	def _update_timeline(self) -> None:
		visible_corrections: dict[str, list[dict[str, object]]] = {}
		correction_mode = self.correction_combo.currentData()
		if correction_mode in ("manual", "both") and self.quantized_events is not None:
			visible_corrections["manual"] = self.quantized_events
		if correction_mode in ("music21", "both") and self.music21_quantized_events is not None:
			visible_corrections["music21"] = self.music21_quantized_events
		if correction_mode == "editable" and self.editable_events is not None:
			visible_corrections["editable"] = self.editable_events

		midi_by_note = {
			str(event["note"]): int(event["midi_note"])
			for event in [
				*self.events,
				*self.waterfall_events,
			*(self.editable_events or []),
			]
		}
		timeline_notes = sorted(midi_by_note, key=midi_by_note.__getitem__)
		note_rows = {note: index for index, note in enumerate(timeline_notes)}
		unit_scale = self.fps if self.units_combo.currentData() == "frames" else 1.0
		colors = {"left": "#28a9bd", "right": "#42a45f"}
		raw_bars: list[dict[str, object]] = []
		corrected_bars: list[dict[str, object]] = []
		all_ends: list[float] = []
		reference_mode = str(self.reference_source_combo.currentData())
		if reference_mode == "waterfall":
			reference_events = self.waterfall_events
			reference_label = "Waterfall"
		elif reference_mode == "combined":
			reference_events = combine_detection_and_waterfall(
				self.events, self.waterfall_events, self.fps
			)
			reference_label = "Mean Detection + Waterfall"
		elif reference_mode == "normalized_detection":
			reference_events = self.normalized_detection_events
			reference_label = "Detection normalized"
		elif reference_mode == "normalized_waterfall":
			reference_events = self.normalized_waterfall_events
			reference_label = "Waterfall normalized"
		elif reference_mode == "normalized" and self.normalized_events is not None:
			reference_events = self.normalized_events
			reference_label = "Mean normalized"
		else:
			reference_events = self.events
			reference_label = "Detection"
		correction_by_id = {
			source: {int(item["event_id"]): item for item in items}
			for source, items in visible_corrections.items()
		}

		for event in reference_events:
			note = str(event["note"])
			if note not in note_rows:
				continue
			row = note_rows[note]
			hand_color = colors[str(event["hand"])]
			is_normalized = reference_mode.startswith("normalized")
			raw_start, raw_end = self._event_bounds_seconds(
				event, corrected=is_normalized, correction_source="normalized"
			)
			raw_start *= unit_scale
			raw_end *= unit_scale
			if raw_end <= raw_start:
				raw_end = raw_start + unit_scale / self.fps
			if self.reference_visible_checkbox.isChecked():
				raw_bars.append({
					"row": row,
					"start": raw_start,
					"end": raw_end,
					"color": hand_color,
					"source": (
						"normalized"
						if is_normalized
						else "combined-reference"
						if reference_label == "Mean Detection + Waterfall"
						else "waterfall-reference"
						if reference_label == "Waterfall"
						else "detected"
					),
					"event_id": int(event["event_id"]),
				})
			all_ends.append(raw_end)
		for event in reference_events:
			note = str(event["note"])
			if note not in note_rows:
				continue
			row = note_rows[note]
			for source, by_id in correction_by_id.items():
				corrected = by_id.get(int(event["event_id"]))
				if corrected is None or corrected["offset_frame"] is None:
					continue
				start, end = self._event_bounds_seconds(
					corrected, corrected=True, correction_source=source
				)
				start *= unit_scale
				end *= unit_scale
				if end <= start:
					end = start + unit_scale / self.fps
				corrected_bars.append({
					"row": row,
					"start": start,
					"end": end,
					"color": hand_color,
					"source": source,
					"event_id": int(event["event_id"]),
				})
				all_ends.append(end)
		unit_name = "seconds" if unit_scale == 1.0 else "samples (frames)"
		measure_seconds = 4 * 60.0 / self.bpm_slider.value()
		visible_measures = int(self.visible_measures_combo.currentData())
		visible_span = measure_seconds * visible_measures * unit_scale
		total_end = max(all_ends, default=visible_span)
		total_measures = max(1, math.ceil(total_end / (measure_seconds * unit_scale)))
		maximum_start = max(
			0,
			math.ceil(total_end / (measure_seconds * unit_scale) - visible_measures / 2),
		)
		was_blocked = self.timeline_start_slider.blockSignals(True)
		self.timeline_start_slider.setRange(0, maximum_start)
		if not self.timeline_widget._timeline_initialized:
			first_onset_seconds = min(
				(int(event["onset_frame"]) / self.fps for event in self.events),
				default=0.0,
			)
			initial_measure = int(first_onset_seconds / measure_seconds)
			self.timeline_start_slider.setValue(min(maximum_start, initial_measure))
			self.timeline_widget._timeline_initialized = True
		self.timeline_start_slider.blockSignals(was_blocked)
		base_start_seconds = self.timeline_start_slider.value() * measure_seconds
		maximum_start_seconds = max(
			0.0,
			total_end / unit_scale - visible_measures * measure_seconds / 2,
		)
		self.timeline_max_start_seconds = maximum_start_seconds
		if self.timeline_window_start_seconds is None:
			window_start_seconds = base_start_seconds
		else:
			window_start_seconds = max(
				0.0,
				min(maximum_start_seconds, self.timeline_window_start_seconds),
			)
		self.timeline_window_start_seconds = window_start_seconds
		start_measure = int(window_start_seconds / measure_seconds)
		was_blocked = self.timeline_start_slider.blockSignals(True)
		self.timeline_start_slider.setValue(start_measure)
		self.timeline_start_slider.blockSignals(was_blocked)
		window_start = window_start_seconds * unit_scale
		window_end = window_start + visible_span
		self.timeline_navigation_label.setText(f"{start_measure + 1} / {max(1, total_measures)}")
		if reference_mode == "waterfall" and self.waterfall_events:
			origin_seconds = min(
				int(event["onset_frame"]) / self.fps
				for event in self.waterfall_events
			)
		elif reference_mode == "combined":
			combined_origin_frame = self._selected_grid_origin_frame()
			origin_seconds = (
				combined_origin_frame / self.fps
				if combined_origin_frame is not None
				else 0.0
			)
		elif reference_mode.startswith("normalized"):
			normalization_settings = {
				"normalized_detection": self.normalization_detection_settings,
				"normalized_waterfall": self.normalization_waterfall_settings,
				"normalized": self.normalization_settings,
			}.get(reference_mode)
			origin_seconds = (
				float(normalization_settings["origin_seconds"])
				if normalization_settings is not None
				else 0.0
			)
		elif self.quantization_settings is not None:
			origin_seconds = float(self.quantization_settings["origin_seconds"])
		else:
			origin_seconds = min(
				(int(event["onset_frame"]) / self.fps for event in self.events),
				default=0.0,
			)
		origin = origin_seconds * unit_scale
		self.timeline_widget.set_timeline_data(
			timeline_notes,
			raw_bars,
			corrected_bars,
			(window_start, window_end),
			unit_scale,
			unit_name,
			origin,
			self.bpm_slider.value(),
			self.detected_alpha,
			self.corrected_alpha,
			set(visible_corrections),
			reference_label,
		)
		self.timeline_widget.set_playhead(self.playback_current_seconds * unit_scale)
		visible_detected_count = len(self.timeline_widget.visible_raw_bars)
		corrected_count = len(corrected_bars)
		self.summary_label.setText(
			f"{len(self.events)} detected hits; {visible_detected_count} in view; "
			f"{corrected_count} corrected bars shown"
			+ ("; drag a corrected bar edge to edit" if correction_mode != "music21" else "")
		)

	def _selected_playback_events(self) -> list[dict[str, object]]:
		source = self.playback_source_combo.currentData()
		if source == "editable" and self.editable_events is not None:
			return self.editable_events
		if source == "waterfall":
			return self.waterfall_events
		if source == "combined":
			return combine_detection_and_waterfall(
				self.events, self.waterfall_events, self.fps
			)
		if source == "normalized_detection":
			return self.normalized_detection_events or []
		if source == "normalized_waterfall":
			return self.normalized_waterfall_events or []
		if source == "normalized" and self.normalized_events is not None:
			return self.normalized_events
		if source == "music21" and self.music21_quantized_events is not None:
			return self.music21_quantized_events
		if source == "manual" and self.quantized_events is not None:
			return self.quantized_events
		if source == "detected":
			return self.events
		return []

	def _toggle_playback(self) -> None:
		if self.playback_timer.isActive():
			self._stop_playback()
			return
		if self.plot_type_combo.currentData() != "timeline":
			self.plot_type_combo.setCurrentIndex(
				self.plot_type_combo.findData("timeline")
			)
		source = self.playback_source_combo.currentData()
		if source == "manual" and self.quantized_events is None:
			QMessageBox.information(
				self, "Manual timing unavailable", "Quantize manually and accept first."
			)
			return
		if source == "music21" and self.music21_quantized_events is None:
			QMessageBox.information(
				self, "music21 timing unavailable", "Run music21 quantization and accept first."
			)
			return
		if source.startswith("normalized") and not self._selected_playback_events():
			QMessageBox.information(
				self, "Normalization unavailable", "Normalize detections first."
			)
			return
		if source == "waterfall" and not self.waterfall_events:
			QMessageBox.information(
				self, "Waterfall timing unavailable", "No waterfall crossings were detected."
			)
			return
		if source == "combined" and not (self.events or self.waterfall_events):
			QMessageBox.information(
				self, "Combined timing unavailable", "No detected or waterfall notes are available."
			)
			return
		try:
			self.midi_output = mido.open_output()
		except (OSError, mido.MidiError) as error:
			QMessageBox.warning(
				self,
				"MIDI output unavailable",
				f"Could not open a MIDI output device: {error}",
			)
			return

		measure_seconds = 4 * 60.0 / self.bpm_slider.value()
		self.playback_start_seconds = (
			self.timeline_window_start_seconds
			if self.timeline_window_start_seconds is not None
			else self.timeline_start_slider.value() * measure_seconds
		)
		self.playback_current_seconds = self.playback_start_seconds
		self._build_playback_schedule(self.playback_start_seconds)
		self._start_playback_scheduler()
		self.play_button.setText("Pause")
		unit_scale = self.fps if self.units_combo.currentData() == "frames" else 1.0
		self.timeline_widget.set_playhead(self.playback_start_seconds * unit_scale)
		self.playback_timer.start()
		self._update_timeline_interaction()

	def _build_playback_schedule(self, start_seconds: float) -> None:
		self.playback_notes = []
		playback_source = str(self.playback_source_combo.currentData())
		for event in self._selected_playback_events():
			onset, offset = self._event_bounds_seconds(
				event,
				corrected=playback_source.startswith("normalized") or playback_source in ("manual", "music21"),
				correction_source=playback_source,
			)
			if offset <= start_seconds:
				continue
			channel = 0 if event["hand"] == "left" else 1
			note_number = int(event["midi_note"])
			self.playback_notes.append((max(onset, start_seconds), 1, channel, note_number))
			self.playback_notes.append((offset, 0, channel, note_number))
		self.playback_notes.sort()
		self.playback_start_wall = time.perf_counter()

	def _start_playback_scheduler(self) -> None:
		if self.midi_output is None:
			return
		self.playback_scheduler = PlaybackScheduler(
			self.midi_output,
			self.playback_notes,
			self.playback_start_wall,
			self.playback_start_seconds,
		)
		self.playback_scheduler.start()

	def _stop_playback_scheduler(self) -> None:
		if self.playback_scheduler is not None:
			self.playback_scheduler.stop_event.set()
			self.playback_scheduler.join(timeout=1.0)
			self.playback_scheduler = None
		self.midi_output = None

	def _refresh_playback_schedule(self) -> None:
		if not self.playback_timer.isActive():
			return
		self._stop_playback_scheduler()
		try:
			self.midi_output = mido.open_output()
		except (OSError, mido.MidiError) as error:
			QMessageBox.warning(self, "MIDI output unavailable", str(error))
			self._stop_playback()
			return
		self.playback_start_seconds = self.playback_current_seconds
		self._build_playback_schedule(self.playback_start_seconds)
		self._start_playback_scheduler()

	def _playback_tick(self) -> None:
		if self.playback_scheduler is None:
			self._stop_playback()
			return
		elapsed = time.perf_counter() - self.playback_start_wall
		current_seconds = self.playback_start_seconds + elapsed
		self.playback_current_seconds = current_seconds
		unit_scale = self.fps if self.units_combo.currentData() == "frames" else 1.0
		if self.autoscroll_checkbox.isChecked():
			self._follow_playhead(current_seconds)
		self.timeline_widget.set_playhead(current_seconds * unit_scale)
		if self.playback_scheduler.completed_event.is_set():
			self._stop_playback()

	def _follow_playhead(self, current_seconds: float) -> None:
		measure_seconds = 4 * 60.0 / self.bpm_slider.value()
		visible_measures = int(self.visible_measures_combo.currentData())
		window_start = current_seconds - measure_seconds * visible_measures / 2
		maximum_start = self.timeline_max_start_seconds
		window_start = max(0.0, min(maximum_start, window_start))
		current_start = self.timeline_widget.window[0] / (
			self.fps if self.units_combo.currentData() == "frames" else 1.0
		)
		if abs(window_start - current_start) >= 0.5 / self.fps:
			self._set_timeline_window_start_seconds(window_start, lightweight=True)

	def _scroll_timeline(self, direction: int) -> None:
		unit_scale = self.fps if self.units_combo.currentData() == "frames" else 1.0
		window_start = self.timeline_widget.window[0] / unit_scale
		window_span = (self.timeline_widget.window[1] - self.timeline_widget.window[0]) / unit_scale
		self._set_timeline_window_start_seconds(window_start + direction * window_span * 0.25)

	def _pan_timeline(self, delta_units: float) -> None:
		unit_scale = self.fps if self.units_combo.currentData() == "frames" else 1.0
		window_start = self.timeline_widget.window[0] / unit_scale
		self._set_timeline_window_start_seconds(window_start + delta_units / unit_scale)

	def _set_timeline_window_start_seconds(
		self, start_seconds: float, lightweight: bool = False
	) -> None:
		self.timeline_window_start_seconds = max(
			0.0, min(self.timeline_max_start_seconds, start_seconds)
		)
		if lightweight:
			measure_seconds = 4 * 60.0 / self.bpm_slider.value()
			start_measure = int(self.timeline_window_start_seconds / measure_seconds)
			was_blocked = self.timeline_start_slider.blockSignals(True)
			self.timeline_start_slider.setValue(start_measure)
			self.timeline_start_slider.blockSignals(was_blocked)
			self.timeline_navigation_label.setText(
				f"{start_measure + 1} / "
				f"{max(1, self.timeline_start_slider.maximum() + int(self.visible_measures_combo.currentData()))}"
			)
			unit_scale = self.fps if self.units_combo.currentData() == "frames" else 1.0
			visible_span = measure_seconds * int(self.visible_measures_combo.currentData()) * unit_scale
			window_start = self.timeline_window_start_seconds * unit_scale
			self.timeline_widget.set_window((window_start, window_start + visible_span))
			return
		self._update_plot()

	def _zoom_timeline(self, direction: int, anchor_position: float) -> None:
		current_index = self.visible_measures_combo.currentIndex()
		new_index = max(
			0,
			min(self.visible_measures_combo.count() - 1, current_index - direction),
		)
		if new_index == current_index:
			return
		unit_scale = self.fps if self.units_combo.currentData() == "frames" else 1.0
		old_span = self.timeline_widget.window[1] - self.timeline_widget.window[0]
		anchor_fraction = (anchor_position - self.timeline_widget.window[0]) / max(1.0, old_span)
		new_measures = int(self.visible_measures_combo.itemData(new_index))
		measure_span = 4 * 60.0 / self.bpm_slider.value() * unit_scale
		new_start = anchor_position - anchor_fraction * measure_span * new_measures
		self.timeline_window_start_seconds = new_start / unit_scale
		self.visible_measures_combo.setCurrentIndex(new_index)

	def _timeline_center_position(self) -> float:
		return (self.timeline_widget.window[0] + self.timeline_widget.window[1]) / 2

	def eventFilter(self, watched, event) -> bool:
		if (
			(
				watched is self.timeline_chart_scroll
				or watched is self.timeline_chart_scroll.viewport()
			)
			and event.type() == QEvent.Type.Wheel
			and self.plot_type_combo.currentData() == "timeline"
		):
			if not self.timeline_widget.interaction_enabled:
				return True
			delta = event.angleDelta().y() or event.angleDelta().x()
			if not delta:
				return True
			if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
				position = self.timeline_widget.mapFromGlobal(
					event.globalPosition().toPoint()
				)
				self._zoom_timeline(
					1 if delta > 0 else -1,
					self.timeline_widget._x_value(position.x()),
				)
			else:
				self._scroll_timeline(-1 if delta > 0 else 1)
			return True
		return super().eventFilter(watched, event)

	def _stop_playback(self) -> None:
		self.playback_timer.stop()
		self._stop_playback_scheduler()
		self._update_timeline_interaction()
		self.play_button.setText("Play")

	def _update_timeline_interaction(self, _checked: bool = False) -> None:
		locked = self.playback_timer.isActive() and self.autoscroll_checkbox.isChecked()
		self.timeline_widget.set_interaction_enabled(not locked)
		self.timeline_chart_scroll.verticalScrollBar().setEnabled(not locked)

	def closeEvent(self, event) -> None:
		self._stop_playback()
		super().closeEvent(event)

	def _timeline_drag_start(self, source: str, event_id: int, side: str) -> bool:
		if source == "editable":
			source_events = self.editable_events
		elif source == "music21":
			source_events = self.music21_quantized_events
		else:
			source_events = self.quantized_events
		if source_events is None:
			return False
		corrected_event = next(
			(item for item in source_events if int(item["event_id"]) == event_id),
			None,
		)
		if corrected_event is None:
			return False
		if source == "editable":
			if self.snap_edit_checkbox.isChecked():
				grid, origin = self._editable_snap_grid()
				onset_grid = round((int(corrected_event["onset_frame"]) / self.fps - origin) / grid)
			else:
				grid = 1.0 / self.fps
				origin = 0.0
				onset_grid = int(corrected_event["onset_frame"])
			offset_frame = corrected_event["offset_frame"]
			if offset_frame is not None:
				offset_seconds = int(offset_frame) / self.fps
				offset_grid = round((offset_seconds - origin) / grid)
			else:
				offset_grid = onset_grid + 1
		elif source == "music21":
			grid = float(corrected_event["music21_grid_seconds"])
			origin = float(corrected_event["music21_origin_seconds"])
			onset_grid = int(corrected_event["music21_onset_grid_index"])
			offset_grid = int(corrected_event["music21_offset_grid_index"])
		else:
			if self.quantization_settings is None:
				return False
			grid = float(corrected_event["midi_grid_seconds"])
			origin = float(self.quantization_settings["origin_seconds"])
			onset_grid = int(corrected_event["midi_onset_grid_index"])
			offset_grid = int(corrected_event["midi_offset_grid_index"])
		self.timeline_drag = {
			"event": corrected_event,
			"source": source,
			"side": side,
			"grid": grid,
			"origin": origin,
			"onset_grid": onset_grid,
			"offset_grid": offset_grid,
			"preview_onset_grid": onset_grid,
			"preview_offset_grid": offset_grid,
		}
		return True

	def _update_drag_preview(self, x_position: float) -> None:
		if self.timeline_drag is None:
			return
		unit_scale = self.fps if self.units_combo.currentData() == "frames" else 1.0
		position_seconds = float(x_position) / unit_scale
		grid = float(self.timeline_drag["grid"])
		origin = float(self.timeline_drag["origin"])
		grid_index = max(0, round((position_seconds - origin) / grid))
		if self.timeline_drag["side"] == "onset":
			grid_index = min(grid_index, int(self.timeline_drag["offset_grid"]) - 1)
			start_index = grid_index
			end_index = int(self.timeline_drag["offset_grid"])
		else:
			grid_index = max(grid_index, int(self.timeline_drag["onset_grid"]) + 1)
			start_index = int(self.timeline_drag["onset_grid"])
			end_index = grid_index
		self.timeline_drag["preview_onset_grid"] = start_index
		self.timeline_drag["preview_offset_grid"] = end_index
		event_record = self.timeline_drag["event"]
		row = self.timeline_widget.notes.index(str(event_record["note"]))
		color = "#28a9bd" if event_record["hand"] == "left" else "#42a45f"
		self.timeline_widget.set_drag_preview({
			"row": row,
			"start": (origin + start_index * grid) * unit_scale,
			"end": (origin + end_index * grid) * unit_scale,
			"color": color,
			"source": str(self.timeline_drag["source"]),
		})

	def _timeline_mouse_release(self, x_position: float) -> None:
		if self.timeline_drag is None:
			return
		self._update_drag_preview(x_position)
		drag = self.timeline_drag
		event_record = drag["event"]
		onset_grid = int(drag["preview_onset_grid"])
		offset_grid = int(drag["preview_offset_grid"])
		grid = float(drag["grid"])
		origin = float(drag["origin"])
		if drag["source"] == "editable":
			onset_seconds = origin + onset_grid * grid
			offset_seconds = origin + offset_grid * grid
			event_record["onset_frame"] = round(onset_seconds * self.fps)
			event_record["offset_frame"] = round(offset_seconds * self.fps)
			event_record["video_onset_seconds"] = onset_seconds
			event_record["video_offset_seconds"] = offset_seconds
			event_record["duration_seconds"] = offset_seconds - onset_seconds
		elif drag["source"] == "music21":
			event_record["music21_onset_grid_index"] = onset_grid
			event_record["music21_offset_grid_index"] = offset_grid
			event_record["music21_onset_seconds"] = origin + onset_grid * grid
			event_record["music21_offset_seconds"] = origin + offset_grid * grid
		else:
			event_record["midi_onset_grid_index"] = onset_grid
			event_record["midi_offset_grid_index"] = offset_grid
			event_record["midi_duration_grid_units"] = offset_grid - onset_grid
			event_record["midi_onset_seconds"] = origin + onset_grid * grid
			event_record["midi_offset_seconds"] = origin + offset_grid * grid
		self.timeline_drag = None
		self.timeline_widget.set_drag_preview(None)
		self._update_plot()
		if self.playback_timer.isActive():
			self._refresh_playback_schedule()

	def _plot_type_changed(self, _index: int = 0) -> None:
		is_timeline = self.plot_type_combo.currentData() == "timeline"
		self.correction_label.setVisible(is_timeline)
		self.correction_combo.setVisible(is_timeline)
		self.measure_label.setVisible(not is_timeline)
		self.measure_combo.setVisible(not is_timeline)
		self.visible_measures_label.setVisible(is_timeline)
		self.visible_measures_combo.setVisible(is_timeline)
		self.zoom_in_button.setVisible(is_timeline)
		self.zoom_out_button.setVisible(is_timeline)
		self.timeline_navigation.setVisible(is_timeline)
		self.navigation_toolbar.setVisible(not is_timeline)
		self.plot_scroll_area.setVisible(not is_timeline)
		self.timeline_panel.setVisible(is_timeline)
		self._update_plot()

	def _timeline_start_changed(self, _value: int) -> None:
		self._pause_follow_for_manual_navigation()
		self.timeline_window_start_seconds = None
		self._update_plot()

	def _pause_follow_for_manual_navigation(self) -> None:
		if self.playback_timer.isActive() and self.autoscroll_checkbox.isChecked():
			self.autoscroll_checkbox.setChecked(False)

	def _update_plot(self, _index: int = 0) -> None:
		if self.plot_type_combo.currentData() == "timeline":
			self.timeline_navigation.setVisible(True)
			self._update_timeline()
			return
		self.timeline_navigation.setVisible(False)

		measure = self.measure_combo.currentData()
		unit = self.units_combo.currentData()
		events = (
			self.quantized_events
			if self.plot_quantized_preview and self.quantized_events is not None
			else self.events
		)
		values: list[float | int] = []
		grid_seconds = (
			float(self.quantization_settings["grid_seconds"])
			if self.plot_quantized_preview and self.quantization_settings is not None
			else None
		)
		grid_origin_seconds = (
			float(self.quantization_settings["origin_seconds"])
			if self.plot_quantized_preview and self.quantization_settings is not None
			else 0.0
		)
		for event in events:
			offset_frame = event["offset_frame"]
			if self.plot_quantized_preview and grid_seconds is not None:
				onset_grid = int(event["midi_onset_grid_index"])
				offset_grid = event.get("midi_offset_grid_index")
				if measure == "onset":
					values.append(onset_grid)
				elif offset_grid is not None and measure == "offset":
					values.append(int(offset_grid))
				elif offset_grid is not None and measure == "duration":
					values.append(int(offset_grid) - onset_grid)
				continue

			onset_frame = int(event["onset_frame"])
			if measure == "onset":
				values.append(onset_frame)
			elif offset_frame is not None and measure == "offset":
				values.append(int(offset_frame))
			elif offset_frame is not None and measure == "duration":
				duration_frames = int(offset_frame) - onset_frame
				if duration_frames > 0:
					values.append(duration_frames)

		if self.plot_quantized_preview and grid_seconds is not None:
			scale = grid_seconds if unit == "seconds" else grid_seconds * self.fps
			origin = (
				grid_origin_seconds
				if measure in ("onset", "offset")
				else 0.0
			)
			if unit == "frames":
				origin *= self.fps
		else:
			scale = 1.0 / self.fps if unit == "seconds" else 1.0
			origin = 0.0
		axis_values = [value * scale + origin for value in values]

		self.axes.clear()
		measure_name = str(self.measure_combo.currentText()).lower()
		if axis_values:
			counts = sorted(Counter(axis_values).items())
			positions = [value for value, _ in counts]
			if len(positions) > 1:
				bar_width = min(
					right - left
					for left, right in zip(positions, positions[1:])
				) * 0.7
			else:
				bar_width = 0.7 / self.fps if unit == "seconds" else 0.7
			self.axes.bar(
				positions,
				[count for _, count in counts],
				width=bar_width,
				color="#3f8065",
				edgecolor="#183d32",
			)
			labels = (
				[f"{position:.5f}" for position in positions]
				if unit == "seconds"
				else [
					str(int(position))
					if position.is_integer()
					else f"{position:.2f}"
					for position in positions
				]
			)
			self.axes.set_xticks(positions, labels, rotation=60, ha="right")
			unit_label = "seconds" if unit == "seconds" else "samples (frames)"
			self.axes.set_xlabel(f"{measure_name.title()} ({unit_label})")
			self.axes.set_ylabel("Key hits")
			self.axes.grid(axis="y", alpha=0.25)
			self._add_histogram_grid(min(positions), max(positions))
		else:
			self.axes.text(
				0.5,
				0.5,
				f"No {measure_name} values available",
				ha="center",
				va="center",
				transform=self.axes.transAxes,
			)
			self.axes.set_axis_off()

		self.summary_label.setText(
			f"{len(axis_values)} hits; {len(set(axis_values))} distinct "
			f"{measure_name} values"
		)
		self.figure.tight_layout()
		figure_width = max(8.0, len(set(axis_values)) * 0.34)
		self.figure.set_size_inches(figure_width, 4.4, forward=True)
		self.canvas.setMinimumWidth(round(figure_width * 100))
		self.canvas.draw_idle()

	def _add_histogram_grid(self, minimum: float, maximum: float) -> None:
		if minimum == maximum:
			return
		step = 60.0 / self.bpm_slider.value() / 4
		measure = self.measure_combo.currentData()
		if measure == "duration":
			origin_seconds = 0.0
		elif self.plot_quantized_preview and self.quantization_settings is not None:
			origin_seconds = float(self.quantization_settings["origin_seconds"])
		else:
			origin_seconds = min(
				int(event["onset_frame"]) / self.fps for event in self.events
			)
		unit_scale = self.fps if self.units_combo.currentData() == "frames" else 1.0
		origin = origin_seconds * unit_scale
		step *= unit_scale
		first = math.ceil((minimum - origin) / step)
		last = math.floor((maximum - origin) / step)
		positions = [origin + index * step for index in range(first, last + 1)]
		if positions:
			self.axes.vlines(
				positions,
				0,
				1,
				transform=self.axes.get_xaxis_transform(),
				color="#718096",
				linewidth=0.6,
				alpha=0.35,
				zorder=0,
			)

class PianoRollReaderWindow(QMainWindow):
	def __init__(self, video_path: Path | None = None) -> None:
		super().__init__()
		self.setWindowTitle("Piano Roll Reader")
		self.resize(1480, 900)
		self.setMinimumSize(980, 640)
		self.setStyleSheet(
			"QMainWindow { background: #eef1f4; }"
			"QWidget { color: #18212b; font-family: 'Segoe UI'; font-size: 10pt; }"
			"QPushButton { color: #18212b; background: #ffffff; padding: 7px 13px; "
			"border: 1px solid #aeb8c4; border-radius: 3px; }"
			"QPushButton:hover { background: #e1edf5; border-color: #607d91; }"
			"QPushButton:pressed { background: #c8dbe8; }"
			"QPushButton:disabled { color: #687582; background: #e1e5e9; "
			"border-color: #c4cbd2; }"
			"QComboBox { color: #18212b; background: #ffffff; padding: 5px 8px; "
			"border: 1px solid #aeb8c4; }"
			"QComboBox:disabled { color: #687582; background: #e1e5e9; }"
			"QComboBox QAbstractItemView { color: #18212b; background: #ffffff; "
			"selection-color: #18212b; selection-background-color: #c8dbe8; }"
			"QTableView { color: #18212b; background: #ffffff; "
			"gridline-color: #d9dee5; }"
			"QHeaderView::section { color: #18212b; background: #e3e8ed; "
			"padding: 5px; border: 0; border-bottom: 1px solid #c4cbd2; }"
			"QProgressBar { color: #18212b; background: #ffffff; "
			"border: 1px solid #aeb8c4; text-align: center; }"
			"QProgressBar::chunk { background: #3f8065; }"
		)

		self.video_path: Path | None = None
		self.capture: cv2.VideoCapture | None = None
		self.baseline_frame: np.ndarray | None = None
		self.keys: list[dict[str, object]] = []
		self.key_masks: list[np.ndarray] = []
		self.key_width = 0.0
		self.fps = 30.0
		self.frame_count = 0
		self.frame_index = -1
		self.output_path: Path | None = None
		self.events: list[dict[str, object]] = []
		self.waterfall_events: list[dict[str, object]] = []
		self.active_waterfall_hits: dict[str, dict[str, object]] = {}
		self.normalized_events: list[dict[str, object]] | None = None
		self.normalization_settings: dict[str, object] | None = None
		self.quantized_events: list[dict[str, object]] | None = None
		self.music21_quantized_events: list[dict[str, object]] | None = None
		self.quantization_settings: dict[str, object] | None = None
		self.music21_quantization_settings: dict[str, object] | None = None
		self.active_hits: dict[str, dict[str, object]] = {}
		self.keys_by_note: dict[str, dict[str, object]] = {}
		self.pending_table_events: dict[int, dict[str, object]] = {}
		self.current_frame: np.ndarray | None = None
		self.midi_tempo = 500_000
		self.midi_ticks_per_beat = 480
		self.timer = QTimer(self)
		self.timer.timeout.connect(self._process_next_frame)
		self.table_flush_timer = QTimer(self)
		self.table_flush_timer.setSingleShot(True)
		self.table_flush_timer.timeout.connect(self._flush_table_updates)

		self._build_ui()
		if video_path is not None:
			self.open_video(video_path)

	def _build_ui(self) -> None:
		central = QWidget()
		self.setCentralWidget(central)
		main_layout = QVBoxLayout(central)
		main_layout.setContentsMargins(12, 12, 12, 12)
		main_layout.setSpacing(10)

		toolbar = QHBoxLayout()
		self.open_button = QPushButton("Open video")
		self.open_button.clicked.connect(self.choose_video)
		toolbar.addWidget(self.open_button)
		self.play_button = QPushButton("Play")
		self.play_button.setEnabled(False)
		self.play_button.clicked.connect(self.toggle_play)
		toolbar.addWidget(self.play_button)
		self.step_button = QPushButton("Step frame")
		self.step_button.setEnabled(False)
		self.step_button.clicked.connect(self.step_frame)
		toolbar.addWidget(self.step_button)
		self.restart_button = QPushButton("Restart")
		self.restart_button.setEnabled(False)
		self.restart_button.clicked.connect(self.restart)
		toolbar.addWidget(self.restart_button)
		self.save_button = QPushButton("Export MIDI")
		self.save_button.setEnabled(False)
		self.save_button.clicked.connect(self.save_midi_as)
		toolbar.addWidget(self.save_button)
		toolbar.addWidget(QLabel("MIDI timing"))
		self.export_timing_combo = QComboBox()
		self.export_timing_combo.addItem("Original", "original")
		self.export_timing_combo.setEnabled(False)
		self.export_timing_combo.setToolTip(
			"Choose original detected timings or the quantized copy for MIDI export."
		)
		toolbar.addWidget(self.export_timing_combo)
		self.timeline_button = QPushButton("Timeline")
		self.timeline_button.setEnabled(False)
		self.timeline_button.clicked.connect(self.show_timeline_dialog)
		toolbar.addWidget(self.timeline_button)
		self.statistics_button = QPushButton("Statistics")
		self.statistics_button.setEnabled(False)
		self.statistics_button.clicked.connect(self.show_statistics_dialog)
		toolbar.addWidget(self.statistics_button)
		toolbar.addStretch(1)
		self.status_label = QLabel("Open a video to begin")
		self.status_label.setAlignment(
			Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
		)
		toolbar.addWidget(self.status_label)
		main_layout.addLayout(toolbar)

		splitter = QSplitter(Qt.Orientation.Horizontal)
		main_layout.addWidget(splitter, 1)

		video_panel = QWidget()
		video_layout = QVBoxLayout(video_panel)
		video_layout.setContentsMargins(0, 0, 8, 0)
		self.video_label = QLabel("Open a video to see detections")
		self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
		self.video_label.setMinimumSize(320, 180)
		self.video_label.setSizePolicy(
			QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
		)
		self.video_label.setStyleSheet("background: #101317; color: #d5dbe2;")
		self.video_label.installEventFilter(self)
		video_layout.addWidget(self.video_label, 1)
		self.frame_label = QLabel("Frame: -")
		video_layout.addWidget(self.frame_label)
		splitter.addWidget(video_panel)

		event_panel = QWidget()
		event_layout = QVBoxLayout(event_panel)
		event_layout.setContentsMargins(8, 0, 0, 0)
		self.left_hand_label = QLabel("Left hand (cyan): none")
		self.left_hand_label.setStyleSheet("color: #087fbd; font-weight: 700;")
		event_layout.addWidget(self.left_hand_label)
		self.right_hand_label = QLabel("Right hand (green): none")
		self.right_hand_label.setStyleSheet("color: #16833a; font-weight: 700;")
		event_layout.addWidget(self.right_hand_label)
		heading = QHBoxLayout()
		title = QLabel("Detected key hits")
		title.setStyleSheet("font-size: 12pt; font-weight: 700; padding-top: 6px;")
		heading.addWidget(title)
		heading.addStretch(1)
		self.table_updates_checkbox = QCheckBox("Update table")
		self.table_updates_checkbox.setChecked(True)
		self.table_updates_checkbox.setToolTip(
			"Pause table writes while continuing to track key hits. "
			"Re-enabling catches the table up."
		)
		self.table_updates_checkbox.toggled.connect(self._set_table_updates_enabled)
		heading.addWidget(self.table_updates_checkbox)
		event_layout.addLayout(heading)

		self.event_model = KeyHitTableModel(self)
		self.event_table = QTableView()
		self.event_table.setModel(self.event_model)
		self.event_table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
		self.event_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
		self.event_table.setAlternatingRowColors(True)
		self.event_table.verticalHeader().setVisible(False)
		header = self.event_table.horizontalHeader()
		header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
		for column, width in enumerate((58, 70, 55, 88, 82, 92, 82, 82)):
			self.event_table.setColumnWidth(column, width)
		header.setStretchLastSection(True)
		event_layout.addWidget(self.event_table, 1)
		self.progress_bar = QProgressBar()
		self.progress_bar.setRange(0, 1)
		event_layout.addWidget(self.progress_bar)
		self.output_label = QLabel("MIDI will be saved beside the video.")
		self.output_label.setWordWrap(True)
		self.output_label.setStyleSheet("color: #5c6672;")
		event_layout.addWidget(self.output_label)
		splitter.addWidget(event_panel)
		splitter.setStretchFactor(0, 3)
		splitter.setStretchFactor(1, 1)
		splitter.setCollapsible(0, False)
		splitter.setCollapsible(1, True)
		splitter.setSizes([1200, 400])
		splitter.splitterMoved.connect(lambda *_args: self._resize_video())

	def choose_video(self) -> None:
		path, _ = QFileDialog.getOpenFileName(
			self,
			"Open piano tutorial video",
			str(DEFAULT_VIDEO.parent),
			"Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)",
		)
		if path:
			self.open_video(Path(path))

	def open_video(self, video_path: Path) -> None:
		self.pause()
		if self.capture is not None:
			self.capture.release()
		capture = cv2.VideoCapture(str(video_path))
		if not capture.isOpened():
			QMessageBox.critical(self, "Cannot open video", str(video_path))
			return

		self.capture = capture
		self.video_path = video_path
		self.fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
		if not np.isfinite(self.fps) or self.fps <= 0:
			self.fps = 30.0
		self.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
		try:
			self.baseline_frame = read_video_frame(video_path, 0)
			baseline_path = video_path.with_name(
				f"{video_path.stem}_key_contours.json"
			)
			self.keys, self.key_width = save_frame_zero_baseline(
				self.baseline_frame, video_path, baseline_path
			)
			self.keys_by_note = {str(key["note"]): key for key in self.keys}
			self._prepare_key_masks()
		except (SystemExit, cv2.error, OSError, ValueError) as error:
			capture.release()
			self.capture = None
			QMessageBox.critical(self, "Cannot analyze keyboard", str(error))
			return

		self.output_path = self._unused_output_path(
			video_path.with_name(f"{video_path.stem}_key_hits.mid")
		)
		self.output_label.setText(f"MIDI: {self.output_path}")
		self.events.clear()
		self.waterfall_events.clear()
		self.active_waterfall_hits.clear()
		self.normalized_events = None
		self.normalization_settings = None
		self._clear_quantized_export()
		self.active_hits.clear()
		self.pending_table_events.clear()
		self.table_flush_timer.stop()
		self.event_model.clear()
		self.frame_index = -1
		self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
		self.progress_bar.setRange(0, max(1, self.frame_count))
		self.progress_bar.setValue(0)
		for button in (
			self.play_button,
			self.step_button,
			self.restart_button,
			self.save_button,
			self.timeline_button,
			self.statistics_button,
		):
			button.setEnabled(True)
		self.export_timing_combo.setEnabled(True)
		self.status_label.setText(video_path.name)
		self._process_next_frame()

	@staticmethod
	def _unused_output_path(path: Path) -> Path:
		if not path.exists():
			return path
		for number in range(2, 1000):
			candidate = path.with_name(f"{path.stem}_{number}{path.suffix}")
			if not candidate.exists():
				return candidate
		return path.with_name(f"{path.stem}_{time.time_ns()}{path.suffix}")

	def _prepare_key_masks(self) -> None:
		kernel_size = max(3, round(self.key_width * 0.28) | 1)
		shrink_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
		self.key_masks = []
		for key in self.keys:
			x, y, width, height = key["bbox"]
			contour = np.array(key["contour"], dtype=np.int32).reshape(-1, 1, 2)
			local_contour = contour - (x, y)
			mask = np.zeros((height, width), dtype=np.uint8)
			cv2.drawContours(mask, [local_contour], -1, 255, cv2.FILLED)
			mask = cv2.erode(mask, shrink_kernel)
			self.key_masks.append(mask > 0)

	def _score_frame(self, frame: np.ndarray) -> dict[str, str]:
		if self.baseline_frame is None or self.baseline_frame.shape != frame.shape:
			return {}
		pressed: dict[str, str] = {}
		for key, interior_mask in zip(self.keys, self.key_masks):
			x, y, width, height = key["bbox"]
			mask_pixel_count = np.count_nonzero(interior_mask)
			if not mask_pixel_count:
				continue
			frame_region = frame[y : y + height, x : x + width]
			baseline_region = self.baseline_frame[y : y + height, x : x + width]
			region_hsv = cv2.cvtColor(frame_region, cv2.COLOR_BGR2HSV)
			color_delta = cv2.absdiff(frame_region, baseline_region).max(axis=2)
			changed_region = (region_hsv[:, :, 1] > 100) & (color_delta > 80)
			interior_changes = changed_region & interior_mask
			if np.count_nonzero(interior_changes) / mask_pixel_count < 0.55:
				continue
			hues = region_hsv[:, :, 0][interior_changes]
			cyan_fraction = float(((hues >= 85) & (hues <= 110)).mean())
			green_fraction = float(((hues >= 45) & (hues <= 80)).mean())
			if max(cyan_fraction, green_fraction) < 0.5:
				continue
			pressed[str(key["note"])] = (
				"left" if cyan_fraction > green_fraction else "right"
			)
		return pressed

	def toggle_play(self) -> None:
		if self.timer.isActive():
			self.pause()
		elif self.capture is not None:
			self.timer.start(max(1, round(1000 / self.fps)))
			self.play_button.setText("Pause")

	def pause(self) -> None:
		self.timer.stop()
		if hasattr(self, "play_button"):
			self.play_button.setText("Play")

	def step_frame(self) -> None:
		if not self.timer.isActive():
			self._process_next_frame()

	def restart(self) -> None:
		if self.capture is None:
			return
		self.pause()
		self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
		self.frame_index = -1
		self.events.clear()
		self.waterfall_events.clear()
		self.active_waterfall_hits.clear()
		self.normalized_events = None
		self.normalization_settings = None
		self._clear_quantized_export()
		self.active_hits.clear()
		self.pending_table_events.clear()
		self.table_flush_timer.stop()
		self.event_model.clear()
		self._process_next_frame()

	def _process_next_frame(self) -> None:
		if self.capture is None or self.baseline_frame is None:
			return
		ok, frame = self.capture.read()
		if not ok:
			self._finish_active_hits(self.frame_index + 1)
			self._finish_waterfall_hits(self.frame_index + 1)
			self.pause()
			self.status_label.setText("Processing complete")
			return

		self.frame_index += 1
		pressed = self._score_frame(frame)
		waterfall_notes = score_waterfall_notes(frame, self.keys, self.key_width)
		self._update_waterfall_events(waterfall_notes, self.frame_index)
		changed = self._update_events(pressed, self.frame_index)
		display_stride = max(1, round(self.fps / 30))
		if changed or self.frame_index % display_stride == 0:
			self._update_hand_labels(pressed)
			self._show_frame(frame, pressed)
			self._update_progress()

	def _update_events(self, pressed: dict[str, str], frame_number: int) -> bool:
		changed = False
		for note in list(self.active_hits):
			if note not in pressed:
				self._close_event(self.active_hits.pop(note), frame_number)
				changed = True

		for note, hand in pressed.items():
			if note in self.active_hits:
				continue
			event: dict[str, object] = {
				"event_id": len(self.events) + 1,
				"note": note,
				"midi_note": NOTE_TO_MIDI[note],
				"hand": hand,
				"onset_frame": frame_number,
				"offset_frame": None,
				"video_onset_seconds": frame_number / self.fps,
				"video_offset_seconds": None,
				"duration_seconds": None,
			}
			self.events.append(event)
			self.active_hits[note] = event
			if self.table_updates_checkbox.isChecked():
				self._queue_table_event(event)
			changed = True
		if changed:
			self._clear_quantized_export()
		return changed

	def _update_waterfall_events(
		self, waterfall_notes: dict[str, str], frame_number: int
	) -> None:
		for note, event in list(self.active_waterfall_hits.items()):
			last_seen = int(event["last_seen_frame"])
			if note not in waterfall_notes and frame_number - last_seen > 1:
				event["offset_frame"] = last_seen + 1
				event.pop("last_seen_frame", None)
				self.active_waterfall_hits.pop(note)
		for note, hand in waterfall_notes.items():
			if note in self.active_waterfall_hits:
				self.active_waterfall_hits[note]["last_seen_frame"] = frame_number
				continue
			event = {
				"event_id": len(self.waterfall_events) + 1,
				"note": note,
				"midi_note": NOTE_TO_MIDI[note],
				"hand": hand,
				"onset_frame": frame_number,
				"offset_frame": None,
				"last_seen_frame": frame_number,
			}
			self.waterfall_events.append(event)
			self.active_waterfall_hits[note] = event

	def _finish_waterfall_hits(self, offset_frame: int) -> None:
		for event in self.active_waterfall_hits.values():
			last_seen = int(event.pop("last_seen_frame", offset_frame - 1))
			event["offset_frame"] = min(offset_frame, last_seen + 1)
		self.active_waterfall_hits.clear()

	def _close_event(self, event: dict[str, object], offset_frame: int) -> None:
		onset_frame = int(event["onset_frame"])
		video_offset = offset_frame / self.fps
		event["offset_frame"] = offset_frame
		event["video_offset_seconds"] = video_offset
		event["duration_seconds"] = video_offset - float(
			event["video_onset_seconds"]
		)
		if self.table_updates_checkbox.isChecked():
			self._queue_table_event(event)

	def _finish_active_hits(self, offset_frame: int) -> None:
		if self.active_hits and any((
			self.quantized_events is not None,
			self.music21_quantized_events is not None,
			self.normalized_events is not None,
		)):
			self._clear_quantized_export()
		for event in list(self.active_hits.values()):
			self._close_event(event, offset_frame)
		self.active_hits.clear()

	def _upsert_event_row(self, event: dict[str, object], active: bool) -> None:
		self.event_model.upsert_event(event)

	def _queue_table_event(self, event: dict[str, object]) -> None:
		self.pending_table_events[int(event["event_id"])] = event
		if not self.table_flush_timer.isActive():
			self.table_flush_timer.start(TABLE_FLUSH_INTERVAL_MS)

	def _flush_table_updates(self) -> None:
		if not self.table_updates_checkbox.isChecked():
			self.pending_table_events.clear()
			return

		event_ids = list(self.pending_table_events)[:TABLE_FLUSH_BATCH_SIZE]
		for event_id in event_ids:
			event = self.pending_table_events.pop(event_id)
			self._upsert_event_row(
				event,
				active=event["offset_frame"] is None,
			)

		if self.pending_table_events:
			self.table_flush_timer.start(TABLE_FLUSH_INTERVAL_MS)

	def _set_table_updates_enabled(self, enabled: bool) -> None:
		if not enabled:
			self.table_flush_timer.stop()
			self.pending_table_events.clear()
			return
		self.pending_table_events = {
			int(event["event_id"]): event for event in self.events
		}
		if self.pending_table_events:
			self.table_flush_timer.start(1)

	def _update_hand_labels(self, pressed: dict[str, str]) -> None:
		left = sorted(note for note, hand in pressed.items() if hand == "left")
		right = sorted(note for note, hand in pressed.items() if hand == "right")
		self.left_hand_label.setText(
			f"Left hand (cyan): {', '.join(left) if left else 'none'}"
		)
		self.right_hand_label.setText(
			f"Right hand (green): {', '.join(right) if right else 'none'}"
		)

	def _show_frame(self, frame: np.ndarray, pressed: dict[str, str]) -> None:
		result = frame.copy()
		for key in self.keys:
			note = str(key["note"])
			if note not in pressed:
				color = (150, 150, 150)
			elif key["kind"] == "white":
				color = (139, 0, 0)
			else:
				color = (0, 0, 139)
			contour = np.array(key["contour"], dtype=np.int32).reshape(-1, 1, 2)
			cv2.drawContours(result, [contour], -1, color, 2)
			if note in pressed:
				x, y, _, _ = key["bbox"]
				cv2.putText(
					result,
					note,
					(x, max(15, y - 5)),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.4,
					color,
					1,
					cv2.LINE_AA,
				)
		self.current_frame = result
		self._resize_video()

	def _resize_video(self) -> None:
		if self.current_frame is None:
			return
		frame_height, frame_width = self.current_frame.shape[:2]
		rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
		image = QImage(
			rgb.data,
			frame_width,
			frame_height,
			rgb.strides[0],
			QImage.Format.Format_RGB888,
		).copy()
		pixmap = QPixmap.fromImage(image).scaled(
			self.video_label.size(),
			Qt.AspectRatioMode.IgnoreAspectRatio,
			Qt.TransformationMode.SmoothTransformation,
		)
		self.video_label.setPixmap(pixmap)
		self.frame_label.setText(
			f"Frame {self.frame_index} / {max(0, self.frame_count - 1)}"
		)

	def resizeEvent(self, event) -> None:
		super().resizeEvent(event)
		self._resize_video()

	def eventFilter(self, watched, event) -> bool:
		if watched is self.video_label and event.type() == QEvent.Type.Resize:
			self._resize_video()
		return super().eventFilter(watched, event)

	def _update_progress(self) -> None:
		if self.frame_count:
			self.progress_bar.setValue(min(self.frame_index + 1, self.frame_count))
		elapsed = self.frame_index / self.fps
		total = self.frame_count / self.fps if self.frame_count else 0.0
		self.status_label.setText(f"{elapsed:.2f}s / {total:.2f}s at {self.fps:.3f} fps")

	def save_midi_as(self) -> None:
		if self.video_path is None:
			return
		folder = QFileDialog.getExistingDirectory(
			self,
			"Choose folder for all four MIDI versions",
			str(self.video_path.parent),
		)
		if not folder:
			return
		try:
			self._ensure_quantization_versions()
		except Exception as error:
			QMessageBox.critical(self, "Could not prepare MIDI versions", str(error))
			return

		original_output_path = self.output_path
		original_timing_index = self.export_timing_combo.currentIndex()
		export_paths = []
		try:
			for version, suffix in (
				("original", "captured"),
				("normalized", "mean_normalized"),
				("manual", "manual"),
				("music21", "music21"),
			):
				self.export_timing_combo.setCurrentIndex(
					self.export_timing_combo.findData(version)
				)
				self.output_path = Path(folder) / (
					f"{self.video_path.stem}_key_hits_{suffix}.mid"
				)
				self._write_midi(quiet=True)
				export_paths.append(self.output_path)
		finally:
			self.output_path = original_output_path
			self.export_timing_combo.setCurrentIndex(original_timing_index)

		self.output_label.setText(
			"Exported captured, normalized, manual, and music21 MIDI files to "
			f"{folder}"
		)
		self.status_label.setText("Exported all four MIDI timing versions")

	def _ensure_quantization_versions(self) -> None:
		if (
			self.quantized_events is not None
			and self.music21_quantized_events is not None
			and self.normalized_events is not None
		):
			return
		settings = self.quantization_settings or self.music21_quantization_settings
		bpm = int(settings["bpm"]) if settings is not None else SONG_REFERENCE_BPM
		beat_length = (
			float(settings["beat_length"])
			if settings is not None
			else 0.5
		)
		if self.quantized_events is None:
			self.quantized_events, self.quantization_settings, _ = (
				quantize_events_manually(
					self.events,
					self.fps,
					bpm,
					beat_length,
					self.frame_index,
				)
			)
		if self.music21_quantized_events is None:
			self.music21_quantized_events = quantize_events_with_music21(
				self.events,
				self.fps,
				bpm,
				beat_length,
				self.frame_index,
			)
			self.music21_quantization_settings = {
				"bpm": bpm,
				"beat_length": beat_length,
			}
		if self.normalized_events is None:
			self.normalized_events, self.normalization_settings = normalize_events_to_reference(
				self.events,
				self.waterfall_events,
				self.fps,
				SONG_REFERENCE_BPM,
				0.5,
				self.frame_index,
			)
		for label, data in (
			("Mean normalized", "normalized"),
			("Manual", "manual"),
			("music21", "music21"),
		):
			if self.export_timing_combo.findData(data) < 0:
				self.export_timing_combo.addItem(label, data)
		self.export_timing_combo.setEnabled(True)

	def _set_normalized_export(
		self,
		events: list[dict[str, object]],
		settings: dict[str, object],
	) -> None:
		self.normalized_events = [event.copy() for event in events]
		self.normalization_settings = settings.copy()
		if self.export_timing_combo.findData("normalized") < 0:
			self.export_timing_combo.addItem("Mean normalized", "normalized")
		self.export_timing_combo.setEnabled(True)
		self.status_label.setText(
			f"Mean normalized to {settings['bpm']} BPM, "
			f"{self._beat_label(float(settings['beat_length']))} grid"
		)

	def show_timeline_dialog(self) -> None:
		self._show_analysis_dialog("timeline")

	def show_statistics_dialog(self) -> None:
		self._show_analysis_dialog("statistics")

	def _show_analysis_dialog(self, view_mode: str) -> None:
		was_playing = self.timer.isActive()
		if was_playing:
			self.pause()
		dialog = DurationAnalysisDialog(
			self.events,
			self.fps,
			self,
			self._set_quantized_export if view_mode == "timeline" else None,
			self._set_music21_export if view_mode == "timeline" else None,
			current_frame=self.frame_index,
			view_mode=view_mode,
			waterfall_events=self.waterfall_events,
			on_normalized_accept=(
				self._set_normalized_export if view_mode == "timeline" else None
			),
		)
		if view_mode == "timeline":
			dialog.setWindowState(
				dialog.windowState() | Qt.WindowState.WindowMaximized
			)
		dialog.exec()
		if was_playing:
			self.toggle_play()

	def _set_quantized_export(
		self,
		events: list[dict[str, object]],
		settings: dict[str, object],
	) -> None:
		self.quantized_events = [event.copy() for event in events]
		self.quantization_settings = settings.copy()
		if self.export_timing_combo.findData("manual") < 0:
			self.export_timing_combo.addItem("Manual", "manual")
		self.export_timing_combo.setEnabled(True)
		self.export_timing_combo.setCurrentIndex(
			self.export_timing_combo.findData("manual")
		)
		self.status_label.setText(
			f"Quantized export ready: {settings['bpm']} BPM, "
			f"{self._beat_label(float(settings['beat_length']))} grid"
		)

	def _set_music21_export(
		self,
		events: list[dict[str, object]],
		settings: dict[str, object],
	) -> None:
		self.music21_quantized_events = [event.copy() for event in events]
		self.music21_quantization_settings = settings.copy()
		if self.export_timing_combo.findData("music21") < 0:
			self.export_timing_combo.addItem("music21", "music21")
		self.export_timing_combo.setEnabled(True)
		self.export_timing_combo.setCurrentIndex(
			self.export_timing_combo.findData("music21")
		)
		self.status_label.setText(
			f"music21 export ready: {settings['bpm']} BPM, "
			f"{self._beat_label(float(settings['beat_length']))} grid"
		)

	@staticmethod
	def _beat_label(beat_length: float) -> str:
		labels = {
			1.0: "quarter note (1 beat)",
			0.5: "eighth note (1/2 beat)",
			0.25: "16th note (1/4 beat)",
			0.125: "32nd note (1/8 beat)",
			0.0625: "64th note (1/16 beat)",
			0.03125: "128th note (1/32 beat)",
		}
		return labels.get(beat_length, f"{beat_length:g} beat")

	def _clear_quantized_export(self) -> None:
		if (
			self.quantized_events is None
			and self.music21_quantized_events is None
			and self.normalized_events is None
		):
			return
		self.quantized_events = None
		self.music21_quantized_events = None
		self.normalized_events = None
		self.quantization_settings = None
		self.music21_quantization_settings = None
		self.normalization_settings = None
		for data in ("manual", "music21", "normalized"):
			index = self.export_timing_combo.findData(data)
			if index >= 0:
				self.export_timing_combo.removeItem(index)
		self.export_timing_combo.setCurrentIndex(
			self.export_timing_combo.findData("original")
		)
		self.status_label.setText(
			"New hits were detected; re-quantize before exporting quantized timing."
		)

	def _write_midi(self, quiet: bool) -> None:
		if self.output_path is None or self.video_path is None:
			return
		try:
			timing_mode = self.export_timing_combo.currentData()
			if timing_mode == "music21" and self.music21_quantized_events is not None:
				export_events = self.music21_quantized_events
				quantization_settings = self.music21_quantization_settings
				midi_onset_key = "music21_onset_seconds"
				midi_offset_key = "music21_offset_seconds"
			elif timing_mode == "normalized" and self.normalized_events is not None:
				export_events = self.normalized_events
				quantization_settings = self.normalization_settings
				midi_onset_key = "normalized_onset_seconds"
				midi_offset_key = "normalized_offset_seconds"
			elif timing_mode == "manual" and self.quantized_events is not None:
				export_events = self.quantized_events
				quantization_settings = self.quantization_settings
				midi_onset_key = "midi_onset_seconds"
				midi_offset_key = "midi_offset_seconds"
			else:
				export_events = self.events
				quantization_settings = None
				midi_onset_key = "video_onset_seconds"
				midi_offset_key = "video_offset_seconds"
			export_tempo = self.midi_tempo
			if quantization_settings is not None:
				export_tempo = round(
					60_000_000 / int(quantization_settings["bpm"])
				)
			midi_file = mido.MidiFile(
				type=1, ticks_per_beat=self.midi_ticks_per_beat
			)
			for channel, hand in enumerate(("left", "right")):
				track = mido.MidiTrack()
				midi_file.tracks.append(track)
				track.append(mido.MetaMessage("track_name", name=f"{hand.title()} hand"))
				if channel == 0:
					track.append(
						mido.MetaMessage("set_tempo", tempo=export_tempo)
					)
				track.append(
					mido.Message("program_change", channel=channel, program=0)
				)
				midi_events: list[tuple[int, int, int, str]] = []
				for event in export_events:
					if event["hand"] != hand:
						continue
					onset_seconds = float(event.get(midi_onset_key, event["video_onset_seconds"]))
					offset_frame = event["offset_frame"]
					offset_seconds = event.get(midi_offset_key)
					if offset_seconds is None:
						offset_seconds = (
							int(offset_frame)
							if offset_frame is not None
							else self.frame_index + 1
						) / self.fps
					offset_seconds = float(offset_seconds)
					note = int(event["midi_note"])
					onset_tick = round(
						mido.second2tick(
							onset_seconds,
							self.midi_ticks_per_beat,
							export_tempo,
						)
					)
					offset_tick = max(
						onset_tick + 1,
						round(
							mido.second2tick(
								offset_seconds,
								self.midi_ticks_per_beat,
								export_tempo,
							)
						),
					)
					midi_events.append((onset_tick, 1, note, "note_on"))
					midi_events.append((offset_tick, 0, note, "note_off"))

				previous_tick = 0
				for tick, _, note, event_type in sorted(midi_events):
					track.append(
						mido.Message(
							event_type,
							channel=channel,
							note=note,
							velocity=90 if event_type == "note_on" else 0,
							time=tick - previous_tick,
						)
					)
					previous_tick = tick
				track.append(mido.MetaMessage("end_of_track"))

			self.output_path.parent.mkdir(parents=True, exist_ok=True)
			midi_file.save(str(self.output_path))
			if not quiet:
				self.status_label.setText(
					f"Exported {len(export_events)} {timing_mode} key hits to MIDI"
				)
		except (OSError, ValueError) as error:
			if not quiet:
				QMessageBox.critical(self, "Could not export MIDI", str(error))

	def closeEvent(self, event) -> None:
		self.pause()
		if self.capture is not None:
			self.capture.release()
		event.accept()


def main() -> None:
	parser = argparse.ArgumentParser(description="Track piano key hits in a video.")
	parser.add_argument("video", nargs="?", type=Path, default=DEFAULT_VIDEO)
	args = parser.parse_args()
	app = QApplication([])
	window = PianoRollReaderWindow(args.video)
	window.showMaximized()
	app.exec()


if __name__ == "__main__":
	main()