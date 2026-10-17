import argparse
import math
import shutil
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

import cv2
import mido
import numpy as np
from pianorolltempo import track_beats
from music21 import note as music21_note
from music21 import stream as music21_stream
from music21 import note as music21_note
from music21 import stream as music21_stream
from matplotlib.backends.backend_qtagg import (
	FigureCanvasQTAgg,
	NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6.QtCore import (
	QAbstractTableModel,
	QEvent,
	QModelIndex,
	QProcess,
	QPointF,
	QRectF,
	QSize,
	Qt,
	QTimer,
)
from PySide6.QtPdf import QPdfDocument
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
	QSurfaceFormat,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
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
	QMenu,
	QMessageBox,
	QPlainTextEdit,
	QProgressBar,
	QPushButton,
	QScrollArea,
	QSizePolicy,
	QSlider,
	QSpinBox,
	QSplitter,
	QTableView,
	QToolTip,
	QVBoxLayout,
	QWidget,
)

from pianorollscore import (
	ScoreLayout,
	ScoreSource,
	adjust_duration,
	build_score,
	current_length,
	find_note,
	insert_note,
	lilypond_excerpt,
	video_time_map,
	find_lilypond,
	load_layout,
	write_score_files,
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
SONG_REFERENCE_BPM = 160


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


def normalize_events_tempo_adaptive(
	events: list[dict[str, object]],
	waterfall_events: list[dict[str, object]],
	fps: float,
	bpm: int = SONG_REFERENCE_BPM,
	beat_length: float = 0.5,
	current_frame: int = 0,
	origin_frame: int | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
	"""Follow the performance tempo, then render the beats at a fixed tempo."""
	normalized, settings = normalize_events_to_reference(
		events, waterfall_events, fps, bpm, beat_length, current_frame, origin_frame
	)
	if not normalized:
		return normalized, settings
	slots_per_beat = 1.0 / beat_length
	warp, info = track_beats(
		[int(event["onset_frame"]) / fps for event in normalized],
		float(bpm),
		slots_per_beat,
	)
	grid_seconds = float(settings["grid_seconds"])
	origin_seconds = float(settings["origin_seconds"])
	for event in normalized:
		onset_slot = max(0, round(warp(int(event["onset_frame"]) / fps)))
		offset_frame = event["offset_frame"]
		if offset_frame is None:
			offset_frame = max(current_frame, int(event["onset_frame"]) + 1)
		offset_slot = max(
			onset_slot + 1, round(warp(int(offset_frame) / fps))
		)
		event["normalized_onset_grid_index"] = onset_slot
		event["normalized_offset_grid_index"] = offset_slot
		event["normalized_onset_seconds"] = origin_seconds + onset_slot * grid_seconds
		event["normalized_offset_seconds"] = origin_seconds + offset_slot * grid_seconds
	settings = dict(settings)
	settings["tracked_initial_bpm"] = info["initial_bpm"]
	settings["tracked_final_bpm"] = info["final_bpm"]
	return normalized, settings


def compute_all_normalizations(
	events: list[dict[str, object]],
	waterfall_events: list[dict[str, object]],
	fps: float,
	current_frame: int = 0,
) -> dict[str, tuple[list[dict[str, object]], dict[str, object]]]:
	"""Every normalization variant, keyed like the timeline dialog sources."""
	waterfall_origin = min(
		(int(event["onset_frame"]) for event in waterfall_events), default=None
	)
	combined_origin = min(
		(
			int(event["onset_frame"])
			for event in combine_detection_and_waterfall(events, waterfall_events, fps)
		),
		default=None,
	)
	specs = {
		"detection": (events, [], None),
		"waterfall": (waterfall_events, [], waterfall_origin),
		"mean": (events, waterfall_events, combined_origin),
	}
	results = {}
	for name, (source, extra, origin) in specs.items():
		if not source:
			continue
		for prefix, function in (
			("normalized", normalize_events_to_reference),
			("normalized_adaptive", normalize_events_tempo_adaptive),
		):
			key = f"{prefix}_{name}" if name != "mean" or prefix != "normalized" else "normalized"
			if name == "mean" and prefix == "normalized_adaptive":
				key = "normalized_adaptive"
			results[key] = function(
				source, extra, fps, SONG_REFERENCE_BPM, 0.5, current_frame, origin
			)
	return results


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


class PianoRollTimeline(QOpenGLWidget):
	def __init__(self, parent=None) -> None:
		super().__init__(parent)
		surface_format = self.format()
		surface_format.setSwapInterval(1)
		surface_format.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
		self.setFormat(surface_format)
		self.setUpdateBehavior(QOpenGLWidget.UpdateBehavior.PartialUpdate)
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
		self.active_notes: dict[str, str] = {}
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
		is_reference = bar.get("source") in ("detected", "normalized", "waterfall-reference", "combined-reference")
		if (1 if self.raw_bars else 0) + len(self.visible_sources) <= 1:
			center_y = base_y
			bar_height = max(4.0, min(18.0, row_height * 0.72))
		else:
			lane = -1 if is_reference else (0 if bar.get("source") == "manual" else 1)
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

	def paintGL(self) -> None:
		painter = QPainter(self)
		dirty = QRectF(self.rect())
		painter.fillRect(self.rect(), QColor("#0b1218"))
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
			for row, note_name in enumerate(self.notes):
				row_top = chart.bottom() - (row + 1) * row_height
				row_bottom = chart.bottom() - row * row_height
				if row_bottom < dirty.top() or row_top > dirty.bottom():
					continue
				midi_note = NOTE_TO_MIDI[note_name]
				is_white_key = midi_note % 12 in {0, 2, 4, 5, 7, 9, 11}
				active_hand = self.active_notes.get(note_name)
				key_color = (
					"#28a9bd" if active_hand == "left" else "#42a45f"
				) if active_hand is not None else ("#f1eee4" if is_white_key else "#172027")
				key_width = 76 if is_white_key else 52
				key_rect = QRectF(chart.left() - key_width - 18, row_top, key_width, row_height)
				painter.fillRect(key_rect, QColor(key_color))
				painter.setPen(QPen(QColor("#52616a" if is_white_key else "#72808a"), 0.8))
				painter.drawRect(key_rect)
				painter.setPen(QColor("#172027" if is_white_key else "#f0f5f7"))
				painter.setFont(QFont(painter.font().family(), 7, QFont.Weight.DemiBold))
				painter.drawText(
					key_rect.adjusted(2, 0, -3, 0),
					Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
					note_name,
				)
				if active_hand is not None:
					painter.setPen(QPen(QColor("#ffe49a"), 1.5))
					painter.drawRect(key_rect.adjusted(1, 1, -1, -1))

		painter.save()
		painter.setClipRect(chart)
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
		painter.restore()

		painter.save()
		painter.setClipRect(chart)
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
		painter.restore()
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
		if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
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
		if event.key() in (Qt.Key.Key_Plus, Qt.Key.Key_Equal, Qt.Key.Key_Minus) and self.on_zoom is not None:
			center = (self.window[0] + self.window[1]) / 2
			self.on_zoom(-1 if event.key() == Qt.Key.Key_Minus else 1, center)
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

	def set_active_notes(self, active_notes: dict[str, str]) -> None:
		if active_notes == self.active_notes:
			return
		changed_notes = set(self.active_notes) | set(active_notes)
		previous_active_notes = self.active_notes
		self.active_notes = active_notes.copy()
		row_height = self.chart_rect.height() / max(1, len(self.notes))
		for row, note_name in enumerate(self.notes):
			if note_name not in changed_notes:
				continue
			if previous_active_notes.get(note_name) == self.active_notes.get(note_name):
				continue
			y = self.chart_rect.bottom() - (row + 1) * row_height
			self.update(0, round(y), round(self.chart_rect.left()), max(1, round(row_height) + 1))

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


class ScoreView(QWidget):
	"""Renders the compiled score PDF page by page and paints the playhead."""

	PAGE_MARGIN = 14
	DPI_SCALE = 96.0 / 72.0

	def __init__(self, parent=None) -> None:
		super().__init__(parent)
		self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
		self.document = QPdfDocument(self)
		self.zoom = 1.0
		self.images: list[QImage] = []
		self.page_rects: list[QRectF] = []
		self.playhead: tuple[int, float, float, float] | None = None
		self.playhead_visible = True
		self.on_click = None
		self.on_context = None
		self.on_zoom = None
		self.on_toggle_play = None
		self.setMinimumSize(400, 300)

	def load(self, pdf_path: str) -> bool:
		self.document.load(pdf_path)
		if self.document.status() != QPdfDocument.Status.Ready:
			return False
		self._render_pages()
		return True

	def clear(self) -> None:
		self.document.close()
		self.images = []
		self.page_rects = []
		self.playhead = None
		self.setFixedSize(400, 300)
		self.update()

	def page_width_pixels(self) -> float:
		if self.document.pageCount() == 0:
			return 1.0
		return self.document.pagePointSize(0).width() * self.DPI_SCALE

	def set_zoom(self, zoom: float) -> None:
		zoom = max(0.3, min(4.0, zoom))
		if abs(zoom - self.zoom) < 1e-6 and self.images:
			return
		self.zoom = zoom
		self._render_pages()

	def _render_pages(self) -> None:
		self.images = []
		self.page_rects = []
		y = float(self.PAGE_MARGIN)
		width = 0.0
		for page in range(self.document.pageCount()):
			point_size = self.document.pagePointSize(page)
			size = QSize(
				round(point_size.width() * self.DPI_SCALE * self.zoom),
				round(point_size.height() * self.DPI_SCALE * self.zoom),
			)
			image = self.document.render(page, size)
			self.images.append(image)
			self.page_rects.append(QRectF(self.PAGE_MARGIN, y, size.width(), size.height()))
			y += size.height() + self.PAGE_MARGIN
			width = max(width, size.width())
		self.setFixedSize(round(width + 2 * self.PAGE_MARGIN), round(y))
		self.update()

	def playhead_point(self) -> tuple[float, float] | None:
		if self.playhead is None:
			return None
		page, fx, top, bottom = self.playhead
		rect = self.page_rects[page]
		return rect.left() + fx * rect.width(), rect.top() + (top + bottom) / 2 * rect.height()

	def set_playhead(self, location: tuple[int, float, float, float] | None) -> None:
		if location is not None and location[0] >= len(self.page_rects):
			location = None
		self.playhead = location
		self.update()

	def set_playhead_visible(self, visible: bool) -> None:
		self.playhead_visible = visible
		self.update()

	def paintEvent(self, event) -> None:
		painter = QPainter(self)
		painter.fillRect(self.rect(), QColor("#8a949c"))
		for rect, image in zip(self.page_rects, self.images):
			if not rect.intersects(QRectF(event.rect())):
				continue
			painter.fillRect(rect, QColor("#ffffff"))
			painter.drawImage(rect.topLeft(), image)
		if self.playhead_visible and self.playhead is not None:
			page, fx, top, bottom = self.playhead
			rect = self.page_rects[page]
			x = round(rect.left() + fx * rect.width())
			painter.setPen(QPen(QColor("#ff4b45"), 2.0))
			painter.drawLine(x, round(rect.top() + top * rect.height()), x, round(rect.top() + bottom * rect.height()))

	def mousePressEvent(self, event) -> None:
		self.setFocus(Qt.FocusReason.MouseFocusReason)
		if event.button() != Qt.MouseButton.LeftButton or self.on_click is None:
			return
		position = event.position()
		for page, rect in enumerate(self.page_rects):
			if rect.contains(position):
				self.on_click(
					page,
					(position.x() - rect.left()) / rect.width(),
					(position.y() - rect.top()) / rect.height(),
				)
				break

	def contextMenuEvent(self, event) -> None:
		if self.on_context is None:
			return
		position = event.pos()
		for page, rect in enumerate(self.page_rects):
			if rect.contains(QPointF(position)):
				self.on_context(
					page,
					(position.x() - rect.left()) / rect.width(),
					(position.y() - rect.top()) / rect.height(),
					event.globalPos(),
				)
				break

	def keyPressEvent(self, event) -> None:
		if event.key() in (Qt.Key.Key_Plus, Qt.Key.Key_Equal) and self.on_zoom is not None:
			self.on_zoom(1)
		elif event.key() == Qt.Key.Key_Minus and self.on_zoom is not None:
			self.on_zoom(-1)
		elif event.key() == Qt.Key.Key_Space and self.on_toggle_play is not None:
			self.on_toggle_play()
		else:
			super().keyPressEvent(event)
			return
		event.accept()


class ScoreDialog(QDialog):
	"""Engraves the edited notes with LilyPond and plays them along the score."""

	def __init__(
		self,
		sources: list[tuple[str, str]],
		note_provider,
		bpm: int,
		parent=None,
		current_source: str | None = None,
		video_path: Path | None = None,
		on_jump=None,
	) -> None:
		super().__init__(parent)
		self.setWindowTitle("Score")
		self.on_jump = on_jump
		self.video_path = video_path
		self.video_capture: cv2.VideoCapture | None = None
		self.video_last_frame = -10
		self.video_map = None
		self.resize(1120, 760)
		self.setMinimumSize(1000, 620)
		self.note_provider = note_provider
		self.score_source: ScoreSource | None = None
		self.score_layout: ScoreLayout | None = None
		self.score_notes: list[dict[str, object]] = []
		self.work_dir: Path | None = None
		self.lilypond_path = find_lilypond()
		self.process: QProcess | None = None
		self.compile_step = 0
		self.start_slot = 0
		self._restore_view: tuple[float, int, int, int] | None = None
		self.playback_timer = QTimer(self)
		self.playback_timer.setTimerType(Qt.TimerType.PreciseTimer)
		self.playback_timer.setInterval(16)
		self.playback_timer.timeout.connect(self._playback_tick)
		self.midi_output = None
		self.playback_scheduler: PlaybackScheduler | None = None
		self.playback_notes: list[tuple[float, int, int, int]] = []
		self.playback_start_wall = 0.0
		self.playback_start_seconds = 0.0
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
		self.summary_label = QLabel("Generate the score to see it here.")
		self.summary_label.setStyleSheet("font-size: 12pt; font-weight: 700;")
		layout.addWidget(self.summary_label)

		panel = QWidget(self)
		panel_layout = QHBoxLayout(panel)
		panel_layout.setContentsMargins(0, 0, 0, 0)
		panel_layout.setSpacing(12)
		self.score_view = ScoreView()
		self.score_view.on_click = self._score_clicked
		self.score_view.on_context = self._score_context_menu
		self.score_view.on_zoom = self._zoom_step
		self.score_view.on_toggle_play = self._toggle_playback
		self.score_scroll = QScrollArea()
		self.score_scroll.setWidget(self.score_view)
		self.score_scroll.setAlignment(Qt.AlignmentFlag.AlignHCenter)
		panel_layout.addWidget(self.score_scroll, 1)

		controls_scroll = QScrollArea()
		controls_scroll.setWidgetResizable(True)
		controls_scroll.setFixedWidth(430 if self.video_path else 300)
		controls_scroll.setStyleSheet(
			"QScrollArea { border: 0; background: #e8edf1; }"
			"QScrollArea > QWidget > QWidget { background: #e8edf1; }"
		)
		controls = QWidget()
		controls.setStyleSheet(
			"QLabel { background: transparent; color: #263746; font-weight: 600; }"
			"QCheckBox { background: #f7f9fa; color: #18212b; padding: 7px; "
			"border: 1px solid #c2ccd3; border-radius: 4px; }"
			"QCheckBox::indicator { width: 16px; height: 16px; }"
			"QPushButton { background: #ffffff; color: #18212b; padding: 8px 10px; "
			"border: 1px solid #aebbc5; border-radius: 4px; font-weight: 600; }"
			"QPushButton:hover { background: #dce8ee; border-color: #6c8797; }"
			"QPushButton:disabled { color: #8b98a3; background: #eef1f4; }"
			"QComboBox, QLineEdit { background: #ffffff; color: #18212b; "
			"border: 1px solid #aebbc5; border-radius: 4px; padding: 6px; }"
		)
		controls_layout = QVBoxLayout(controls)
		controls_layout.setContentsMargins(8, 4, 8, 4)
		controls_layout.setSpacing(10)

		def section_heading(text: str) -> QLabel:
			label = QLabel(text.upper())
			label.setStyleSheet(
				"QLabel { color: #526775; background: transparent; "
				"font-size: 8pt; font-weight: 700; padding: 4px 0 3px 0; "
				"border-bottom: 1px solid #c3cdd4; }"
			)
			return label

		controls_layout.addWidget(section_heading("Video"))
		self.video_label = QLabel("No video loaded." if not self.video_path else "Generate the score first.")
		self.video_label.setFixedSize(396, 223)
		self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
		self.video_label.setStyleSheet("QLabel { background: #101418; color: #9aa7b2; }")
		self.video_label.setVisible(bool(self.video_path))
		controls_layout.addWidget(self.video_label)

		controls_layout.addWidget(section_heading("Source data"))
		self.source_combo = QComboBox()
		for label, key in sources:
			self.source_combo.addItem(label, key)
		if current_source is not None and self.source_combo.findData(current_source) >= 0:
			self.source_combo.setCurrentIndex(self.source_combo.findData(current_source))
		controls_layout.addWidget(self.source_combo)

		controls_layout.addWidget(section_heading("Score"))
		self.title_entry = QLineEdit("Piano roll")
		controls_layout.addWidget(QLabel("Title"))
		controls_layout.addWidget(self.title_entry)
		bpm_row = QHBoxLayout()
		bpm_row.addWidget(QLabel("BPM"))
		self.bpm_entry = QLineEdit(str(bpm))
		self.bpm_entry.setValidator(QIntValidator(20, 300, self))
		self.bpm_entry.setFixedWidth(64)
		bpm_row.addWidget(self.bpm_entry)
		bpm_row.addStretch(1)
		controls_layout.addLayout(bpm_row)
		self.generate_button = QPushButton("Generate score")
		self.generate_button.clicked.connect(self._generate_score)
		controls_layout.addWidget(self.generate_button)
		self.save_button = QPushButton("Save PDF and LilyPond files...")
		self.save_button.setEnabled(False)
		self.save_button.clicked.connect(self._save_score)
		controls_layout.addWidget(self.save_button)

		controls_layout.addWidget(section_heading("View"))
		zoom_row = QHBoxLayout()
		zoom_out = QPushButton("Zoom -")
		zoom_out.clicked.connect(lambda: self._zoom_step(-1))
		zoom_row.addWidget(zoom_out)
		zoom_in = QPushButton("Zoom +")
		zoom_in.clicked.connect(lambda: self._zoom_step(1))
		zoom_row.addWidget(zoom_in)
		controls_layout.addLayout(zoom_row)
		fit_button = QPushButton("Fit width")
		fit_button.clicked.connect(self._fit_width)
		controls_layout.addWidget(fit_button)
		self.playhead_visible_checkbox = QCheckBox("Show playhead")
		self.playhead_visible_checkbox.setChecked(True)
		self.playhead_visible_checkbox.toggled.connect(self.score_view.set_playhead_visible)
		controls_layout.addWidget(self.playhead_visible_checkbox)

		controls_layout.addWidget(section_heading("Playback"))
		playback_buttons = QHBoxLayout()
		self.play_button = QPushButton("Play")
		self.play_button.setEnabled(False)
		self.play_button.clicked.connect(self._toggle_playback)
		playback_buttons.addWidget(self.play_button)
		self.stop_button = QPushButton("Stop")
		self.stop_button.clicked.connect(self._stop_playback)
		playback_buttons.addWidget(self.stop_button)
		controls_layout.addLayout(playback_buttons)
		self.autoscroll_checkbox = QCheckBox("Follow playback")
		self.autoscroll_checkbox.setChecked(True)
		controls_layout.addWidget(self.autoscroll_checkbox)
		self.status_label = QLabel(
			"Click into the score to choose where playback starts. "
			"Right-click a note or chord to adjust its duration. "
			"Zoom with the + and - keys."
		)
		self.status_label.setStyleSheet("color: #53616d;")
		self.status_label.setWordWrap(True)
		controls_layout.addWidget(self.status_label)
		controls_layout.addStretch(1)
		controls_scroll.setWidget(controls)
		panel_layout.addWidget(controls_scroll)
		layout.addWidget(panel, 1)

		buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
		buttons.rejected.connect(self.reject)
		layout.addWidget(buttons)

	def _set_status(self, text: str) -> None:
		self.status_label.setText(text)

	def _ensure_lilypond(self) -> bool:
		if self.lilypond_path and Path(self.lilypond_path).exists():
			return True
		chosen, _ = QFileDialog.getOpenFileName(
			self, "Locate lilypond.exe", "", "LilyPond (lilypond.exe lilypond)"
		)
		if not chosen:
			return False
		self.lilypond_path = chosen
		return True

	def _generate_score(self) -> None:
		if self.process is not None:
			return
		if not self._ensure_lilypond():
			self._set_status("LilyPond was not found. Install it or set LILYPOND_PATH.")
			return
		self._stop_playback()
		notes, origin_seconds, grid_bpm = self.note_provider(str(self.source_combo.currentData()))
		if not notes:
			QMessageBox.information(self, "No notes", "The selected source contains no completed notes.")
			return
		try:
			bpm = int(self.bpm_entry.text())
		except ValueError:
			bpm = 120
		if grid_bpm is not None and grid_bpm != bpm:
			# Times of this source lie on its own BPM grid; any other value shifts notes.
			bpm = grid_bpm
			self.bpm_entry.setText(str(bpm))
		self.score_notes = notes
		self._compile_notes(notes, origin_seconds, bpm)

	def _compile_notes(
		self,
		notes: list[dict[str, object]],
		origin_seconds: float,
		bpm: int,
		keep_view: bool = False,
	) -> None:
		self._restore_view = None
		if keep_view:
			self._restore_view = (
				self.score_view.zoom,
				self.score_scroll.horizontalScrollBar().value(),
				self.score_scroll.verticalScrollBar().value(),
				self.start_slot,
			)
		self.score_notes = notes
		self.video_map = video_time_map(notes)
		self.score_source = build_score(
			notes, bpm, origin_seconds, self.title_entry.text().strip() or "Piano roll"
		)
		self._cleanup_work_dir()
		self.work_dir = Path(tempfile.mkdtemp(prefix="pianoroll_score_"))
		write_score_files(self.score_source, self.work_dir)
		self.score_layout = None
		self.play_button.setEnabled(False)
		self.save_button.setEnabled(False)
		self.generate_button.setEnabled(False)
		self.compile_step = 0
		self.summary_label.setText(f"Compiling {self.score_source.measures} measures with LilyPond...")
		self._start_compile_step()

	def _start_compile_step(self) -> None:
		arguments = [
			["--pdf", "-dno-point-and-click", "-o", "score", "score.ly"],
			["-dbackend=svg", "-dpoint-and-click", "-o", "score", "score.ly"],
		][self.compile_step]
		self._set_status(
			"Engraving the PDF..." if self.compile_step == 0 else "Reading note positions..."
		)
		self.process = QProcess(self)
		self.process.setWorkingDirectory(str(self.work_dir))
		self.process.finished.connect(self._compile_finished)
		self.process.errorOccurred.connect(self._compile_error)
		self.process.start(str(self.lilypond_path), arguments)

	def _compile_error(self, _error) -> None:
		if self.process is None or self.process.state() != QProcess.ProcessState.NotRunning:
			return
		self.process.deleteLater()
		self.process = None
		self.generate_button.setEnabled(True)
		self.summary_label.setText("LilyPond could not be started.")

	def _compile_finished(self, exit_code: int, _status) -> None:
		process = self.process
		if process is None:
			return
		output = bytes(process.readAllStandardError()).decode("utf-8", errors="replace")
		process.deleteLater()
		self.process = None
		if exit_code != 0:
			self.generate_button.setEnabled(True)
			self.summary_label.setText("LilyPond reported an error.")
			message = QMessageBox(self)
			message.setIcon(QMessageBox.Icon.Critical)
			message.setWindowTitle("LilyPond failed")
			message.setText("LilyPond could not compile the score.")
			message.setDetailedText(output[-4000:])
			message.exec()
			return
		if self.compile_step == 0:
			self.compile_step = 1
			self._start_compile_step()
			return
		self.generate_button.setEnabled(True)
		self._load_result()

	def _load_result(self) -> None:
		assert self.score_source is not None and self.work_dir is not None
		first_load = not self.score_view.images
		restore = self._restore_view
		self._restore_view = None
		if not self.score_view.load(str(self.work_dir / "score.pdf")):
			self.summary_label.setText("The compiled PDF could not be loaded.")
			return
		self.score_layout = load_layout(self.work_dir, self.score_source)
		self.start_slot = 0
		if restore is not None:
			zoom, scroll_x, scroll_y, self.start_slot = restore
			self.score_view.set_zoom(zoom)
			QTimer.singleShot(0, lambda: (
				self.score_scroll.horizontalScrollBar().setValue(scroll_x),
				self.score_scroll.verticalScrollBar().setValue(scroll_y),
			))
		elif first_load:
			self._fit_width()
		self.play_button.setEnabled(bool(self.score_layout.systems))
		self.save_button.setEnabled(True)
		self.summary_label.setText(
			f"{self.score_source.measures} measures, "
			f"{self.score_view.document.pageCount()} page(s), {len(self.score_notes)} notes"
		)
		self._set_status("Score ready. Click into the score to choose the start position.")
		self._show_start_marker()

	def _cleanup_work_dir(self) -> None:
		if self.work_dir is not None:
			self.score_view.clear()
			shutil.rmtree(self.work_dir, ignore_errors=True)
			self.work_dir = None

	def _save_score(self) -> None:
		if self.score_source is None or self.work_dir is None:
			return
		chosen, _ = QFileDialog.getSaveFileName(self, "Save score", "score.pdf", "PDF (*.pdf)")
		if not chosen:
			return
		target = Path(chosen).with_suffix(".pdf")
		base = target.stem
		shutil.copyfile(self.work_dir / "score.pdf", target)
		exported = build_score(
			self.score_notes,
			int(self.bpm_entry.text() or 120),
			self.score_source.time_origin,
			self.title_entry.text().strip() or "Piano roll",
			base,
		)
		write_score_files(exported, target.parent)
		self._set_status(f"Saved {target.name} and its LilyPond sources.")

	def _zoom_step(self, direction: int) -> None:
		self.score_view.set_zoom(self.score_view.zoom * (1.2 if direction > 0 else 1 / 1.2))

	def _fit_width(self) -> None:
		available = self.score_scroll.viewport().width() - 2 * ScoreView.PAGE_MARGIN - 4
		self.score_view.set_zoom(available / self.score_view.page_width_pixels())

	def _slot_seconds(self, slot: float) -> float:
		assert self.score_source is not None
		return self.score_source.time_origin + slot * self.score_source.grid_seconds

	def _seconds_slot(self, seconds: float) -> float:
		assert self.score_source is not None
		return (seconds - self.score_source.time_origin) / self.score_source.grid_seconds

	def _show_start_marker(self) -> None:
		if self.score_layout is not None:
			self.score_view.set_playhead(self.score_layout.locate(self.start_slot))
			self._show_video_at(self._slot_seconds(self.start_slot))

	def _show_video_at(self, score_seconds: float) -> None:
		if not self.video_path or self.video_map is None:
			return
		if self.video_capture is None:
			self.video_capture = cv2.VideoCapture(str(self.video_path))
			self.video_last_frame = -10
		capture = self.video_capture
		fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
		target = max(0, round(self.video_map(score_seconds) * fps))
		if target == self.video_last_frame:
			return
		ahead = target - self.video_last_frame
		if 0 < ahead <= 6:
			for _ in range(ahead - 1):
				capture.grab()
		else:
			capture.set(cv2.CAP_PROP_POS_FRAMES, target)
		ok, frame = capture.read()
		if not ok:
			return
		self.video_last_frame = target
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		height, width = rgb.shape[:2]
		image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888)
		self.video_label.setPixmap(
			QPixmap.fromImage(image).scaled(
				self.video_label.size(),
				Qt.AspectRatioMode.KeepAspectRatio,
				Qt.TransformationMode.SmoothTransformation,
			)
		)

	def _score_clicked(self, page: int, fx: float, fy: float) -> None:
		if self.score_layout is None:
			return
		slot = self.score_layout.slot_at(page, fx, fy)
		if slot is None:
			return
		was_playing = self.playback_timer.isActive()
		self._stop_playback()
		self.start_slot = slot
		self._show_start_marker()
		if was_playing:
			self._toggle_playback()

	DURATION_CHOICES = (
		(1, "Sixteenth"), (2, "Eighth"), (3, "Dotted eighth"), (4, "Quarter"),
		(6, "Dotted quarter"), (8, "Half"), (12, "Dotted half"), (16, "Whole"),
		(24, "Dotted whole"), (32, "Two wholes"),
	)

	def _score_context_menu(self, page: int, fx: float, fy: float, global_pos) -> None:
		if self.score_layout is None or self.score_source is None or self.process is not None:
			return
		hit = self.score_layout.hit_test(page, fx, fy)
		if hit is None:
			return
		hand, slot, pitch = hit
		origin, grid = self.score_source.time_origin, self.score_source.grid_seconds
		length = (
			current_length(self.score_notes, hand, slot, origin, grid, pitch)
			if pitch is not None else None
		)
		note = (
			find_note(self.score_notes, hand, slot, origin, grid, pitch)
			if pitch is not None else None
		)
		menu = QMenu(self)
		menu.setStyleSheet(
			"QMenu { background: #ffffff; color: #18212b; border: 1px solid #aebbc5; }"
			"QMenu::item { padding: 6px 22px; background: transparent; color: #18212b; }"
			"QMenu::item:selected { background: #cfe3f1; color: #18212b; }"
			"QMenu::item:disabled { color: #9aa7b2; }"
		)
		adjust_action = menu.addAction(
			f"Adjust duration of {midi_note_name(pitch)}..." if length is not None else "Adjust duration..."
		)
		adjust_action.setEnabled(length is not None)
		insert_action = menu.addAction("Insert note here...")
		menu.addSeparator()
		video_action = menu.addAction("Show in video")
		video_action.setEnabled(self.video_map is not None)
		jump_action = menu.addAction("Jump to timeline")
		jump_action.setEnabled(
			self.on_jump is not None and note is not None and "video_seconds" in note
		)
		chosen = menu.exec(global_pos)
		if chosen is adjust_action and length is not None:
			self._adjust_duration_dialog(hand, slot, length, pitch)
		elif chosen is insert_action:
			self._insert_note_dialog(hand, slot, pitch)
		elif chosen is video_action and self.video_map is not None:
			self.start_slot = slot
			self._show_start_marker()
			if note is not None:
				self._show_video_at(float(note["start"]))
		elif chosen is jump_action and note is not None and self.on_jump is not None:
			self.on_jump(float(note["video_seconds"]))
			self.accept()

	def _edit_dialog(self, title: str) -> tuple[QDialog, QVBoxLayout]:
		dialog = QDialog(self)
		dialog.setStyleSheet(
			"QDialog { background: #eef1f4; }"
			"QLabel, QCheckBox { color: #18212b; background: transparent; }"
			"QComboBox, QSpinBox { background: #ffffff; color: #18212b; border: 1px solid #aebbc5; padding: 5px; }"
			"QComboBox QAbstractItemView { background: #ffffff; color: #18212b; }"
			"QPlainTextEdit { background: #ffffff; color: #18212b; border: 1px solid #aebbc5; "
			"font-family: Consolas, monospace; }"
			"QPushButton { background: #ffffff; color: #18212b; border: 1px solid #aebbc5; padding: 6px 12px; }"
		)
		dialog.setWindowTitle(title)
		dialog.resize(640, 520)
		return dialog, QVBoxLayout(dialog)

	def _edit_preview(self, form: QVBoxLayout):
		"""Add change list and old/new LilyPond views; returns refresh(new_notes, hand)."""
		form.addWidget(QLabel("Affected notes"))
		changes = QPlainTextEdit()
		changes.setReadOnly(True)
		changes.setMaximumHeight(110)
		form.addWidget(changes)
		columns = QHBoxLayout()
		views = []
		for title in ("LilyPond before", "LilyPond after"):
			column = QVBoxLayout()
			column.addWidget(QLabel(title))
			view = QPlainTextEdit()
			view.setReadOnly(True)
			view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
			column.addWidget(view)
			columns.addLayout(column)
			views.append(view)
		form.addLayout(columns, 1)
		source = self.score_source
		assert source is not None
		origin, grid = source.time_origin, source.grid_seconds
		old_notes = self.score_notes

		def where(slot: int) -> str:
			return f"bar {slot // 16 + 1}, beat {slot % 16 / 4 + 1:g}"

		def span(note: dict[str, object]) -> tuple[int, int]:
			start = round((float(note["start"]) - origin) / grid)
			return start, max(round((float(note["end"]) - origin) / grid), start + 1)

		def refresh(new_notes: list[dict[str, object]] | None, hand: str) -> None:
			if new_notes is None:
				changes.setPlainText("Nothing changes.")
				views[0].setPlainText("")
				views[1].setPlainText("")
				return
			lines: list[str] = []
			touched: list[int] = []
			for index, new in enumerate(new_notes):
				name = f"{midi_note_name(int(new['midi_note']))} ({new['hand']})"
				new_start, new_end = span(new)
				if index >= len(old_notes):
					lines.append(f"added {name}: {where(new_start)}, {new_end - new_start}/16")
					touched += [new_start, new_end]
					continue
				old_start, old_end = span(old_notes[index])
				if (old_start, old_end) == (new_start, new_end):
					continue
				if new_start != old_start:
					lines.append(f"moved {name}: {where(old_start)} -> {where(new_start)}")
				if new_end - new_start != old_end - old_start:
					lines.append(
						f"{name} at {where(new_start)}: "
						f"{old_end - old_start}/16 -> {new_end - new_start}/16"
					)
				touched += [old_start, new_start, old_end, new_end]
			changes.setPlainText("\n".join(lines) or "Nothing changes.")
			if not touched:
				views[0].setPlainText("")
				views[1].setPlainText("")
				return
			try:
				bpm = int(self.bpm_entry.text())
			except ValueError:
				bpm = round(15.0 / grid)
			preview = build_score(new_notes, bpm, origin, "preview")
			first = min(touched) // 16 * 16
			last = (max(touched) // 16 + 1) * 16
			for view, score in zip(views, (source, preview)):
				excerpt = lilypond_excerpt(score, hand, first, last)
				view.setPlainText(f"% {hand} hand, bars {first // 16 + 1}-{last // 16}\n{excerpt}")

		return refresh

	def _adjust_duration_dialog(self, hand: str, slot: int, length: int, pitch: int | None) -> None:
		assert self.score_source is not None
		origin, grid = self.score_source.time_origin, self.score_source.grid_seconds
		dialog, form = self._edit_dialog("Adjust duration")
		subject = midi_note_name(pitch) if pitch is not None else "chord"
		form.addWidget(QLabel(
			f"{subject}, {'right' if hand == 'right' else 'left'} hand, bar {slot // 16 + 1}, "
			f"current length {length}/16"
		))
		combo = QComboBox()
		for slots, label in self.DURATION_CHOICES:
			combo.addItem(f"{label} ({slots}/16)", slots)
		index = combo.findData(length)
		if index < 0:
			combo.addItem(f"Current ({length}/16)", length)
			index = combo.count() - 1
		combo.setCurrentIndex(index)
		form.addWidget(combo)
		pause = QCheckBox("Add pause (following notes stay in place)")
		form.addWidget(pause)
		hint = QLabel("")
		hint.setStyleSheet("color: #53616d;")
		form.addWidget(hint)
		refresh = self._edit_preview(form)
		state: dict[str, object] = {"edited": None}

		def update() -> None:
			new_length = int(combo.currentData())
			hint.setText(
				"Lengthening: following notes always move along (pause is ignored)."
				if new_length > length
				else "Shortening: with pause the following notes stay, otherwise they move up."
			)
			edited = None
			if new_length != length:
				edited = adjust_duration(
					self.score_notes, hand, slot, new_length, pause.isChecked(), origin, grid, pitch
				)
			state["edited"] = edited
			refresh(edited, hand)

		combo.currentIndexChanged.connect(update)
		pause.toggled.connect(update)
		update()
		buttons = QDialogButtonBox(
			QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
		)
		buttons.accepted.connect(dialog.accept)
		buttons.rejected.connect(dialog.reject)
		form.addWidget(buttons)
		if dialog.exec() != QDialog.DialogCode.Accepted or state["edited"] is None:
			return
		self._apply_edit(state["edited"])

	def _insert_note_dialog(self, hand: str, slot: int, pitch: int | None) -> None:
		assert self.score_source is not None
		origin, grid = self.score_source.time_origin, self.score_source.grid_seconds
		dialog, form = self._edit_dialog("Insert note")
		form.addWidget(QLabel(f"{'Right' if hand == 'right' else 'Left'} hand"))
		pitch_combo = QComboBox()
		for midi in range(108, 20, -1):
			pitch_combo.addItem(midi_note_name(midi), midi)
		pitch_combo.setCurrentIndex(
			pitch_combo.findData(pitch if pitch is not None else (60 if hand == "right" else 48))
		)
		form.addWidget(QLabel("Pitch"))
		form.addWidget(pitch_combo)
		position = QSpinBox()
		position.setRange(0, self.score_source.total_slots + 64)
		position.setValue(slot)
		position_label = QLabel("")
		form.addWidget(QLabel("Start (in sixteenths from the beginning)"))
		form.addWidget(position)
		form.addWidget(position_label)
		length_combo = QComboBox()
		for slots, label in self.DURATION_CHOICES:
			length_combo.addItem(f"{label} ({slots}/16)", slots)
		length_combo.setCurrentIndex(length_combo.findData(4))
		form.addWidget(QLabel("Length"))
		form.addWidget(length_combo)
		push = QCheckBox("Push following notes of this hand")
		form.addWidget(push)
		refresh = self._edit_preview(form)
		state: dict[str, object] = {"edited": None}

		def update() -> None:
			start = position.value()
			position_label.setText(f"bar {start // 16 + 1}, beat {start % 16 / 4 + 1:g}")
			position_label.setStyleSheet("color: #53616d;")
			video_seconds = (
				self.video_map(origin + start * grid) if self.video_map is not None else None
			)
			edited = insert_note(
				self.score_notes, hand, int(pitch_combo.currentData()), start,
				int(length_combo.currentData()), push.isChecked(), origin, grid, video_seconds,
			)
			state["edited"] = edited
			refresh(edited, hand)

		pitch_combo.currentIndexChanged.connect(update)
		length_combo.currentIndexChanged.connect(update)
		position.valueChanged.connect(update)
		push.toggled.connect(update)
		update()
		buttons = QDialogButtonBox(
			QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
		)
		buttons.accepted.connect(dialog.accept)
		buttons.rejected.connect(dialog.reject)
		form.addWidget(buttons)
		if dialog.exec() != QDialog.DialogCode.Accepted or state["edited"] is None:
			return
		self._apply_edit(state["edited"])

	def _apply_edit(self, edited: list[dict[str, object]]) -> None:
		assert self.score_source is not None
		self._stop_playback()
		try:
			bpm = int(self.bpm_entry.text())
		except ValueError:
			bpm = round(15.0 / self.score_source.grid_seconds)
		self._compile_notes(edited, self.score_source.time_origin, bpm, keep_view=True)

	def _toggle_playback(self) -> None:
		if self.playback_timer.isActive():
			self._stop_playback()
			return
		if self.score_layout is None or self.score_source is None:
			return
		try:
			self.midi_output = mido.open_output()
		except (OSError, mido.MidiError) as error:
			QMessageBox.warning(
				self, "MIDI output unavailable", f"Could not open a MIDI output device: {error}"
			)
			return
		self.playback_start_seconds = self._slot_seconds(self.start_slot)
		self.playback_notes = []
		for note in self.score_notes:
			if float(note["end"]) <= self.playback_start_seconds:
				continue
			channel = 0 if note["hand"] == "left" else 1
			midi_note = int(note["midi_note"])
			self.playback_notes.append(
				(max(float(note["start"]), self.playback_start_seconds), 1, channel, midi_note)
			)
			self.playback_notes.append((float(note["end"]), 0, channel, midi_note))
		self.playback_notes.sort()
		self.playback_start_wall = time.perf_counter()
		self.playback_scheduler = PlaybackScheduler(
			self.midi_output,
			self.playback_notes,
			self.playback_start_wall,
			self.playback_start_seconds,
		)
		self.playback_scheduler.start()
		self.play_button.setText("Pause")
		self.playback_timer.start()

	def _stop_playback(self) -> None:
		self.playback_timer.stop()
		if self.playback_scheduler is not None:
			self.playback_scheduler.stop_event.set()
			self.playback_scheduler.join(timeout=1.0)
			self.playback_scheduler = None
		self.midi_output = None
		self.play_button.setText("Play")
		self._show_start_marker()

	def _playback_tick(self) -> None:
		if self.playback_scheduler is None or self.score_layout is None:
			self._stop_playback()
			return
		seconds = self.playback_start_seconds + time.perf_counter() - self.playback_start_wall
		self._show_video_at(seconds)
		location = self.score_layout.locate(self._seconds_slot(seconds))
		self.score_view.set_playhead(location)
		if self.autoscroll_checkbox.isChecked():
			point = self.score_view.playhead_point()
			if point is not None:
				self.score_scroll.ensureVisible(
					round(point[0]), round(point[1]), 80, round(self.score_scroll.viewport().height() * 0.3)
				)
		if self.playback_scheduler.completed_event.is_set():
			self._stop_playback()

	def done(self, result: int) -> None:
		self._stop_playback()
		if self.video_capture is not None:
			self.video_capture.release()
			self.video_capture = None
		if self.process is not None:
			self.process.kill()
			self.process.waitForFinished(2000)
			self.process = None
		self._cleanup_work_dir()
		super().done(result)


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
		self.pending_jump: float | None = None
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
		self._compute_adaptive_normalizations()
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
		self.playback_timer.setTimerType(Qt.TimerType.PreciseTimer)
		self.playback_timer.setInterval(16)
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
		def section_heading(text: str) -> QLabel:
			label = QLabel(text.upper())
			label.setStyleSheet(
				"QLabel { color: #526775; background: transparent; "
				"font-size: 8pt; font-weight: 700; padding: 4px 0 3px 0; "
				"border-bottom: 1px solid #c3cdd4; }"
			)
			return label

		self.timeline_controls_scroll.setWidget(self.timeline_controls_panel)
		timeline_panel_layout.addWidget(self.timeline_controls_scroll)
		layout.addWidget(self.timeline_panel, 1)
		self.timeline_panel.hide()

		self.timeline_navigation = QWidget()
		timeline_navigation_layout = QVBoxLayout(self.timeline_navigation)
		timeline_navigation_layout.setContentsMargins(0, 0, 0, 0)
		timeline_navigation_layout.setSpacing(5)
		timeline_navigation_layout.addWidget(section_heading("Navigation"))
		timeline_navigation_layout.addWidget(QLabel("Start measure"))
		self.timeline_start_slider = QSlider(Qt.Orientation.Horizontal)
		self.timeline_start_slider.setRange(0, 0)
		self.timeline_start_slider.valueChanged.connect(self._timeline_start_changed)
		timeline_navigation_layout.addWidget(self.timeline_start_slider, 1)
		self.timeline_navigation_label = QLabel("1 / 1")
		self.timeline_navigation_label.setAlignment(Qt.AlignmentFlag.AlignRight)
		timeline_navigation_layout.addWidget(self.timeline_navigation_label)
		timeline_navigation_layout.addWidget(section_heading("Source data"))
		reference_row = QHBoxLayout()
		self.reference_source_combo = QComboBox()
		self.reference_source_combo.addItem("Detection", "detected")
		self.reference_source_combo.addItem("Waterfall", "waterfall")
		self.reference_source_combo.addItem("Mean: Detection + Waterfall", "combined")
		self.reference_source_combo.addItem("Detection normalized", "normalized_detection")
		self.reference_source_combo.addItem("Waterfall normalized", "normalized_waterfall")
		self.reference_source_combo.addItem("Mean normalized", "normalized")
		for label, data in self._normalization_choices()[3:]:
			self.reference_source_combo.addItem(label, data)
		self.reference_source_combo.currentIndexChanged.connect(
			self._reference_source_changed
		)
		reference_row.addWidget(self.reference_source_combo, 1)
		timeline_navigation_layout.addLayout(reference_row)
		self.reference_visible_checkbox = QCheckBox("Show source track")
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
		timeline_navigation_layout.addWidget(section_heading("Playback"))
		listen_row.addWidget(QLabel("Listen"))
		self.playback_source_combo = QComboBox()
		self.playback_source_combo.addItem("Detected", "detected")
		self.playback_source_combo.addItem("Waterfall", "waterfall")
		self.playback_source_combo.addItem("Mean: Detection + Waterfall", "combined")
		self.playback_source_combo.addItem("Detection normalized", "normalized_detection")
		self.playback_source_combo.addItem("Waterfall normalized", "normalized_waterfall")
		self.playback_source_combo.addItem("Mean normalized", "normalized")
		for label, data in self._normalization_choices()[3:]:
			self.playback_source_combo.addItem(label, data)
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
		settings.addWidget(section_heading("Quantization"))
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
		settings.addWidget(section_heading("Edit copy"))
		self.editable_source_label = QLabel(
			"Uses the selected reference; edits never change the source."
		)
		self.editable_source_label.setWordWrap(True)
		self.editable_source_label.setStyleSheet("color: #53616d; background: transparent;")
		settings.addWidget(self.editable_source_label)
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
		settings.addWidget(section_heading("Score"))
		self.score_button = QPushButton("Create score (LilyPond)...")
		self.score_button.setToolTip(
			"Engrave the notes of a chosen source (e.g. the editable copy) as a "
			"two-staff piano score and play along."
		)
		self.score_button.clicked.connect(self._open_score_dialog)
		settings.addWidget(self.score_button)
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

	def _score_sources(self) -> list[tuple[str, str]]:
		sources: list[tuple[str, str]] = []
		if self.editable_events is not None:
			sources.append(("Editable copy", "editable"))
		sources.extend((
			("Manual quantization (of selected source)", "manual"),
			("music21 quantization (of selected source)", "music21"),
			("Detection", "detected"),
			("Waterfall", "waterfall"),
			("Mean: Detection + Waterfall", "combined"),
			("Detection normalized", "normalized_detection"),
			("Waterfall normalized", "normalized_waterfall"),
			("Mean normalized", "normalized"),
			*self._normalization_choices()[3:],
		))
		return sources

	def _score_notes(self, key: str) -> tuple[list[dict[str, object]], float, int | None]:
		corrected = True
		grid_bpm: int | None = None
		origin = self._timeline_origin_seconds()
		if key == "editable":
			events = self.editable_events or []
		elif key == "manual":
			events = self.quantized_events
			if events is None:
				events, settings, _ = quantize_events_manually(
					self._quantization_source_events(),
					self.fps,
					self.bpm_slider.value(),
					float(self.beat_length_combo.currentData()),
					self.current_frame,
					self._selected_grid_origin_frame(),
				)
				origin = float(settings["origin_seconds"])
			elif self.quantization_settings is not None:
				origin = float(self.quantization_settings["origin_seconds"])
		elif key == "music21":
			events = self.music21_quantized_events
			if events is None:
				events = quantize_events_with_music21(
					self._quantization_source_events(),
					self.fps,
					self.bpm_slider.value(),
					float(self.beat_length_combo.currentData()),
					self.current_frame,
					self._selected_grid_origin_frame(),
				)
			if events:
				origin = float(events[0].get("music21_origin_seconds", origin))
		elif key.startswith("normalized"):
			events = self._normalization_events_map().get(key) or []
			settings = self._normalization_settings_map().get(key)
			if settings is not None:
				origin = float(settings["origin_seconds"])
				grid_bpm = int(settings["bpm"])
			key = "normalized"
		else:
			corrected = False
			if key == "waterfall":
				events = self.waterfall_events
			elif key == "combined":
				events = combine_detection_and_waterfall(
					self.events, self.waterfall_events, self.fps
				)
			else:
				events = self.events
		notes: list[dict[str, object]] = []
		for event in events:
			if event.get("offset_frame") is None:
				continue
			start, end = self._event_bounds_seconds(
				event, corrected=corrected, correction_source=key
			)
			if end <= start:
				end = start + 1 / self.fps
			notes.append({
				"hand": str(event["hand"]),
				"midi_note": int(event["midi_note"]),
				"start": start,
				"end": end,
				"video_seconds": int(event["onset_frame"]) / self.fps,
			})
		return notes, origin, grid_bpm

	def _open_score_dialog(self) -> None:
		self._stop_playback()
		reference = str(self.reference_source_combo.currentData())
		current = (
			"editable" if self.editable_events is not None
			else reference if reference != "detected" else "detected"
		)
		dialog = ScoreDialog(
			self._score_sources(),
			self._score_notes,
			self.bpm_slider.value(),
			self,
			current_source=current,
			video_path=getattr(self.parent(), "video_path", None),
			on_jump=lambda seconds: setattr(self, "pending_jump", seconds),
		)
		dialog.setWindowState(dialog.windowState() | Qt.WindowState.WindowMaximized)
		dialog.exec()
		if self.pending_jump is not None and self.isVisible():
			seconds, self.pending_jump = self.pending_jump, None
			self.jump_to_seconds(seconds)

	def jump_to_seconds(self, seconds: float) -> None:
		"""Show the timeline with the given video time near the left of the window."""
		self.plot_type_combo.setCurrentIndex(self.plot_type_combo.findData("timeline"))
		measure_seconds = 4 * 60.0 / self.bpm_slider.value()
		span = measure_seconds * int(self.visible_measures_combo.currentData())
		self.timeline_window_start_seconds = max(0.0, seconds - 0.25 * span)
		self._update_plot()
		unit_scale = self.fps if self.units_combo.currentData() == "frames" else 1.0
		self.timeline_widget.set_playhead(seconds * unit_scale)

	def _create_editable_copy(self) -> None:
		source_events = self._quantization_source_events()
		if not source_events:
			QMessageBox.information(self, "No notes", "There are no notes to copy.")
			return
		self.editable_events = [event.copy() for event in source_events]
		# Keep exact (non frame-rounded) times so grid-aligned notes stay on the grid.
		for event in self.editable_events:
			if event.get("offset_frame") is None:
				continue
			event["editable_onset_seconds"] = float(
				event.get("video_onset_seconds", int(event["onset_frame"]) / self.fps)
			)
			event["editable_offset_seconds"] = float(
				event.get("video_offset_seconds", int(event["offset_frame"]) / self.fps)
			)
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
		onset_seconds = max(0.0, onset_seconds)
		onset_frame = round(onset_seconds * self.fps)
		duration_seconds = 60.0 / self.bpm_slider.value() * 0.25
		duration_frames = max(1, round(duration_seconds * self.fps))
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
			"editable_onset_seconds": onset_seconds,
			"editable_offset_seconds": onset_seconds + duration_seconds,
			"duration_seconds": duration_seconds,
			"editable_only": True,
		}
		self.editable_events.append(event)
		self.timeline_widget.selected_bar = ("editable", event_id)
		self.selected_editable_event_id = event_id
		self.remove_note_button.setEnabled(True)
		self._update_plot()

	def _editable_snap_grid(self) -> tuple[float, float]:
		grid_seconds = 60.0 / self.bpm_slider.value() / 4
		# Must match the origin of the grid lines drawn in the timeline.
		return grid_seconds, self._timeline_origin_seconds()

	def _timeline_origin_seconds(self) -> float:
		reference_mode = str(self.reference_source_combo.currentData())
		if reference_mode == "waterfall" and self.waterfall_events:
			return min(
				int(event["onset_frame"]) / self.fps
				for event in self.waterfall_events
			)
		if reference_mode == "combined":
			origin_frame = self._selected_grid_origin_frame()
			return origin_frame / self.fps if origin_frame is not None else 0.0
		if reference_mode.startswith("normalized"):
			settings = self._normalization_settings_map().get(reference_mode)
			return float(settings["origin_seconds"]) if settings is not None else 0.0
		if self.quantization_settings is not None:
			return float(self.quantization_settings["origin_seconds"])
		return min(
			(int(event["onset_frame"]) / self.fps for event in self.events),
			default=0.0,
		)

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
		normalized_source = self._normalization_events_map().get(source)
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
		normalization_settings = self._normalization_settings_map().get(reference_source)
		if normalization_settings is not None:
			return int(normalization_settings["origin_frame"])
		return None

	def _compute_adaptive_normalizations(self) -> None:
		waterfall_origin = min(
			(int(event["onset_frame"]) for event in self.waterfall_events),
			default=None,
		)
		combined_origin = min(
			(
				int(event["onset_frame"])
				for event in combine_detection_and_waterfall(
					self.events, self.waterfall_events, self.fps
				)
			),
			default=None,
		)
		specs = {
			"normalized_adaptive_detection": (self.events, [], None),
			"normalized_adaptive_waterfall": (self.waterfall_events, [], waterfall_origin),
			"normalized_adaptive": (self.events, self.waterfall_events, combined_origin),
		}
		self.adaptive_events = {}
		self.adaptive_settings = {}
		for key, (source, extra, origin) in specs.items():
			try:
				events, settings = normalize_events_tempo_adaptive(
					source, extra, self.fps, SONG_REFERENCE_BPM, 0.5,
					self.current_frame, origin,
				)
			except ValueError:
				events, settings = [], None
			self.adaptive_events[key] = events
			if settings is not None:
				self.adaptive_settings[key] = settings

	def _normalization_choices(self) -> list[tuple[str, str]]:
		return [
			("Detection normalized", "normalized_detection"),
			("Waterfall normalized", "normalized_waterfall"),
			("Mean normalized", "normalized"),
			("Detection tempo-adaptive", "normalized_adaptive_detection"),
			("Waterfall tempo-adaptive", "normalized_adaptive_waterfall"),
			("Mean tempo-adaptive", "normalized_adaptive"),
		]

	def _normalization_events_map(self) -> dict[str, list[dict[str, object]] | None]:
		return {
			"normalized_detection": self.normalized_detection_events,
			"normalized_waterfall": self.normalized_waterfall_events,
			"normalized": self.normalized_events,
			**self.adaptive_events,
		}

	def _normalization_settings_map(self) -> dict[str, dict[str, object]]:
		return {
			"normalized_detection": self.normalization_detection_settings,
			"normalized_waterfall": self.normalization_waterfall_settings,
			"normalized": self.normalization_settings,
			**self.adaptive_settings,
		}

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
		self._compute_adaptive_normalizations()
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
		if correction_source == "editable":
			return (
				float(event.get("editable_onset_seconds", raw_onset)),
				float(event.get("editable_offset_seconds", raw_end)),
			)
		if not corrected:
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
		elif reference_mode.startswith("normalized") and reference_mode in self._normalization_events_map():
			reference_events = self._normalization_events_map()[reference_mode] or []
			reference_label = dict(self._normalization_choices()).get(reference_mode, "Normalized")
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
				corrected_hand_color = colors[str(corrected["hand"])]
				corrected_bars.append({
					"row": row,
					"start": start,
					"end": end,
					"color": corrected_hand_color,
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
			normalization_settings = self._normalization_settings_map().get(reference_mode)
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
		normalization_map = self._normalization_events_map()
		if source in normalization_map:
			return normalization_map[source] or []
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
		self._update_active_piano_keys(self.playback_start_seconds)
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
		self._update_active_piano_keys(current_seconds)
		if self.autoscroll_checkbox.isChecked():
			self._follow_playhead(current_seconds)
		self.timeline_widget.set_playhead(current_seconds * unit_scale)
		if self.playback_scheduler.completed_event.is_set():
			self._stop_playback()

	def _update_active_piano_keys(self, current_seconds: float) -> None:
		source = str(self.playback_source_combo.currentData())
		corrected = source.startswith("normalized") or source in ("manual", "music21")
		active_notes: dict[str, str] = {}
		for event in self._selected_playback_events():
			onset, offset = self._event_bounds_seconds(
				event, corrected=corrected, correction_source=source
			)
			if onset <= current_seconds < offset:
				active_notes[str(event["note"])] = str(event["hand"])
		self.timeline_widget.set_active_notes(active_notes)

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
				return True
			self._scroll_timeline(-1 if delta > 0 else 1)
			return True
		return super().eventFilter(watched, event)

	def _stop_playback(self) -> None:
		self.playback_timer.stop()
		self._stop_playback_scheduler()
		self.timeline_widget.set_active_notes({})
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
			onset_seconds, offset_seconds = self._event_bounds_seconds(
				corrected_event, corrected=True, correction_source="editable"
			)
		else:
			onset_seconds, offset_seconds = self._event_bounds_seconds(
				corrected_event,
				corrected=True,
				correction_source=source,
			)
		if self.snap_edit_checkbox.isChecked():
			grid, origin = self._editable_snap_grid()
		else:
			grid, origin = 1.0 / self.fps, 0.0
		onset_grid = round((onset_seconds - origin) / grid)
		offset_grid = max(onset_grid + 1, round((offset_seconds - origin) / grid))
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
			event_record["editable_onset_seconds"] = onset_seconds
			event_record["editable_offset_seconds"] = offset_seconds
			event_record["duration_seconds"] = offset_seconds - onset_seconds
		elif drag["source"] == "music21":
			event_record["music21_onset_seconds"] = origin + onset_grid * grid
			event_record["music21_offset_seconds"] = origin + offset_grid * grid
			event_record["edit_onset_grid_index"] = onset_grid
			event_record["edit_offset_grid_index"] = offset_grid
			event_record["edit_grid_seconds"] = grid
			event_record["edit_origin_seconds"] = origin
		else:
			event_record["midi_onset_seconds"] = origin + onset_grid * grid
			event_record["midi_offset_seconds"] = origin + offset_grid * grid
			event_record["edit_onset_grid_index"] = onset_grid
			event_record["edit_offset_grid_index"] = offset_grid
			event_record["edit_grid_seconds"] = grid
			event_record["edit_origin_seconds"] = origin
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
		self.export_normalizations: dict[str, tuple[list[dict[str, object]], dict[str, object]]] = {}
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
		self.score_main_button = QPushButton("Score")
		self.score_main_button.setEnabled(False)
		self.score_main_button.clicked.connect(self.show_score_dialog)
		toolbar.addWidget(self.score_main_button)
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
			self.score_main_button,
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
		versions = [
			("original", "captured"),
			("manual", "manual"),
			("music21", "music21"),
			("normalized_detection", "detection_normalized"),
			("normalized_waterfall", "waterfall_normalized"),
			("normalized", "mean_normalized"),
			("normalized_adaptive_detection", "detection_tempo_adaptive"),
			("normalized_adaptive_waterfall", "waterfall_tempo_adaptive"),
			("normalized_adaptive", "mean_tempo_adaptive"),
		]
		failures: list[str] = []
		try:
			for version, suffix in versions:
				index = self.export_timing_combo.findData(version)
				if index < 0:
					continue
				self.export_timing_combo.setCurrentIndex(index)
				self.output_path = Path(folder) / (
					f"{self.video_path.stem}_key_hits_{suffix}.mid"
				)
				try:
					self._write_midi(quiet=True)
				except Exception as error:
					failures.append(f"{suffix}: {type(error).__name__}: {error}")
					continue
				export_paths.append(self.output_path)
		finally:
			self.output_path = original_output_path
			self.export_timing_combo.setCurrentIndex(original_timing_index)

		self.output_label.setText(
			f"Exported {len(export_paths)} MIDI versions to {folder}"
		)
		self.status_label.setText(f"Exported {len(export_paths)} MIDI timing versions")
		if failures:
			QMessageBox.warning(
				self, "Some MIDI versions failed", "\n".join(failures)
			)

	def _ensure_quantization_versions(self) -> None:
		if (
			self.quantized_events is not None
			and self.music21_quantized_events is not None
			and self.normalized_events is not None
			and self.export_normalizations
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
		labels = dict(
			[
				("Detection normalized", "normalized_detection"),
				("Waterfall normalized", "normalized_waterfall"),
				("Mean normalized", "normalized"),
				("Detection tempo-adaptive", "normalized_adaptive_detection"),
				("Waterfall tempo-adaptive", "normalized_adaptive_waterfall"),
				("Mean tempo-adaptive", "normalized_adaptive"),
			]
		)
		if not self.export_normalizations:
			computed = compute_all_normalizations(
				self.events, self.waterfall_events, self.fps, self.frame_index
			)
			if self.normalized_events is not None:
				computed.pop("normalized", None)
			self.export_normalizations = computed
		for label, data in (
			("Manual", "manual"),
			("music21", "music21"),
			("Mean normalized", "normalized"),
			*[
				(label, data)
				for label, data in labels.items()
				if data in self.export_normalizations
			],
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

	def show_score_dialog(self) -> None:
		was_playing = self.timer.isActive()
		if was_playing:
			self.pause()
		# The analysis dialog owns all quantizations and normalizations the score can use.
		source_dialog = DurationAnalysisDialog(
			self.events,
			self.fps,
			self,
			current_frame=self.frame_index,
			view_mode="timeline",
			waterfall_events=self.waterfall_events,
		)
		source_dialog._open_score_dialog()
		jump = source_dialog.pending_jump
		source_dialog.deleteLater()
		if jump is not None:
			self._show_analysis_dialog("timeline", jump_seconds=jump)
		elif was_playing:
			self.toggle_play()

	def _show_analysis_dialog(self, view_mode: str, jump_seconds: float | None = None) -> None:
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
			if jump_seconds is not None:
				dialog.jump_to_seconds(jump_seconds)
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
			and not self.export_normalizations
		):
			return
		self.quantized_events = None
		self.music21_quantized_events = None
		self.normalized_events = None
		self.export_normalizations = {}
		self.quantization_settings = None
		self.music21_quantization_settings = None
		self.normalization_settings = None
		for index in reversed(range(self.export_timing_combo.count())):
			if self.export_timing_combo.itemData(index) != "original":
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
			elif timing_mode in self.export_normalizations:
				export_events, quantization_settings = self.export_normalizations[timing_mode]
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
					onset_seconds = event.get(midi_onset_key)
					if onset_seconds is None:
						onset_seconds = int(event["onset_frame"]) / self.fps
					onset_seconds = float(onset_seconds)
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
			if quiet:
				raise
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