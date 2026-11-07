"""Reusable detection profiles and their video calibration dialog."""

import json

import cv2
import numpy as np
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
	QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
	QHBoxLayout, QInputDialog, QLabel, QMessageBox, QPushButton, QSlider,
	QSpinBox, QVBoxLayout,
)

from pianorolldetection import DEFAULT_KEY_SATURATION, DEFAULT_PROFILE, HUE_FIELDS, hue_range, score_frame, validate_profile, waterfall_hues
from pianorollwaterfall import strike_line, waterfall_segments


def load_profiles():
	text = QSettings("PianoRollReader", "PianoRollReader").value("detection_profiles", "[]")
	try:
		profiles = json.loads(text)
		if not isinstance(profiles, list):
			raise ValueError("Profiles must be a list.")
		profiles = [validate_profile(item) for item in profiles]
		names = [item["name"].casefold() for item in profiles]
		if "original" in names or len(names) != len(set(names)):
			raise ValueError("Profile names must be unique and cannot be Original.")
		return profiles
	except (TypeError, ValueError) as error:
		raise ValueError(f"Cannot load saved detection profiles: {error}") from error


def save_profiles(profiles):
	settings = QSettings("PianoRollReader", "PianoRollReader")
	settings.setValue("detection_profiles", json.dumps(profiles))
	settings.sync()
	if settings.status() != QSettings.Status.NoError:
		raise OSError("Could not save detection profiles to application settings.")


class CalibrationPreview(QLabel):
	def __init__(self, clicked):
		super().__init__()
		self.clicked = clicked
		self.image_size = None
		self.image = None
		self.drag_enabled = False
		self.setAlignment(Qt.AlignmentFlag.AlignCenter)
		self.setMinimumSize(480, 270)
		self.setText("No frame")

	def set_frame(self, frame):
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		height, width = rgb.shape[:2]
		self.image_size = (width, height)
		self.image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()
		self._resize_image()

	def _resize_image(self):
		if self.image is None:
			return
		self.setPixmap(QPixmap.fromImage(self.image).scaled(
			self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
		))

	def resizeEvent(self, event):
		super().resizeEvent(event)
		self._resize_image()

	def mousePressEvent(self, event):
		self._click(event)

	def mouseMoveEvent(self, event):
		if self.drag_enabled and event.buttons() & Qt.MouseButton.LeftButton:
			self._click(event)

	def _click(self, event):
		pixmap = self.pixmap()
		if self.image_size is None or pixmap is None or pixmap.isNull():
			return
		x = event.position().x() - (self.width() - pixmap.width()) / 2
		y = event.position().y() - (self.height() - pixmap.height()) / 2
		if 0 <= x < pixmap.width() and 0 <= y < pixmap.height():
			self.clicked(int(x * self.image_size[0] / pixmap.width()), int(y * self.image_size[1] / pixmap.height()))


class DetectionProfileDialog(QDialog):
	def __init__(self, video_path, frame_index, baseline, keys, masks, key_width, active_profile, parent=None):
		super().__init__(parent)
		self.setWindowTitle("Detection profiles")
		self.setStyleSheet(
			"QDialog { background: #eef1f4; }"
			"QWidget { color: #18212b; font-family: 'Segoe UI'; font-size: 10pt; }"
			"QLabel { color: #18212b; background: transparent; }"
			"QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { color: #18212b; background: #ffffff; "
			"border: 1px solid #aeb8c4; padding: 3px; }"
			"QComboBox QAbstractItemView { color: #18212b; background: #ffffff; "
			"selection-color: #18212b; selection-background-color: #c8dbe8; }"
			"QPushButton { color: #18212b; background: #ffffff; border: 1px solid #aeb8c4; padding: 5px 9px; }"
			"QPushButton:hover { background: #e1edf5; }"
		)
		self.resize(1000, 850)
		self.baseline, self.keys, self.masks, self.key_width = baseline, keys, masks, key_width
		self.capture = cv2.VideoCapture(str(video_path))
		if not self.capture.isOpened():
			self.capture.release()
			raise OSError(f"Cannot open video for calibration: {video_path}")
		try:
			self.profiles = load_profiles()
		except ValueError:
			self.capture.release()
			raise
		self.frame = None
		self.result_profile = None
		layout = QVBoxLayout(self)
		top = QHBoxLayout()
		self.profile_combo = QComboBox()
		self.profile_combo.addItem("Original", DEFAULT_PROFILE)
		for profile in self.profiles:
			self.profile_combo.addItem(profile["name"], profile)
		active_index = next((i for i in range(self.profile_combo.count())
							 if self.profile_combo.itemData(i) == active_profile), -1)
		if active_index < 0:
			self.profile_combo.addItem(f"{active_profile['name']} (project / unsaved)", active_profile)
			active_index = self.profile_combo.count() - 1
		self.profile_combo.setCurrentIndex(active_index)
		top.addWidget(QLabel("Profile"))
		top.addWidget(self.profile_combo, 1)
		save = QPushButton("Save as...")
		save.clicked.connect(self._save_profile)
		top.addWidget(save)
		delete = QPushButton("Delete saved profile")
		delete.clicked.connect(self._delete_profile)
		top.addWidget(delete)
		layout.addLayout(top)
		self.preview = CalibrationPreview(self._sample)
		layout.addWidget(self.preview, 1)
		self.slider = QSlider(Qt.Orientation.Horizontal)
		self.slider.setRange(0, max(0, int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1))
		self.slider.setValue(max(0, min(frame_index, self.slider.maximum())))
		layout.addWidget(self.slider)
		self.frame_label = QLabel()
		layout.addWidget(self.frame_label)
		sample_row = QHBoxLayout()
		self.sample_mode = QComboBox()
		for label, key in (
			("Place keyboard line", "line"),
			("Left white key", "key_left"), ("Left black key", "key_left_black"),
			("Right white key", "key_right"), ("Right black key", "key_right_black"),
			("Left white-key bar", "bar_left"), ("Left black-key bar", "bar_left_black"),
			("Right white-key bar", "bar_right"), ("Right black-key bar", "bar_right_black"),
		):
			self.sample_mode.addItem(label, key)
		self.sample_mode.currentIndexChanged.connect(
			lambda: setattr(self.preview, "drag_enabled", self.sample_mode.currentData() == "line")
		)
		self.preview.drag_enabled = True
		sample_row.addWidget(QLabel("Click preview:"))
		sample_row.addWidget(self.sample_mode)
		sample_row.addWidget(QLabel("Sample proposes hue ±5 and key saturation; refine below."))
		layout.addLayout(sample_row)
		form = QFormLayout()
		self.hue_controls = {}
		for key, label in (("key_left", "Left key"), ("key_right", "Right key"),
						   ("bar_left", "Left bar"), ("bar_right", "Right bar")):
			row = QHBoxLayout()
			for field, kind in ((key, "White"), (f"{key}_black", "Black")):
				row.addWidget(QLabel(f"{kind} min / max"))
				low, high = QSpinBox(), QSpinBox()
				for spin in (low, high):
					spin.setRange(0, 179)
					spin.valueChanged.connect(self._refresh)
					row.addWidget(spin)
				self.hue_controls[field] = (low, high)
			form.addRow(label + " hue", row)
		self.saturation_controls = {}
		for hand in ("left", "right"):
			row = QHBoxLayout()
			for kind, suffix in (("White", ""), ("Black", "_black")):
				row.addWidget(QLabel(kind))
				spin = QSpinBox()
				spin.setRange(0, 254)
				spin.valueChanged.connect(self._refresh)
				row.addWidget(spin)
				self.saturation_controls[f"key_{hand}{suffix}_saturation"] = spin
			form.addRow(f"{hand.title()} key saturation >", row)
		self.position = QDoubleSpinBox()
		self.position.setRange(10, 99)
		self.position.setDecimals(2)
		self.position.setSuffix(" %")
		self.position.valueChanged.connect(self._refresh)
		self.use_original_position = QComboBox()
		self.use_original_position.addItem("Original fixed position (75%)", True)
		self.use_original_position.addItem("Custom keyboard position", False)
		self.use_original_position.currentIndexChanged.connect(self._refresh)
		form.addRow("Keyboard position mode", self.use_original_position)
		form.addRow("Keyboard top", self.position)
		self.offset = QSpinBox()
		self.offset.setRange(0, 200)
		self.offset.setSuffix(" px")
		self.offset.valueChanged.connect(self._refresh)
		form.addRow("Tracking line above keyboard", self.offset)
		layout.addLayout(form)
		self.info = QLabel()
		self.info.setWordWrap(True)
		layout.addWidget(self.info)
		buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
		buttons.accepted.connect(self._accept)
		buttons.rejected.connect(self.reject)
		layout.addWidget(buttons)
		self.profile_combo.currentIndexChanged.connect(self._select_profile)
		self.slider.valueChanged.connect(self._schedule_frame)
		self.frame_timer = QTimer(self)
		self.frame_timer.setSingleShot(True)
		self.frame_timer.timeout.connect(self._read_frame)
		self._set_profile(active_profile)
		self._read_frame()

	def done(self, result):
		self.capture.release()
		super().done(result)

	def _set_profile(self, profile):
		profile = validate_profile(profile)
		for key, (low, high) in self.hue_controls.items():
			value = hue_range(profile, key.removesuffix("_black"), "black" if key.endswith("_black") else "white")
			low.setValue(value[0])
			high.setValue(value[1])
		for key, control in self.saturation_controls.items():
			control.setValue(profile.get(key, DEFAULT_KEY_SATURATION))
		self.use_original_position.setCurrentIndex(0 if profile["keyboard_fraction"] is None else 1)
		self.position.setValue(100 * (profile["keyboard_fraction"] if profile["keyboard_fraction"] is not None else 0.75))
		self.offset.setValue(profile["line_offset"])
		self.profile_name = profile["name"]
		self._refresh()

	def _profile(self):
		profile = {"name": self.profile_name}
		for key in HUE_FIELDS:
			low, high = self.hue_controls[key]
			profile[key] = [low.value(), high.value()]
			low, high = self.hue_controls[f"{key}_black"]
			black = [low.value(), high.value()]
			if black != profile[key]:
				profile[f"{key}_black"] = black
		for key, control in self.saturation_controls.items():
			if control.value() != DEFAULT_KEY_SATURATION:
				profile[key] = control.value()
		profile["keyboard_fraction"] = None if self.use_original_position.currentData() else self.position.value() / 100
		profile["line_offset"] = self.offset.value()
		return validate_profile(profile)

	def _select_profile(self):
		self._set_profile(self.profile_combo.currentData())

	def _save_profile(self):
		name, ok = QInputDialog.getText(self, "Save detection profile", "Profile name:", text=self.profile_name if self.profile_name != "Original" else "")
		if not ok:
			return
		name = name.strip()
		if name.casefold() == "original":
			QMessageBox.warning(self, "Save profile", "Original is read-only. Choose another name.")
			return
		try:
			profile = self._profile()
			profile["name"] = name
			profile = validate_profile(profile)
			existing = next((p for p in self.profiles if p["name"].casefold() == name.casefold()), None)
			if existing and QMessageBox.question(self, "Replace profile", f"Replace '{existing['name']}'?") != QMessageBox.StandardButton.Yes:
				return
			profiles = [p for p in self.profiles if p["name"].casefold() != name.casefold()] + [profile]
			save_profiles(profiles)
		except (ValueError, OSError) as error:
			QMessageBox.warning(self, "Save profile", str(error))
			return
		self.profiles = profiles
		self.profile_combo.blockSignals(True)
		self.profile_combo.clear()
		self.profile_combo.addItem("Original", DEFAULT_PROFILE)
		for item in profiles:
			self.profile_combo.addItem(item["name"], item)
		self.profile_combo.setCurrentIndex(self.profile_combo.count() - 1)
		self.profile_combo.blockSignals(False)
		self.profile_name = name

	def _delete_profile(self):
		index = self.profile_combo.currentIndex()
		if index == 0:
			QMessageBox.information(self, "Delete profile", "Original cannot be deleted.")
			return
		name = self.profile_combo.currentText()
		if not any(p["name"] == name for p in self.profiles):
			QMessageBox.information(self, "Delete profile", "This profile is not saved in application settings.")
			return
		if QMessageBox.question(self, "Delete profile", f"Delete '{name}'?") != QMessageBox.StandardButton.Yes:
			return
		profiles = [p for p in self.profiles if p["name"] != name]
		try:
			save_profiles(profiles)
		except OSError as error:
			QMessageBox.warning(self, "Delete profile", str(error))
			return
		self.profiles = profiles
		self.profile_combo.removeItem(index)
		self.profile_combo.setCurrentIndex(0)

	def _schedule_frame(self):
		self.frame_timer.start(100)

	def _read_frame(self):
		index = self.slider.value()
		self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
		ok, frame = self.capture.read()
		if not ok:
			self.frame = None
			self.preview.clear()
			self.preview.setText("Cannot read frame")
			self.info.setText(f"Could not read frame {index}.")
			return
		self.frame = frame
		fps = self.capture.get(cv2.CAP_PROP_FPS)
		self.frame_label.setText(f"Frame {index}" + (f"  ({index / fps:.2f} s)" if fps > 0 else ""))
		self._refresh()

	def _sample(self, x, y):
		if self.frame is None:
			return
		mode = self.sample_mode.currentData()
		if mode == "line":
			self.use_original_position.setCurrentIndex(1)
			self.position.setValue(100 * y / self.frame.shape[0])
		else:
			h, s, v = cv2.cvtColor(self.frame[y:y+1, x:x+1], cv2.COLOR_BGR2HSV)[0, 0]
			is_key = mode.startswith("key_")
			if s <= (30 if is_key else 120) or v <= 100:
				QMessageBox.warning(
					self, "Sample color",
					"Choose a colored, pressed key or bar, away from its edge."
					if is_key else "Choose a clearly colored bar, away from its edge.",
				)
				return
			low, high = self.hue_controls[mode]
			low.setValue(max(0, int(h) - 5))
			high.setValue(min(179, int(h) + 5))
			if is_key:
				self.saturation_controls[f"{mode}_saturation"].setValue(
					min(DEFAULT_KEY_SATURATION, max(0, int(s) - 25))
				)

	def _refresh(self):
		if self.frame is None:
			return
		try:
			profile = self._profile()
		except ValueError as error:
			self.info.setText(str(error))
			return
		frame = self.frame.copy()
		pressed = score_frame(frame, self.baseline, self.keys, self.masks, profile)
		for key in self.keys:
			if str(key["note"]) in pressed:
				cv2.polylines(frame, [np.array(key["contour"], dtype=np.int32)], True, (0, 0, 255), 3)
		white_hues, black_hues = waterfall_hues(profile)
		segments = waterfall_segments(self.frame, self.keys, self.key_width, profile["keyboard_fraction"], white_hues, profile["line_offset"], black_hues)
		keyboard = round(frame.shape[0] * (profile["keyboard_fraction"] if profile["keyboard_fraction"] is not None else 0.75))
		line = strike_line(frame.shape[0], profile["keyboard_fraction"], profile["line_offset"])
		centers = {str(key["note"]): int(key["bbox"][0] + key["bbox"][2] / 2) for key in self.keys}
		for note, hand, top, bottom in segments:
			cv2.line(frame, (centers[note], top), (centers[note], bottom),
					 (255, 255, 0) if hand == "left" else (0, 255, 0), 3)
		cv2.line(frame, (0, keyboard), (frame.shape[1]-1, keyboard), (0, 255, 255), 2)
		cv2.line(frame, (0, line), (frame.shape[1]-1, line), (0, 0, 255), 2)
		self.preview.set_frame(frame)
		self.info.setText("Red outlines: detected keys; colored lines: waterfall segments. Yellow: keyboard top; red: tracking line. "
						  f"Keys: {', '.join(sorted(pressed)) or 'none'}; waterfall segments: {len(segments)}.")

	def _accept(self):
		try:
			self.result_profile = self._profile()
			if self.result_profile["name"] == "Original" and self.result_profile != DEFAULT_PROFILE:
				self.result_profile["name"] = "Custom (unsaved)"
		except ValueError as error:
			QMessageBox.warning(self, "Invalid profile", str(error))
			return
		if self.frame_timer.isActive():
			self.frame_timer.stop()
			self._read_frame()
		self.accept()
