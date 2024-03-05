from core import StyleTTS2Core
from PyQt5.QtCore import (pyqtSignal, Qt, QBuffer, QSize, QUrl, QMimeData,
    QByteArray, QMargins)
from PyQt5.QtGui import (QIcon, QDrag, QIntValidator, QDoubleValidator)
from PyQt5.QtWidgets import (QWidget, QApplication, QMainWindow, QVBoxLayout,
    QHBoxLayout, QFrame, QPushButton, QPlainTextEdit, QGroupBox, QRadioButton,
    QSizePolicy, QGridLayout, QLineEdit, QLabel, QListWidget, QListWidgetItem,
    QComboBox, QDialog, QProgressBar, QFileDialog, QMessageBox, QCheckBox,
    QAbstractItemView, QSlider, QStyle)
from PyQt5.QtMultimedia import (QMediaPlayer, QMediaContent)
#import pygame
import PyQt5
import sys
import os
import torch
import time
import soundfile as sf
import numpy as np
from pathlib import Path
from config import load_config, save_config
from gui_models import find_models
from log import logger
from io import BytesIO
from nltk import sent_tokenize
from utils import sanitize_filename
import traceback

class FileButton(QPushButton):
    fileDropped = pyqtSignal(list)
    def __init__(self, file_cb, label = "Files to Convert"):
        super().__init__(label)
        self.setAcceptDrops(True)
        self.pressed.connect(self.dialog)
        self.label = label
        self.file_cb = file_cb
        self.files = None

    def dialog(self):
        file = QFileDialog.getOpenFileName(self, self.label)
        self.files = [file[0]]
        if self.file_cb is not None:
            self.file_cb(self.files)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            clean_files = []
            for url in event.mimeData().urls():
                if not url.toLocalFile():
                    continue
                clean_files.append(url.toLocalFile())
            self.files = clean_files
            if self.file_cb is not None:
                self.file_cb(self.files)
            event.acceptProposedAction()
        else:
            event.ignore()
        pass

class FieldWidget(QFrame):
    def __init__(self, label, field):
        super().__init__()
        self.layout = QHBoxLayout(self)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0,0,0,0)
        label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(label)
        self.label = label
        self.field = field
        field.setAlignment(Qt.AlignRight)
        field.sizeHint = lambda: QSize(60, 32)
        field.setSizePolicy(QSizePolicy.Maximum,
            QSizePolicy.Preferred)
        self.layout.addWidget(field)

    def setEnabled(self, val):
        self.field.setEnabled(val)

    def text(self):
        return self.field.text()

class AudioPreviewWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.vlayout = QVBoxLayout(self)
        self.vlayout.setSpacing(0)
        self.vlayout.setContentsMargins(0,0,0,0)

        self.playing_label = QLabel("Preview")
        self.playing_label.setWordWrap(True)
        self.vlayout.addWidget(self.playing_label)

        self.player_frame = QFrame()
        self.vlayout.addWidget(self.player_frame)

        self.player_layout = QHBoxLayout(self.player_frame)
        self.player_layout.setSpacing(4)
        self.player_layout.setContentsMargins(0,0,0,0)

        #self.playing_label.hide()

        self.player = QMediaPlayer()
        self.player.setNotifyInterval(500)

        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setSizePolicy(QSizePolicy.Expanding,
            QSizePolicy.Preferred)
        self.player_layout.addWidget(self.seek_slider)

        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(
            getattr(QStyle, 'SP_MediaPlay')))
        self.player_layout.addWidget(self.play_button)
        self.play_button.clicked.connect(self.toggle_play)
        self.play_button.setSizePolicy(QSizePolicy.Maximum,
            QSizePolicy.Minimum)
        self.play_button.mouseMoveEvent = self.drag_hook

        self.seek_slider.sliderMoved.connect(self.seek)
        self.player.positionChanged.connect(self.update_seek_slider)
        self.player.stateChanged.connect(self.state_changed)
        self.player.durationChanged.connect(self.duration_changed)

        self.local_file = ""

    def set_text(self, text=""):
        if len(text) > 0:
            self.playing_label.show()
            self.playing_label.setText(text)
        else:
            self.playing_label.hide()

    def from_file(self, path):
        try:
            self.player.stop()
            if hasattr(self, 'audio_buffer'):
                self.audio_buffer.close()

            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(
                os.path.abspath(path))))

            self.play_button.setIcon(self.style().standardIcon(
                getattr(QStyle, 'SP_MediaPlay')))

            self.local_file = path
        except Exception as e:
            print(e)
            pass

    def drag_hook(self, e):
        if e.buttons() != Qt.LeftButton:
            return
        self.try_drag_from_file()

    def try_drag_from_file(self):
        if not len(self.local_file):
            return
        mime_data = QMimeData()
        mime_data.setUrls([QUrl.fromLocalFile(
            os.path.abspath(self.local_file))])
        drag = QDrag(self)
        drag.setMimeData(mime_data)
        drag.exec_(Qt.CopyAction)

# This is not possible
#    def try_drag_from_memory(self):
#        if not hasattr(self, 'audio_data'):
#            return
#        mime_data = QMimeData()
#        mime_data.setData("application/octet-stream", self.audio_data_bytestring)
#        mime_data.setProperty("fileName", self.audio_default_filename)
#        drag = QDrag(self)
#        drag.setMimeData(mime_data)
#        drag.exec_(Qt.CopyAction)
#
    def from_memory(self, data, default_filename="none"):
        self.player.stop()
        if hasattr(self, 'audio_buffer'):
            self.audio_buffer.close()

        self.audio_data = QByteArray(data)
        self.audio_data_bytestring = data
        self.audio_default_filename = default_filename
        self.audio_buffer = QBuffer()
        self.audio_buffer.setData(self.audio_data)
        self.audio_buffer.open(QBuffer.ReadOnly)
        self.player.setMedia(QMediaContent(), self.audio_buffer)

    def state_changed(self, state):
        if (state == QMediaPlayer.StoppedState) or (
            state == QMediaPlayer.PausedState):
            self.play_button.setIcon(self.style().standardIcon(
                getattr(QStyle, 'SP_MediaPlay')))

    def duration_changed(self, dur):
        self.seek_slider.setRange(0, self.player.duration())

    def toggle_play(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        elif self.player.mediaStatus() != QMediaPlayer.NoMedia:
            self.player.play()
            self.play_button.setIcon(self.style().standardIcon(
                getattr(QStyle, 'SP_MediaPause')))

    def update_seek_slider(self, position):
        self.seek_slider.setValue(position)

    def seek(self, position):
        self.player.setPosition(position)

class StyleTTS2GUI(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.setWindowTitle('StyleTTS2')
        #self.setFixedWidth(2000)

        self.core = StyleTTS2Core(device=config.device)
        self.config = config

        self.central_widget = QFrame()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.layout.addWidget(self.top_frame())

        self.text_box = QGroupBox("Text to infer")
        self.text_box.setMinimumHeight(420)
        self.layout.addWidget(self.text_box)
        self.text_lay = QVBoxLayout(self.text_box)
        self.text_lay.setAlignment(Qt.AlignTop)

        self.text_field = QPlainTextEdit()
        self.text_lay.addWidget(self.text_field, stretch=2)

        self.phonemes_label = QLabel("Phonemes")
        self.phonemes_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.phonemes_label.setWordWrap(True)
        self.text_lay.addWidget(self.phonemes_label)

        self.layout.addWidget(self.infer_frame())

        self.find_models()

        os.makedirs(self.config['output_dir'], exist_ok=True)
    pass

    def top_frame(self):
        frame = QFrame()
        frame.setMaximumHeight(220)
        layout = QHBoxLayout(frame)

        self.model_frame = QGroupBox("Model")
        layout.addWidget(self.model_frame)
        model_lay = QVBoxLayout(self.model_frame)
        model_lay.setAlignment(Qt.AlignTop)

        self.model_dropdown = QComboBox()
        model_lay.addWidget(self.model_dropdown)
        self.model_dropdown.currentTextChanged.connect(self.load_model_cb)

        self.acoustic_frame = QGroupBox("Acoustic style")
        layout.addWidget(self.acoustic_frame)
        acoustic_lay = QVBoxLayout(self.acoustic_frame)
        acoustic_lay.setAlignment(Qt.AlignTop)

        # 1. Dropdown menu for preset styles
        self.acoustic_dropdown = QComboBox()
        acoustic_lay.addWidget(self.acoustic_dropdown)

        # 2. Users should be able to select their own acoustic style
        self.acoustic_custom_label = QLabel("File:") # Reads out the loaded file
        def acoustic_select_cb(files):
            file = files[0]
            self.acoustic_custom_label.setText(file)
        self.acoustic_custom_select = FileButton(
            file_cb=acoustic_select_cb, label="Load custom acoustic from audio")
        acoustic_lay.addWidget(self.acoustic_custom_select)
        acoustic_lay.addWidget(self.acoustic_custom_label)
        acoustic_lay.addStretch()


        # 3. Select whether to use the preset or custom
        self.acoustic_from_preset = QRadioButton("Use acoustic style from dropdown")
        acoustic_lay.addWidget(self.acoustic_from_preset)
        self.acoustic_from_custom = QRadioButton("Use acoustic style from file")
        acoustic_lay.addWidget(self.acoustic_from_custom)
        self.acoustic_from_preset.setChecked(True)

        self.prosodic_frame = QGroupBox("Prosodic style")
        prosodic_lay = QVBoxLayout(self.prosodic_frame)
        prosodic_lay.setAlignment(Qt.AlignTop)
        
        # 1. Select own prosodic style
        layout.addWidget(self.prosodic_frame)
        self.prosodic_custom_label = QLabel("File:")
        def prosodic_select_cb(files):
            file = files[0]
            self.prosodic_custom_label.setText(file)
        self.prosodic_custom_select = FileButton(
            file_cb=prosodic_select_cb, label="Load custom prosodic from audio")
        prosodic_lay.addWidget(self.prosodic_custom_select)
        prosodic_lay.addWidget(self.prosodic_custom_label)
        prosodic_lay.addStretch()

        # 2. Select to use from preset or custom
        self.prosodic_from_preset = QRadioButton("Use prosodic style associated with acoustic")
        prosodic_lay.addWidget(self.prosodic_from_preset)
        self.prosodic_from_custom = QRadioButton("Use prosodic style from file")
        prosodic_lay.addWidget(self.prosodic_from_custom)
        self.prosodic_from_preset.setChecked(True)

        layout.addWidget(self.prosodic_frame)
        return frame

    def infer_frame(self):
        frame = QFrame()
        layout = QHBoxLayout(frame)

        infer_settings_frame = QGroupBox("Inference settings")
        layout.addWidget(infer_settings_frame)
        infer_settings_lay = QVBoxLayout(infer_settings_frame)
        infer_settings_lay.setAlignment(Qt.AlignTop)

        self.alpha = FieldWidget(
            QLabel("Alpha"), QLineEdit("0.1"))
        self.alpha.field.setValidator(QDoubleValidator(0.0, 1.0, 3))
        self.alpha.label.setToolTip(
            "Controls how much synthesized audio timbre is conditioned"
            "on generated text vs. the reference audio timbre. "
            "A lower alpha will have a closer timbre to reference.")
        infer_settings_lay.addWidget(self.alpha)

        self.beta = FieldWidget(
            QLabel("Beta"), QLineEdit("0.6"))
        self.beta.field.setValidator(QDoubleValidator(0.0, 1.0, 3))
        self.beta.label.setToolTip(
            "Controls how much synthesized audio prosody is conditioned"
            "on generated text vs. the reference audio prosody. "
            "A lower beta will have a closer prosody to reference.")
        infer_settings_lay.addWidget(self.beta)

        self.diffusion_steps = FieldWidget(
            QLabel("Diffusion steps"), QLineEdit("10"))
        self.diffusion_steps.field.setValidator(QIntValidator(1,10))
        self.diffusion_steps.label.setToolTip(
            "Number of diffusion steps used to predict style vector. "
            "Higher steps increase processing time but will have higher diversity. "
            "Quality also increases with higher steps to an extent.")
        infer_settings_lay.addWidget(self.diffusion_steps)

        self.embedding_scale = FieldWidget(
            QLabel("Embedding scale"), QLineEdit("2.5"))

        self.embedding_scale.label.setToolTip(
            "aka CFG scale. Higher settings increase the conditioning of the style vectors "
            "on the text, increasing emotiveness at the cost of output quality at high values")
        self.embedding_scale.field.setValidator(QDoubleValidator(0.1, 30.0, 3))
        infer_settings_lay.addWidget(self.embedding_scale)

        self.target_wpm = FieldWidget(
            QLabel("Target wpm"), QLineEdit("170"))
        self.target_wpm.field.setValidator(QIntValidator(1,1000))
        infer_settings_lay.addWidget(self.target_wpm)

        self.dur_switch = QCheckBox("Enable duration scaling")
        self.dur_switch.setToolTip("Uses the target wpm setting to scale duration of generated audio.")
        self.dur_switch.setChecked(True)
        infer_settings_lay.addWidget(self.dur_switch)

        self.style_blend = FieldWidget(
            QLabel("Style blend"), QLineEdit("0.7"))
        self.style_blend.field.setValidator(QDoubleValidator(0.0,1.0,2))
        self.style_blend.label.setToolTip(
            "Used for blending styles of subsequent sentences in long form inference mode."
        )
        infer_settings_lay.addWidget(self.style_blend)

        self.f0 = FieldWidget(
            QLabel("F0 adjust (Hz)"), QLineEdit("0"))
        self.f0.field.setValidator(QIntValidator(-5000,5000))
        infer_settings_lay.addWidget(self.f0)

        self.longform = QCheckBox("Long form inference mode")
        infer_settings_lay.addWidget(self.longform)

        self.longform_single = QCheckBox("Only infer one iteration in longform mode")
        self.longform_single.setChecked(True)
        infer_settings_lay.addWidget(self.longform_single)

        infer_outputs_frame = QGroupBox("Outputs")
        layout.addWidget(infer_outputs_frame)
        infer_outputs_lay = QVBoxLayout(infer_outputs_frame)
        infer_outputs_lay.setAlignment(Qt.AlignTop)

        self.audio_previews = [None]*self.config.n_infer
        for i in range(self.config.n_infer):
            self.audio_previews[i] = AudioPreviewWidget()
            infer_outputs_lay.addWidget(self.audio_previews[i])

        self.infer_button = QPushButton("Infer")
        infer_outputs_lay.addWidget(self.infer_button)
        self.infer_button.pressed.connect(self.infer_cb)
        
        infer_outputs_lay.addStretch(1)
        self.infer_time = QLabel("Inference time:")
        infer_outputs_lay.addWidget(self.infer_time)

        self.infer_progress = QProgressBar()
        infer_outputs_lay.addWidget(self.infer_progress)
        
        return frame

    def load_model_cb(self):
        key = self.model_dropdown.currentText()
        spec = self.model_specs[key]
        self.core.load_model(spec['config'], spec['ckpt'])

        self.acoustic_dropdown.clear()
        for key,style in spec['style_index_data'].items():
            self.acoustic_dropdown.addItem(key)
            # Styles are not computed until inference time

    def current_spec(self):
        key = self.model_dropdown.currentText()
        return self.model_specs[key]

    def output_name(self, j):
        if not self.acoustic_from_custom.isChecked():
            tag = self.acoustic_dropdown.currentText()
        else:
            tag = ""
        return os.path.join(
            self.config['output_dir'],
            sanitize_filename(
                tag+'_'+
                self.model_dropdown.currentText()+'_'+
                self.text_field.toPlainText()[0:40]+str(j)+'.flac'))

    def find_models(self):
        self.model_specs = find_models(config)
        self.model_dropdown.clear()
        if not len(self.model_specs):
            return
        for key,spec in self.model_specs.items():
            self.model_dropdown.addItem(key)

        self.acoustic_dropdown.clear()
        self.load_model_cb()

    def infer_cb(self):
        self.infer_progress.setValue(0)
        # 1. Gather styles according to settings
        if self.acoustic_from_custom.isChecked():
            style_files = self.acoustic_custom_select.files
            if style_files is None or not len(style_files):
                logger.warn("No file supplied for acoustic style")
                return
            style_files = style_files[0]
            t = self.core.style_from_path_with_components(style_files)
            if t is None:
                logger.warn(
                    f"File {style_files} is too short for style calculations")
                return
            _, acoustic_vec, prosodic_vec = t
            acoustic_vec = acoustic_vec.cpu().numpy()
            prosodic_vec = prosodic_vec.cpu().numpy()
        else:
            key = self.acoustic_dropdown.currentText()
            style_vec = self.current_spec()['style_index_data'][key]
            style_vec = np.array(style_vec)

            acoustic_vec = style_vec[:,0:128]
            prosodic_vec = style_vec[:,128:256]

        if self.prosodic_from_custom.isChecked():
            style_files = self.prosodic_custom_select.files
            if style_files is None or not len(style_files):
                logger.warn("No file supplied for prosodic style")
                return
            style_files = style_files[0]
            t = self.core.style_from_path_with_components(style_files)
            if t is None:
                logger.warn(
                    f"File {style_files} is too short for style calculations")
                return
            _, _, prosodic_vec = t
            prosodic_vec = prosodic_vec.cpu().numpy()
        
        style_vec = torch.cat([
            torch.from_numpy(acoustic_vec), 
            torch.from_numpy(prosodic_vec)], dim=1).to(
                self.config.device).to(torch.float32)

        start_time = time.time()
        n_infer = self.config.n_infer
        if not self.longform.isChecked():
            for i in range(n_infer):
                self.infer_progress.setValue(
                    int((i+1)*100/(self.config.n_infer)))
                j = i
                while os.path.exists(self.output_name(j)):
                    j += 1
                output_name = self.output_name(j)
                output_basename = Path(output_name).name
                if self.dur_switch.isChecked():
                    target_wpm = int(self.target_wpm.text())
                else:
                    target_wpm = None
                try:
                    out, ps = self.core.inference(self.text_field.toPlainText(),
                        style_vec,
                        alpha=float(self.alpha.text()),
                        beta=float(self.beta.text()),
                        diffusion_steps=int(self.diffusion_steps.text()),
                        embedding_scale=float(self.embedding_scale.text()),
                        target_wpm=target_wpm,
                        f0_adjust=int(self.f0.text()))
                except Exception as e:
                    traceback.print_exception(e)
                    logger.error(e)
                    return
                sf.write(output_name, out, self.core.sr, format='flac')
                self.audio_previews[i].from_file(output_name)
                self.audio_previews[i].set_text(output_basename)
        else:
            for i in range(n_infer):
                self.infer_progress.setValue(
                    int((i+1)*100/(self.config.n_infer)))
                j = i
                while os.path.exists(self.output_name(j)):
                    j += 1
                output_name = self.output_name(j)
                output_basename = Path(output_name).name
                if self.dur_switch.isChecked():
                    target_wpm = int(self.target_wpm.text())
                else:
                    target_wpm = None
                textlist = sent_tokenize(self.text_field.toPlainText())
                s_prev = None
                wavs = []
                for text in textlist:
                    try:
                        wav, s_prev, ps = self.core.LFinference(text,
                            s_prev,
                            style_vec,
                            alpha=float(self.alpha.text()),
                            beta=float(self.beta.text()),
                            t=float(self.style_blend.text()),
                            diffusion_steps=int(self.diffusion_steps.text()),
                            embedding_scale=float(self.embedding_scale.text()),
                            target_wpm=target_wpm,
                            f0_adjust=int(self.f0.text()))
                    except Exception as e:
                        traceback.print_exception(e)
                        logger.error(e)
                    wavs.append(wav)
                out = np.concatenate(wavs)
                sf.write(output_name, out, self.core.sr, format='flac')
                self.audio_previews[i].from_file(output_name)
                self.audio_previews[i].set_text(output_basename)
        dur = time.time() - start_time
        self.phonemes_label.setText(f"Phonemes: {ps}")
        self.infer_time.setText(f"Inference time: {dur}s ("
            f"{dur/self.config.n_infer}s/it)")


def gui_main(config):
    app = QApplication(sys.argv)

    if config.get('dark_mode', False):
        with open('ManjaroMix.qss', 'r') as f:
            style = f.read()
            app.setStyleSheet(style)

    w = StyleTTS2GUI(config)
    w.show()
    app.exec()

if __name__ == '__main__':
    config = load_config()
    gui_main(config)
    save_config(config)