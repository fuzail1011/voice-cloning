import sys
from pathlib import Path
from time import sleep
from typing import List, Set
from warnings import filterwarnings, warn

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf
import umap
from PyQt5.QtCore import Qt, QStringListModel
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtGui import QPixmap
from encoder.inference import plot_embedding_as_heatmap
from toolbox.utterance import Utterance

filterwarnings("ignore")


colormap = (
    np.array(
        [
            [0, 127, 70],
            [255, 0, 0],
            [255, 217, 38],
            [0, 135, 255],
            [165, 0, 165],
            [255, 167, 255],
            [97, 142, 151],
            [0, 255, 255],
            [255, 96, 38],
            [142, 76, 0],
            [33, 0, 127],
            [0, 0, 0],
            [183, 183, 183],
            [76, 255, 0],
        ],
        dtype=np.float,
    )
    / 255
)

default_text = "Welcome to the Team 22 Voice cloning module! This enables us to create cloned voice for audiobook creation"


class UI(QDialog):
    min_umap_points = 4
    max_log_lines = 5
    max_saved_utterances = 20

    def save_audio_file(self, wav, sample_rate):
        dialog = QFileDialog()
        dialog.setDefaultSuffix(".wav")
        fpath, _ = dialog.getSaveFileName(
            parent=self,
            caption="Select a path to save the audio file",
            filter="Audio Files (*.flac *.wav *.mp3)",
        )
        if fpath:
            # Default format is wav
            if Path(fpath).suffix == "":
                fpath += ".wav"
            sf.write(fpath, wav, sample_rate)

    def setup_audio_devices(self, sample_rate):
        input_devices = []
        output_devices = []
        for device in sd.query_devices():
            # Check if valid input
            try:
                sd.check_input_settings(device=device["name"], samplerate=sample_rate)
                input_devices.append(device["name"])
            except:
                pass

            # Check if valid output
            try:
                sd.check_output_settings(device=device["name"], samplerate=sample_rate)
                output_devices.append(device["name"])
            except Exception as e:
                # Log a warning only if the device is not an input
                if not device["name"] in input_devices:
                    warn(
                        "Unsupported output device %s for the sample rate: %d \nError: %s"
                        % (device["name"], sample_rate, str(e))
                    )

        if len(input_devices) == 0:
            self.log("No audio input device detected. Recording may not work.")
            self.audio_in_device = None
        else:
            self.audio_in_device = input_devices[0]

        if len(output_devices) == 0:
            self.log("No supported output audio devices were found! Audio output may not work.")
            self.audio_out_devices_cb.addItems(["None"])
            self.audio_out_devices_cb.setDisabled(True)
        else:
            self.audio_out_devices_cb.clear()
            self.audio_out_devices_cb.addItems(output_devices)
            self.audio_out_devices_cb.currentTextChanged.connect(self.set_audio_device)

        self.set_audio_device()

    def set_audio_device(self):

        output_device = self.audio_out_devices_cb.currentText()
        if output_device == "None":
            output_device = None

        # If None, sounddevice queries portaudio
        sd.default.device = (self.audio_in_device, output_device)

    def play(self, wav, sample_rate):
        try:
            sd.stop()
            sd.play(wav, sample_rate)
        except Exception as e:
            print(e)
            self.log("Error in audio playback. Try selecting a different audio output device.")
            self.log("Your device must be connected before you start the toolbox.")

    def stop(self):
        sd.stop()

    def record_one(self, sample_rate, duration):
        self.record_button.setText("Recording...")
        self.record_button.setDisabled(True)

        self.log("Recording %d seconds of audio" % duration)
        sd.stop()
        try:
            wav = sd.rec(duration * sample_rate, sample_rate, 1)
        except Exception as e:
            print(e)
            self.log("Could not record anything. Is your recording device enabled?")
            self.log("Your device must be connected before you start the toolbox.")
            return None

        for i in np.arange(0, duration, 0.1):
            self.set_loading(i, duration)
            sleep(0.1)
        self.set_loading(duration, duration)
        sd.wait()

        self.log("Done recording.")
        self.record_button.setText("Record")
        self.record_button.setDisabled(False)

        return wav.squeeze()

    @property
    def current_dataset_name(self):
        return self.dataset_box.currentText()

    @property
    def current_speaker_name(self):
        return self.speaker_box.currentText()

    @property
    def current_utterance_name(self):
        return self.utterance_box.currentText()

    def browse_file(self):
        fpath = QFileDialog().getOpenFileName(
            parent=self,
            caption="Select an audio file",
            filter="Audio Files (*.mp3 *.flac *.wav *.m4a)",
        )
        return Path(fpath[0]) if fpath[0] != "" else ""

    @staticmethod
    def repopulate_box(box, items, random=False):
        """
        Resets a box and adds a list of items. Pass a list of (item, data) pairs instead to join
        data to the items
        """
        box.blockSignals(True)
        box.clear()
        for item in items:
            item = list(item) if isinstance(item, tuple) else [item]
            box.addItem(str(item[0]), *item[1:])
        if len(items) > 0:
            box.setCurrentIndex(np.random.randint(len(items)) if random else 0)
        box.setDisabled(len(items) == 0)
        box.blockSignals(False)

    def populate_browser(
        self, datasets_root: Path, recognized_datasets: List, level: int, random=True
    ):
        # Select a random dataset
        if level <= 0:
            if datasets_root is not None:
                datasets = [datasets_root.joinpath(d) for d in recognized_datasets]
                datasets = [d.relative_to(datasets_root) for d in datasets if d.exists()]
                self.browser_load_button.setDisabled(len(datasets) == 0)
            if datasets_root is None or len(datasets) == 0:
                msg = "Warning: you d" + (
                    "id not pass a root directory for datasets as argument"
                    if datasets_root is None
                    else "o not have any of the recognized datasets" " in %s" % datasets_root
                )
                self.log(msg)
                msg += (
                    ".\nThe recognized datasets are:\n\t%s\nFeel free to add your own. You "
                    "can still use the toolbox by recording samples yourself."
                    % ("\n\t".join(recognized_datasets))
                )
                print(msg, file=sys.stderr)
                return
            self.repopulate_box(self.dataset_box, datasets, random)

        # Select a random speaker
        if level <= 1:
            speakers_root = datasets_root.joinpath(self.current_dataset_name)
            speaker_names = [d.stem for d in speakers_root.glob("*") if d.is_dir()]
            self.repopulate_box(self.speaker_box, speaker_names, random)

        # Select a random utterance
        if level <= 2:
            utterances_root = datasets_root.joinpath(
                self.current_dataset_name, self.current_speaker_name
            )
            utterances = []
            for extension in ["mp3", "flac", "wav", "m4a"]:
                utterances.extend(Path(utterances_root).glob("**/*.%s" % extension))
            utterances = [fpath.relative_to(utterances_root) for fpath in utterances]
            self.repopulate_box(self.utterance_box, utterances, random)

    def browser_select_next(self):
        index = (self.utterance_box.currentIndex() + 1) % len(self.utterance_box)
        self.utterance_box.setCurrentIndex(index)

    @property
    def current_encoder_fpath(self):
        return self.encoder_box.itemData(self.encoder_box.currentIndex())

    @property
    def current_synthesizer_fpath(self):
        return self.synthesizer_box.itemData(self.synthesizer_box.currentIndex())

    @property
    def current_vocoder_fpath(self):
        return self.vocoder_box.itemData(self.vocoder_box.currentIndex())

    def populate_models(self, models_dir: Path):
        # Encoder
        encoder_fpaths = list(models_dir.glob("*/encoder.pt"))
        if len(encoder_fpaths) == 0:
            raise Exception("No encoder models found in %s" % models_dir)
        self.repopulate_box(self.encoder_box, [(f.parent.name, f) for f in encoder_fpaths])

        # Synthesizer
        synthesizer_fpaths = list(models_dir.glob("*/synthesizer.pt"))
        if len(synthesizer_fpaths) == 0:
            raise Exception("No synthesizer models found in %s" % models_dir)
        self.repopulate_box(self.synthesizer_box, [(f.parent.name, f) for f in synthesizer_fpaths])

        # Vocoder
        vocoder_fpaths = list(models_dir.glob("*/vocoder.pt"))
        vocoder_items = [(f.parent.name, f) for f in vocoder_fpaths] + [("Griffin-Lim", None)]
        self.repopulate_box(self.vocoder_box, vocoder_items)

    @property
    def selected_utterance(self):
        return self.utterance_history.itemData(self.utterance_history.currentIndex())

    def register_utterance(self, utterance: Utterance):
        self.utterance_history.blockSignals(True)
        self.utterance_history.insertItem(0, utterance.name, utterance)
        self.utterance_history.setCurrentIndex(0)
        self.utterance_history.blockSignals(False)

        if len(self.utterance_history) > self.max_saved_utterances:
            self.utterance_history.removeItem(self.max_saved_utterances)

        self.play_button.setDisabled(False)
        self.generate_button.setDisabled(False)
        self.synthesize_button.setDisabled(False)

    def log(self, line, mode="newline"):
        if mode == "newline":
            self.logs.append(line)
            if len(self.logs) > self.max_log_lines:
                del self.logs[0]
        elif mode == "append":
            self.logs[-1] += line
        elif mode == "overwrite":
            self.logs[-1] = line
        log_text = "\n".join(self.logs)

        self.log_window.setText(log_text)
        self.app.processEvents()

    def set_loading(self, value, maximum=1):
        self.loading_bar.setValue(value * 100)
        self.loading_bar.setMaximum(maximum * 100)
        self.loading_bar.setTextVisible(value != 0)
        self.app.processEvents()

    def populate_gen_options(self, seed, trim_silences):
        if seed is not None:
            self.random_seed_checkbox.setChecked(True)
            self.seed_textbox.setText(str(seed))
            self.seed_textbox.setEnabled(True)
        else:
            self.random_seed_checkbox.setChecked(False)
            self.seed_textbox.setText(str(0))
            self.seed_textbox.setEnabled(False)

        if not trim_silences:
            self.trim_silences_checkbox.setChecked(False)
            self.trim_silences_checkbox.setDisabled(True)

    def update_seed_textbox(self):
        if self.random_seed_checkbox.isChecked():
            self.seed_textbox.setEnabled(True)
        else:
            self.seed_textbox.setEnabled(False)

    def reset_interface(self):
        self.set_loading(0)
        self.play_button.setDisabled(True)
        self.generate_button.setDisabled(True)
        self.synthesize_button.setDisabled(True)
        self.vocode_button.setDisabled(True)
        self.replay_wav_button.setDisabled(True)
        self.export_wav_button.setDisabled(True)
        [self.log("") for _ in range(self.max_log_lines)]

    def update_current_spectrogram(self, spectrogram_data):
        # Update the current spectrogram plot
        self.current_ax[0].clear()
        self.current_ax[0].imshow(spectrogram_data, aspect="auto", origin="lower", cmap="viridis")
        self.current_ax[0].set_title("Current Spectrogram")
        self.current_ax[0].set_xlabel("Time")
        self.current_ax[0].set_ylabel("Frequency")
        self.current_fig.canvas.draw()

    def update_generated_spectrogram(self, spectrogram_data):
        # Update the generated spectrogram plot
        self.gen_ax[0].clear()
        self.gen_ax[0].imshow(spectrogram_data, aspect="auto", origin="lower", cmap="viridis")
        self.gen_ax[0].set_title("Generated Spectrogram")
        self.gen_ax[0].set_xlabel("Time")
        self.gen_ax[0].set_ylabel("Frequency")
        self.gen_fig.canvas.draw()

    def __init__(self):
        ## Initialize the application
        self.app = QApplication(sys.argv)
        super().__init__(None)
        self.setWindowTitle("Voice Clone Model")
        self.setStyleSheet("background-color:  #222222;color: white;")

        ## Main layouts
        # Root
        root_layout = QGridLayout()
        self.setLayout(root_layout)

        # Browser
        browser_layout = QGridLayout()
        root_layout.addLayout(browser_layout, 0, 0, 1, 2)

        # Generation
        gen_layout = QVBoxLayout()
        root_layout.addLayout(gen_layout, 0, 2, 1, 2)

        # Visualizations
        vis_layout = QVBoxLayout()
        root_layout.addLayout(vis_layout, 1, 1, 1, 3)

        layout = QVBoxLayout()
        self.current_fig, self.current_ax = plt.subplots(
            1, 2, figsize=(10, 2.25), facecolor="#222222"
        )
        self.gen_fig, self.gen_ax = plt.subplots(1, 2, figsize=(10, 2.25), facecolor="#999999")

        # Add canvases to layout
        layout.addWidget(FigureCanvas(self.current_fig))
        layout.addWidget(FigureCanvas(self.gen_fig))

        self.setLayout(layout)
        i = 0
        button_style = "QPushButton { background-color: orange; color: black; }"
        # Utterance box
        browser_layout.addWidget(QLabel("<b>Dataset Input </b>"), i, 0)
        i += 1
        # Random & next utterance buttons
        self.browser_browse_button = QPushButton("Browse")
        self.browser_browse_button.setStyleSheet(button_style)
        browser_layout.addWidget(self.browser_browse_button, i, 0, 1, 2)
        self.record_button = QPushButton("Record")
        browser_layout.addWidget(self.record_button, i, 2, 1, 2)
        self.play_button = QPushButton("Play")
        browser_layout.addWidget(self.play_button, i + 1, 0, 1, 2)
        self.stop_button = QPushButton("Stop")
        browser_layout.addWidget(self.stop_button, i + 1, 2, 1, 2)
        self.record_button.setStyleSheet(button_style)
        self.play_button.setStyleSheet(button_style)
        self.stop_button.setStyleSheet(button_style)
        i += 2
        browser_layout.addWidget(QLabel("<b>Uploaded Audio file:</b>"), i, 0)
        self.utterance_history = QComboBox()
        browser_layout.addWidget(self.utterance_history, i, 1, 1, 2)

        i += 1

        # Model and audio output selection
        self.encoder_box = QComboBox()
        browser_layout.addWidget(QLabel("<b>Encoder</b>"), i, 0)
        browser_layout.addWidget(self.encoder_box, i + 1, 0)
        self.synthesizer_box = QComboBox()
        browser_layout.addWidget(QLabel("<b>Synthesizer</b>"), i, 1)
        browser_layout.addWidget(self.synthesizer_box, i + 1, 1)
        self.vocoder_box = QComboBox()
        browser_layout.addWidget(QLabel("<b>Vocoder</b>"), i, 2)
        browser_layout.addWidget(self.vocoder_box, i + 1, 2)

        self.audio_out_devices_cb = QComboBox()
        browser_layout.addWidget(QLabel("<b>Audio Output</b>"), i, 3)
        browser_layout.addWidget(self.audio_out_devices_cb, i + 1, 3)
        i += 1

        # Replay & Save Audio
        browser_layout.addWidget(QLabel("<b>Clone Voice:</b>"), i, 0)
        self.waves_cb = QComboBox()
        self.waves_cb_model = QStringListModel()
        self.waves_cb.setModel(self.waves_cb_model)
        self.waves_cb.setToolTip(
            "Select one of the last generated waves in this section for replaying or exporting"
        )
        browser_layout.addWidget(self.waves_cb, i, 1)
        self.replay_wav_button = QPushButton("Replay")
        self.replay_wav_button.setStyleSheet(button_style)
        self.replay_wav_button.setToolTip("Replay last generated vocoder")
        browser_layout.addWidget(self.replay_wav_button, i + 4, 0, 1, 4)
        self.export_wav_button = QPushButton("Export")
        self.export_wav_button.setStyleSheet(button_style)
        self.export_wav_button.setToolTip(
            "Save last generated vocoder audio in filesystem as a wav file"
        )
        browser_layout.addWidget(self.export_wav_button, i + 5, 0, 1, 4)

        self.image_label = QLabel(self)
        pixmap = QPixmap("Daco_4143941.png")
        scaled_pixmap = pixmap.scaled(1500, 200)
        self.image_label.setPixmap(scaled_pixmap)
        self.layout().addWidget(self.image_label, i, 0, 1, 4)

        ## Generation
        self.text_prompt = QPlainTextEdit(default_text)
        gen_layout.addWidget(self.text_prompt, stretch=1)

        self.generate_button = QPushButton("Synthesize and vocode")
        self.generate_button.setStyleSheet(button_style)
        gen_layout.addWidget(self.generate_button)

        layout = QHBoxLayout()
        self.synthesize_button = QPushButton("Synthesize only")
        self.synthesize_button.setStyleSheet(button_style)
        layout.addWidget(self.synthesize_button)
        self.vocode_button = QPushButton("Vocode only")
        self.vocode_button.setStyleSheet(button_style)
        layout.addWidget(self.vocode_button)
        gen_layout.addLayout(layout)

        layout_seed = QGridLayout()
        self.random_seed_checkbox = QCheckBox("Random seed:")
        self.random_seed_checkbox.setToolTip(
            "When checked, makes the synthesizer and vocoder deterministic."
        )
        layout_seed.addWidget(self.random_seed_checkbox, 0, 0)
        self.seed_textbox = QLineEdit()
        self.seed_textbox.setMaximumWidth(80)
        layout_seed.addWidget(self.seed_textbox, 0, 1)
        self.trim_silences_checkbox = QCheckBox("Enhance vocoder output")
        self.trim_silences_checkbox.setToolTip(
            "When checked, trims excess silence in vocoder output."
            " This feature requires `webrtcvad` to be installed."
        )
        layout_seed.addWidget(self.trim_silences_checkbox, 0, 2, 1, 2)
        gen_layout.addLayout(layout_seed)

        self.loading_bar = QProgressBar()
        gen_layout.addWidget(self.loading_bar)

        self.log_window = QLabel()
        self.log_window.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
        gen_layout.addWidget(self.log_window)
        self.logs = []
        gen_layout.addStretch()

        ## Set the size of the window and of the elements
        max_size = QDesktopWidget().availableGeometry(self).size() * 0.8
        self.resize(max_size)

        ## Finalize the display
        self.reset_interface()
        self.show()

    def start(self):
        self.app.exec_()
