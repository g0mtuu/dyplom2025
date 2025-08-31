import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy.linalg import solve_toeplitz
import sounddevice as sd
import threading
from datetime import datetime
import os

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class WavAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analizator WAV") # ZMIENIONA NAZWA
        self.root.geometry("1400x900")

        # Zmienne audio
        self.samplerate, self.audio_data, self.normalized_audio_data, self.filepath, self.duration_ms = [None] * 5
        self.result_data = None
        self.convolution_ir_data = None
        self.convolution_ir_path = None
        self.convolution_ir_samplerate = None

        # Zmienne Tkinter
        self.zoom_ms_var = tk.DoubleVar(value=20.0)
        self.offset_coarse_ms_var = tk.DoubleVar(value=0.0)
        self.offset_fine_ms_var = tk.DoubleVar(value=0.0)
        self.invert_phase_vars = [tk.BooleanVar(value=False) for _ in range(3)]
        self.phase_shift_vars = [tk.DoubleVar(value=0.0) for _ in range(3)]
        self.analysis_mode_var = tk.StringVar(value="Odejmowanie A-B")
        self.analysis_ch_A_var = tk.IntVar(value=1)
        self.analysis_ch_B_var = tk.IntVar(value=2)
        self.convolution_source_channel_var = tk.IntVar(value=1)

        # Zmienne do analizy spektralnej
        self.spectrogram_ch_A_var = tk.IntVar(value=1)
        self.spectrogram_ch_B_var = tk.IntVar(value=2)
        self.spec_f_min_var = tk.StringVar(value="100")
        self.spec_f_max_var = tk.StringVar(value="2000")
        self.analysis_start_ms_var = tk.DoubleVar(value=0.0)
        self.analysis_end_ms_var = tk.DoubleVar(value=0.0)

        # Inne zmienne stanu
        self.is_playing = False
        self.play_buttons = []

        self.create_widgets()
        self.setup_plot()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill="both", expand=True)
        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1, minsize=350)

        self.plot_frame = ttk.Frame(main_frame)
        self.plot_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        controls_panel = ttk.Frame(main_frame)
        controls_panel.grid(row=0, column=1, sticky="nsew")

        open_file_frame = ttk.Frame(controls_panel)
        open_file_frame.pack(fill='x', pady=(0, 5))
        open_file_frame.columnconfigure(1, weight=1)
        open_button = ttk.Button(open_file_frame, text="Wybierz plik WAV...", command=self.select_file)
        open_button.pack(side='left', padx=(0,10))
        self.filepath_label = ttk.Label(open_file_frame, text="Nie wybrano pliku", anchor="w", relief="sunken", padding=5)
        self.filepath_label.pack(fill='x', expand=True)

        combined_controls_frame = ttk.LabelFrame(controls_panel, text="Nawigacja i Kontrola", padding="10")
        combined_controls_frame.pack(fill="x", pady=5)
        combined_controls_frame.columnconfigure(1, weight=1)
        
        ttk.Label(combined_controls_frame, text="Długość (ms):").grid(row=0, column=0, sticky="w")
        self.zoom_slider = ttk.Scale(combined_controls_frame, from_=1, to=100, variable=self.zoom_ms_var, orient='horizontal', command=self.redraw_plots)
        self.zoom_slider.grid(row=0, column=1, sticky="ew", padx=5)
        zoom_entry = ttk.Entry(combined_controls_frame, textvariable=self.zoom_ms_var, width=8)
        zoom_entry.grid(row=0, column=2)
        zoom_entry.bind("<Return>", self.redraw_plots)
        ttk.Label(combined_controls_frame, text="Offset zgrubnie (ms):").grid(row=1, column=0, sticky="w")
        self.offset_coarse_slider = ttk.Scale(combined_controls_frame, from_=0, to=1000, variable=self.offset_coarse_ms_var, orient='horizontal', command=self.redraw_plots)
        self.offset_coarse_slider.grid(row=1, column=1, sticky="ew", padx=5)
        offset_coarse_entry = ttk.Entry(combined_controls_frame, textvariable=self.offset_coarse_ms_var, width=8)
        offset_coarse_entry.grid(row=1, column=2)
        offset_coarse_entry.bind("<Return>", self.redraw_plots)
        ttk.Label(combined_controls_frame, text="Offset precyzyjnie (ms):").grid(row=2, column=0, sticky="w")
        self.offset_fine_slider = ttk.Scale(combined_controls_frame, from_=0, to=1000, variable=self.offset_fine_ms_var, orient='horizontal', command=self.redraw_plots)
        self.offset_fine_slider.grid(row=2, column=1, sticky="ew", padx=5)
        offset_fine_entry = ttk.Entry(combined_controls_frame, textvariable=self.offset_fine_ms_var, width=8)
        offset_fine_entry.grid(row=2, column=2)
        offset_fine_entry.bind("<Return>", self.redraw_plots)
        
        ttk.Separator(combined_controls_frame, orient='horizontal').grid(row=3, column=0, columnspan=3, sticky='ew', pady=10)
        channel_ctrl_container = ttk.Frame(combined_controls_frame)
        channel_ctrl_container.grid(row=4, column=0, columnspan=3, sticky='ew')
        channel_ctrl_container.columnconfigure((0, 1, 2), weight=1)
        for i in range(3):
            ch_frame = ttk.Frame(channel_ctrl_container, padding=5)
            ch_frame.grid(row=0, column=i, sticky="ew")
            ch_frame.columnconfigure(0, weight=1)
            ttk.Label(ch_frame, text=f"Kanał {i+1}", font="-weight bold").grid(row=0, column=0, columnspan=2)
            invert_check = ttk.Checkbutton(ch_frame, text="Odwróć fazę", variable=self.invert_phase_vars[i], command=self.redraw_plots)
            invert_check.grid(row=1, column=0, columnspan=2, sticky='w')
            ttk.Label(ch_frame, text="Przesunięcie (ms):").grid(row=2, column=0, columnspan=2, sticky='w', pady=(5,0))
            shift_slider = ttk.Scale(ch_frame, from_=-5.0, to=5.0, variable=self.phase_shift_vars[i], orient='horizontal', command=self.redraw_plots)
            shift_slider.grid(row=3, column=0, sticky='ew')
            shift_entry = ttk.Entry(ch_frame, textvariable=self.phase_shift_vars[i], width=6)
            shift_entry.grid(row=3, column=1, padx=5)
            shift_entry.bind("<Return>", self.redraw_plots)

        convolution_frame = ttk.LabelFrame(controls_panel, text="Splot (konwolucja)", padding="10")
        convolution_frame.pack(fill="x", pady=5)
        convolution_frame.columnconfigure(1, weight=1)
        ir_button = ttk.Button(convolution_frame, text="Wybierz plik IR (.wav)", command=self.select_ir_file)
        ir_button.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        self.ir_path_label = ttk.Label(convolution_frame, text="Nie wybrano filtru", anchor='w', relief='sunken')
        self.ir_path_label.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        ttk.Label(convolution_frame, text="Kanał źródłowy:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.convolution_channel_cb = ttk.Combobox(convolution_frame, textvariable=self.convolution_source_channel_var, state="readonly", width=5)
        self.convolution_channel_cb.grid(row=1, column=1, sticky='w', padx=5, pady=5)
        apply_conv_button = ttk.Button(convolution_frame, text="Zastosuj splot", command=self.apply_convolution)
        apply_conv_button.grid(row=2, column=0, columnspan=2, sticky='ew', padx=5, pady=5)

        analysis_frame = ttk.LabelFrame(controls_panel, text="Analiza Różnicowa i Odtwarzanie", padding="10")
        analysis_frame.pack(fill="x", pady=5)
        analysis_frame.columnconfigure(1, weight=1)
        ttk.Label(analysis_frame, text="Tryb:").grid(row=0, column=0, sticky="w", padx=5)
        mode_cb = ttk.Combobox(analysis_frame, textvariable=self.analysis_mode_var, values=["Odejmowanie A-B", "Dekonwolucja iFFT(FFT[A]/FFT[B])"], state="readonly")
        mode_cb.grid(row=0, column=1, columnspan=3, sticky="ew", padx=5)
        ttk.Label(analysis_frame, text="Kanał A:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.ch_A_cb = ttk.Combobox(analysis_frame, textvariable=self.analysis_ch_A_var, state="readonly", width=5)
        self.ch_A_cb.grid(row=1, column=1, sticky="w", padx=5)
        ttk.Label(analysis_frame, text="Kanał B:").grid(row=1, column=2, sticky="e", padx=5)
        self.ch_B_cb = ttk.Combobox(analysis_frame, textvariable=self.analysis_ch_B_var, state="readonly", width=5)
        self.ch_B_cb.grid(row=1, column=3, sticky="w", padx=5)
        calc_button = ttk.Button(analysis_frame, text="Oblicz i pokaż wynik", command=self.recalculate_analysis)
        calc_button.grid(row=2, column=0, columnspan=4, sticky="ew", padx=5, pady=5)
        
        playback_frame = ttk.Frame(analysis_frame)
        playback_frame.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(10,0))
        playback_frame.columnconfigure((0,1,2,3,4), weight=1)
        self.play_ch1_btn = ttk.Button(playback_frame, text="Odtw. K1", command=lambda: self.play_audio(0))
        self.play_ch2_btn = ttk.Button(playback_frame, text="Odtw. K2", command=lambda: self.play_audio(1))
        self.play_ch3_btn = ttk.Button(playback_frame, text="Odtw. K3", command=lambda: self.play_audio(2))
        self.play_res_btn = ttk.Button(playback_frame, text="Odtw. Wynik", command=lambda: self.play_audio(3))
        self.stop_btn = ttk.Button(playback_frame, text="⏹ Stop", command=self.stop_audio, state="disabled")
        self.play_ch1_btn.grid(row=0, column=0, sticky='ew'); self.play_ch2_btn.grid(row=0, column=1, sticky='ew')
        self.play_ch3_btn.grid(row=0, column=2, sticky='ew'); self.play_res_btn.grid(row=0, column=3, sticky='ew')
        self.stop_btn.grid(row=0, column=4, sticky='ew')
        self.play_buttons = [self.play_ch1_btn, self.play_ch2_btn, self.play_ch3_btn, self.play_res_btn]

        save_frame = ttk.Frame(analysis_frame)
        save_frame.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(10,0))
        save_frame.columnconfigure((0,1), weight=1)
        save_4ch_button = ttk.Button(save_frame, text="Zapisz wynik (4-kanałowy WAV)", command=self.save_result_file_4ch)
        save_4ch_button.grid(row=0, column=0, sticky="ew", padx=(0,5))
        save_1ch_button = ttk.Button(save_frame, text="Zapisz tylko wynik (Mono WAV)", command=self.save_result_file_1ch)
        save_1ch_button.grid(row=0, column=1, sticky="ew", padx=(5,0))

        spectral_frame = ttk.LabelFrame(controls_panel, text="Analiza Spektralna", padding="10")
        spectral_frame.pack(fill="x", pady=5)
        spectral_frame.columnconfigure((1, 3), weight=1)
        
        ttk.Label(spectral_frame, text="Kanał A:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.spec_ch_A_cb = ttk.Combobox(spectral_frame, textvariable=self.spectrogram_ch_A_var, state="readonly", width=5)
        self.spec_ch_A_cb.grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Label(spectral_frame, text="Kanał B:").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        self.spec_ch_B_cb = ttk.Combobox(spectral_frame, textvariable=self.spectrogram_ch_B_var, state="readonly", width=5)
        self.spec_ch_B_cb.grid(row=0, column=3, sticky="ew", padx=5)
        
        ttk.Label(spectral_frame, text="Min Freq (Hz):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        spec_f_min_entry = ttk.Entry(spectral_frame, textvariable=self.spec_f_min_var, width=8)
        spec_f_min_entry.grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Label(spectral_frame, text="Max Freq (Hz):").grid(row=1, column=2, sticky="w", padx=5, pady=2)
        spec_f_max_entry = ttk.Entry(spectral_frame, textvariable=self.spec_f_max_var, width=8)
        spec_f_max_entry.grid(row=1, column=3, sticky="ew", padx=5)
        
        ttk.Label(spectral_frame, text="Początek (ms):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.analysis_start_slider = ttk.Scale(spectral_frame, from_=0, to=1000, variable=self.analysis_start_ms_var, orient='horizontal')
        self.analysis_start_slider.grid(row=2, column=1, columnspan=2, sticky="ew", padx=5)
        ttk.Entry(spectral_frame, textvariable=self.analysis_start_ms_var, width=8).grid(row=2, column=3)
        
        ttk.Label(spectral_frame, text="Koniec (ms):").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.analysis_end_slider = ttk.Scale(spectral_frame, from_=0, to=1000, variable=self.analysis_end_ms_var, orient='horizontal')
        self.analysis_end_slider.grid(row=3, column=1, columnspan=2, sticky="ew", padx=5)
        ttk.Entry(spectral_frame, textvariable=self.analysis_end_ms_var, width=8).grid(row=3, column=3)
        
        buttons_frame = ttk.Frame(spectral_frame)
        buttons_frame.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        buttons_frame.columnconfigure((0, 1, 2), weight=1)

        show_spec_button = ttk.Button(buttons_frame, text="Pokaż Spektrogramy", command=self.show_spectrogram_window)
        show_spec_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        show_fft_button = ttk.Button(buttons_frame, text="Pokaż FFT", command=self.show_fft_window)
        show_fft_button.grid(row=0, column=1, sticky="ew", padx=5)

        show_f0_button = ttk.Button(buttons_frame, text="Pokaż Statystyki Głosu", command=self.show_voice_stats)
        show_f0_button.grid(row=0, column=2, sticky="ew", padx=(5, 0))

        self.status_label = ttk.Label(controls_panel, text="Gotowy", anchor="w", relief="sunken", padding=5)
        self.status_label.pack(side='bottom', fill='x', pady=(10,0))

    def setup_plot(self):
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.axs = self.fig.subplots(4, 1, sharex=True, sharey=True)
        self.fig.set_constrained_layout(True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.initialize_axes()

    def initialize_axes(self):
        labels = ["Kanał 1", "Kanał 2", "Kanał 3", "Wynik"]
        for i, ax in enumerate(self.axs):
            ax.clear()
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_ylabel(labels[i], rotation=0, ha='right', va='center', labelpad=25)
            ax.tick_params(axis='x', labelbottom=True)
        self.axs[-1].set_xlabel("Czas (s)")
        self.canvas.draw()

    def select_file(self):
        filepath = filedialog.askopenfilename(title="Wybierz plik audio", filetypes=[("Pliki WAV", "*.wav")])
        if not filepath: return
        self.filepath = filepath; self.filepath_label.config(text=self.filepath)
        self.load_and_plot_wav()

    def load_and_plot_wav(self):
        if not self.filepath: return
        try: self.samplerate, self.audio_data = wavfile.read(self.filepath)
        except Exception as e: messagebox.showerror("Błąd odczytu", f"Nie można wczytać pliku:\n{e}"); return
        if self.audio_data.size == 0: messagebox.showwarning("Pusty plik", "Plik audio nie zawiera danych."); return

        if np.issubdtype(self.audio_data.dtype, np.integer):
            max_val = np.iinfo(self.audio_data.dtype).max
            self.normalized_audio_data = self.audio_data.astype(np.float32) / max_val
        else: self.normalized_audio_data = self.audio_data

        if self.normalized_audio_data.ndim == 1:
             self.normalized_audio_data = np.expand_dims(self.normalized_audio_data, axis=1)

        num_samples, num_channels = self.normalized_audio_data.shape
        self.duration_ms = (num_samples / self.samplerate) * 1000
        self.status_label.config(text=f"Próbkowanie: {self.samplerate} Hz | Kanały: {num_channels} | Długość: {self.duration_ms/1000:.2f} s")
        
        ch_list = [str(i+1) for i in range(num_channels)]
        self.ch_A_cb['values'] = ch_list
        self.ch_B_cb['values'] = ch_list
        self.convolution_channel_cb['values'] = ch_list
        self.spec_ch_A_cb['values'] = ch_list
        self.spec_ch_B_cb['values'] = ch_list
        if num_channels > 0: 
            self.analysis_ch_A_var.set(1); self.convolution_source_channel_var.set(1); self.spectrogram_ch_A_var.set(1)
        if num_channels > 1: 
            self.analysis_ch_B_var.set(2); self.spectrogram_ch_B_var.set(2)
        else:
            self.analysis_ch_B_var.set(1); self.spectrogram_ch_B_var.set(1)
        
        default_zoom = 20.0; max_zoom = self.duration_ms
        default_offset = self.duration_ms / 2.0
        for var in [self.zoom_ms_var, self.offset_coarse_ms_var, self.offset_fine_ms_var]:
            if var.trace_info(): var.trace_remove("write", var.trace_info()[0][1])
        self.zoom_slider.config(to=max_zoom)
        self.zoom_ms_var.set(default_zoom if self.duration_ms > default_zoom else self.duration_ms)
        self.offset_coarse_slider.config(to=self.duration_ms)
        self.offset_coarse_ms_var.set(default_offset)
        self.offset_fine_slider.config(to=default_zoom)
        self.offset_fine_ms_var.set(0)
        for var in [self.zoom_ms_var, self.offset_coarse_ms_var, self.offset_fine_ms_var]:
            var.trace_add("write", self.redraw_plots)

        self.analysis_start_slider.config(to=self.duration_ms)
        self.analysis_end_slider.config(to=self.duration_ms)
        self.analysis_start_ms_var.set(0.0)
        self.analysis_end_ms_var.set(self.duration_ms)

        self.result_data = None
        self.redraw_plots()
    
    def get_modified_channel_data(self, index):
        if self.normalized_audio_data is None or not (0 <= index < self.normalized_audio_data.shape[1]): return None
        channel_data = self.normalized_audio_data[:, index].copy()
        if self.invert_phase_vars[index].get(): channel_data *= -1
        time_shift_ms = self.phase_shift_vars[index].get()
        if time_shift_ms != 0:
            shift_samples = int((time_shift_ms / 1000.0) * self.samplerate)
            channel_data = np.roll(channel_data, shift_samples)
        return channel_data

    def recalculate_analysis(self):
        if self.normalized_audio_data is None: return
        ch_A_idx, ch_B_idx = self.analysis_ch_A_var.get() - 1, self.analysis_ch_B_var.get() - 1
        ch_A, ch_B = self.get_modified_channel_data(ch_A_idx), self.get_modified_channel_data(ch_B_idx)
        if ch_A is None or ch_B is None: messagebox.showerror("Błąd", "Wybrano nieprawidłowe kanały."); return
        mode = self.analysis_mode_var.get()
        try:
            if mode == "Odejmowanie A-B": self.result_data = ch_A - ch_B
            elif mode == "Dekonwolucja iFFT(FFT[A]/FFT[B])":
                epsilon = 1e-12
                fft_A, fft_B = np.fft.fft(ch_A), np.fft.fft(ch_B)
                self.result_data = np.real(np.fft.ifft(fft_A / (fft_B + epsilon)))
            if self.result_data is not None:
                max_abs = np.max(np.abs(self.result_data))
                if max_abs > 0: self.result_data /= max_abs
        except Exception as e: messagebox.showerror("Błąd Obliczeń", f"Wystąpił błąd:\n{e}"); self.result_data = None
        self.redraw_plots()
    
    def select_ir_file(self):
        filepath = filedialog.askopenfilename(title="Wybierz plik odpowiedzi impulsowej (IR)", filetypes=[("Pliki WAV", "*.wav")])
        if not filepath: return
        try:
            self.convolution_ir_samplerate, ir_data_raw = wavfile.read(filepath)
            if self.samplerate and self.convolution_ir_samplerate != self.samplerate:
                messagebox.showwarning("Niezgodne próbkowanie", f"Plik IR ma inne próbkowanie ({self.convolution_ir_samplerate} Hz) niż plik główny ({self.samplerate} Hz). Wynik może być nieprawidłowy.")
            
            if np.issubdtype(ir_data_raw.dtype, np.integer):
                max_val = np.iinfo(ir_data_raw.dtype).max
                ir_data = ir_data_raw.astype(np.float32) / max_val
            else: ir_data = ir_data_raw
            if ir_data.ndim > 1: ir_data = ir_data.mean(axis=1)
            self.convolution_ir_data = ir_data
            self.ir_path_label.config(text=os.path.basename(filepath))
        except Exception as e: messagebox.showerror("Błąd odczytu pliku IR", f"Nie można wczytać pliku:\n{e}")

    def apply_convolution(self):
        if self.normalized_audio_data is None: messagebox.showwarning("Brak danych", "Najpierw wczytaj główny plik audio."); return
        if self.convolution_ir_data is None: messagebox.showwarning("Brak danych", "Najpierw wczytaj plik z odpowiedzią impulsową (IR)."); return
        source_ch_idx = self.convolution_source_channel_var.get() - 1
        source_data = self.get_modified_channel_data(source_ch_idx)
        if source_data is None: messagebox.showerror("Błąd", "Wybrano nieprawidłowy kanał źródłowy."); return
        
        try:
            convolved_signal = signal.convolve(source_data, self.convolution_ir_data, mode='full', method='fft')
            max_abs = np.max(np.abs(convolved_signal))
            if max_abs > 0: convolved_signal /= max_abs
            self.result_data = convolved_signal
            self.redraw_plots()
        except Exception as e: messagebox.showerror("Błąd konwolucji", f"Wystąpił błąd podczas obliczeń:\n{e}")

    def _get_spectral_analysis_params(self):
        if self.normalized_audio_data is None:
            messagebox.showwarning("Brak danych", "Najpierw wczytaj plik audio.")
            return None
        try:
            ch_A_idx = self.spectrogram_ch_A_var.get() - 1
            ch_B_idx = self.spectrogram_ch_B_var.get() - 1
            f_min = float(self.spec_f_min_var.get())
            f_max = float(self.spec_f_max_var.get())
            start_ms = self.analysis_start_ms_var.get()
            end_ms = self.analysis_end_ms_var.get()

            if f_min >= f_max or start_ms >= end_ms:
                messagebox.showerror("Błąd wartości", "Wartości minimalne muszą być mniejsze od maksymalnych.")
                return None
            
            start_sample = int(start_ms / 1000 * self.samplerate)
            end_sample = int(end_ms / 1000 * self.samplerate)

            ch_A_data = self.normalized_audio_data[start_sample:end_sample, ch_A_idx]
            ch_B_data = self.normalized_audio_data[start_sample:end_sample, ch_B_idx]
            
            return ch_A_idx, ch_B_idx, ch_A_data, ch_B_data, f_min, f_max, start_sample, end_sample

        except (ValueError, tk.TclError):
            messagebox.showerror("Błąd wartości", "Wprowadź poprawne wartości liczbowe.")
            return None
        except IndexError:
            messagebox.showerror("Błąd kanału", "Wybrano nieprawidłowy numer kanału. Sprawdź, czy plik audio ma wystarczającą liczbę kanałów.")
            return None

    def show_spectrogram_window(self):
        params = self._get_spectral_analysis_params()
        if params is None: return
        ch_A_idx, ch_B_idx, ch_A_data, ch_B_data, f_min, f_max, _, _ = params

        spec_window = tk.Toplevel(self.root)
        spec_window.title(f"Analiza Spektrogramu (Kanały {ch_A_idx+1} i {ch_B_idx+1})")
        spec_window.geometry("1200x700")

        fig = Figure(figsize=(12, 6), dpi=100)
        ax1, ax2 = fig.subplots(1, 2, sharey=True)
        
        def plot_single_spectrogram(ax, data, title):
            f, t, Sxx = signal.spectrogram(data, self.samplerate, nperseg=1024)
            mesh = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-9), shading='gouraud', rasterized=True)
            ax.set_title(title)
            ax.set_ylabel('Częstotliwość [Hz]')
            ax.set_xlabel('Czas [s]')
            return mesh

        plot_single_spectrogram(ax1, ch_A_data, f'Kanał {ch_A_idx+1}')
        mesh2 = plot_single_spectrogram(ax2, ch_B_data, f'Kanał {ch_B_idx+1}')
        
        ax1.set_ylim(f_min, f_max)
        fig.colorbar(mesh2, ax=ax2, format='%+2.0f dB', label='Intensywność (dB)')
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=spec_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, spec_window)
        toolbar.update()
        canvas.get_tk_widget().pack()

    def show_fft_window(self):
        params = self._get_spectral_analysis_params()
        if params is None: return
        ch_A_idx, ch_B_idx, ch_A_data, ch_B_data, _, _, _, _ = params
        
        if len(ch_A_data) == 0:
            messagebox.showwarning("Brak danych", "Wybrany fragment nie zawiera próbek audio.")
            return

        fft_window = tk.Toplevel(self.root)
        fft_window.title(f"Analiza FFT (Kanały {ch_A_idx+1} i {ch_B_idx+1})")
        fft_window.geometry("1000x700")
        
        N = len(ch_A_data)
        fft_A = np.fft.fft(ch_A_data)
        fft_B = np.fft.fft(ch_B_data)
        
        yf_A = 2.0/N * np.abs(fft_A[0:N//2])
        yf_B = 2.0/N * np.abs(fft_B[0:N//2])
        xf = np.fft.fftfreq(N, 1 / self.samplerate)[:N//2]
        
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)

        ax.plot(xf, yf_A, color='orange', label=f'Kanał {ch_A_idx+1}')
        ax.plot(xf, yf_B, color='green', label=f'Kanał {ch_B_idx+1}')
        
        ax.set_title('Analiza Częstotliwości FFT')
        ax.set_xlabel('Częstotliwość [Hz]')
        ax.set_ylabel('Znormalizowana Amplituda')
        ax.set_xscale('log')
        ax.grid(True, which="both", ls="--")
        ax.legend()
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=fft_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, fft_window)
        toolbar.update()
        canvas.get_tk_widget().pack()

    def calculate_f0_autocorr(self, frame, rate):
        if np.sum(np.abs(frame)) == 0: return 0, None
        corr = signal.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]
        energy = corr[0]
        if energy == 0: return 0, None
        normalized_corr = corr / energy
        peaks, _ = signal.find_peaks(normalized_corr)
        min_period_samples = int(rate / 500)
        max_period_samples = int(rate / 75)
        valid_peaks = [p for p in peaks if min_period_samples < p < max_period_samples]
        if not valid_peaks: return 0, normalized_corr
        strongest_peak = max(valid_peaks, key=lambda p: normalized_corr[p])
        f0 = rate / strongest_peak
        return f0, normalized_corr

    def _calculate_hnr(self, normalized_corr, f0, rate):
        if f0 == 0 or normalized_corr is None: return 0
        lag = int(rate / f0)
        if lag >= len(normalized_corr): return 0
        r_T0 = normalized_corr[lag]
        if r_T0 >= 1.0 or r_T0 <= 0: return 0
        hnr = 10 * np.log10(r_T0 / (1 - r_T0))
        return hnr

    def _calculate_formants_lpc(self, frame, rate, num_formants=7):
        order = 2 + int(rate / 1000)
        if len(frame) < order: return [0] * num_formants
        
        frame_emph = np.append(frame[0], frame[1:] - 0.97 * frame[:-1])
        frame_win = frame_emph * np.hamming(len(frame_emph))
        
        r = np.correlate(frame_win, frame_win, mode='full')[len(frame_win)-1:]
        if r[0] == 0: return [0] * num_formants
        
        try:
            a = solve_toeplitz((r[:order], r[:order]), -r[1:order+1])
        except np.linalg.LinAlgError:
            return [0] * num_formants

        roots = np.roots(np.concatenate(([1], a)))
        roots = [r for r in roots if np.imag(r) >= 0]
        
        freqs = sorted([np.arctan2(np.imag(r), np.real(r)) * (rate / (2 * np.pi)) for r in roots])
        freqs = [f for f in freqs if f > 90]
        
        formants = freqs[:num_formants]
        while len(formants) < num_formants:
            formants.append(0)
        return formants

    def _calculate_harmonic_amplitudes(self, frame_fft, f0, rate):
        if f0 == 0: return [0, 0, 0]
        n_fft = len(frame_fft)
        freq_res = rate / (2 * n_fft)
        amplitudes_db = []
        for i in range(1, 4): # H1, H2, H3
            harmonic_freq = i * f0
            if harmonic_freq > rate / 2:
                amplitudes_db.append(0)
                continue
            
            target_bin = int(harmonic_freq / freq_res)
            if target_bin >= n_fft:
                amplitudes_db.append(0)
                continue
            
            amp = np.abs(frame_fft[target_bin])
            amplitudes_db.append(20 * np.log10(amp + 1e-9))
        return amplitudes_db

    def show_voice_stats(self):
        # Ta funkcja nie używa już parametrów ch_A_idx, ch_B_idx z GUI, analizuje kanały 1-3.
        # Pobieramy jednak inne parametry, jak zakres czasu analizy.
        params = self._get_spectral_analysis_params()
        if params is None: return
        
        _, _, _, _, _, _, start_sample, end_sample = params
        num_channels = self.normalized_audio_data.shape[1]

        if (end_sample - start_sample) <= 0:
            messagebox.showwarning("Brak danych", "Wybrany fragment nie zawiera próbek audio.")
            return

        self.status_label.config(text="Obliczanie statystyk głosu dla 3 kanałów...")
        self.root.update_idletasks()

        def _analyze_channel(channel_data):
            frame_size_ms, hop_size_ms = 40, 10
            frame_len = int(self.samplerate * frame_size_ms / 1000)
            hop_len = int(self.samplerate * hop_size_ms / 1000)

            keys = ["f0", "hnr", "amp", "h1", "h2", "h3"] + [f"f{i+1}" for i in range(7)]
            lists = {k: [] for k in keys}
            
            for i in range(0, len(channel_data) - frame_len, hop_len):
                frame = channel_data[i : i + frame_len]
                f0, norm_corr = self.calculate_f0_autocorr(frame, self.samplerate)
                if f0 > 0:
                    lists["f0"].append(f0)
                    lists["hnr"].append(self._calculate_hnr(norm_corr, f0, self.samplerate))
                    lists["amp"].append(np.sqrt(np.mean(frame**2))) # RMS Amplitude
                    
                    formants = self._calculate_formants_lpc(frame, self.samplerate, num_formants=7)
                    for fi in range(7): lists[f"f{fi+1}"].append(formants[fi])
                    
                    frame_fft = np.fft.fft(frame * np.hanning(len(frame)))
                    harmonics = self._calculate_harmonic_amplitudes(frame_fft[:frame_len//2], f0, self.samplerate)
                    lists["h1"].append(harmonics[0]); lists["h2"].append(harmonics[1]); lists["h3"].append(harmonics[2])
            
            if not lists["f0"]: return None
            
            stats = {"voiced_frames": len(lists["f0"])}
            for key, values in lists.items():
                if key not in ["f0", "amp"]:
                    if values:
                        arr = np.array(values)
                        arr = arr[arr > 0]
                        stats[f"{key}_mean"] = np.mean(arr) if len(arr) > 0 else 0
            
            f0_arr = np.array(lists["f0"])
            stats["f0_mean"] = np.mean(f0_arr); stats["f0_std"] = np.std(f0_arr)
            stats["f0_median"] = np.median(f0_arr); stats["f0_min"] = np.min(f0_arr)
            stats["f0_max"] = np.max(f0_arr)
            
            if len(lists["f0"]) > 1:
                periods = 1 / f0_arr
                stats["jitter_percent"] = 100 * np.mean(np.abs(np.diff(periods))) / np.mean(periods)
                
                amp_arr = np.array(lists["amp"])
                amp_arr_prev = amp_arr[:-1]
                amp_arr_curr = amp_arr[1:]
                valid_indices = (amp_arr_prev > 1e-9) & (amp_arr_curr > 1e-9)
                if np.any(valid_indices):
                    stats["shimmer_db"] = np.mean(np.abs(20 * np.log10(amp_arr_curr[valid_indices] / amp_arr_prev[valid_indices])))
                else:
                    stats["shimmer_db"] = 0
            return stats

        all_stats = []
        for i in range(3):
            if i < num_channels:
                channel_data_slice = self.normalized_audio_data[start_sample:end_sample, i]
                stats = _analyze_channel(channel_data_slice)
                all_stats.append(stats)
            else:
                all_stats.append(None) # Dodaj None, jeśli kanał nie istnieje

        result_window = tk.Toplevel(self.root)
        result_window.title("Wyniki analizy głosu dla kanałów 1-3")
        result_window.geometry("800x600")
        text_widget = tk.Text(result_window, wrap='none', font=("Courier New", 10))
        text_widget.pack(pady=10, padx=10, fill="both", expand=True)

        header = f"Analiza dla fragmentu: {self.analysis_start_ms_var.get():.1f}-{self.analysis_end_ms_var.get():.1f} ms\n\n"
        text_widget.insert("end", header)

        def format_line(param, key):
            stats_ch1, stats_ch2, stats_ch3 = all_stats
            val_1 = f"{stats_ch1.get(key, 0):.2f}" if stats_ch1 else "b.d."
            val_2 = f"{stats_ch2.get(key, 0):.2f}" if stats_ch2 else "b.d."
            val_3 = f"{stats_ch3.get(key, 0):.2f}" if stats_ch3 else "b.d."
            return f"{param:<24} | {val_1:^16} | {val_2:^16} | {val_3:^16}\n"

        table = f"{'Parametr':<24} | {'Kanał 1':^16} | {'Kanał 2':^16} | {'Kanał 3':^16}\n"
        table += f"{'─'*24}┼{'─'*18}┼{'─'*18}┼{'─'*18}\n"
        
        table += format_line("Średnia F0 [Hz]", "f0_mean")
        table += format_line("Odch. stand. F0 [Hz]", "f0_std")
        table += format_line("Mediana F0 [Hz]", "f0_median")
        
        min_max_1_str = f"{all_stats[0].get('f0_min', 0):.1f}/{all_stats[0].get('f0_max', 0):.1f}" if all_stats[0] else "b.d."
        min_max_2_str = f"{all_stats[1].get('f0_min', 0):.1f}/{all_stats[1].get('f0_max', 0):.1f}" if all_stats[1] else "b.d."
        min_max_3_str = f"{all_stats[2].get('f0_min', 0):.1f}/{all_stats[2].get('f0_max', 0):.1f}" if all_stats[2] else "b.d."
        table += f"{'Min/Max F0 [Hz]':<24} | {min_max_1_str:^16} | {min_max_2_str:^16} | {min_max_3_str:^16}\n"

        table += f"{'─'*24}┼{'─'*18}┼{'─'*18}┼{'─'*18}\n"
        table += format_line("Jitter (relative) [%]", "jitter_percent")
        table += format_line("Shimmer (local) [dB]", "shimmer_db")
        table += format_line("Średni HNR [dB]", "hnr_mean")
        table += f"{'─'*24}┼{'─'*18}┼{'─'*18}┼{'─'*18}\n"
        for i in range(7): table += format_line(f"Średni Formant F{i+1} [Hz]", f"f{i+1}_mean")
        table += f"{'─'*24}┼{'─'*18}┼{'─'*18}┼{'─'*18}\n"
        for i in range(3): table += format_line(f"Śr. Amplituda H{i+1} [dB]", f"h{i+1}_mean")
        table += f"{'─'*24}┴{'─'*18}┴{'─'*18}┴{'─'*18}\n"
        
        frames_1 = all_stats[0]['voiced_frames'] if all_stats[0] else 0
        frames_2 = all_stats[1]['voiced_frames'] if all_stats[1] else 0
        frames_3 = all_stats[2]['voiced_frames'] if all_stats[2] else 0
        footer = f"Przeanalizowano ramek dźwięcznych: {frames_1} (K1), {frames_2} (K2), {frames_3} (K3)"
        
        text_widget.insert("end", table)
        text_widget.insert("end", footer)
        text_widget.config(state="disabled")
        self.status_label.config(text="Gotowy")

    def redraw_plots(self, *args):
        if self.normalized_audio_data is None: 
            self.initialize_axes()
            return

        num_channels = self.normalized_audio_data.shape[1]
        time_axis = np.linspace(0., self.duration_ms/1000, self.normalized_audio_data.shape[0])
        
        labels = ["Kanał 1", "Kanał 2", "Kanał 3", "Wynik"]
        axs_flat = self.axs.flatten()
        for i, ax in enumerate(axs_flat):
            ax.clear()
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_ylabel(labels[i], rotation=0, ha='right', va='center', labelpad=25)
            ax.tick_params(axis='x', labelbottom=True)

            if i < 3:
                if i < num_channels:
                    data_to_plot = self.get_modified_channel_data(i)
                    if data_to_plot is not None:
                        ax.plot(time_axis, data_to_plot, linewidth=0.5, color=f'C{i}')
                else:
                    ax.axis('off')
            else: 
                ax.set_xlabel("Czas (s)")
                if self.result_data is not None:
                    current_time_axis = np.linspace(0., len(self.result_data) / self.samplerate, num=len(self.result_data))
                    ax.plot(current_time_axis, self.result_data, linewidth=0.5, color='purple')
                else:
                    ax.text(0.5, 0.5, 'Oczekiwanie na obliczenie', ha='center', va='center', transform=ax.transAxes)
        
        self.apply_zoom_and_offset()

    def apply_zoom_and_offset(self):
        try:
            zoom, offset_coarse, offset_fine = self.zoom_ms_var.get(), self.offset_coarse_ms_var.get(), self.offset_fine_ms_var.get()
        except tk.TclError: return
        if self.offset_fine_slider.cget('to') != zoom: self.offset_fine_slider.config(to=zoom)
        start_sec = (offset_coarse + offset_fine) / 1000.0
        end_sec = start_sec + (zoom / 1000.0)
        
        max_time = self.duration_ms / 1000.0
        if self.result_data is not None:
                max_time = max(max_time, len(self.result_data) / self.samplerate)
        if end_sec > max_time: end_sec = max_time

        self.axs[0].set_xlim(start_sec, end_sec)
        self.axs[0].set_ylim(-1.1, 1.1)
        self.canvas.draw()
    
    def manage_playback_buttons(self, is_playing):
        self.is_playing = is_playing
        for btn in self.play_buttons: btn.config(state="disabled" if is_playing else "normal")
        self.stop_btn.config(state="normal" if is_playing else "disabled")

    def stop_audio(self):
        sd.stop()
        self.manage_playback_buttons(False)

    def play_audio(self, plot_index):
        if self.is_playing: self.stop_audio()
        if self.normalized_audio_data is None: return
        
        data_to_play = None
        if plot_index < self.normalized_audio_data.shape[1]:
            data_to_play = self.get_modified_channel_data(plot_index)
        elif plot_index == 3 and self.result_data is not None:
                data_to_play = self.result_data

        if data_to_play is not None:
            def playback_task():
                self.manage_playback_buttons(True)
                sd.play(data_to_play, self.samplerate)
                sd.wait()
                self.root.after(0, self.manage_playback_buttons, False)
            threading.Thread(target=playback_task, daemon=True).start()
    
    def save_result_file_4ch(self):
        if self.normalized_audio_data is None or self.result_data is None: messagebox.showwarning("Brak danych", "Najpierw wczytaj plik i oblicz wynik."); return
        num_channels_orig = self.normalized_audio_data.shape[1]
        
        # Sprawdź, czy są dostępne co najmniej 3 kanały do zapisu
        if num_channels_orig < 3:
            messagebox.showwarning("Brak danych", "Plik źródłowy musi mieć co najmniej 3 kanały, aby zapisać wynik 4-kanałowy.")
            return

        save_path = filedialog.asksaveasfilename(title="Zapisz plik 4-kanałowy", defaultextension=".wav", filetypes=[("Pliki WAV", "*.wav")])
        if not save_path: return

        try:
            len_orig, len_res = len(self.normalized_audio_data), len(self.result_data)
            output_len = max(len_orig, len_res)
            output_data = np.zeros((output_len, 4), dtype=np.float32)

            # Bezpieczne kopiowanie kanałów
            output_data[:len_orig, 0] = self.get_modified_channel_data(0)
            output_data[:len_orig, 1] = self.get_modified_channel_data(1)
            output_data[:len_orig, 2] = self.get_modified_channel_data(2)
            output_data[:len_res, 3] = self.result_data

            wavfile.write(save_path, self.samplerate, (output_data * 32767).astype(np.int16))
            messagebox.showinfo("Sukces", f"Plik został zapisany w:\n{save_path}")
        except Exception as e: messagebox.showerror("Błąd zapisu", f"Nie udało się zapisać pliku:\n{e}")

    def save_result_file_1ch(self):
        if self.result_data is None: messagebox.showwarning("Brak danych", "Najpierw oblicz wynik."); return
        save_path = filedialog.asksaveasfilename(title="Zapisz kanał wynikowy (Mono)", defaultextension=".wav", filetypes=[("Pliki WAV", "*.wav")])
        if not save_path: return
        try:
            wavfile.write(save_path, self.samplerate, (self.result_data * 32767).astype(np.int16))
            messagebox.showinfo("Sukces", f"Plik został zapisany w:\n{save_path}")
        except Exception as e: messagebox.showerror("Błąd zapisu", f"Nie udało się zapisać pliku:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = WavAnalyzerApp(root)
    root.mainloop()