import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import threading
from datetime import datetime
import os
from PIL import Image, ImageTk

class AudioRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rejestrator z Analizą Audio ")
        self.root.geometry("1200x750")

        # --- Inicjalizacja Zmiennych Stanu ---
        self.is_recording = False
        self.is_monitoring = False
        self.stream = None
        self.audio_data = [] # Bufor na nagrywane dane
        self.channel_colors = [] # Przechowuje aktualny kolor (zielony/czerwony) dla każdego oscyloskopu
        self.clip_timers = []  # Przechowuje timery do resetowania koloru po przesterowaniu
        self.gui_update_scheduled = False # Flaga zapobiegająca nadmiernemu odświeżaniu GUI

        # --- Główne Ustawienia Aplikacji ---
        self.save_path = os.getcwd() # Domyślna ścieżka zapisu to folder z programem
        self.supported_samplerates = [44100, 48000, 88200, 96000]
        self.supported_bitdepths = {"16-bit": 'int16', "24-bit": 'int32', "32-bit Float": 'float32'}
        self.block_size = 1024 # Ilość próbek w jednym bloku audio (wpływa na "zoom" oscyloskopu)
        
        # --- Zmienne dla Widgetów Tkinter ---
        self.samplerate_var = tk.IntVar(value=44100)
        self.bitdepth_str_var = tk.StringVar(value="16-bit")
        self.prefix_var = tk.StringVar(value="nagranie")
        self.trigger_enabled_var = tk.BooleanVar(value=True)
        self.trigger_level_var = tk.DoubleVar(value=0.0)
        self.spectrum_channel_var = tk.IntVar(value=1)
        self.spectrum_min_freq_var = tk.StringVar(value="20")
        self.spectrum_max_freq_var = tk.StringVar(value="5000")
        self.spectrum_threshold_var = tk.DoubleVar(value=-48.0)
        self.channel_vars, self.oscilloscopes = [], []

        # --- Inicjalizacja Urządzenia Audio ---
        self.device_id = self.find_first_input_device()
        self.channels = self.get_max_channels(self.device_id)
        
        # Bufor przechowujący ostatnie 2 bloki audio, używany do stabilizacji triggera
        self.display_buffer = np.zeros((self.block_size * 2, self.channels if self.channels > 0 else 1), dtype=np.float32)

        # --- Inicjalizacja Danych dla Spektrogramu ---
        self.num_freq_bins = self.block_size // 2 + 1 # Ilość "koszyków" częstotliwości z FFT
        self.spectrogram_height_px = 256 # Stała rozdzielczość pionowa bufora danych
        self.spectrogram_width_slices = 300 # Ilość "plastrów" czasu w buforze danych
        # Bufor 2D przechowujący dane głośności dla każdej częstotliwości w czasie
        self.spectrogram_data = np.zeros((self.num_freq_bins, self.spectrogram_width_slices))
        self.spectrogram_photo_image = None # Referencja do obrazu, musi być trzymana, aby Tkinter jej nie usunął

        # Mapa kolorów (colormap) zdefiniowana jako punkty (wartość 0-1, [R, G, B])
        self.cmap_values = np.array([0.0, 0.5, 0.75, 1.0]) # Punkty kontrolne
        self.cmap_colors = np.array([[0.2, 0.0, 0.4], [0.9, 0.1, 0.1], [1.0, 0.5, 0.0], [1.0, 1.0, 0.5]]) # Kolory (Fiolet->Czerwień->Pomarańcz->Żółty)

        # Uruchomienie budowy interfejsu i monitoringu
        self.create_widgets()
        self.start_monitoring()

    def find_first_input_device(self):
        """Znajduje domyślne urządzenie wejściowe w systemie."""
        try: return sd.default.device[0]
        except Exception: return None

    def get_max_channels(self, device_id):
        """Pobiera maksymalną liczbę kanałów dla danego urządzenia."""
        if device_id is not None: return sd.query_devices(device_id)['max_input_channels']
        return 0

    def create_widgets(self):
        """Tworzy i rozmieszcza wszystkie elementy graficzne w oknie aplikacji."""
        # Główny kontener na dwie kolumny
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill="both", expand=True)
        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1, minsize=350) # Lewa kolumna z kontrolkami
        main_frame.columnconfigure(1, weight=3) # Prawa kolumna z oscyloskopami (3x szersza)

        # --- Panel lewy (kontrolki) ---
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(3, weight=1) # Pozwól analizatorowi się rozciągać

        settings_frame = ttk.LabelFrame(left_panel, text="Ustawienia Nagrywania", padding="10")
        settings_frame.pack(fill="x", pady=(0, 10))
        settings_frame.columnconfigure(1, weight=1)

        # Kontrolki w ramce ustawień (Urządzenie, Próbkowanie, itp.)
        ttk.Label(settings_frame, text="Urządzenie:").grid(row=0, column=0, sticky="w", padx=5)
        devices = sd.query_devices()
        device_names = [f"{i}: {dev['name']}" for i, dev in enumerate(devices) if dev['max_input_channels'] > 0]
        self.device_var = tk.StringVar(value=next((name for name in device_names if name.startswith(str(self.device_id))), "Brak urządzeń"))
        self.device_menu = ttk.Combobox(settings_frame, textvariable=self.device_var, values=device_names, state="readonly")
        self.device_menu.grid(row=0, column=1, columnspan=3, sticky="ew", padx=5)
        self.device_menu.bind("<<ComboboxSelected>>", self.change_device)
        
        ttk.Label(settings_frame, text="Próbkowanie:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.samplerate_menu = ttk.Combobox(settings_frame, textvariable=self.samplerate_var, values=self.supported_samplerates, state="readonly", width=10)
        self.samplerate_menu.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        ttk.Label(settings_frame, text="Głębia bitowa:").grid(row=1, column=2, sticky="w", padx=5, pady=5)
        self.bitdepth_menu = ttk.Combobox(settings_frame, textvariable=self.bitdepth_str_var, values=list(self.supported_bitdepths.keys()), state="readonly", width=12)
        self.bitdepth_menu.grid(row=1, column=3, sticky="w", padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Prefix pliku:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.prefix_entry = ttk.Entry(settings_frame, textvariable=self.prefix_var)
        self.prefix_entry.grid(row=2, column=1, columnspan=3, sticky="ew", padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Folder zapisu:").grid(row=3, column=0, sticky="w", padx=5)
        self.path_label = ttk.Label(settings_frame, text=self.save_path, relief="sunken", anchor="w", padding=(5,2))
        self.path_label.grid(row=3, column=1, columnspan=2, sticky="ew", padx=5)
        self.path_button = ttk.Button(settings_frame, text="Zmień...", command=self.select_save_path)
        self.path_button.grid(row=3, column=3, padx=5)
        
        trigger_check = ttk.Checkbutton(settings_frame, text="Włącz trigger", variable=self.trigger_enabled_var)
        trigger_check.grid(row=4, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Poziom:").grid(row=5, column=0, sticky="w", padx=5)
        trigger_slider = ttk.Scale(settings_frame, from_=0.0, to=1.0, variable=self.trigger_level_var, orient='horizontal')
        trigger_slider.grid(row=5, column=1, columnspan=3, sticky="ew", padx=5)
        
        self.channel_select_frame = ttk.LabelFrame(left_panel, text="Wybór Kanałów (Zapis/Podgląd)", padding="10")
        self.channel_select_frame.pack(fill="x", pady=(0, 10))
        
        self.record_button = tk.Button(left_panel, text="🔴 Nagrywaj", font=("Helvetica", 14, "bold"), bg="red", fg="white", command=self.toggle_recording,
                                       activebackground="darkred", activeforeground="white", relief="raised", borderwidth=3)
        self.record_button.pack(fill="x", pady=(10, 10), ipady=10)
        
        self.spectrum_frame = ttk.LabelFrame(left_panel, text="Spektrogram", padding="10")
        self.spectrum_frame.pack(fill="both", expand=True)
        
        spec_ctrl_frame = ttk.Frame(self.spectrum_frame)
        spec_ctrl_frame.pack(fill='x', pady=(0,5), anchor='n')
        
        row1 = ttk.Frame(spec_ctrl_frame); row1.pack(fill='x')
        ttk.Label(row1, text="Źródło:").pack(side='left')
        self.spectrum_channel_menu = ttk.Combobox(row1, textvariable=self.spectrum_channel_var, state="readonly", width=5)
        self.spectrum_channel_menu.pack(side='left', padx=(2,10))
        self.spectrum_channel_menu.bind("<<ComboboxSelected>>", self.update_spectrum_title)

        row2 = ttk.Frame(spec_ctrl_frame); row2.pack(fill='x', pady=(5,0))
        ttk.Label(row2, text="Zakres (Hz):").pack(side='left')
        min_freq_entry = ttk.Entry(row2, textvariable=self.spectrum_min_freq_var, width=6)
        min_freq_entry.pack(side='left', padx=2)
        ttk.Label(row2, text="-").pack(side='left')
        max_freq_entry = ttk.Entry(row2, textvariable=self.spectrum_max_freq_var, width=6)
        max_freq_entry.pack(side='left', padx=2)
        
        row3 = ttk.Frame(spec_ctrl_frame); row3.pack(fill='x', pady=(5,0))
        ttk.Label(row3, text="Próg (dBr):").pack(side='left')
        threshold_slider = ttk.Scale(row3, from_=-60, to=0, variable=self.spectrum_threshold_var, orient='horizontal')
        threshold_slider.pack(fill='x', expand=True, side='left', padx=2)
        threshold_label = ttk.Label(row3, width=5)
        threshold_label.pack(side='left')
        self.spectrum_threshold_var.trace_add("write", lambda *args: threshold_label.config(text=f"{self.spectrum_threshold_var.get():.0f}"))
        threshold_label.config(text=f"{self.spectrum_threshold_var.get():.0f}")

        spec_display_frame = ttk.Frame(self.spectrum_frame)
        spec_display_frame.pack(fill="both", expand=True)
        spec_display_frame.rowconfigure(0, weight=1)
        spec_display_frame.columnconfigure(1, weight=1)
        
        # Osobny canvas na legendę osi Y (częstotliwości)
        self.spectrogram_scale_canvas = tk.Canvas(spec_display_frame, bg='black', width=50)
        self.spectrogram_scale_canvas.grid(row=0, column=0, sticky='ns')
        # Główny canvas na obraz spektrogramu
        self.spectrum_canvas = tk.Canvas(spec_display_frame, bg='black')
        self.spectrum_canvas.grid(row=0, column=1, sticky='nsew')
        # Przerysowanie skali przy zmianie rozmiaru okna
        self.spectrogram_scale_canvas.bind("<Configure>", self.draw_spectrogram_scale)

        self.scope_frame = ttk.LabelFrame(main_frame, text="Podgląd na żywo", padding="10")
        self.scope_frame.grid(row=0, column=1, sticky="nsew")
        self.scope_frame.columnconfigure(0, weight=1)
        
        self.update_channel_layout()

    def update_spectrum_title(self, event=None):
        """Aktualizuje tytuł ramki spektrogramu o numer wybranego kanału."""
        self.spectrum_frame.config(text=f"Spektrogram (Kanał {self.spectrum_channel_var.get()})")

    def update_spectrum_channel_selector(self):
        """Aktualizuje listę dostępnych kanałów w menu wyboru źródła dla analizatora."""
        channels_list = [f"{i+1}" for i in range(self.channels)]
        self.spectrum_channel_menu['values'] = channels_list
        if self.channels > 0:
            if self.spectrum_channel_var.get() > self.channels: self.spectrum_channel_var.set(1)
            self.spectrum_channel_menu.config(state="readonly")
        else: self.spectrum_channel_menu.config(state="disabled")
        self.update_spectrum_title()
        
    def update_channel_layout(self):
        """Przebudowuje layout kanałów (checkboxy i oscyloskopy) po zmianie urządzenia."""
        for widget in self.scope_frame.winfo_children(): widget.destroy()
        for widget in self.channel_select_frame.winfo_children(): widget.destroy()
        self.oscilloscopes, self.channel_vars = [], []
        self.channel_colors, self.clip_timers = ['#00FF00']*self.channels, [None]*self.channels
        self.display_buffer = np.zeros((self.block_size*2, self.channels if self.channels>0 else 1), dtype=np.float32)

        if self.channels > 0:
            for i in range(self.channels):
                var = tk.BooleanVar(value=(i < 3)); chk = ttk.Checkbutton(self.channel_select_frame, text=f"Kanał {i+1}", variable=var, command=self.refresh_scope_grid)
                chk.grid(row=0, column=i, padx=5, pady=5); self.channel_vars.append(var)
                canvas = tk.Canvas(self.scope_frame, bg='black'); self.oscilloscopes.append(canvas)
            self.refresh_scope_grid()
            self.update_spectrum_channel_selector()
        else: ttk.Label(self.channel_select_frame, text="Brak dostępnych kanałów.").pack()

    def refresh_scope_grid(self):
        """Pokazuje/ukrywa oscyloskopy w zależności od stanu checkboxów."""
        visible_scopes = [c for i,c in enumerate(self.oscilloscopes) if self.channel_vars[i].get()]
        for canvas in self.oscilloscopes: canvas.grid_forget()
        for row, canvas in enumerate(visible_scopes): canvas.grid(row=row, column=0, sticky="nsew", pady=4)
        if visible_scopes: self.scope_frame.rowconfigure(list(range(len(visible_scopes))), weight=1)

    def change_device(self, event=None):
        """Obsługuje zmianę urządzenia audio - zatrzymuje stary strumień, tworzy nowy layout i uruchamia go."""
        self.stop_monitoring()
        selection = self.device_var.get(); self.device_id = int(selection.split(':')[0])
        self.channels = self.get_max_channels(self.device_id); self.update_channel_layout(); self.start_monitoring()

    def select_save_path(self):
        """Otwiera okno wyboru folderu do zapisu plików."""
        path = filedialog.askdirectory()
        if path: self.save_path = path; self.path_label.config(text=self.save_path)

    def audio_callback(self, indata, frames, time, status):
        """
        Funkcja wywoływana przez bibliotekę `sounddevice` dla każdego nowego bloku danych audio.
        Działa w osobnym wątku, więc nie może bezpośrednio modyfikować GUI.
        """
        if status: print(status)
        # Przesuń stare dane w buforze i dopisz nowe na końcu
        self.display_buffer = np.roll(self.display_buffer, -frames, axis=0)
        self.display_buffer[-frames:, :] = indata
        
        # Ograniczenie odświeżania GUI do ok. 25 FPS, aby nie obciążać procesora.
        if not self.gui_update_scheduled:
            self.gui_update_scheduled = True
            # Sprawdź, które kanały są przesterowane
            clipped = np.any(np.abs(indata) >= 1.0, axis=0)
            # Przekaż dane do przetworzenia w głównym wątku GUI
            self.root.after(40, self.process_gui_update, self.display_buffer.copy(), clipped)

        if self.is_recording: self.audio_data.append(indata.copy())

    def process_gui_update(self, data_buffer, clipped_channels):
        """Główna funkcja przetwarzająca dane i aktualizująca GUI. Działa w głównym wątku."""
        # Zarządzanie przesterowaniem (czerwony kolor oscyloskopów)
        for i, clipped in enumerate(clipped_channels):
            if i < len(self.channel_colors) and clipped:
                self.channel_colors[i] = '#FF0000'
                if self.clip_timers[i]: self.root.after_cancel(self.clip_timers[i])
                self.clip_timers[i] = self.root.after(500, lambda idx=i: self.reset_clip_color(idx))
        
        # Domyślnie, do rysowania użyj ostatniego bloku danych ("płynący")
        data_to_display = data_buffer[-self.block_size:, :]; fft_mags = None
        
        # Sprawdź, czy którykolwiek kanał jest źródłem dla analizy
        analysis_idx = self.spectrum_channel_var.get() - 1
        if 0 <= analysis_idx < self.channels:
            # Jeśli trigger jest włączony, znajdź punkt synchronizacji
            if self.trigger_enabled_var.get():
                trigger_level = self.trigger_level_var.get()
                trigger_data = data_buffer[:, analysis_idx]
                # Szukaj próbki, która przekracza próg triggera (po nie-przekraczającej)
                crossings = np.where(np.logical_and(trigger_data[:-1] <= trigger_level, trigger_data[1:] > trigger_level))[0]
                valid_crossings = crossings[crossings < self.block_size]
                if len(valid_crossings) > 0:
                    # Jeśli znaleziono punkt, wytnij nowy blok danych do wyświetlenia
                    data_to_display = data_buffer[valid_crossings[0] : valid_crossings[0] + self.block_size, :]

            # Oblicz FFT dla kanału źródłowego, używając danych (ew. zsynchronizowanych)
            fft_data = data_to_display[:, analysis_idx]
            if np.sqrt(np.mean(fft_data**2)) > 0.001: # Próg głośności dla FFT
                fft_mags = np.abs(np.fft.rfft(fft_data * np.hanning(len(fft_data))))
        
        # Wywołaj funkcje rysujące z przygotowanymi danymi
        self.update_visuals(data_to_display, fft_mags)
        self.gui_update_scheduled = False # Zezwól na zaplanowanie kolejnego odświeżenia

    def reset_clip_color(self, index):
        """Przywraca zielony kolor oscyloskopu po przesterowaniu."""
        if index < len(self.channel_colors): self.channel_colors[index] = '#00FF00'; self.clip_timers[index] = None

    def update_visuals(self, data, fft_mags):
        """Wywołuje obie funkcje rysujące."""
        self.update_oscilloscopes(data)
        self.update_spectrogram(fft_mags)

    def colormap_vectorized(self, norm_data):
        """Szybka, zwektoryzowana mapa kolorów używająca NumPy do interpolacji."""
        r = np.interp(norm_data, self.cmap_values, self.cmap_colors[:, 0])
        g = np.interp(norm_data, self.cmap_values, self.cmap_colors[:, 1])
        b = np.interp(norm_data, self.cmap_values, self.cmap_colors[:, 2])
        # `np.stack` prawidłowo składa tablice R,G,B w obraz 3D o wymiarach (wysokość, szerokość, 3)
        return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

    def draw_spectrogram_scale(self, event=None):
        """Rysuje statyczną legendę (oś Y) dla spektrogramu."""
        canvas = self.spectrogram_scale_canvas; canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width <= 1 or height <= 1: return
        
        try: min_freq, max_freq = float(self.spectrum_min_freq_var.get()), float(self.spectrum_max_freq_var.get())
        except ValueError: min_freq, max_freq = 20, 5000
        min_freq, max_freq = max(20, min_freq), min(max_freq, self.samplerate_var.get()/2)
        if min_freq >= max_freq: min_freq, max_freq = 20, 5000
        log_min, log_max = np.log10(min_freq), np.log10(max_freq)

        # Rysuj etykiety dla typowych częstotliwości audio, jeśli mieszczą się w zakresie
        for freq_label in [100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
            if min_freq <= freq_label <= max_freq:
                log_f = np.log10(freq_label)
                # Oblicz pozycję Y na podstawie skali logarytmicznej
                y = height - ((log_f - log_min) / (log_max - log_min)) * height
                canvas.create_line(width-10, y, width, y, fill="#606060")
                label = f"{int(freq_label/1000)}k" if freq_label >= 1000 else str(freq_label)
                canvas.create_text(width-15, y, text=label, fill="grey", anchor="e", font=("TkFixedFont", 8))

    def update_spectrogram(self, fft_mags):
        """Aktualizuje bufor danych spektrogramu i rysuje go na ekranie."""
        # 1. Aktualizacja bufora danych
        self.spectrogram_data = np.roll(self.spectrogram_data, -1, axis=1)
        if fft_mags is None:
            self.spectrogram_data[:, -1] = -100 # Wartość dla ciszy
        else:
            db_mags = 20 * np.log10(fft_mags + 1e-9)
            peak_db = np.max(db_mags); db_normalized = db_mags - peak_db
            self.spectrogram_data[:, -1] = db_normalized

        canvas = self.spectrum_canvas; canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width <= 1: return
        
        # 2. Pobranie i walidacja ustawień
        samplerate = self.samplerate_var.get()
        try: min_freq, max_freq = float(self.spectrum_min_freq_var.get()), float(self.spectrum_max_freq_var.get())
        except ValueError: min_freq, max_freq = 20, 5000
        min_freq, max_freq = max(20, min_freq), min(max_freq, samplerate/2)
        if min_freq >= max_freq: min_freq, max_freq = 20, 5000
        
        db_min_rel, db_max_rel = -60, 0
        threshold_db = self.spectrum_threshold_var.get()
        
        # 3. Resampling - kluczowy krok do poprawnego skalowania
        # Liniowa oś częstotliwości z FFT
        freqs = np.fft.rfftfreq(self.block_size, 1 / samplerate)
        # Logarytmiczna oś częstotliwości, na którą będziemy rzutować (o wysokości bufora)
        log_freq_axis = np.logspace(np.log10(min_freq), np.log10(max_freq), num=self.spectrogram_height_px)
        
        # Interpolacja każdej kolumny czasu z bufora na nową, logarytmiczną oś
        resampled_data = np.array([np.interp(log_freq_axis, freqs, self.spectrogram_data[:, j]) 
                                   for j in range(self.spectrogram_width_slices)]).T
        
        # 4. Konwersja danych na kolory
        resampled_data[resampled_data < threshold_db] = db_min_rel
        norm_data = np.clip((resampled_data - db_min_rel) / (db_max_rel - db_min_rel), 0, 1)
        
        rgb_data = self.colormap_vectorized(norm_data)
        rgb_data = np.flipud(rgb_data) # Odwróć, aby niskie częstotliwości były na dole

        # 5. Rysowanie obrazu
        if rgb_data.size > 0:
            image = Image.fromarray(rgb_data, 'RGB').resize((width, height), Image.Resampling.NEAREST)
            self.spectrogram_photo_image = ImageTk.PhotoImage(image=image)
            canvas.create_image(0, 0, image=self.spectrogram_photo_image, anchor='nw')
        
        self.draw_spectrogram_scale()

    def update_oscilloscopes(self, data):
        """Aktualizuje wszystkie widoczne oscyloskopy."""
        duration_ms = (self.block_size / self.samplerate_var.get()) * 1000
        analysis_idx = self.spectrum_channel_var.get() - 1

        for i, canvas in enumerate(self.oscilloscopes):
            if self.channel_vars[i].get() and i < data.shape[1]:
                canvas.delete("all"); width, height = canvas.winfo_width(), canvas.winfo_height()
                if width <= 1: continue

                # Rysowanie siatki poziomej (amplituda)
                for level, label in [(-1.0,"-1.0"), (-0.5,"-0.5"), (0.0," 0.0"), (0.5,"+0.5"), (1.0,"+1.0")]:
                    y = (height/2)*(1 - level)
                    canvas.create_line(0, y, width, y, fill="#303030" if level != 0.0 else "#606060", dash=(2, 4) if level != 0.0 else ())
                    if -1.0 < level < 1.0:
                        canvas.create_text(25, y, text=label, fill="grey", anchor="e", font=("TkFixedFont", 8))
                
                # Rysowanie siatki pionowej (czas)
                for j in range(1, 9): # 8 linii
                    x = width * (j/8)
                    canvas.create_line(x, 0, x, height, fill="#303030", dash=(2, 4))
                    canvas.create_text(x, height-5, text=f"{duration_ms*j/8:.1f}", fill="grey", anchor="s", font=("TkFixedFont", 8))
                
                canvas.create_text(width-5, 5, text="ms", fill="grey", anchor="ne", font=("TkFixedFont", 8))
                canvas.create_text(5, 5, text=f"Kanał {i+1}", fill="grey", anchor="nw", font=("TkFixedFont", 8))
                
                # Rysowanie linii triggera na odpowiednim kanale
                if self.trigger_enabled_var.get() and i == analysis_idx:
                    y_trigger = (height/2)*(1 - self.trigger_level_var.get())
                    canvas.create_line(0, y_trigger, width, y_trigger, fill="#FFFF00", dash=(4, 4))

                # Rysowanie fali
                points = []; num_samples = len(data)
                if num_samples > 1:
                    channel_data = data[:, i]
                    for j, s in enumerate(channel_data):
                        x, y = (j/(num_samples-1))*width, (height/2)*(1-s); points.extend([x, y])
                    canvas.create_line(points, fill=self.channel_colors[i], width=1.5)

    def start_monitoring(self):
        """Uruchamia strumień audio i oblicza wymiary bufora spektrogramu."""
        samplerate = self.samplerate_var.get()
        time_per_block = self.block_size / samplerate
        self.spectrogram_width_slices = int(5.0 / time_per_block) # ok. 5 sekund historii
        
        self.num_freq_bins = self.block_size // 2 + 1
        self.spectrogram_data = np.zeros((self.num_freq_bins, self.spectrogram_width_slices))

        if self.device_id is None or self.channels == 0: return
        if not self.is_monitoring:
            try: self.stream = sd.InputStream(device=self.device_id, channels=self.channels, samplerate=samplerate,
                                             callback=self.audio_callback, blocksize=self.block_size, dtype='float32'); self.stream.start()
            except Exception as e: messagebox.showerror("Błąd strumienia", f"Nie można uruchomić podglądu:\n{e}")

    # ... (reszta metod bez zmian)
    def stop_monitoring(self):
        if self.stream: self.stream.stop(); self.stream.close(); self.stream = None
    def toggle_recording(self):
        if self.is_recording:
            self.is_recording = False; self.record_button.config(text="🔴 Nagrywaj", bg="red", fg="white")
            threading.Thread(target=self.save_recording).start()
        else:
            if not any(v.get() for v in self.channel_vars): messagebox.showwarning("Brak kanałów", "Wybierz kanał."); return
            self.audio_data = []; self.is_recording = True
            self.record_button.config(text="⏹️ Trwa nagrywanie...", bg="#cccccc", fg="black")
    def save_recording(self):
        if not self.audio_data: return
        recording = np.concatenate(self.audio_data, axis=0)
        selected_indices = [i for i, v in enumerate(self.channel_vars) if v.get()]
        if not selected_indices: return
        final_recording = recording[:, selected_indices]
        prefix = self.prefix_var.get().strip() or "nagranie"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(self.save_path, f"{prefix}_{timestamp}.wav")
        try:
            dtype = self.supported_bitdepths[self.bitdepth_str_var.get()]
            if dtype == 'int16': final_recording = (final_recording * 32767).astype(np.int16)
            elif dtype == 'int32': final_recording = (final_recording * 8388607).astype(np.int32)
            write(filename, self.samplerate_var.get(), final_recording)
            self.root.after(0, lambda: messagebox.showinfo("Sukces", f"Plik zapisany:\n{filename}"))
        except Exception as e: self.root.after(0, lambda: messagebox.showerror("Błąd zapisu", f"Nie udało się zapisać pliku:\n{e}"))
    def on_closing(self):
        if self.is_recording and not messagebox.askyesno("Ostrzeżenie", "Nagrywanie trwa. Wyjść?"): return
        self.stop_monitoring(); self.root.destroy()

if __name__ == "__main__":
    try:
        sd.query_devices()
        root = tk.Tk()
        app = AudioRecorderApp(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
    except Exception as e:
        print(f"FATAL: Błąd inicjalizacji audio: {e}")
        root_err = tk.Tk(); root_err.withdraw()
        messagebox.showerror("Krytyczny błąd", f"Nie można zainicjalizować systemu audio.\nSzczegóły: {e}")