# -*- coding: utf-8 -*-

# --------------------------------------------------------------------------------
# Import niezbędnych bibliotek
# --------------------------------------------------------------------------------

# Biblioteka do tworzenia graficznego interfejsu użytkownika (GUI)
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# NumPy - podstawowa biblioteka do obliczeń naukowych, zwłaszcza na tablicach (macierzach)
import numpy as np

# SciPy - biblioteka do zaawansowanych obliczeń naukowych i technicznych
from scipy.io import wavfile           # Do wczytywania i zapisywania plików .wav
from scipy import signal               # Do przetwarzania sygnałów (np. spektrogram, splot)
from scipy.linalg import solve_toeplitz # Do rozwiązywania układów równań (używane w LPC)

# Sounddevice - biblioteka do odtwarzania i nagrywania dźwięku
import sounddevice as sd

# Threading - pozwala na uruchamianie zadań w tle (np. odtwarzanie audio), aby interfejs się nie blokował
import threading

# Datetime i OS - do operacji na datach i systemie plików (np. pobieranie nazwy pliku)
from datetime import datetime
import os

# Matplotlib - biblioteka do tworzenia wykresów
import matplotlib
matplotlib.use("TkAgg")  # Ustawienie "backendu" Matplotlib, aby był kompatybilny z Tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# --------------------------------------------------------------------------------
# Główna klasa aplikacji
# --------------------------------------------------------------------------------

class WavAnalyzerApp:
    """
    Główna klasa aplikacji, która zawiera całą logikę i elementy interfejsu.
    Struktura oparta na klasie pozwala na łatwe zarządzanie stanem aplikacji
    poprzez zmienne instancji (self.zmienna).
    """

    # -------------------
    # METODA INICJALIZACYJNA (__init__)
    # -------------------
    def __init__(self, root):
        """
        Konstruktor klasy. Wywoływany przy tworzeniu nowego obiektu (aplikacji).
        Tutaj inicjalizujemy okno, wszystkie zmienne i tworzymy widgety.

        :param root: Główne okno aplikacji (obiekt tk.Tk())
        """
        # --- Główne okno ---
        self.root = root  # Przechowujemy referencję do głównego okna
        self.root.title("Analizator WAV")  # Ustawiamy tytuł okna
        self.root.geometry("1400x900")  # Ustawiamy domyślny rozmiar okna

        # --- Zmienne przechowujące dane audio ---
        # None oznacza, że na początku nie mają żadnej wartości
        self.samplerate, self.audio_data, self.normalized_audio_data, self.filepath, self.duration_ms = [None] * 5
        # self.samplerate: Częstotliwość próbkowania [Hz]
        # self.audio_data: Surowe dane audio wczytane z pliku (zazwyczaj int16)
        # self.normalized_audio_data: Dane audio przeskalowane do zakresu [-1.0, 1.0] (float32)
        # self.filepath: Ścieżka do wczytanego pliku .wav
        # self.duration_ms: Długość pliku audio w milisekundach

        # Dane wynikowe po operacjach (np. odejmowanie, dekonwolucja)
        self.result_data = None

        # Zmienne związane z operacją splotu (konwolucji)
        self.convolution_ir_data = None         # Dane odpowiedzi impulsowej (IR)
        self.convolution_ir_path = None         # Ścieżka do pliku IR
        self.convolution_ir_samplerate = None   # Częstotliwość próbkowania pliku IR

        # --- Zmienne stanu interfejsu (połączone z widgetami) ---
        # Używamy specjalnych zmiennych Tkintera (tk.DoubleVar, tk.BooleanVar, etc.),
        # aby automatycznie aktualizować widgety, gdy zmienna się zmienia.

        # Zmienne do nawigacji na wykresie
        self.zoom_ms_var = tk.DoubleVar(value=20.0)             # Długość okna widoku [ms]
        self.offset_coarse_ms_var = tk.DoubleVar(value=0.0)     # Przesunięcie zgrubne [ms]
        self.offset_fine_ms_var = tk.DoubleVar(value=0.0)       # Przesunięcie precyzyjne [ms]

        # Zmienne do kontroli kanałów (lista dla każdego z 3 kanałów)
        self.invert_phase_vars = [tk.BooleanVar(value=False) for _ in range(3)] # Czy odwrócić fazę?
        self.phase_shift_vars = [tk.DoubleVar(value=0.0) for _ in range(3)]     # Przesunięcie czasowe [ms]

        # Zmienne do analizy różnicowej
        self.analysis_mode_var = tk.StringVar(value="Odejmowanie A-B")  # Tryb analizy
        self.analysis_ch_A_var = tk.IntVar(value=1)                     # Numer kanału A
        self.analysis_ch_B_var = tk.IntVar(value=2)                     # Numer kanału B

        # Zmienna do splotu
        self.convolution_source_channel_var = tk.IntVar(value=1)        # Kanał źródłowy dla splotu

        # Zmienne do analizy spektralnej (FFT, spektrogram)
        self.spectrogram_ch_A_var = tk.IntVar(value=1)      # Kanał A do analizy spektralnej
        self.spectrogram_ch_B_var = tk.IntVar(value=2)      # Kanał B do analizy spektralnej
        self.spec_f_min_var = tk.StringVar(value="100")     # Minimalna częstotliwość do wyświetlenia [Hz]
        self.spec_f_max_var = tk.StringVar(value="2000")    # Maksymalna częstotliwość do wyświetlenia [Hz]
        self.analysis_start_ms_var = tk.DoubleVar(value=0.0)# Początek fragmentu do analizy [ms]
        self.analysis_end_ms_var = tk.DoubleVar(value=0.0)  # Koniec fragmentu do analizy [ms]

        # --- Inne zmienne stanu ---
        self.is_playing = False     # Flaga, czy aktualnie odtwarzany jest dźwięk
        self.play_buttons = []      # Lista przycisków odtwarzania, do łatwego zarządzania ich stanem

        # --- Tworzenie interfejsu ---
        self.create_widgets() # Metoda budująca wszystkie elementy GUI
        self.setup_plot()     # Metoda przygotowująca obszar wykresów Matplotlib

    # -------------------
    # TWORZENIE INTERFEJSU
    # -------------------
    def create_widgets(self):
        """
        Tworzy i rozmieszcza wszystkie elementy interfejsu (przyciski, suwaki, etykiety itp.).
        Używamy 'ttk' zamiast 'tk' tam, gdzie to możliwe, dla nowocześniejszego wyglądu.
        Rozmieszczenie elementów odbywa się za pomocą metod .pack() i .grid().
        """
        # Główny kontener, który dzieli okno na dwie kolumny: wykres (lewo) i panel kontrolny (prawo)
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill="both", expand=True)
        main_frame.rowconfigure(0, weight=1)  # Wiersz 0 ma się rozciągać pionowo
        main_frame.columnconfigure(0, weight=3) # Kolumna 0 (wykres) jest 3x szersza
        main_frame.columnconfigure(1, weight=1, minsize=350) # Kolumna 1 (kontrole) ma stałą minimalną szerokość

        # Ramka na wykresy
        self.plot_frame = ttk.Frame(main_frame)
        self.plot_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10)) # sticky="nsew" rozciąga ramkę we wszystkich kierunkach

        # Panel z wszystkimi kontrolkami po prawej stronie
        controls_panel = ttk.Frame(main_frame)
        controls_panel.grid(row=0, column=1, sticky="nsew")

        # --- Sekcja: Otwieranie pliku ---
        open_file_frame = ttk.Frame(controls_panel)
        open_file_frame.pack(fill='x', pady=(0, 5))
        open_file_frame.columnconfigure(1, weight=1)
        open_button = ttk.Button(open_file_frame, text="Wybierz plik WAV...", command=self.select_file)
        open_button.pack(side='left', padx=(0,10))
        self.filepath_label = ttk.Label(open_file_frame, text="Nie wybrano pliku", anchor="w", relief="sunken", padding=5)
        self.filepath_label.pack(fill='x', expand=True)

        # --- Sekcja: Nawigacja i Kontrola Kanałów ---
        combined_controls_frame = ttk.LabelFrame(controls_panel, text="Nawigacja i Kontrola", padding="10")
        combined_controls_frame.pack(fill="x", pady=5)
        combined_controls_frame.columnconfigure(1, weight=1) # Środkowa kolumna ma się rozciągać

        # Suwaki do zoomu i offsetu
        ttk.Label(combined_controls_frame, text="Długość (ms):").grid(row=0, column=0, sticky="w")
        self.zoom_slider = ttk.Scale(combined_controls_frame, from_=1, to=100, variable=self.zoom_ms_var, orient='horizontal', command=self.redraw_plots)
        self.zoom_slider.grid(row=0, column=1, sticky="ew", padx=5)
        zoom_entry = ttk.Entry(combined_controls_frame, textvariable=self.zoom_ms_var, width=8)
        zoom_entry.grid(row=0, column=2)
        zoom_entry.bind("<Return>", self.redraw_plots) # Przerysuj po wciśnięciu Enter

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

        # Separator i kontrolki dla poszczególnych kanałów
        ttk.Separator(combined_controls_frame, orient='horizontal').grid(row=3, column=0, columnspan=3, sticky='ew', pady=10)
        channel_ctrl_container = ttk.Frame(combined_controls_frame)
        channel_ctrl_container.grid(row=4, column=0, columnspan=3, sticky='ew')
        channel_ctrl_container.columnconfigure((0, 1, 2), weight=1) # Wszystkie 3 kolumny równej szerokości
        # Pętla tworząca kontrolki dla kanałów 1, 2 i 3
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

        # --- Sekcja: Splot (konwolucja) ---
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

        # --- Sekcja: Analiza Różnicowa i Odtwarzanie ---
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

        # Kontrolki do odtwarzania
        playback_frame = ttk.Frame(analysis_frame)
        playback_frame.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(10,0))
        playback_frame.columnconfigure((0,1,2,3,4), weight=1) # 5 przycisków równej szerokości
        self.play_ch1_btn = ttk.Button(playback_frame, text="Odtw. K1", command=lambda: self.play_audio(0))
        self.play_ch2_btn = ttk.Button(playback_frame, text="Odtw. K2", command=lambda: self.play_audio(1))
        self.play_ch3_btn = ttk.Button(playback_frame, text="Odtw. K3", command=lambda: self.play_audio(2))
        self.play_res_btn = ttk.Button(playback_frame, text="Odtw. Wynik", command=lambda: self.play_audio(3))
        self.stop_btn = ttk.Button(playback_frame, text="⏹ Stop", command=self.stop_audio, state="disabled")
        self.play_ch1_btn.grid(row=0, column=0, sticky='ew'); self.play_ch2_btn.grid(row=0, column=1, sticky='ew')
        self.play_ch3_btn.grid(row=0, column=2, sticky='ew'); self.play_res_btn.grid(row=0, column=3, sticky='ew')
        self.stop_btn.grid(row=0, column=4, sticky='ew')
        self.play_buttons = [self.play_ch1_btn, self.play_ch2_btn, self.play_ch3_btn, self.play_res_btn]

        # Przyciski do zapisywania wyników
        save_frame = ttk.Frame(analysis_frame)
        save_frame.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(10,0))
        save_frame.columnconfigure((0,1), weight=1)
        save_4ch_button = ttk.Button(save_frame, text="Zapisz wynik (4-kanałowy WAV)", command=self.save_result_file_4ch)
        save_4ch_button.grid(row=0, column=0, sticky="ew", padx=(0,5))
        save_1ch_button = ttk.Button(save_frame, text="Zapisz tylko wynik (Mono WAV)", command=self.save_result_file_1ch)
        save_1ch_button.grid(row=0, column=1, sticky="ew", padx=(5,0))

        # --- Sekcja: Analiza Spektralna ---
        spectral_frame = ttk.LabelFrame(controls_panel, text="Analiza Spektralna", padding="10")
        spectral_frame.pack(fill="x", pady=5)
        spectral_frame.columnconfigure((1, 3), weight=1)

        # Wybór kanałów i zakresu częstotliwości
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

        # Wybór fragmentu do analizy
        ttk.Label(spectral_frame, text="Początek (ms):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.analysis_start_slider = ttk.Scale(spectral_frame, from_=0, to=1000, variable=self.analysis_start_ms_var, orient='horizontal')
        self.analysis_start_slider.grid(row=2, column=1, columnspan=2, sticky="ew", padx=5)
        ttk.Entry(spectral_frame, textvariable=self.analysis_start_ms_var, width=8).grid(row=2, column=3)

        ttk.Label(spectral_frame, text="Koniec (ms):").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.analysis_end_slider = ttk.Scale(spectral_frame, from_=0, to=1000, variable=self.analysis_end_ms_var, orient='horizontal')
        self.analysis_end_slider.grid(row=3, column=1, columnspan=2, sticky="ew", padx=5)
        ttk.Entry(spectral_frame, textvariable=self.analysis_end_ms_var, width=8).grid(row=3, column=3)

        # Przyciski do uruchamiania analiz
        buttons_frame = ttk.Frame(spectral_frame)
        buttons_frame.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        buttons_frame.columnconfigure((0, 1, 2), weight=1)

        show_spec_button = ttk.Button(buttons_frame, text="Pokaż Spektrogramy", command=self.show_spectrogram_window)
        show_spec_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        show_fft_button = ttk.Button(buttons_frame, text="Pokaż FFT", command=self.show_fft_window)
        show_fft_button.grid(row=0, column=1, sticky="ew", padx=5)

        show_f0_button = ttk.Button(buttons_frame, text="Pokaż Statystyki Głosu", command=self.show_voice_stats)
        show_f0_button.grid(row=0, column=2, sticky="ew", padx=(5, 0))

        # --- Pasek statusu na dole ---
        self.status_label = ttk.Label(controls_panel, text="Gotowy", anchor="w", relief="sunken", padding=5)
        self.status_label.pack(side='bottom', fill='x', pady=(10,0))

    # -------------------
    # METODY ZWIĄZANE Z WYKRESAMI
    # -------------------
    def setup_plot(self):
        """
        Inicjalizuje obiekt wykresu Matplotlib i osadza go w ramce Tkinter.
        """
        # Stworzenie "figury" - głównego kontenera na wszystkie wykresy
        self.fig = Figure(figsize=(10, 8), dpi=100)

        # Stworzenie 4 pod-wykresów (axes) w jednej kolumnie, z wspólnymi osiami X i Y
        self.axs = self.fig.subplots(4, 1, sharex=True, sharey=True)

        # Ustawienie ciasnego układu, aby etykiety się nie nakładały
        self.fig.set_constrained_layout(True)

        # Stworzenie "płótna" (canvas), które jest widgetem Tkinter do rysowania figury Matplotlib
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Dodanie standardowego paska narzędzi Matplotlib (zoom, przesuwanie, zapis)
        toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Inicjalizacja osi (siatka, etykiety)
        self.initialize_axes()

    def initialize_axes(self):
        """
        Czyści i ustawia początkowy wygląd osi wykresów (etykiety, siatka).
        """
        labels = ["Kanał 1", "Kanał 2", "Kanał 3", "Wynik"]
        for i, ax in enumerate(self.axs):
            ax.clear()  # Wyczyść poprzednią zawartość osi
            ax.grid(True, linestyle='--', alpha=0.6) # Dodaj siatkę
            # Ustaw etykietę osi Y, obróconą dla lepszej czytelności
            ax.set_ylabel(labels[i], rotation=0, ha='right', va='center', labelpad=25)
            ax.tick_params(axis='x', labelbottom=True) # Upewnij się, że etykiety osi X są widoczne
        self.axs[-1].set_xlabel("Czas (s)") # Dodaj etykietę osi X do ostatniego wykresu
        self.canvas.draw() # Odśwież płótno

    def redraw_plots(self, *args):
        """
        Główna funkcja do przerysowywania wszystkich wykresów.
        Wywoływana po każdej zmianie (wczytanie pliku, zmiana suwaków, obliczenia).
        """
        # Jeśli nie ma danych, tylko wyczyść wykresy
        if self.normalized_audio_data is None:
            self.initialize_axes()
            return

        # Przygotowanie danych do rysowania
        num_channels = self.normalized_audio_data.shape[1] # Liczba kanałów
        # Utworzenie osi czasu w sekundach
        time_axis = np.linspace(0., self.duration_ms/1000, self.normalized_audio_data.shape[0])

        labels = ["Kanał 1", "Kanał 2", "Kanał 3", "Wynik"]
        axs_flat = self.axs.flatten() # Spłaszczenie tablicy osi do jednowymiarowej listy

        # Pętla przez 4 osie wykresów
        for i, ax in enumerate(axs_flat):
            ax.clear() # Wyczyść oś
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_ylabel(labels[i], rotation=0, ha='right', va='center', labelpad=25)
            ax.tick_params(axis='x', labelbottom=True)

            # Rysowanie pierwszych 3 kanałów
            if i < 3:
                # Sprawdź, czy kanał o danym indeksie istnieje w pliku
                if i < num_channels:
                    # Pobierz dane kanału z uwzględnieniem modyfikacji (faza, przesunięcie)
                    data_to_plot = self.get_modified_channel_data(i)
                    if data_to_plot is not None:
                        ax.plot(time_axis, data_to_plot, linewidth=0.5, color=f'C{i}')
                else:
                    # Jeśli kanał nie istnieje, wyłącz oś
                    ax.axis('off')
            # Rysowanie czwartego wykresu (wynikowego)
            else:
                ax.set_xlabel("Czas (s)")
                if self.result_data is not None:
                    # Oś czasu dla wyniku może mieć inną długość (np. po splocie)
                    current_time_axis = np.linspace(0., len(self.result_data) / self.samplerate, num=len(self.result_data))
                    ax.plot(current_time_axis, self.result_data, linewidth=0.5, color='purple')
                else:
                    # Jeśli nie ma wyniku, wyświetl informację
                    ax.text(0.5, 0.5, 'Oczekiwanie na obliczenie', ha='center', va='center', transform=ax.transAxes)

        # Zastosuj aktualne ustawienia zoomu i przesunięcia
        self.apply_zoom_and_offset()

    def apply_zoom_and_offset(self):
        """
        Pobiera wartości z suwaków zoom/offset i ustawia odpowiedni widok na osi X.
        """
        try:
            # Pobierz wartości z zmiennych Tkinter
            zoom = self.zoom_ms_var.get()
            offset_coarse = self.offset_coarse_ms_var.get()
            offset_fine = self.offset_fine_ms_var.get()
        except tk.TclError:
            return # Zignoruj błędy, które mogą pojawić się przy zamykaniu okna

        # Dynamicznie dostosuj zakres suwaka precyzyjnego do wartości zoomu
        if self.offset_fine_slider.cget('to') != zoom:
            self.offset_fine_slider.config(to=zoom)

        # Oblicz początek i koniec widocznego fragmentu w sekundach
        start_sec = (offset_coarse + offset_fine) / 1000.0
        end_sec = start_sec + (zoom / 1000.0)

        # Upewnij się, że nie wyjdziemy poza zakres danych
        max_time = self.duration_ms / 1000.0
        if self.result_data is not None:
            max_time = max(max_time, len(self.result_data) / self.samplerate)
        if end_sec > max_time:
            end_sec = max_time

        # Ustaw limity dla osi X i Y na pierwszym (i przez to na wszystkich) wykresie
        self.axs[0].set_xlim(start_sec, end_sec)
        self.axs[0].set_ylim(-1.1, 1.1)
        self.canvas.draw() # Odśwież widok

    # -------------------
    # METODY OBSŁUGI PLIKÓW I DANYCH
    # -------------------
    def select_file(self):
        """
        Otwiera okno dialogowe do wyboru pliku .wav.
        """
        filepath = filedialog.askopenfilename(title="Wybierz plik audio", filetypes=[("Pliki WAV", "*.wav")])
        if not filepath: return # Jeśli użytkownik zamknął okno, nic nie rób
        self.filepath = filepath
        self.filepath_label.config(text=self.filepath)
        self.load_and_plot_wav() # Wczytaj i narysuj dane z wybranego pliku

    def load_and_plot_wav(self):
        """
        Wczytuje dane z pliku .wav, normalizuje je i aktualizuje interfejs.
        """
        if not self.filepath: return
        try:
            # Wczytaj plik WAV. Zwraca częstotliwość próbkowania i dane.
            self.samplerate, self.audio_data = wavfile.read(self.filepath)
        except Exception as e:
            messagebox.showerror("Błąd odczytu", f"Nie można wczytać pliku:\n{e}")
            return
        if self.audio_data.size == 0:
            messagebox.showwarning("Pusty plik", "Plik audio nie zawiera danych.")
            return

        # --- Normalizacja danych ---
        # Przekształca dane z formatu całkowitoliczbowego (np. int16, zakres -32768 do 32767)
        # na format zmiennoprzecinkowy (float) w zakresie od -1.0 do 1.0.
        # Jest to standardowa procedura w cyfrowym przetwarzaniu sygnałów.
        if np.issubdtype(self.audio_data.dtype, np.integer):
            max_val = np.iinfo(self.audio_data.dtype).max
            self.normalized_audio_data = self.audio_data.astype(np.float32) / max_val
        else: # Jeśli dane są już w formacie float
            self.normalized_audio_data = self.audio_data

        # Upewnij się, że dane mają 2 wymiary (próbki, kanały), nawet dla plików mono
        if self.normalized_audio_data.ndim == 1:
            self.normalized_audio_data = np.expand_dims(self.normalized_audio_data, axis=1)

        # --- Aktualizacja interfejsu ---
        num_samples, num_channels = self.normalized_audio_data.shape
        self.duration_ms = (num_samples / self.samplerate) * 1000
        self.status_label.config(text=f"Próbkowanie: {self.samplerate} Hz | Kanały: {num_channels} | Długość: {self.duration_ms/1000:.2f} s")

        # Zaktualizuj listy wyboru kanałów w całym interfejsie
        ch_list = [str(i+1) for i in range(num_channels)]
        self.ch_A_cb['values'] = ch_list
        self.ch_B_cb['values'] = ch_list
        self.convolution_channel_cb['values'] = ch_list
        self.spec_ch_A_cb['values'] = ch_list
        self.spec_ch_B_cb['values'] = ch_list
        # Ustaw domyślne wartości dla list wyboru
        if num_channels > 0:
            self.analysis_ch_A_var.set(1)
            self.convolution_source_channel_var.set(1)
            self.spectrogram_ch_A_var.set(1)
        if num_channels > 1:
            self.analysis_ch_B_var.set(2)
            self.spectrogram_ch_B_var.set(2)
        else: # Dla plików mono, oba kanały analizy to kanał 1
            self.analysis_ch_B_var.set(1)
            self.spectrogram_ch_B_var.set(1)

        # Skonfiguruj suwaki nawigacji na podstawie długości pliku
        default_zoom = 20.0; max_zoom = self.duration_ms
        default_offset = self.duration_ms / 2.0
        # Zaktualizuj zakresy suwaków
        self.zoom_slider.config(to=max_zoom)
        self.zoom_ms_var.set(default_zoom if self.duration_ms > default_zoom else self.duration_ms)
        self.offset_coarse_slider.config(to=self.duration_ms)
        self.offset_coarse_ms_var.set(default_offset)
        self.offset_fine_slider.config(to=default_zoom)
        self.offset_fine_ms_var.set(0)

        # Zaktualizuj suwaki do wyboru fragmentu analizy
        self.analysis_start_slider.config(to=self.duration_ms)
        self.analysis_end_slider.config(to=self.duration_ms)
        self.analysis_start_ms_var.set(0.0)
        self.analysis_end_ms_var.set(self.duration_ms)

        # Zresetuj dane wynikowe i przerysuj wykresy
        self.result_data = None
        self.redraw_plots()

    def get_modified_channel_data(self, index):
        """
        Zwraca dane dla wybranego kanału, uwzględniając modyfikacje
        (odwrócenie fazy, przesunięcie w czasie).
        :param index: Indeks kanału (0, 1, 2...)
        :return: Tablica NumPy z danymi kanału lub None, jeśli kanał nie istnieje.
        """
        if self.normalized_audio_data is None or not (0 <= index < self.normalized_audio_data.shape[1]):
            return None

        # Stwórz kopię, aby nie modyfikować oryginalnych danych
        channel_data = self.normalized_audio_data[:, index].copy()

        # Odwrócenie fazy (mnożenie przez -1)
        if self.invert_phase_vars[index].get():
            channel_data *= -1

        # Przesunięcie w czasie (cykliczne przesunięcie próbek w tablicy)
        time_shift_ms = self.phase_shift_vars[index].get()
        if time_shift_ms != 0:
            shift_samples = int((time_shift_ms / 1000.0) * self.samplerate)
            channel_data = np.roll(channel_data, shift_samples)

        return channel_data

    # -------------------
    # METODY OBLICZENIOWE
    # -------------------
    def recalculate_analysis(self):
        """
        Wykonuje główną analizę (odejmowanie lub dekonwolucję) na podstawie
        wybranych kanałów i trybu.
        """
        if self.normalized_audio_data is None: return

        # Pobierz indeksy kanałów A i B (pamiętaj, że użytkownik widzi 1, 2.., a my potrzebujemy 0, 1..)
        ch_A_idx, ch_B_idx = self.analysis_ch_A_var.get() - 1, self.analysis_ch_B_var.get() - 1

        # Pobierz zmodyfikowane dane dla obu kanałów
        ch_A = self.get_modified_channel_data(ch_A_idx)
        ch_B = self.get_modified_channel_data(ch_B_idx)

        if ch_A is None or ch_B is None:
            messagebox.showerror("Błąd", "Wybrano nieprawidłowe kanały.")
            return

        mode = self.analysis_mode_var.get()
        try:
            if mode == "Odejmowanie A-B":
                # Proste odejmowanie "próbka po próbce"
                self.result_data = ch_A - ch_B
            elif mode == "Dekonwolucja iFFT(FFT[A]/FFT[B])":
                # Dekonwolucja w dziedzinie częstotliwości.
                # Jest to operacja odwrotna do splotu, która pozwala np. "usunąć"
                # wpływ pomieszczenia (B) z nagrania z mikrofonu (A).
                epsilon = 1e-12 # Mała wartość, aby uniknąć dzielenia przez zero
                fft_A = np.fft.fft(ch_A)
                fft_B = np.fft.fft(ch_B)
                # Wynik jest liczbą zespoloną, bierzemy tylko część rzeczywistą
                self.result_data = np.real(np.fft.ifft(fft_A / (fft_B + epsilon)))

            # Normalizuj wynik, aby mieścił się w zakresie [-1, 1]
            if self.result_data is not None:
                max_abs = np.max(np.abs(self.result_data))
                if max_abs > 0:
                    self.result_data /= max_abs
        except Exception as e:
            messagebox.showerror("Błąd Obliczeń", f"Wystąpił błąd:\n{e}")
            self.result_data = None

        self.redraw_plots() # Pokaż wynik na wykresie

    def select_ir_file(self):
        """
        Otwiera okno dialogowe do wyboru pliku z odpowiedzią impulsową (IR)
        dla operacji splotu.
        """
        filepath = filedialog.askopenfilename(title="Wybierz plik odpowiedzi impulsowej (IR)", filetypes=[("Pliki WAV", "*.wav")])
        if not filepath: return
        try:
            self.convolution_ir_samplerate, ir_data_raw = wavfile.read(filepath)
            # Sprawdź, czy częstotliwości próbkowania są zgodne
            if self.samplerate and self.convolution_ir_samplerate != self.samplerate:
                messagebox.showwarning("Niezgodne próbkowanie", f"Plik IR ma inne próbkowanie ({self.convolution_ir_samplerate} Hz) niż plik główny ({self.samplerate} Hz). Wynik może być nieprawidłowy.")

            # Znormalizuj dane IR
            if np.issubdtype(ir_data_raw.dtype, np.integer):
                max_val = np.iinfo(ir_data_raw.dtype).max
                ir_data = ir_data_raw.astype(np.float32) / max_val
            else:
                ir_data = ir_data_raw
            # Jeśli plik IR jest stereo, uśrednij kanały
            if ir_data.ndim > 1:
                ir_data = ir_data.mean(axis=1)

            self.convolution_ir_data = ir_data
            self.ir_path_label.config(text=os.path.basename(filepath))
        except Exception as e:
            messagebox.showerror("Błąd odczytu pliku IR", f"Nie można wczytać pliku:\n{e}")

    def apply_convolution(self):
        """
        Wykonuje operację splotu (konwolucji) sygnału źródłowego z wczytaną
        odpowiedzią impulsową (IR). Pozwala to nałożyć na sygnał charakterystykę
        innego systemu, np. pogłos (reverb) sali koncertowej.
        """
        if self.normalized_audio_data is None:
            messagebox.showwarning("Brak danych", "Najpierw wczytaj główny plik audio.")
            return
        if self.convolution_ir_data is None:
            messagebox.showwarning("Brak danych", "Najpierw wczytaj plik z odpowiedzią impulsową (IR).")
            return

        source_ch_idx = self.convolution_source_channel_var.get() - 1
        source_data = self.get_modified_channel_data(source_ch_idx)
        if source_data is None:
            messagebox.showerror("Błąd", "Wybrano nieprawidłowy kanał źródłowy.")
            return

        try:
            # Użyj funkcji `convolve` z biblioteki SciPy.
            # Metoda 'fft' jest szybsza dla długich sygnałów.
            # Tryb 'full' zwraca cały wynik splotu.
            convolved_signal = signal.convolve(source_data, self.convolution_ir_data, mode='full', method='fft')

            # Znormalizuj wynik
            max_abs = np.max(np.abs(convolved_signal))
            if max_abs > 0:
                convolved_signal /= max_abs

            self.result_data = convolved_signal
            self.redraw_plots()
        except Exception as e:
            messagebox.showerror("Błąd konwolucji", f"Wystąpił błąd podczas obliczeń:\n{e}")
            
    # -------------------
    # METODY ANALIZY SPEKTRALNEJ
    # -------------------
    def _get_spectral_analysis_params(self):
        """
        Funkcja pomocnicza, która zbiera i waliduje wszystkie parametry potrzebne
        do analizy spektralnej (FFT, spektrogram, statystyki głosu) z interfejsu.
        :return: Krotka z parametrami lub None w przypadku błędu.
        """
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
            
            # Przelicz czas w ms na indeksy próbek
            start_sample = int(start_ms / 1000 * self.samplerate)
            end_sample = int(end_ms / 1000 * self.samplerate)

            # Wytnij odpowiednie fragmenty danych
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
        """
        Oblicza i wyświetla spektrogramy dla dwóch wybranych kanałów w nowym oknie.
        Spektrogram pokazuje, jak zmienia się widmo częstotliwości sygnału w czasie.
        """
        params = self._get_spectral_analysis_params()
        if params is None: return
        ch_A_idx, ch_B_idx, ch_A_data, ch_B_data, f_min, f_max, _, _ = params

        # Stworzenie nowego okna
        spec_window = tk.Toplevel(self.root)
        spec_window.title(f"Analiza Spektrogramu (Kanały {ch_A_idx+1} i {ch_B_idx+1})")
        spec_window.geometry("1200x700")

        fig = Figure(figsize=(12, 6), dpi=100)
        ax1, ax2 = fig.subplots(1, 2, sharey=True) # Dwa wykresy obok siebie, ze wspólną osią Y
        
        def plot_single_spectrogram(ax, data, title):
            # Oblicz spektrogram za pomocą funkcji z SciPy
            # nperseg - długość okna FFT
            f, t, Sxx = signal.spectrogram(data, self.samplerate, nperseg=1024)
            # Narysuj spektrogram jako mapę kolorów. Skala w decybelach (logarytmiczna).
            mesh = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-9), shading='gouraud', rasterized=True)
            ax.set_title(title)
            ax.set_ylabel('Częstotliwość [Hz]')
            ax.set_xlabel('Czas [s]')
            return mesh

        plot_single_spectrogram(ax1, ch_A_data, f'Kanał {ch_A_idx+1}')
        mesh2 = plot_single_spectrogram(ax2, ch_B_data, f'Kanał {ch_B_idx+1}')
        
        # Ustaw zakres częstotliwości na osi Y
        ax1.set_ylim(f_min, f_max)
        # Dodaj pasek kolorów (legendę)
        fig.colorbar(mesh2, ax=ax2, format='%+2.0f dB', label='Intensywność (dB)')
        fig.tight_layout()
        
        # Osadź wykres w oknie Tkinter
        canvas = FigureCanvasTkAgg(fig, master=spec_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, spec_window)
        toolbar.update()
        canvas.get_tk_widget().pack()

    def show_fft_window(self):
        """
        Oblicza i wyświetla widmo amplitudowe (FFT) dla dwóch wybranych kanałów w nowym oknie.
        FFT pokazuje, jakie częstotliwości składają się na sygnał.
        """
        params = self._get_spectral_analysis_params()
        if params is None: return
        ch_A_idx, ch_B_idx, ch_A_data, ch_B_data, _, _, _, _ = params
        
        if len(ch_A_data) == 0:
            messagebox.showwarning("Brak danych", "Wybrany fragment nie zawiera próbek audio.")
            return

        fft_window = tk.Toplevel(self.root)
        fft_window.title(f"Analiza FFT (Kanały {ch_A_idx+1} i {ch_B_idx+1})")
        fft_window.geometry("1000x700")
        
        # Długość sygnału
        N = len(ch_A_data)
        # Oblicz Szybką Transformatę Fouriera
        fft_A = np.fft.fft(ch_A_data)
        fft_B = np.fft.fft(ch_B_data)
        
        # Przelicz wynik FFT na znormalizowaną amplitudę.
        # Bierzemy tylko pierwszą połowę wyników, bo druga jest lustrzanym odbiciem.
        yf_A = 2.0/N * np.abs(fft_A[0:N//2])
        yf_B = 2.0/N * np.abs(fft_B[0:N//2])
        # Oblicz wektor częstotliwości dla osi X
        xf = np.fft.fftfreq(N, 1 / self.samplerate)[:N//2]
        
        # Rysowanie wykresu
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)

        ax.plot(xf, yf_A, color='orange', label=f'Kanał {ch_A_idx+1}')
        ax.plot(xf, yf_B, color='green', label=f'Kanał {ch_B_idx+1}')
        
        ax.set_title('Analiza Częstotliwości FFT')
        ax.set_xlabel('Częstotliwość [Hz]')
        ax.set_ylabel('Znormalizowana Amplituda')
        ax.set_xscale('log') # Użyj skali logarytmicznej dla częstotliwości
        ax.grid(True, which="both", ls="--")
        ax.legend()
        fig.tight_layout()
        
        # Osadzenie w oknie Tkinter
        canvas = FigureCanvasTkAgg(fig, master=fft_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, fft_window)
        toolbar.update()
        canvas.get_tk_widget().pack()
        
    # -------------------
    # METODY ANALIZY GŁOSU
    # -------------------
    def calculate_f0_autocorr(self, frame, rate):
        """
        Oblicza częstotliwość podstawową (F0, czyli wysokość głosu) dla danej ramki audio
        przy użyciu metody autokorelacji.

        :param frame: Ramka audio (fragment sygnału).
        :param rate: Częstotliwość próbkowania.
        :return: Krotka (f0, znormalizowana_korelacja).
        """
        if np.sum(np.abs(frame)) == 0: return 0, None # Pusta ramka

        # Autokorelacja to miara podobieństwa sygnału do jego przesuniętej wersji.
        # Dla sygnałów okresowych (jak głos), korelacja będzie miała maksima w miejscach
        # odpowiadających okresowi podstawowemu.
        corr = signal.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:] # Bierzemy tylko drugą połowę wyniku

        # Normalizacja
        energy = corr[0]
        if energy == 0: return 0, None
        normalized_corr = corr / energy

        # Znajdź szczyty (peaks) w sygnale korelacji
        peaks, _ = signal.find_peaks(normalized_corr)

        # Ogranicz wyszukiwanie do rozsądnego zakresu dla ludzkiego głosu (np. 75-500 Hz)
        min_period_samples = int(rate / 500) # Maks. F0 = 500 Hz
        max_period_samples = int(rate / 75)  # Min. F0 = 75 Hz
        valid_peaks = [p for p in peaks if min_period_samples < p < max_period_samples]

        if not valid_peaks: return 0, normalized_corr

        # Wybierz najsilniejszy szczyt w dozwolonym zakresie
        strongest_peak = max(valid_peaks, key=lambda p: normalized_corr[p])
        f0 = rate / strongest_peak # F0 = 1 / Okres

        return f0, normalized_corr

    def _calculate_hnr(self, normalized_corr, f0, rate):
        """
        Oblicza HNR (Harmonics-to-Noise Ratio), czyli stosunek energii harmonicznych do szumu.
        Jest to miara "czystości" lub "dźwięczności" głosu.
        """
        if f0 == 0 or normalized_corr is None: return 0
        lag = int(rate / f0) # Przesunięcie odpowiadające okresowi F0
        if lag >= len(normalized_corr): return 0
        r_T0 = normalized_corr[lag] # Wartość korelacji w punkcie T0

        # Wzór na HNR na podstawie autokorelacji
        if r_T0 >= 1.0 or r_T0 <= 0: return 0
        hnr = 10 * np.log10(r_T0 / (1 - r_T0))
        return hnr

    def _calculate_formants_lpc(self, frame, rate, num_formants=7):
        """
        Oblicza częstotliwości formantowe przy użyciu metody LPC (Linear Predictive Coding).
        Formanty to piki w widmie głosu, które odpowiadają za barwę i artykulację samogłosek.
        """
        # Rząd modelu LPC, zależny od częstotliwości próbkowania
        order = 2 + int(rate / 1000)
        if len(frame) < order: return [0] * num_formants
        
        # Preemfaza (wzmocnienie wysokich częstotliwości) i okienkowanie
        frame_emph = np.append(frame[0], frame[1:] - 0.97 * frame[:-1])
        frame_win = frame_emph * np.hamming(len(frame_emph))
        
        # Obliczenie autokorelacji dla okienkowanej ramki
        r = np.correlate(frame_win, frame_win, mode='full')[len(frame_win)-1:]
        if r[0] == 0: return [0] * num_formants
        
        try:
            # Rozwiązanie równań Yule-Walkera za pomocą macierzy Toeplitza
            # w celu znalezienia współczynników filtru LPC `a`.
            a = solve_toeplitz((r[:order], r[:order]), -r[1:order+1])
        except np.linalg.LinAlgError:
            return [0] * num_formants

        # Znalezienie pierwiastków wielomianu filtru 1/A(z)
        roots = np.roots(np.concatenate(([1], a)))
        # Interesują nas tylko pierwiastki w górnej połowie płaszczyzny zespolonej
        roots = [r for r in roots if np.imag(r) >= 0]
        
        # Przeliczenie kątów pierwiastków na częstotliwości formantów
        freqs = sorted([np.arctan2(np.imag(r), np.real(r)) * (rate / (2 * np.pi)) for r in roots])
        freqs = [f for f in freqs if f > 90] # Odrzuć nierealistycznie niskie wartości
        
        # Zwróć `num_formants` pierwszych znalezionych formantów
        formants = freqs[:num_formants]
        while len(formants) < num_formants:
            formants.append(0) # Uzupełnij zerami, jeśli znaleziono mniej
        return formants

    def _calculate_harmonic_amplitudes(self, frame_fft, f0, rate):
        """
        Oblicza amplitudy (w dB) pierwszych trzech harmonicznych (H1, H2, H3)
        na podstawie FFT ramki i jej częstotliwości podstawowej F0.
        """
        if f0 == 0: return [0, 0, 0]
        n_fft = len(frame_fft)
        freq_res = rate / (2 * n_fft) # Rozdzielczość częstotliwościowa FFT
        amplitudes_db = []
        for i in range(1, 4): # Pętla dla H1, H2, H3
            harmonic_freq = i * f0
            if harmonic_freq > rate / 2: # Częstotliwość nie może być większa niż Nyquista
                amplitudes_db.append(0)
                continue
            
            # Znajdź "koszyk" (bin) FFT najbliższy częstotliwości harmonicznej
            target_bin = int(harmonic_freq / freq_res)
            if target_bin >= n_fft:
                amplitudes_db.append(0)
                continue
            
            # Pobierz amplitudę i przelicz na decybele
            amp = np.abs(frame_fft[target_bin])
            amplitudes_db.append(20 * np.log10(amp + 1e-9))
        return amplitudes_db

    def show_voice_stats(self):
        """
        Główna funkcja do analizy głosu. Przetwarza wybrany fragment audio ramka po ramce,
        oblicza szereg parametrów (F0, Jitter, Shimmer, HNR, formanty) i wyświetla
        uśrednione wyniki w nowym oknie.
        """
        params = self._get_spectral_analysis_params()
        if params is None: return
        
        _, _, _, _, _, _, start_sample, end_sample = params
        num_channels = self.normalized_audio_data.shape[1]

        if (end_sample - start_sample) <= 0:
            messagebox.showwarning("Brak danych", "Wybrany fragment nie zawiera próbek audio.")
            return

        self.status_label.config(text="Obliczanie statystyk głosu dla 3 kanałów...")
        self.root.update_idletasks() # Odśwież interfejs, aby pokazać nową etykietę statusu

        def _analyze_channel(channel_data):
            """Funkcja wewnętrzna do analizy pojedynczego kanału."""
            # Parametry analizy ramkowej
            frame_size_ms, hop_size_ms = 40, 10
            frame_len = int(self.samplerate * frame_size_ms / 1000)
            hop_len = int(self.samplerate * hop_size_ms / 1000)

            # Słownik do przechowywania list wyników dla każdej ramki
            keys = ["f0", "hnr", "amp", "h1", "h2", "h3"] + [f"f{i+1}" for i in range(7)]
            lists = {k: [] for k in keys}
            
            # Pętla przez sygnał z krokiem `hop_len`
            for i in range(0, len(channel_data) - frame_len, hop_len):
                frame = channel_data[i : i + frame_len]
                f0, norm_corr = self.calculate_f0_autocorr(frame, self.samplerate)
                
                # Obliczenia wykonuj tylko dla ramek dźwięcznych (gdzie F0 > 0)
                if f0 > 0:
                    lists["f0"].append(f0)
                    lists["hnr"].append(self._calculate_hnr(norm_corr, f0, self.samplerate))
                    lists["amp"].append(np.sqrt(np.mean(frame**2))) # Amplituda RMS
                    
                    formants = self._calculate_formants_lpc(frame, self.samplerate, num_formants=7)
                    for fi in range(7): lists[f"f{fi+1}"].append(formants[fi])
                    
                    frame_fft = np.fft.fft(frame * np.hanning(len(frame)))
                    harmonics = self._calculate_harmonic_amplitudes(frame_fft[:frame_len//2], f0, self.samplerate)
                    lists["h1"].append(harmonics[0]); lists["h2"].append(harmonics[1]); lists["h3"].append(harmonics[2])
            
            if not lists["f0"]: return None # Jeśli nie znaleziono żadnych ramek dźwięcznych
            
            # --- Obliczanie statystyk (średnia, odchylenie itp.) ---
            stats = {"voiced_frames": len(lists["f0"])}
            
            # Oblicz średnie dla HNR, formantów i harmonicznych
            for key, values in lists.items():
                if key not in ["f0", "amp"]:
                    if values:
                        arr = np.array(values)
                        arr = arr[arr > 0] # Weź pod uwagę tylko dodatnie wartości
                        stats[f"{key}_mean"] = np.mean(arr) if len(arr) > 0 else 0
            
            # Oblicz statystyki dla F0
            f0_arr = np.array(lists["f0"])
            stats["f0_mean"] = np.mean(f0_arr); stats["f0_std"] = np.std(f0_arr)
            stats["f0_median"] = np.median(f0_arr); stats["f0_min"] = np.min(f0_arr)
            stats["f0_max"] = np.max(f0_arr)
            
            # Oblicz Jitter (zmienność F0) i Shimmer (zmienność amplitudy)
            if len(lists["f0"]) > 1:
                # Jitter - średnia względna zmiana okresu F0
                periods = 1 / f0_arr
                stats["jitter_percent"] = 100 * np.mean(np.abs(np.diff(periods))) / np.mean(periods)
                
                # Shimmer - średnia zmiana amplitudy w dB
                amp_arr = np.array(lists["amp"])
                amp_arr_prev = amp_arr[:-1]
                amp_arr_curr = amp_arr[1:]
                valid_indices = (amp_arr_prev > 1e-9) & (amp_arr_curr > 1e-9)
                if np.any(valid_indices):
                    stats["shimmer_db"] = np.mean(np.abs(20 * np.log10(amp_arr_curr[valid_indices] / amp_arr_prev[valid_indices])))
                else:
                    stats["shimmer_db"] = 0
            return stats

        # Wykonaj analizę dla każdego z pierwszych 3 kanałów
        all_stats = []
        for i in range(3):
            if i < num_channels:
                channel_data_slice = self.normalized_audio_data[start_sample:end_sample, i]
                stats = _analyze_channel(channel_data_slice)
                all_stats.append(stats)
            else:
                all_stats.append(None) # Jeśli kanał nie istnieje

        # --- Wyświetlanie wyników w nowym oknie ---
        result_window = tk.Toplevel(self.root)
        result_window.title("Wyniki analizy głosu dla kanałów 1-3")
        result_window.geometry("800x600")
        text_widget = tk.Text(result_window, wrap='none', font=("Courier New", 10))
        text_widget.pack(pady=10, padx=10, fill="both", expand=True)

        header = f"Analiza dla fragmentu: {self.analysis_start_ms_var.get():.1f}-{self.analysis_end_ms_var.get():.1f} ms\n\n"
        text_widget.insert("end", header)

        # Funkcja do formatowania pojedynczej linii tabeli
        def format_line(param, key):
            stats_ch1, stats_ch2, stats_ch3 = all_stats
            val_1 = f"{stats_ch1.get(key, 0):.2f}" if stats_ch1 else "b.d."
            val_2 = f"{stats_ch2.get(key, 0):.2f}" if stats_ch2 else "b.d."
            val_3 = f"{stats_ch3.get(key, 0):.2f}" if stats_ch3 else "b.d."
            return f"{param:<24} | {val_1:^16} | {val_2:^16} | {val_3:^16}\n"

        # Budowanie tabeli jako string
        table = f"{'Parametr':<24} | {'Kanał 1':^16} | {'Kanał 2':^16} | {'Kanał 3':^16}\n"
        table += f"{'─'*24}┼{'─'*18}┼{'─'*18}┼{'─'*18}\n"
        
        table += format_line("Średnia F0 [Hz]", "f0_mean")
        table += format_line("Odch. stand. F0 [Hz]", "f0_std")
        table += format_line("Mediana F0 [Hz]", "f0_median")
        
        # Specjalne formatowanie dla min/max
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
        
        # Wstawienie gotowego tekstu do widgetu i zablokowanie edycji
        text_widget.insert("end", table)
        text_widget.insert("end", footer)
        text_widget.config(state="disabled")
        self.status_label.config(text="Gotowy")

    # -------------------
    # METODY ODTWARZANIA I ZAPISU
    # -------------------
    def manage_playback_buttons(self, is_playing):
        """
        Zarządza stanem (aktywny/nieaktywny) przycisków odtwarzania i stop.
        """
        self.is_playing = is_playing
        for btn in self.play_buttons:
            btn.config(state="disabled" if is_playing else "normal")
        self.stop_btn.config(state="normal" if is_playing else "disabled")

    def stop_audio(self):
        """
        Zatrzymuje odtwarzanie dźwięku.
        """
        sd.stop()
        self.manage_playback_buttons(False)

    def play_audio(self, plot_index):
        """
        Odtwarza dźwięk z wybranego kanału lub z wyniku.
        :param plot_index: Indeks wykresu (0-2 dla kanałów, 3 dla wyniku).
        """
        if self.is_playing:
            self.stop_audio()
        if self.normalized_audio_data is None: return
        
        # Wybierz dane do odtworzenia
        data_to_play = None
        if plot_index < self.normalized_audio_data.shape[1]:
            # Odtwarzanie jednego z oryginalnych (zmodyfikowanych) kanałów
            data_to_play = self.get_modified_channel_data(plot_index)
        elif plot_index == 3 and self.result_data is not None:
            # Odtwarzanie wyniku
            data_to_play = self.result_data

        if data_to_play is not None:
            # Uruchom odtwarzanie w osobnym wątku, aby nie blokować interfejsu
            def playback_task():
                self.manage_playback_buttons(True)
                sd.play(data_to_play, self.samplerate)
                sd.wait() # Czekaj na zakończenie odtwarzania
                # Po zakończeniu, wróć do głównego wątku GUI, aby zaktualizować przyciski
                self.root.after(0, self.manage_playback_buttons, False)

            threading.Thread(target=playback_task, daemon=True).start()
    
    def save_result_file_4ch(self):
        """
        Zapisuje wynik jako 4-kanałowy plik WAV, zawierający:
        Kanał 1: zmodyfikowany kanał 1 z oryginału
        Kanał 2: zmodyfikowany kanał 2 z oryginału
        Kanał 3: zmodyfikowany kanał 3 z oryginału
        Kanał 4: obliczony wynik
        """
        if self.normalized_audio_data is None or self.result_data is None:
            messagebox.showwarning("Brak danych", "Najpierw wczytaj plik i oblicz wynik.")
            return
        
        num_channels_orig = self.normalized_audio_data.shape[1]
        if num_channels_orig < 3:
            messagebox.showwarning("Brak danych", "Plik źródłowy musi mieć co najmniej 3 kanały, aby zapisać wynik 4-kanałowy.")
            return

        save_path = filedialog.asksaveasfilename(title="Zapisz plik 4-kanałowy", defaultextension=".wav", filetypes=[("Pliki WAV", "*.wav")])
        if not save_path: return

        try:
            len_orig, len_res = len(self.normalized_audio_data), len(self.result_data)
            output_len = max(len_orig, len_res)
            # Stwórz pustą macierz 4-kanałową
            output_data = np.zeros((output_len, 4), dtype=np.float32)

            # Wypełnij macierz danymi, uzupełniając zerami jeśli długości są różne
            output_data[:len_orig, 0] = self.get_modified_channel_data(0)
            output_data[:len_orig, 1] = self.get_modified_channel_data(1)
            output_data[:len_orig, 2] = self.get_modified_channel_data(2)
            output_data[:len_res, 3] = self.result_data

            # Przekonwertuj dane z powrotem do formatu int16 i zapisz plik
            wavfile.write(save_path, self.samplerate, (output_data * 32767).astype(np.int16))
            messagebox.showinfo("Sukces", f"Plik został zapisany w:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Błąd zapisu", f"Nie udało się zapisać pliku:\n{e}")

    def save_result_file_1ch(self):
        """
        Zapisuje tylko kanał wynikowy jako plik mono.
        """
        if self.result_data is None:
            messagebox.showwarning("Brak danych", "Najpierw oblicz wynik.")
            return
        save_path = filedialog.asksaveasfilename(title="Zapisz kanał wynikowy (Mono)", defaultextension=".wav", filetypes=[("Pliki WAV", "*.wav")])
        if not save_path: return
        try:
            # Przekonwertuj i zapisz
            wavfile.write(save_path, self.samplerate, (self.result_data * 32767).astype(np.int16))
            messagebox.showinfo("Sukces", f"Plik został zapisany w:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Błąd zapisu", f"Nie udało się zapisać pliku:\n{e}")


# --------------------------------------------------------------------------------
# Uruchomienie aplikacji
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Ten blok kodu jest wykonywany tylko wtedy, gdy skrypt jest uruchamiany bezpośrednio
    (a nie importowany jako moduł).
    """
    # Stwórz główne okno aplikacji
    root = tk.Tk()
    # Stwórz instancję naszej klasy aplikacji
    app = WavAnalyzerApp(root)
    # Uruchom główną pętlę zdarzeń Tkinter, która czeka na akcje użytkownika
    root.mainloop()
