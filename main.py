import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from settings import *
from plotting import *
from site_eval import *
from lunar_crater_detector import generate_basin_detection
from datetime import datetime
from datetime import datetime, timezone

import threading
import queue


class App(tk.Tk):
    def __init__(self):
        # Setup
        super().__init__()
        self.title('Lunar Landing Site Navigator')
        self.geometry('1500x900')
        self.minsize(1500, 900)
        self.setup_style()

        self.calculator = Calculator()  # handles the maths

        # Performance optimisation variables
        self.cache = {}
        self.messages = queue.Queue()
        self.generation = 0
        self.work_lock = threading.Lock()

        # Check for completed background calculations
        self.after(150, self.poll_messages)

        self.input_frame = ttk.Frame(self, style='Card.TFrame', padding=(20, 12))

        # Text variables
        self.latitude_var = tk.StringVar()
        self.longitude_var = tk.StringVar()
        self.site_var = tk.StringVar()

        # Date selection
        self.start_date_var = tk.StringVar(value="2026-11-14")
        self.end_date_var = tk.StringVar(value="2026-12-16")

        # Dropdown
        self.site_dropdown = ttk.Combobox(
            self.input_frame,
            textvariable=self.site_var,
            values=list(SITES.keys()),
            state="readonly",
            width=18
        )

        self.site_dropdown.set("Select a site")
        self.site_dropdown.bind(
            "<<ComboboxSelected>>",
            self.select_site
        )

        # Header title
        self.header_label = ttk.Label(
            self,
            text='LUNAR LANDING SITE NAVIGATOR',
            style='Header.TLabel'
        )

        # Input elements
        self.button = ttk.Button(
            self.input_frame,
            text='Update Site',
            command=self.update_map,
            style='Primary.TButton'
        )

        self.start_date_label = ttk.Label(
            self.input_frame,
            text="Start date (YYYY-MM-DD):",
            style="Card.TLabel"
        )

        self.start_date_entry = ttk.Entry(
            self.input_frame,
            textvariable=self.start_date_var,
            width=15
        )

        self.end_date_label = ttk.Label(
            self.input_frame,
            text="End date (YYYY-MM-DD):",
            style="Card.TLabel"
        )

        self.end_date_entry = ttk.Entry(
            self.input_frame,
            textvariable=self.end_date_var,
            width=15
        )

        self.latitude_label = ttk.Label(self.input_frame, text='Latitude (°):', style='Card.TLabel')
        self.longitude_label = ttk.Label(self.input_frame, text='Longitude (°):', style='Card.TLabel')

        self.latitude_entry = ttk.Entry(self.input_frame, textvariable=self.latitude_var, width=18)
        self.longitude_entry = ttk.Entry(self.input_frame, textvariable=self.longitude_var, width=18)

        # Allow pressing Enter inside either text box to trigger update
        self.latitude_entry.bind('<Return>', lambda event: self.update_map())
        self.longitude_entry.bind('<Return>', lambda event: self.update_map())
        self.start_date_entry.bind('<Return>', lambda event: self.update_map())
        self.end_date_entry.bind('<Return>', lambda event: self.update_map())

        # Maps
        self.scaled_global_map = self.scale_photo(Image.open('LunarLandingSiteNavigator3/graphics/maps/global_map.png'), 0.25)
        self.displayed_map = self.scaled_global_map.copy()
        self.displayed_map_photo = ImageTk.PhotoImage(self.displayed_map)
        self.displayed_map_image = tk.Label(
            self,
            image=self.displayed_map_photo,
            bd=0,
            highlightthickness=3,
            highlightbackground='#1e293b',
            highlightcolor='#1e293b'
        )

        # Graphs
        self.displayed_elev_graph = self.scale_photo(Image.open('LunarLandingSiteNavigator3/graphics/graph_placeholder.png'), 1)
        self.displayed_elev_graph_photo = ImageTk.PhotoImage(self.displayed_elev_graph)
        self.displayed_elev_graph_image = tk.Label(
            self,
            image=self.displayed_elev_graph_photo,
            bd=0,
            highlightthickness=2,
            highlightbackground='#475569',
            highlightcolor='#475569'
        )

        self.displayed_crater_graph = self.scale_photo(Image.open('LunarLandingSiteNavigator3/graphics/graph_placeholder.png'), 1)
        self.displayed_crater_graph_photo = ImageTk.PhotoImage(self.displayed_crater_graph)
        self.displayed_crater_graph_image = tk.Label(
            self,
            image=self.displayed_crater_graph_photo,
            bd=0,
            highlightthickness=2,
            highlightbackground='#475569',
            highlightcolor='#475569'
        )

        # Lunar horizon graph
        self.displayed_horizon_graph = self.scale_photo(
            Image.open(
                'LunarLandingSiteNavigator3/graphics/graph_placeholder.png'
            ),
            1
        )

        self.displayed_horizon_graph_photo = ImageTk.PhotoImage(
            self.displayed_horizon_graph
        )

        self.displayed_horizon_graph_image = tk.Label(
            self,
            image=self.displayed_horizon_graph_photo,
            bd=0,
            highlightthickness=2,
            highlightbackground='#475569',
            highlightcolor='#475569'
        )

        # Horizon date/time controls
        self.current_result = None

        self.time_index_var = tk.DoubleVar(value=0)

        self.time_label = ttk.Label(
            self,
            text="Select a site to view the lunar horizon",
            style='Caption.TLabel'
        )

        self.time_slider = ttk.Scale(
            self,
            from_=0,
            to=1,
            orient='horizontal',
            variable=self.time_index_var,
            command=self.update_horizon_slider
        )

        self.crater_caption_label = ttk.Label(
            self,
            text='Local basins detected within the selected region',
            style='Caption.TLabel'
        )
        # Three equally sized graph columns
        for col in range(3):
            self.columnconfigure(col, weight=1, uniform="graph_columns")

        # Row sizing
        self.rowconfigure(2, weight=1)

        # Header
        self.header_label.grid(
            row=0, column=0, columnspan=3,
            pady=(12, 8)
        )

        # Global map
        self.displayed_map_image.grid(
            row=1, column=0, columnspan=3,
            pady=(0, 15)
        )

        # Three graph panels
        self.displayed_elev_graph_image.grid(
            row=2, column=0,
            padx=10, pady=5,
            sticky="n"
        )

        self.displayed_crater_graph_image.grid(
            row=2, column=1,
            padx=10, pady=5,
            sticky="n"
        )

        self.displayed_horizon_graph_image.grid(
            row=2, column=2,
            padx=10, pady=5,
            sticky="n"
        )

        # Date/time slider beneath the horizon graph
        self.time_label.grid(
            row=3, column=0, columnspan=3,
            pady=(8, 2)
        )

        self.time_slider.grid(
            row=4, column=0, columnspan=3,
            padx=150, pady=(0, 12),
            sticky="ew"
        )

        # Input card across the bottom
        self.input_frame.configure(
            width=850,
            height=230
        )

        self.input_frame.grid(
            row=5,
            column=0,
            columnspan=3,
            pady=(10, 20)
        )

        # Prevent the frame from automatically expanding
        self.input_frame.grid_propagate(False)

        # Input controls inside the card
        self.site_dropdown.grid(
            row=0, column=3, rowspan=2,
            padx=10, pady=5
        )

        self.latitude_label.grid(
            row=0, column=0, padx=8, pady=5, sticky="e"
        )
        self.latitude_entry.grid(
            row=0, column=1, padx=8, pady=5
        )

        self.longitude_label.grid(
            row=1, column=0, padx=8, pady=5, sticky="e"
        )
        self.longitude_entry.grid(
            row=1, column=1, padx=8, pady=5
        )

        self.button.grid(
            row=0, column=2, rowspan=2,
            padx=15, pady=5, ipady=8
        )

        self.start_date_label.grid(
            row=2, column=0, padx=8, pady=5, sticky="e"
        )
        self.start_date_entry.grid(
            row=2, column=1, padx=8, pady=5
        )

        self.end_date_label.grid(
            row=3, column=0, padx=8, pady=5, sticky="e"
        )
        self.end_date_entry.grid(
            row=3, column=1, padx=8, pady=5
        )


    def setup_style(self):
        bg_main = '#e2e8f0'      
        bg_card = '#ffffff'       
        text_dark = '#0f172a'     
        text_muted = '#334155'    
        accent = '#0284c7'        
        accent_hover = '#0369a1'  

        self.configure(bg=bg_main)

        style = ttk.Style(self)
        style.theme_use('clam')

        # Header & Captions
        style.configure('Header.TLabel', background=bg_main, foreground=text_dark, font=('Segoe UI', 14, 'bold'))
        style.configure('Caption.TLabel', background=bg_main, foreground=text_muted, font=('Segoe UI', 9, 'bold'))

        style.configure('Card.TFrame', background=bg_card, relief='solid', borderwidth=1)
        style.configure('Card.TLabel', background=bg_card, foreground=text_dark, font=('Segoe UI', 10, 'bold'))

        style.configure(
            'TEntry',
            fieldbackground='#f8fafc',
            foreground=text_dark,
            bordercolor='#64748b',
            lightcolor=accent,
            padding=5
        )

        style.configure(
            'Primary.TButton',
            background=accent,
            foreground='#ffffff',
            font=('Segoe UI', 10, 'bold'),
            padding=(18, 6),
            borderwidth=0
        )
        style.map('Primary.TButton', background=[('active', accent_hover)])

    def select_site(self, event=None):
        selected_site = self.site_var.get()

        if selected_site in SITES:
            lat, lon, elevation = SITES[selected_site]

            self.latitude_var.set(str(lat))
            self.longitude_var.set(str(lon))

    def update_map(self):
        try:
            target_lat = float(self.latitude_var.get())
            target_long = float(self.longitude_var.get())

            if not -90 <= target_lat <= 90:
                raise ValueError("Latitude must be between -90 and 90.")

            # Accept either -180..180 or 0..360 longitude input, then convert to -180..180 for the global map and local projection.
            if not -180 <= target_long <= 360:
                raise ValueError("Longitude must be between -180 and 360.")

            target_long = ((target_long + 180) % 360) - 180

        except ValueError as err:
            print(f"Invalid coordinates: {err}")
            return

        # Update map immediately
        target_pos = self.calculator.convert_coordinates(
            target_lat,
            target_long,
            self.scaled_global_map.size
        )

        self.displayed_map = self.add_target_pos_to_photo(
            self.scaled_global_map,
            target_pos
        )

        self.displayed_map_photo = ImageTk.PhotoImage(self.displayed_map)

        self.displayed_map_image.config(
            image=self.displayed_map_photo
        )

        self.displayed_map_image.image = self.displayed_map_photo

        # Start background calculations
        self.generation += 1
        generation = self.generation

        threading.Thread(
            target=self.update_graphs,
            args=(target_lat, target_long, generation),
            daemon=True
        ).start()

    def update_graphs(self, lat, long, generation):

        # Prevent simultaneous writes to the same graph files
        with self.work_lock:

            try:
                start = self.start_date_var.get().strip()
                stop = self.end_date_var.get().strip()

                key = (round(lat, 5), round(long, 5), start, stop)

                try:
                    start_date = datetime.strptime(start, "%Y-%m-%d")
                    end_date = datetime.strptime(stop, "%Y-%m-%d")

                    if start_date >= end_date:
                        raise ValueError("End date must be after start date.")

                except ValueError as err:
                    self.messages.put((generation, "error", str(err)))
                    return

                # Check whether this site has already been evaluated
                if key in self.cache:
                    result = self.cache[key]

                else:
                    elev = get_lunar_elevation(lat, long)

                    result = evaluate_site(
                        lat,
                        long,
                        elev,
                        start=start,
                        stop=stop,
                        step="60m"
                    )

                    self.cache[key] = result

                # Generate elevation graph
                fig = plot_site(
                    result,
                    title="Site Selected",
                    save="LunarLandingSiteNavigator3/graphics/site_graph.png"
                )

                if fig is not None:
                    plt.close(fig)

                # Generate crater detection
                generate_basin_detection(
                    centre_lat=lat,
                    centre_lon=long
                )

                # Notify the main GUI thread that processing has finished
                self.messages.put((generation, "success", result))

            except Exception as err:
                self.messages.put((generation, "error", str(err)))

    def add_target_pos_to_photo(self, photo, target_pos):
        new_photo = photo.copy()
        draw = ImageDraw.Draw(new_photo)

        topleft = (target_pos[0] - 6, target_pos[1] - 6)
        bottomright = (target_pos[0] + 6, target_pos[1] + 6)

        draw.ellipse((topleft, bottomright), fill='#00e5ff', outline='#0f172a', width=2)
        return new_photo

    def poll_messages(self):
        try:
            while True:
                generation, status, payload = self.messages.get_nowait()

                # Ignore results from an older site selection
                if generation != self.generation:
                    continue

                if status == "error":
                    print("Calculation failed:", payload)
                    continue

                self.current_result = payload

                self.time_slider.configure(
                    from_=0,
                    to=len(payload["jd"]) - 1
                )

                self.time_index_var.set(0)
                self.update_horizon_slider(0)

                # Update elevation graph
                self.displayed_elev_graph = self.fit_photo(
                    Image.open(
                        'LunarLandingSiteNavigator3/graphics/site_graph.png'
                    )
                )

                self.displayed_elev_graph_photo = ImageTk.PhotoImage(
                    self.displayed_elev_graph
                )

                self.displayed_elev_graph_image.config(
                    image=self.displayed_elev_graph_photo
                )

                self.displayed_elev_graph_image.image = (
                    self.displayed_elev_graph_photo
                )

                # Update crater graph
                self.displayed_crater_graph = self.fit_photo(
                    Image.open(
                        'LunarLandingSiteNavigator3/graphics/impact_basin_detection.png'
                    )
                )

                self.displayed_crater_graph_photo = ImageTk.PhotoImage(
                    self.displayed_crater_graph
                )

                self.displayed_crater_graph_image.config(
                    image=self.displayed_crater_graph_photo
                )

                self.displayed_crater_graph_image.image = (
                    self.displayed_crater_graph_photo
                )

        except queue.Empty:
            pass

        # Check again in 150 milliseconds
        self.after(150, self.poll_messages)

    def scale_photo(self, photo, scale):
        width, height = photo.size
        new_width = int(width * scale)
        new_height = int(height * (new_width / width))
        adjusted_image = photo.resize((new_width, new_height), Image.BICUBIC)
        return adjusted_image


    def fit_photo(self, photo, max_width=400, max_height=450):
        width, height = photo.size

        scale = min(
            max_width / width,
            max_height / height
        )

        new_width = int(width * scale)
        new_height = int(height * scale)

        return photo.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )


    def update_horizon_slider(self, value):
        if self.current_result is None:
            return

        index = int(float(value))
        result = self.current_result

        jd = result["jd"][index]

        selected_time = datetime.fromtimestamp(
            (jd - 2440587.5) * 86400,
            tz=timezone.utc
        )

        self.time_label.config(
            text=selected_time.strftime("%d %B %Y  |  %H:%M UTC")
        )

        fig = plot_horizon_view(result, index=index)

        plt.close(fig)

        self.displayed_horizon_graph = self.fit_photo(
            Image.open(
                'LunarLandingSiteNavigator3/graphics/horizon_view.png'
            )
        )

        self.displayed_horizon_graph_photo = ImageTk.PhotoImage(
            self.displayed_horizon_graph
        )

        self.displayed_horizon_graph_image.config(
            image=self.displayed_horizon_graph_photo
        )

        self.displayed_horizon_graph_image.image = (
            self.displayed_horizon_graph_photo
        )

class Calculator:
    def __init__(self):
        pass

    def convert_coordinates(self, lat, long, map_size):  # Converts latitude and longitude into image coordinates
        map_width, map_height = map_size

        x = int((long + 180) * (map_width / 360))
        y = int((90 - lat) * (map_height / 180))

        return x, y

if __name__ == "__main__":
    app = App()
    app.mainloop()
