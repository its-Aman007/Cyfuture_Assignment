import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

from .model import EmotionDetector
from . import config

class EmotionDetectorGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title(config.WINDOW_TITLE)
        self.window.geometry(config.WINDOW_SIZE)
        self.window.configure(bg=config.BG_COLOR)
        
        # Load the model
        try:
            self.model = EmotionDetector.load_model()
            self.model_loaded = True
        except:
            self.model_loaded = False
        
        self.setup_gui()
    
    def setup_gui(self):
        """
        Set up the GUI components
        """
        # Main frame
        main_frame = ttk.Frame(self.window)
        main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="Emotion Detection System",
            font=(config.FONT_FAMILY, config.FONT_SIZE_TITLE, "bold")
        )
        title_label.pack(pady=10)
        
        # Text input area
        input_label = ttk.Label(
            main_frame,
            text="Enter text to analyze:",
            font=(config.FONT_FAMILY, config.FONT_SIZE_TEXT)
        )
        input_label.pack(pady=5)
        
        self.text_input = scrolledtext.ScrolledText(
            main_frame,
            height=5,
            width=50,
            font=(config.FONT_FAMILY, config.FONT_SIZE_TEXT)
        )
        self.text_input.pack(pady=10)
        
        # Analyze button
        self.analyze_button = ttk.Button(
            main_frame,
            text="Detect Emotion",
            command=self.analyze_text
        )
        self.analyze_button.pack(pady=10)
        
        # Results area
        results_frame = ttk.LabelFrame(main_frame, text="Results")
        results_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.result_label = ttk.Label(
            results_frame,
            text="",
            font=(config.FONT_FAMILY, config.FONT_SIZE_TEXT)
        )
        self.result_label.pack(pady=10)
        
        # Create figure for visualization
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=results_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_bar = ttk.Label(
            self.window,
            text="Ready" if self.model_loaded else "Error: Model not loaded",
            font=(config.FONT_FAMILY, 10)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # If model isn't loaded, disable the analyze button
        if not self.model_loaded:
            self.analyze_button.config(state='disabled')
    
    def analyze_text(self):
        """
        Analyze the input text and display results
        """
        # Get input text
        text = self.text_input.get("1.0", tk.END).strip()
        
        if not text:
            self.show_status("Please enter some text to analyze")
            return
        
        # Disable button during analysis
        self.analyze_button.config(state='disabled')
        self.show_status("Analyzing...")
        
        # Run analysis in a separate thread to prevent GUI freezing
        thread = threading.Thread(target=self._process_text, args=(text,))
        thread.daemon = True
        thread.start()
    
    def _process_text(self, text):
        """
        Process text in a separate thread
        """
        try:
            # Get emotion and probabilities
            emotion = self.model.predict(text)
            probabilities = self.model.predict_proba(text)
            
            # Update GUI in main thread
            self.window.after(0, self._update_results, emotion, probabilities)
            
        except Exception as e:
            self.window.after(0, self.show_status, f"Error: {str(e)}")
            self.window.after(0, self._enable_button)
    
    def _update_results(self, emotion, probabilities):
        """
        Update GUI with results
        """
        # Update result label
        self.result_label.config(
            text=f"Detected Emotion: {emotion.capitalize()}"
        )
        
        # Clear previous plot
        self.ax.clear()
        
        # Create bar plot of probabilities
        emotions = list(probabilities.keys())
        probs = list(probabilities.values())
        
        bars = self.ax.bar(emotions, probs)
        
        # Customize plot
        self.ax.set_title('Emotion Probability Distribution')
        self.ax.set_xlabel('Emotions')
        self.ax.set_ylabel('Probability')
        plt.setp(self.ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Color the bars according to config
        for bar, emotion in zip(bars, emotions):
            bar.set_color(config.EMOTION_COLORS[emotion])
        
        # Adjust layout and redraw
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Update status and re-enable button
        self.show_status("Analysis complete")
        self._enable_button()
    
    def _enable_button(self):
        """
        Re-enable the analyze button
        """
        self.analyze_button.config(state='normal')
    
    def show_status(self, message):
        """
        Update status bar message
        """
        self.status_bar.config(text=message)
    
    def run(self):
        """
        Start the GUI
        """
        self.window.mainloop()

def main():
    app = EmotionDetectorGUI()
    app.run()

if __name__ == "__main__":
    main()