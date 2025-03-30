import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import os

BG_COLOR = "#f0f0f0"
PRIMARY_COLOR = "#2c3e50"
SECONDARY_COLOR = "#3498db"
ACCENT_COLOR = "#e74c3c"
TEXT_COLOR = "#333333"


class TrafficSignPredictor:
    def __init__(self, model_path):
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file '{model_path}' not found")

            self.model = tf.keras.models.load_model(model_path)
            self.categories = {
                0: "Speed limit (20km/h)",
                1: "Speed limit (30km/h)",
                2: "Speed limit (50km/h)",
                3: "Speed limit (60km/h)",
                4: "Speed limit (70km/h)",
                5: "Speed limit (80km/h)",
                6: "End of speed limit (80km/h)",
                7: "Speed limit (100km/h)",
                8: "Speed limit (120km/h)",
                9: "No passing",
                10: "No passing for vehicles over 3.5 tons",
                11: "Right-of-way at next intersection",
                12: "Priority road",
                13: "Yield",
                14: "Stop",
                15: "No vehicles",
                16: "Vehicles over 3.5 tons prohibited",
                17: "No entry",
                18: "General caution",
                19: "Dangerous curve left",
                20: "Dangerous curve right",
                21: "Double curve",
                22: "Bumpy road",
                23: "Slippery road",
                24: "Road narrows on the right",
                25: "Road work",
                26: "Traffic signals",
                27: "Pedestrians",
                28: "Children crossing",
                29: "Bicycles crossing",
                30: "Beware of ice/snow",
                31: "Wild animals crossing",
                32: "End of all speed and passing limits",
                33: "Turn right ahead",
                34: "Turn left ahead",
                35: "Ahead only",
                36: "Go straight or right",
                37: "Go straight or left",
                38: "Keep right",
                39: "Keep left",
                40: "Roundabout mandatory",
                41: "End of no passing",
                42: "End of no passing by vehicles over 3.5 tons"
            }
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            raise

    def predict_image(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None, "Invalid image", 0.0

            image = cv2.resize(image, (30, 30))
            image = image / 255.0
            image = np.expand_dims(image, axis=0)

            prediction = self.model.predict(image)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)

            return predicted_class, self.categories.get(predicted_class, "Unknown"), confidence

        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None, "Prediction failed", 0.0


class TrafficSignGUI:
    def __init__(self, root, model_path="model.h5"):
        self.root = root
        self.setup_window()
        try:
            self.predictor = TrafficSignPredictor(model_path)
            self.create_widgets()
        except Exception:
            root.destroy()

    def setup_window(self):
        self.root.title("Traffic Sign Classifier")
        self.root.geometry("600x700")
        self.root.configure(bg=BG_COLOR)
        self.root.resizable(False, False)

        # Add some padding around the window
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(sticky="nsew")

        header = ttk.Label(
            main_frame,
            text="Traffic Sign Recognition",
            font=("Helvetica", 18, "bold"),
            foreground=PRIMARY_COLOR
        )
        header.grid(row=0, column=0, pady=(0, 20))

        # Image display frame
        img_frame = ttk.LabelFrame(main_frame, text="Preview", padding=10)
        img_frame.grid(row=1, column=0, pady=10, sticky="ew")

        self.image_label = ttk.Label(img_frame)
        self.image_label.pack()

        self.upload_btn = ttk.Button(
            main_frame,
            text="Upload Traffic Sign Image",
            command=self.upload_image,
            style="Accent.TButton"
        )
        self.upload_btn.grid(row=2, column=0, pady=20, ipadx=10, ipady=5)

        results_frame = ttk.LabelFrame(main_frame, text="Results", padding=10)
        results_frame.grid(row=3, column=0, pady=10, sticky="ew")

        self.result_label = ttk.Label(
            results_frame,
            text="Upload an image to get started",
            font=("Helvetica", 12),
            wraplength=400,
            justify="center"
        )
        self.result_label.pack()

        self.confidence_meter = ttk.Progressbar(
            results_frame,
            orient="horizontal",
            length=300,
            mode="determinate"
        )
        self.confidence_meter.pack(pady=10)

        ttk.Label(
            main_frame,
            text="AI Traffic Sign Classifier v1.0",
            foreground=TEXT_COLOR,
            font=("Helvetica", 8)
        ).grid(row=4, column=0, pady=(20, 0))

        # Configure styles
        self.configure_styles()

    def configure_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Configure colors
        style.configure("TFrame", background=BG_COLOR)
        style.configure("TLabel", background=BG_COLOR, foreground=TEXT_COLOR)
        style.configure("TButton", foreground="white", background=SECONDARY_COLOR)
        style.configure("Accent.TButton", foreground="white", background=ACCENT_COLOR)
        style.configure("TLabelframe", background=BG_COLOR)
        style.configure("TLabelframe.Label", background=BG_COLOR)
        style.configure("Horizontal.TProgressbar", troughcolor=BG_COLOR, background="#2ecc71")

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            try:
                # Display image
                image = Image.open(file_path)
                image.thumbnail((400, 400))
                photo = ImageTk.PhotoImage(image)
                self.image_label.configure(image=photo)
                self.image_label.image = photo

                # Get prediction
                class_id, class_name, confidence = self.predictor.predict_image(file_path)
                if class_id is not None:
                    self.result_label.configure(
                        text=f"Prediction: {class_name}\n(ID: {class_id})",
                        foreground=PRIMARY_COLOR
                    )
                    self.confidence_meter["value"] = confidence * 100

                    if confidence > 0.8:
                        self.confidence_meter.configure(style="Green.Horizontal.TProgressbar")
                    elif confidence > 0.5:
                        self.confidence_meter.configure(style="Yellow.Horizontal.TProgressbar")
                    else:
                        self.confidence_meter.configure(style="Red.Horizontal.TProgressbar")

            except Exception as e:
                self.result_label.configure(
                    text=f"Error: {str(e)}",
                    foreground=ACCENT_COLOR
                )
                self.confidence_meter["value"] = 0


if __name__ == "__main__":
    root = tk.Tk()

    style = ttk.Style()
    style.configure("Green.Horizontal.TProgressbar", background="#2ecc71")
    style.configure("Yellow.Horizontal.TProgressbar", background="#f39c12")
    style.configure("Red.Horizontal.TProgressbar", background="#e74c3c")

    MODEL_FILE = "best_model.h5"

    if not os.path.exists(MODEL_FILE):
        messagebox.showerror(
            "Missing Model",
            f"Model file '{MODEL_FILE}' not found.\nPlease train the model first."
        )
    else:
        app = TrafficSignGUI(root, MODEL_FILE)
        root.mainloop()