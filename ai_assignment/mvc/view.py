import tkinter as tk
from tkinter import ttk


class SentimentView:
    def __init__(self, root):
        self.root = root
        self.root.title("Restaurant Review Sentiment Tester")
        self.root.geometry("760x520")
        self.root.minsize(640, 460)

        self.status_var = tk.StringVar(value="Loading model...")
        self.prediction_var = tk.StringVar(value="Prediction will appear here")
        self.cleaned_var = tk.StringVar(value="")

        self.predict_button = None
        self.review_text = None

        self.configure_style()
        self.build_layout()

    def configure_style(self):
        self.root.configure(bg="#f5f7fb")

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#f5f7fb")
        style.configure("TLabel", background="#f5f7fb", foreground="#182033")
        style.configure("TButton", font=("Segoe UI", 10), padding=(12, 8))

    def build_layout(self):
        main = ttk.Frame(self.root, padding=24)
        main.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(
            main,
            text="Restaurant Review Sentiment Tester",
            font=("Segoe UI", 20, "bold"),
        )
        title.pack(anchor=tk.W)

        subtitle = ttk.Label(
            main,
            text="Type a restaurant review and test whether the trained model predicts Positive or Negative.",
            font=("Segoe UI", 10),
        )
        subtitle.pack(anchor=tk.W, pady=(4, 18))

        input_label = ttk.Label(main, text="Review Text", font=("Segoe UI", 11, "bold"))
        input_label.pack(anchor=tk.W)

        self.review_text = tk.Text(
            main,
            height=7,
            wrap=tk.WORD,
            font=("Segoe UI", 11),
            padx=12,
            pady=12,
            relief=tk.SOLID,
            borderwidth=1,
        )
        self.review_text.pack(fill=tk.BOTH, expand=True, pady=(6, 14))
        self.review_text.insert(
            "1.0",
            "The food was absolutely delicious and the service was amazing!",
        )

        controls = ttk.Frame(main)
        controls.pack(fill=tk.X)

        self.predict_button = ttk.Button(
            controls,
            text="Predict Sentiment",
            state=tk.DISABLED,
        )
        self.predict_button.pack(side=tk.LEFT)

        self.clear_button = ttk.Button(controls, text="Clear")
        self.clear_button.pack(side=tk.LEFT, padx=(10, 0))

        status_label = ttk.Label(
            controls,
            textvariable=self.status_var,
            font=("Segoe UI", 10),
        )
        status_label.pack(side=tk.RIGHT)

        result_frame = ttk.Frame(main, padding=(0, 20, 0, 0))
        result_frame.pack(fill=tk.X)

        result_label = ttk.Label(
            result_frame,
            textvariable=self.prediction_var,
            font=("Segoe UI", 18, "bold"),
        )
        result_label.pack(anchor=tk.W)

        cleaned_label = ttk.Label(
            result_frame,
            textvariable=self.cleaned_var,
            wraplength=700,
            font=("Segoe UI", 10),
        )
        cleaned_label.pack(anchor=tk.W, pady=(8, 0))

    def get_review_text(self):
        return self.review_text.get("1.0", tk.END).strip()

    def set_status(self, text):
        self.status_var.set(text)

    def set_prediction(self, label, cleaned_text):
        self.prediction_var.set(f"Prediction: {label}")
        self.cleaned_var.set(f"Cleaned text: {cleaned_text}")

    def reset_result(self):
        self.prediction_var.set("Prediction will appear here")
        self.cleaned_var.set("")

    def clear_input(self):
        self.review_text.delete("1.0", tk.END)

    def enable_prediction(self):
        self.predict_button.config(state=tk.NORMAL)

    def disable_prediction(self):
        self.predict_button.config(state=tk.DISABLED)

    def bind_predict(self, command):
        self.predict_button.config(command=command)

    def bind_clear(self, command):
        self.clear_button.config(command=command)
