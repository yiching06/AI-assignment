from threading import Thread
from tkinter import messagebox


class SentimentController:
    def __init__(self, root, model, view):
        self.root = root
        self.model = model
        self.view = view

        self.view.bind_predict(self.handle_predict)
        self.view.bind_clear(self.handle_clear)
        self.root.after(200, self.load_model_async)

    def load_model_async(self):
        self.view.set_status("Loading model...")
        worker = Thread(target=self.load_model, daemon=True)
        worker.start()

    def load_model(self):
        try:
            self.model.load()
        except Exception as error:
            message = str(error)
            self.root.after(0, lambda: self.handle_model_error(message))
            return

        self.root.after(0, self.handle_model_ready)

    def handle_model_ready(self):
        self.view.set_status("Model ready")
        self.view.enable_prediction()

    def handle_model_error(self, message):
        self.view.set_status("Model failed to load")
        self.view.disable_prediction()
        messagebox.showerror("Model Error", message)

    def handle_predict(self):
        review = self.view.get_review_text()

        try:
            label, cleaned_text = self.model.predict(review)
        except ValueError:
            messagebox.showwarning("Missing Review", "Please enter a review first.")
            return
        except Exception as error:
            messagebox.showerror("Prediction Error", str(error))
            return

        self.view.set_prediction(label, cleaned_text)

    def handle_clear(self):
        self.view.clear_input()
        self.view.reset_result()
