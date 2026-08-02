import tkinter as tk

from ai_assignment.mvc.controller import SentimentController
from ai_assignment.mvc.model import SentimentModel
from ai_assignment.mvc.view import SentimentView


def main():
    root = tk.Tk()
    model = SentimentModel()
    view = SentimentView(root)
    SentimentController(root, model, view)
    root.mainloop()


if __name__ == "__main__":
    main()
