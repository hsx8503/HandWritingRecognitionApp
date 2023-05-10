import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageDraw, ImageGrab, ImageTk
from tkinter import filedialog
import requests


image = None
im = None
class HandWritingRecognitionApp:
    def __init__(self,window):
        self.window=window
        self.window.title("HandWriting Recogniztion")

        self.canvas = tk.Canvas(self.window, width=280, height=280, background='white')
        self.canvas.pack()

        ttk.Button(self.window, text="Select Image", command=self.upload_image).pack(side=tk.TOP)
        ttk.Button(self.window, text="Clear", command=self.clear).pack(side=tk.LEFT)
        ttk.Button(self.window, text="Recognize", command=self.recognize).pack(side=tk.RIGHT)

        self.modle=None

    def upload_image(self):
        global image
        global im

        filename = filedialog.askopenfilename()
        image = Image.open(filename)
        im = ImageTk.PhotoImage(image)

        self.canvas.create_image(280, 280, image=im)

    def clear(self):
        self.canvas.delete("all")

    def recognize(self,image):
        image=image.convert("L")
        resized_image = image.resize((32, 32))
        grayscale_array = np.array(resized_image)
        normalized_array = grayscale_array / 255.0
        flattened_array = normalized_array.flatten()
        normalized_image = Image.fromarray(flattened_array)
















if __name__=="__main__":
    app=HandWritingRecognitionApp(tk.Tk())
    app.window.mainloop()