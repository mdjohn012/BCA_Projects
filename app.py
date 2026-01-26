import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("digit_model.h5")

CANVAS_SIZE = 280
IMAGE_SIZE = 28

root = tk.Tk()
root.title("Handwritten Digit Recognition")

canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black")
canvas.pack()

image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
draw = ImageDraw.Draw(image)

def draw_digit(event):
    x, y = event.x, event.y
    r = 10
    canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
    draw.ellipse((x-r, y-r, x+r, y+r), fill=255)

def clear_canvas():
    canvas.delete("all")
    draw.rectangle((0, 0, CANVAS_SIZE, CANVAS_SIZE), fill=0)
    result_label.config(text="Draw a digit")

def predict_digit():
    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28)
    prediction = model.predict(img)
    digit = np.argmax(prediction)
    result_label.config(text=f"Predicted Digit: {digit}")

canvas.bind("<B1-Motion>", draw_digit)

btn_frame = tk.Frame(root)
btn_frame.pack()

tk.Button(btn_frame, text="Predict", command=predict_digit).pack(side="left")
tk.Button(btn_frame, text="Clear", command=clear_canvas).pack(side="left")

result_label = tk.Label(root, text="Draw a digit", font=("Arial", 16))
result_label.pack()

root.mainloop()
