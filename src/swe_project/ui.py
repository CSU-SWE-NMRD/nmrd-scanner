import tkinter as tk
from tkinter import Label, filedialog
from PIL import Image, ImageTk
from swe_project import classification

def main():
    root = tk.Tk()
    root.title("Chest X-Ray Scanner")
    root.geometry("700x600")

    title = tk.Label(root, text =("Chest X-Ray Scanner"), font=("Arial", 22, "bold"))

    title.pack(pady=20)

    image_label = tk.Label(root)
    image_label.pack(pady=10)

    result_label = tk.Label(root, text = "Select image", font=("Arial", 18))
    result_label.pack(pady=20)

    def select_image():

        file_path = filedialog.askopenfilename(
            title="Select an Image", 
            filetypes=[("Image Files", "*.jpg *.jpeg *.png"),("All Files", "*.*")])

        if not file_path:
            return

        image = Image.open(file_path)
        image.thumbnail((500,400))

        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo

        result = classification.predict(file_path)
        result_label.config(text=f"Result: {result}")

    select_button = tk.Button(root, text = "Select X-Ray", font=("Arial", 14),command=select_image)
    select_button.pack(pady=10)

    def statistics():
        statistics = tk.Toplevel(root)
        statistics.title("Overall Statistics")
        statistics.geometry("1000x700")
        label = tk.Label(statistics, text = "Overall Statistics", font=("Arial", 22, "bold"))
        label.pack(pady=20)
        button = tk.Button(statistics, text = "Close", command = statistics.destroy)
        button.pack(pady=10)
        statistics.mainloop()
    button = tk.Button(root, text = "Overall Statistics", command = statistics)
    button.place(x=15, y=15)



    root.mainloop()
