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

    result_label = tk.Label(root, text = "", font=("Arial", 18))
    result_label.pack(padx=20)
    confidence_label = tk.Label(root, text = "", font=("Arial", 18))
    confidence_label.pack(padx=20)
    retrain_button = tk.Button(root, text = "Retrain Model", command = classification.retrain_model)
    retrain_button.pack(pady=12)
   
    def run_image():
        result, confidence = classification.predict(file_path)
        print(confidence)
        result_label.config(text=f"Result: {result}")
        confidence_label.config(text=f"Confidence: {confidence:.2%}")

        
    run_button = tk.Button(root, text = "Run Scanner",command = run_image)
    run_button.place(x=15, y= 70)
    
    file_path = ""
    def select_image():
        nonlocal file_path

        result_label.config(text="")
        confidence_label.config(text="")

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



    select_button = tk.Button(root, text = "Select X-Ray", font=("Arial", 14),command=select_image)
    select_button.pack(pady=10)
    
    root.mainloop()
