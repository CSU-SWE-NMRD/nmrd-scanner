import tkinter as tk
from tkinter import Label, filedialog
from PIL import Image, ImageTk

from swe_project import classification

def main():
    root = tk.Tk()
    root.title("Chest X-Ray Scanner")
    root.geometry("1000x800")

    BG_COLOR = "#F4F7F6"
    CARD_BG = "#FFFFFF"
    PRIMARY_COLOR = "#007BFF"
    PRIMARY_HOVER = "#0056B3"
    SECONDARY_COLOR = "#6C757D"
    TEXT_MAIN = "#2C3E50"
    ACCENT_COLOR = "#17A2B8"


    root.configure(bg=BG_COLOR)
    
    
    main_frame = tk.Frame(root, bg=BG_COLOR)
    main_frame.pack(fill = tk.BOTH, expand = True, padx=24, pady=24)
    
    header_frame = tk.Frame(main_frame, bg=BG_COLOR)
    header_frame.pack(fill=tk.X, pady=(0,15))


    
    
    title_label = tk.Label(
        header_frame, 
        text =("Chest X-Ray Scanner"), 
        font=("Segoe UI", 20, "bold"),
        bg=BG_COLOR,
        fg=TEXT_MAIN
    )
    title_label.pack(side=tk.LEFT)

    control_frame = tk.Frame(main_frame, bg=BG_COLOR)
    control_frame.pack(fill=tk.X, pady=(0,15))

    file_path = ""
    
    results_card = tk.Frame(main_frame, bg=CARD_BG, bd=1, relief=tk.SOLID)
    results_card.pack(fill=tk.X, pady=(0,15), ipady=12, ipadx=15)
    
    result_label = tk.Label(
        results_card, 
        text = "", 
        font=("Segoe UI", 14, "bold"),
        bg=CARD_BG,
        fg=SECONDARY_COLOR
        )
    result_label.pack(anchor="w", padx=10, pady=2)
    
    confidence_label = tk.Label(
        results_card, 
        text = "", 
        font=("Segoe UI", 12),
        bg=CARD_BG,
        fg=SECONDARY_COLOR
        )
    confidence_label.pack(anchor="w", padx = 10, pady=2)
    
    
    image_card = tk.Frame(main_frame, bg=CARD_BG, bd=1, relief=tk.SOLID, width=520,height=320)
    image_card.pack(fill=tk.BOTH, expand=True, pady=(0,15))
    image_card.pack_propagate(False)

    image_label = tk.Label(image_card, bg=CARD_BG, text ="NO IMAGE SELECTED", font=("Segoe UI", 12),fg=SECONDARY_COLOR)
    image_label.pack(expand=True)

    footer_frame = tk.Frame(main_frame, bg=BG_COLOR)
    footer_frame.pack(fill=tk.X,pady=(5,0))
   
    def run_image():
        if not file_path:
            result_label.config(text="Please select an image first", fg="#DC3545")
            return
        result_label.config(text="Processing", fg=SECONDARY_COLOR)
        root.update_idletasks()

        try:
            result, confidence = classification.predict(file_path)
            result_label.config(text=f"Result: {result}", fg=TEXT_MAIN)
            confidence_label.config(text=f"Confidence: {confidence:.2%}")
        except Exception as e:
            result_label.config(text="Error during processing", fg="#DC3545")
            confidence_label.config(text=str(e))
    
    
    def select_image():
        nonlocal file_path

        result_label.config(text="")
        confidence_label.config(text="")

        file_path = filedialog.askopenfilename(
            title="Select an X-Ray Image", 
            filetypes=[("Image Files", "*.jpg *.jpeg *.png"),("All Files", "*.*")])

        if not file_path:
            return

        image = Image.open(file_path)
        image.thumbnail((650,420))

        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo, text ="")
        image_label.image = photo
        
    select_button = tk.Button(
        control_frame, 
        text = "Select X-Ray", 
        font=("Segoe UI", 11, "bold"),
        bg = ACCENT_COLOR,
        fg="white",
        activebackground="#138496",
        activeforeground="white",
        relief=tk.FLAT,
        padx=16,
        pady=8,
        cursor = "hand2",
        command=select_image
    )
    select_button.pack(side=tk.LEFT, padx=(0,10))

    run_button = tk.Button(
        control_frame, 
        text = "Run Scanner",
        font=("Segoe UI", 11, "bold"),
        bg=PRIMARY_COLOR,
        fg="white",
        activebackground=PRIMARY_HOVER,
        activeforeground="white",
        relief=tk.FLAT,
        padx=16,pady=8,
        cursor ="hand2",
        command = run_image
    )
    run_button.pack(side=tk.LEFT)

    retrain_button = tk.Button(
        footer_frame, 
        text = "Retrain Model",
        font=("Segoe UI", 10),
        bg="#E2E6EA",
        fg=TEXT_MAIN,
        activebackground="#CBD3DA",
        relief=tk.FLAT,
        padx=12,
        pady=6,
        cursor="hand2",
        command = classification.retrain_model
    )
    retrain_button.pack(side=tk.RIGHT)

    
    
    root.mainloop()
if __name__ == "__main__":
    main()