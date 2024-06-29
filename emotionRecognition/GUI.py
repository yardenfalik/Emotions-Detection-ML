import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
from PIL import ImageTk
import numpy as np
import os

# Import your classes and methods from the provided script
from emotionsdetectionproject import Network, DenseLayer, Activation, Model

# Load the model structure
network = Network()
network.add(DenseLayer(input_size=64*64*3, layer_size=128, alpha=0.001, name="dense1"))
network.add(Activation("relu"))
network.add(DenseLayer(input_size=128, layer_size=4, alpha=0.001, name="dense2"))
network.add(Activation("softmax"))


# Load model parameters (make sure to update the path to where your parameters are saved)
model = Model(name="EmotionDetectionModel", iTrainable=network, loss="cross_entropy")
model.iTrainable.load_parameters('parameters')

class EmotionDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detector")
        self.root.geometry("400x500")
        
        self.label = tk.Label(root, text="Upload an Image to Predict Emotion")
        self.label.pack(pady=20)
        
        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)
        
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)
        
        self.result_label = tk.Label(root, text="")
        self.result_label.pack(pady=20)
        
    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        
        try:
            imageToShow = Image.open(file_path).resize((128, 128))
            image = imageToShow.resize((64, 64))
            self.display_image(imageToShow)
            
            # Convert image to numpy array
            image_array = np.array(image)
            if image_array.shape[-1] == 4:  # Remove alpha channel if exists
                image_array = image_array[..., :3]
            image_array = image_array / 255.0  # Normalize
            image_array = image_array.reshape(1, -1).T  # Reshape for the network
            
            # Predict the emotion
            prediction = model.forward_propagation(image_array)
            emotion = np.argmax(prediction)
            emotion_name = ["Angry", "Happy", "Nothing", "Sad"][emotion]
            
            self.result_label.config(text=f"Predicted Emotion: {emotion_name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {e}")
    
    def display_image(self, image):
        img = ImageTk.PhotoImage(image)
        self.image_label.config(image=img)
        self.image_label.image = img

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectorApp(root)
    root.mainloop()
