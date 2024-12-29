import tkinter as tk
from tkinter import messagebox
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Define constants
MAX_LEN = 32
MODEL_SAVE_PATH = 'tense_classifier.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizer and BERT model for sequence classification (multilingual)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3).to(DEVICE)

# Load model function
def load_model(filepath=MODEL_SAVE_PATH):
    try:
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {filepath}.")
    except FileNotFoundError:
        print(f"No checkpoint found at {filepath}. Starting from scratch.")

load_model()

# Label mapping
label_map_reverse = {0: "Past", 1: "Present", 2: "Future"}

# Create GUI
class TenseBotGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TenseSense")
        self.geometry("400x200")
        
        self.label = tk.Label(self, text="Enter a sentence:")
        self.label.pack(pady=10)
        
        self.text_entry = tk.Entry(self, width=50)
        self.text_entry.pack(pady=5)
        
        self.submit_button = tk.Button(self, text="Submit", command=self.predict_tense)
        self.submit_button.pack(pady=10)
        
        self.result_label = tk.Label(self, text="")
        self.result_label.pack(pady=10)
        
    def predict_tense(self):
        sentence = self.text_entry.get()
        
        # Tokenize and prepare input
        encoding = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)

        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, prediction = torch.max(outputs.logits, dim=1)
        
        predicted_tense = label_map_reverse[prediction.item()]
        self.result_label.config(text=f"The sentence is in the {predicted_tense} tense.")
        
        # Ask for feedback if prediction is wrong
        feedback = messagebox.askquestion("Feedback", "Is this correct?")
        if feedback == 'no':
            correct_label = messagebox.askquestion("Correction", "Please provide the correct tense (Past/Present/Future): ")
            if correct_label in label_map_reverse.values():
                correct_label_idx = list(label_map_reverse.values()).index(correct_label)
                correct_label_tensor = torch.tensor([correct_label_idx]).to(DEVICE)
                # Here you can update the model if needed
                # For now, we'll just print a message
                print(f"Correct label received: {correct_label}, but training is disabled.")

# Run the GUI
if __name__ == "__main__":
    app = TenseBotGUI()
    app.mainloop()

