# Tense_Sense

Tense_Sense is a user-friendly GUI application designed to classify the tense of a given sentence into one of three categories: Past, Present, or Future. The application leverages a fine-tuned multilingual BERT model for sequence classification to provide accurate predictions.

---

## Features

- **Interactive GUI**: Built using `tkinter`, the application provides a simple and intuitive interface for users.
- **Multilingual Support**: Utilizes `bert-base-multilingual-cased` to handle input in multiple languages.
- **Real-Time Predictions**: Classifies sentences into tense categories and displays the results instantly.
- **Feedback Mechanism**: Allows users to provide feedback on predictions, enhancing the user experience and potentially supporting model improvement in the future.

---

## Requirements

To run this project, ensure you have the following installed:

- Python 3.8 or higher
- PyTorch
- Transformers library
- tkinter (comes pre-installed with Python)
- GPU (optional but recommended for faster predictions)

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your_username/Tense_Sense.git
   cd Tense_Sense
   ```

2. Install the required Python libraries:
   ```bash
   pip install torch transformers
   ```

3. Place the pre-trained model checkpoint (`tense_classifier.pth`) in the root directory of the project. If unavailable, the application will run without a pre-trained model but may give untrained predictions.

---

## Usage

1. Run the script:
   ```bash
   python Tense_Sense.py
   ```

2. Enter a sentence in the text box and click **Submit**.

3. View the predicted tense displayed on the interface.

4. Provide feedback if the prediction is incorrect. (Note: Model updating based on feedback is currently disabled.)

---

## Project Structure

- **`Tense_Sense.py`**: The main application script.
- **`tense_classifier.pth`**: Pre-trained BERT model checkpoint (place in the root directory).

---

## Future Enhancements

- Implement real-time model updating based on user feedback.
- Extend support for batch sentence input or file-based classification.
- Add visual explanations for predictions using attention weights.
- Improve the feedback interface for better usability.

---

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.


---

## Acknowledgments

- Hugging Face's Transformers library for providing powerful NLP tools.
- PyTorch for the underlying deep learning framework.


