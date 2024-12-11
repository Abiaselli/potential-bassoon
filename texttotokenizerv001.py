import os
from tkinter import filedialog, messagebox, Tk, ttk
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, trainers


class WordTokenizerGUI:
    def __init__(self):
        self.root = Tk()
        self.root.title("Word Tokenizer Creator")
        self.root.geometry("400x200")
        self.tokenizer = None

        self.create_widgets()
        self.root.mainloop()

    def create_widgets(self):
        ttk.Label(self.root, text="Word Tokenizer Creator", font=("Helvetica", 16)).pack(pady=10)

        # Select Text File Button
        ttk.Button(self.root, text="Select Text File", command=self.select_text_file).pack(pady=10)

        # Save Tokenizer Button
        ttk.Button(self.root, text="Save Tokenizer", command=self.save_tokenizer).pack(pady=10)

        # Quit Button
        ttk.Button(self.root, text="Quit", command=self.root.destroy).pack(pady=10)

    def select_text_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Text File for Tokenizer Training",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if not file_path:
            messagebox.showerror("Error", "No file selected.")
            return

        try:
            self.create_word_tokenizer(file_path)
            messagebox.showinfo("Success", "Tokenizer created successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create tokenizer: {e}")

    def create_word_tokenizer(self, file_path):
        # Create a word-level tokenizer
        tokenizer = Tokenizer(models.WordLevel(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        # Define a list of special tokens
        special_tokens = ["<UNK>", "<PAD>", "<BOS>", "<EOS>"]

        # Train the tokenizer on the selected text file
        trainer = trainers.WordLevelTrainer(special_tokens=special_tokens)
        tokenizer.train(files=[file_path], trainer=trainer)

        # Wrap the tokenizer with PreTrainedTokenizerFast
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="<UNK>",
            pad_token="<PAD>",
            bos_token="<BOS>",
            eos_token="<EOS>",
        )

    def save_tokenizer(self):
        if not self.tokenizer:
            messagebox.showerror("Error", "No tokenizer created. Train one first!")
            return

        save_directory = filedialog.askdirectory(title="Select Directory to Save Tokenizer")
        if not save_directory:
            messagebox.showerror("Error", "No directory selected.")
            return

        try:
            self.tokenizer.save_pretrained(save_directory)
            messagebox.showinfo("Success", f"Tokenizer saved to {save_directory}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save tokenizer: {e}")


if __name__ == "__main__":
    WordTokenizerGUI()
