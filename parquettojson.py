import pandas as pd
import json
from tkinter import Tk, filedialog, messagebox, Label, Button, Listbox, Scrollbar, SINGLE, END

class ParquetToJsonGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Parquet to JSON Converter")

        # UI Elements
        Label(root, text="Selected Columns:").pack()

        self.column_listbox = Listbox(root, selectmode=SINGLE, width=50, height=10)
        self.column_listbox.pack()

        scrollbar = Scrollbar(root, command=self.column_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.column_listbox.config(yscrollcommand=scrollbar.set)

        Button(root, text="Load Parquet File", command=self.load_parquet).pack(pady=5)
        Button(root, text="Export to JSON", command=self.export_to_json).pack(pady=5)

        self.dataframe = None

    def load_parquet(self):
        # Open file dialog to select a Parquet file
        file_path = filedialog.askopenfilename(filetypes=[("Parquet Files", "*.parquet")])
        if not file_path:
            return

        try:
            # Load the Parquet file into a DataFrame
            self.dataframe = pd.read_parquet(file_path)
            self.column_listbox.delete(0, END)

            # Populate listbox with column names
            for column in self.dataframe.columns:
                self.column_listbox.insert(END, column)

            messagebox.showinfo("Success", "Parquet file loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load Parquet file:\n{e}")

    def export_to_json(self):
        if self.dataframe is None:
            messagebox.showwarning("Warning", "No Parquet file loaded!")
            return

        selected_columns = [self.column_listbox.get(i) for i in self.column_listbox.curselection()]
        if not selected_columns:
            messagebox.showwarning("Warning", "No columns selected!")
            return

        # Extract selected columns
        extracted_data = self.dataframe[selected_columns]

        # Open file dialog to save JSON file
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
        if not file_path:
            return

        try:
            # Export to JSON
            extracted_data.to_json(file_path, orient="records", indent=4)
            messagebox.showinfo("Success", f"Data exported to JSON file:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export JSON file:\n{e}")

if __name__ == "__main__":
    root = Tk()
    app = ParquetToJsonGUI(root)
    root.mainloop()
