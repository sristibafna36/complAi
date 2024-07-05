import pandas as pd
import os

# Load the CSV file
csv_file_path = 'E:/Bulb/Salary/RBI_Directions/ALL_PDFs/updated_file-Copy.csv'  # Replace with your actual CSV file path
df = pd.read_csv(csv_file_path)

# Define the directory where the PDFs are stored
pdf_directory = 'E:/Bulb/Salary/RBI_Directions/ALL_PDFs'  # Replace with the correct path to your PDF directory

# Define the function to check PDF existence
def check_pdf_existence(filename, directory):
    if isinstance(filename, str):
        return os.path.exists(os.path.join(directory, filename))
    return False

# Check if the PDF files exist and add a column indicating their existence
df['Exists'] = df['PDF Filename'].apply(lambda x: check_pdf_existence(x, pdf_directory))

# Add a column 'Status' indicating "does not exist" if the file is not found
df['Status1'] = df['Exists'].apply(lambda x: '' if x else 'does not exist')

# Drop the 'Exists' column as it's not needed for the final output
df.drop(columns=['Exists'], inplace=True)

# Save the updated DataFrame to a new CSV file
output_csv_file_path = 'E:/Bulb/Salary/RBI_Directions/ALL_PDFs/updated_file1.csv'  # Replace with your desired output file path
df.to_csv(output_csv_file_path, index=False)

print("Updated DataFrame saved to", output_csv_file_path)
