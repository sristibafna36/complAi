from exchangelib import Credentials, Account, DELEGATE, Message, Mailbox
import pandas as pd
import os
from exchangelib import FileAttachment


# Load the Excel/CSV file
email_mapping = pd.read_csv('E:\\Bulb\\Salary\\docs.csv')  # Or pd.read_excel for Excel files

# Directory containing the PDFs
pdf_directory = 'E:\\Bulb\\Salary'

# Exchange server credentials and account details
email_address = 'teamrbih@rbihub.in'
password = 'Surface@2024'
exchange_server = 'https://mail.rbihub.in/owa'  # Adjust if needed

credentials = Credentials(email_address, password)
account = Account(email_address, credentials=credentials, autodiscover=True, access_type=DELEGATE)

# Function to send email
def send_email(to_email, subject, body, attachment_paths):
    m = Message(
        account=account,
        subject=subject,
        body=body,
        to_recipients=[Mailbox(email_address=to_email)]
    )
    
    # Attach files
    for attachment_path in attachment_paths:
        with open(attachment_path, 'rb') as f:
            file_content = f.read()
        # Create a FileAttachment and append it to the attachments list
        attachment = FileAttachment(name=os.path.basename(attachment_path), content=file_content)
        m.attachments.append(attachment)
    
    # Send the email
    m.send_and_save()


# Loop through each PDF in the directory
for pdf_file in os.listdir(pdf_directory):
    if pdf_file.endswith('.pdf'):
        # Extract the first and last name from the file name
        first_name, last_name = pdf_file.replace('.pdf', '').split('_')
        
        # Find the corresponding email
        email_row = email_mapping[(email_mapping['FirstName'].str.lower() == first_name.lower()) & 
                                  (email_mapping['LastName'].str.lower() == last_name.lower())]
        
        if not email_row.empty:
            to_email = email_row['Email'].values[0]
            subject = 'Your Sensitive Document'
            body = 'Please find attached your sensitive document.'
            attachment_paths = [os.path.join(pdf_directory, pdf_file)]
            
            # Send the email
            send_email(to_email, subject, body, attachment_paths)
            print(f'Sent {pdf_file} to {to_email}')
        else:
            print(f'No email found for {first_name} {last_name}')
