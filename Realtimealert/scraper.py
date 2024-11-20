import time
import sqlite3
from selenium import webdriver
from selenium.webdriver.common.by import By

# Initialize the Selenium WebDriver
driver = webdriver.Chrome()

# Connect to SQLite database
conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS rbi_updates (
    id INTEGER PRIMARY KEY,
    title TEXT,
    date TEXT,
    link TEXT,
    status TEXT,
    pdf_filename TEXT,
    last_checked TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')
conn.commit()

# Scraping function
def scrape_rbi_website():
    driver.get("https://website.rbi.org.in/web/rbi/notifications?delta=100")
    time.sleep(5)

    updates = driver.find_elements(By.XPATH, "//div[@class='notification-class']")  # Replace with actual XPath

    for update in updates:
        title = update.find_element(By.XPATH, ".//span[@class='title-class']").text
        date = update.find_element(By.XPATH, ".//span[@class='date-class']").text
        link = update.find_element(By.XPATH, ".//a").get_attribute("href")

        # Check if the entry exists in the database
        cursor.execute("SELECT * FROM rbi_updates WHERE title = ? AND date = ?", (title, date))
        result = cursor.fetchone()

        if not result:
            # New entry, add to database
            cursor.execute("INSERT INTO rbi_updates (title, date, link, status) VALUES (?, ?, ?, ?)", (title, date, link, "New"))
            conn.commit()

# Main loop to run the scraper periodically
if __name__ == "__main__":
    while True:
        scrape_rbi_website()
        print("Scraped RBI website.")
        time.sleep(60)  # Scrape every 60 seconds (adjust as needed)