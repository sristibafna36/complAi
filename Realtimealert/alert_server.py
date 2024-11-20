from flask import Flask, render_template
from flask_socketio import SocketIO
import sqlite3
import time
import threading

app = Flask(__name__)
socketio = SocketIO(app)

# Connect to the database
def get_new_updates():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM rbi_updates WHERE status = 'New'")
    updates = cursor.fetchall()
    conn.close()
    return updates

def mark_updates_as_notified():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("UPDATE rbi_updates SET status = 'Notified' WHERE status = 'New'")
    conn.commit()
    conn.close()

# Background thread to emit updates
def monitor_updates():
    while True:
        updates = get_new_updates()
        if updates:
            for update in updates:
                socketio.emit('new_update', {
                    "title": update[1],
                    "date": update[2],
                    "link": update[3]
                })
            mark_updates_as_notified()
        time.sleep(60)  # Check every 60 seconds

@app.route("/")
def index():
    return render_template("index.html")

# Start the monitoring thread
thread = threading.Thread(target=monitor_updates)
thread.start()

if __name__ == "__main__":
    socketio.run(app, debug=True)