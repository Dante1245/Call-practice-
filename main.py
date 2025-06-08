import eventlet
eventlet.monkey_patch()  # MUST be at the very top before other imports

from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_socketio import SocketIO, emit
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecret")  # Use strong secret key in prod
socketio = SocketIO(app)

# Login credentials
USERNAME = "admin"
PASSWORD = "dantepass"

# Twilio credentials
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

client = Client(TWILIO_SID, TWILIO_AUTH)

# In-memory call logs; consider persistent storage for prod
call_logs = []
active_calls = {}

@app.route("/")
def dashboard():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("index.html", calls=call_logs)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        remember = request.form.get("remember")
        if username == USERNAME and password == PASSWORD:
            session["logged_in"] = True
            if remember:
                session.permanent = True
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/call", methods=["POST"])
def call():
    data = request.json
    to_number = data.get("number")
    message = data.get("message", "Hello from Dante Call Bot")

    call = client.calls.create(
        to=to_number,
        from_=TWILIO_NUMBER,
        url=f"{request.url_root}voice?msg={message}"
    )

    log = {
        "sid": call.sid,
        "to": to_number,
        "status": "initiated",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "direction": "outbound"
    }
    call_logs.append(log)
    active_calls[call.sid] = log
    socketio.emit("call_status", log)
    return jsonify({"message": "Call initiated", "sid": call.sid})

@app.route("/voice", methods=["POST", "GET"])
def voice():
    msg = request.args.get("msg", "This is a call from Dante AI bot.")
    response = VoiceResponse()
    response.say(msg, voice="Polly.Matthew", language="en-US")
    return str(response)

@app.route("/incoming", methods=["POST"])
def incoming():
    from_number = request.form.get("From")
    call_sid = request.form.get("CallSid")

    log = {
        "sid": call_sid,
        "from": from_number,
        "status": "ringing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "direction": "inbound"
    }
    call_logs.append(log)
    active_calls[call_sid] = log
    socketio.emit("call_status", log)

    response = VoiceResponse()
    response.say("Hi. Please wait while someone answers.", voice="Polly.Matthew")
    return str(response)

@app.route("/update_status", methods=["POST"])
def update_status():
    sid = request.json.get("sid")
    status = request.json.get("status")

    if sid in active_calls:
        active_calls[sid]["status"] = status
        socketio.emit("call_status", active_calls[sid])
    return jsonify({"ok": True})

@app.route("/hangup", methods=["POST"])
def hangup():
    sid = request.json.get("sid")
    try:
        client.calls(sid).update(status="completed")
        return jsonify({"message": "Call ended"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@socketio.on("connect")
def handle_connect():
    print("Client connected")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    socketio.run(app, host="0.0.0.0", port=port)
