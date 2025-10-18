from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/hrm/webhook", methods=["GET", "POST"])
def hrm_webhook():
    if request.method == "GET":
        return "✅ HRM Webhook endpoint is ready (awaiting POST data)"
    
    # POST
    data = request.get_json()
    print("[HRM RECEIVED]", data)
    return jsonify({"status": "received", "employee": data.get("employee_id")})

@app.route("/", methods=["GET"])
def index():
    return "✅ HRM Webhook Server is running!"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9000)
