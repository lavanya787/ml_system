from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/train', methods=['POST'])
def train_model():
    data = request.get_json()
    dataset = data.get('dataset')
    model_type = data.get('model_type')

    # You can add actual model training logic here later.
    response = {
        "message": "Training started",
        "dataset": dataset,
        "model_type": model_type
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
