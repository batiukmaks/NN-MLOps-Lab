import logging
from flask import Flask, request
import torch
from model.dnn import DNN

app = Flask(__name__)
# Load the model saved with torch.save(model.state_dict(), filepath)
model = DNN()
model.load_state_dict(torch.load('model/model.pth'))
logging.basicConfig(filename='logs/model.log', level=logging.INFO)

def convert_to_tensor(data):
    # convert the list of inputs into a 2D tensor with shape [1, 10]
    tensor = torch.tensor([data['data']], dtype=torch.float)
    return tensor

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not isinstance(data, dict) or 'data' not in data or len(data['data']) != 10:
        return {'error': 'Bad Request'}, 400
    tensor = convert_to_tensor(data)  
    output = model(tensor)
    output_list = output.tolist()
    
    logging.info(f'Input: {data}, Output: {output_list}')  # Log the input and output
    
    return {'prediction': output_list}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)