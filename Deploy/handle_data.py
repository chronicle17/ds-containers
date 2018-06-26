# handle_data.py
import json
import numpy as np

def invoke_predict(model, raw_json):
    # Decode JSON string, store as dict
    input_dict = json.loads(raw_json)#.decode())

    # Ordered feature list
    feature_list = ['sepal_len', 'sepal_width', 'petal_len', 'petal_width']

    # Initialize empty list for array of inputs
    mapped_inputs = []

    for feature in feature_list:
        mapped_inputs.append(input_dict[feature])#[0])
		
    x_to_predict = np.asarray(mapped_inputs, dtype=np.float64)
    x_to_predict = x_to_predict.reshape(1, -1)
    print(x_to_predict)

    model_output = model.predict(x_to_predict)

    prediction = json.loads(str(model_output))
    return prediction