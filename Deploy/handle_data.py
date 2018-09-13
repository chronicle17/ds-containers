# handle_data.py
import json
import numpy as np
import hashlib

def hashme(s):
    return int(hashlib.sha1(s).hexdigest(), 16) % (10 ** 8)

def invoke_predict(model, raw_json):
    # Decode JSON string, store as dict
    input_dict = json.loads(raw_json)#.decode())

    # Ordered feature list
    
	
    # Initialize empty list for array of inputs
    mapped_inputs = []

    for feature in feature_list:
        mapped_inputs.append(input_dict[feature])#[0])

    for hashfeature in hashfeature_list:
        mapped_inputs.append(hashme(input_dict[hashfeature]))#[0])
		
    x_to_predict = np.asarray(mapped_inputs, dtype=np.float64)
    x_to_predict = x_to_predict.reshape(1, -1)
    print(x_to_predict)

    model_output = model.predict(x_to_predict)

    prediction = json.loads(str(model_output))
    return prediction
