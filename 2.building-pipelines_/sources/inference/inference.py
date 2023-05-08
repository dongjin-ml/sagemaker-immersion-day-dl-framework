import io
import os
import json
import torch
import pickle
import logging
import traceback
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def model_fn(model_dir):
    
    logger.info("### model_fn ###")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net()
    # if torch.cuda.device_count() > 1:
    #     logger.info("Gpu count: {}".format(torch.cuda.device_count()))
    #     model = nn.DataParallel(model)
    #     print ("here")

    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device).eval()

def input_fn(request_body, request_content_type):
  
    logger.info("### input_fn ###")
    logger.info(f"content_type: {request_content_type}")   
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if isinstance(request_body, str): ## json
        print ("string")
    elif isinstance(request_body, io.BytesIO):
        print ("io.BytesIO")
        request_body = request_body.read()
        request_body = bytes.decode(request_body)        
    elif isinstance(request_body, bytes):
        print ("bytes")
        request_body = request_body.decode()
        
    try:
             
        if request_content_type=='application/json':
            deserialized_input = json.loads(request_body)
            input_data = deserialized_input["INPUT"]
            input_dtype = deserialized_input["DTYPE"]
            if input_dtype == "float32": dtype=torch.float32
            #input_data = torch.tensor(input_data, dtype=dtype)
        
        elif request_content_type=='application/pickle':
            deserialized_input = pickle.loads(request_body)
            input_data = deserialized_input["INPUT"]
            input_dtype = deserialized_input["DTYPE"]
            if input_dtype == "float32": dtype=torch.float32
        
        else:
            ValueError("Content type {} is not supported.".format(content_type))
            
        input_data = torch.tensor(input_data, dtype=dtype)
            
    
    except Exception:
        print(traceback.format_exc())  

    return input_data.to(device)

def predict_fn(data, model):
    
    logger.info("### predict_fn ###")    
    
    try:
        logger.info(f"#### type of input data: {type(data)}")                                  
    
        with torch.no_grad():
            predictions = model(data)
            
        predictions_numpy = predictions.detach().cpu().numpy()
      
    except Exception:
        print(traceback.format_exc())        
        
    return predictions_numpy

def output_fn(predictions, content_type="application/json"):
    
    """
    After invoking predict_fn, the model server invokes `output_fn`.
    """
    
    logger.info("### output_fn ###") 
    
    if content_type == "application/json":
        outputs = json.dumps(
            {'pred': predictions.tolist()}
        )             
        return outputs
    else:
        raise ValueError("Content type {} is not supported.".format(content_type))