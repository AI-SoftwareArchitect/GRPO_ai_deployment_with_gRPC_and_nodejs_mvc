import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from ai import * # This import assumes ai.py has its own correct imports now.
import grpc
from concurrent import futures
import time
from typing import List
# Note: You have 'import torch' and 'import torch.nn.functional as F' duplicated below,
# but it doesn't cause an error, just redundancy. Keeping it as per your request for "tam doğru hali".
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import chat_pb2 # Corrected import
import chat_pb2_grpc # Corrected import
# ai-api/cuda.py

class ChatServicer(chat_pb2_grpc.ChatServiceServicer):
    def __init__(self, model_path='grpo_model.pt'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {model_path}...")
        self.model, self.tokenizer = load_model(model_path)
        if self.model is None:
            raise RuntimeError("Could not load model!")
        self.model.eval()
        print("Model loaded and ready for serving!")
    
    def SendMessage(self, request, context):
        try:
            # Process the incoming message
            input_text = request.message
            
            # Encode input
            input_tokens = self.tokenizer.encode(input_text, max_length=64)
            
            # Generate response
            generated_tokens = self.model.generate(
                input_tokens,
                max_new_tokens=50,
                temperature=1.0,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2
            )
            
            # Decode response
            response_text = self.tokenizer.decode(generated_tokens[len(input_tokens):])
            
            return chat_pb2.MessageResponse(reply=response_text)
        
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error generating response: {str(e)}")
            return chat_pb2.MessageResponse()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    chat_pb2_grpc.add_ChatServiceServicer_to_server(ChatServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC server started on port 50051")
    
    try:
        while True:
            time.sleep(86400)  # One day in seconds
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()