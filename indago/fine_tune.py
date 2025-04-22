import json
from ollama import OllamaModel
import os
import json

# def convert_folder_to_jsonl(folder_path):
#     jsonl_data = []

#     # Iterate over each file in the folder
#     for filename in os.listdir(folder_path):
#         print(filename)
#         if filename.endswith('.json'):
#             file_path = os.path.join(folder_path, filename)

#             # Read the JSON file
#             with open(file_path, 'r') as file:
#                 data = json.load(file)

#                 # Extract relevant information
#                 # for example in data:
#                 jsonl_entry = {
#                     "input": f"Environment Config: {data['env_config']}",
#                     "output": "Success" if data["is_success"] == 1 else "Failure"
#                 }
#                 jsonl_data.append(json.dumps(jsonl_entry))

#     # Join all JSONL entries into a single string
#     return "\n".join(jsonl_data)

# # Example usage
# folder_path = 'logs/her/parking-v0_1/json_log'
# jsonl_training_data = convert_folder_to_jsonl(folder_path)

# # Optionally, save to a file
# with open('training_data.jsonl', 'w') as f:
#     f.write(jsonl_training_data)

# print("JSONL conversion complete.")

jsonl_training_data = json.load('training_data.jsonl')

# Load a pre-trained model from Ollama
model = OllamaModel.load('gemma2', url='http://localhost:7869')

# Fine-tune the model
model.fine_tune(jsonl_training_data, epochs=5, learning_rate=0.001)

# Example of generating a new environment and getting a classification
new_environment = {
    "num_lanes": 12,
    "goal_lane_idx": 10,
    "heading_ego": 0.97,
    "parked_vehicles_lane_indices": [7, 8],
    "position_ego": [1.90, -5.00]
}

# Prepare the input for the model
input_text = f"Environment Config: {new_environment}"

# Get the classification
classification = model.classify(input_text)
print(f"Classification: {classification}")
