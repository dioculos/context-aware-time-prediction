import sys
import os
from keras.models import load_model

def print_input_shape(model_path):
    # Load the model
    model = load_model(os.getcwd() + '/' + model_path)

    # Get the input shape
    input_shape = model.input_shape

    # Print the input shape
    print("Input shape:", input_shape)

if __name__ == "__main__":
    # Check if the path argument is provided
    if len(sys.argv) < 2:
        print("Please provide the path to the model file as a command-line argument.")
        sys.exit(1)

    # Get the path from the command-line argument
    model_path = sys.argv[1]

    # Call the function with the provided model path
    print_input_shape(model_path)