import time
from inference_core import process_pipeline

print("\nEnter English text to translate into Hindi (type 'quit' to exit):")

while True:
    source_input = input("> ")
    if source_input.lower() == "quit":
        break

    if source_input.strip():
        start = time.time()
        result = process_pipeline(source_input)
        end = time.time()

        print(f"Translated (Hindi): {result}")
        print(f"Processing Time: {round(end-start, 2)} sec\n")
