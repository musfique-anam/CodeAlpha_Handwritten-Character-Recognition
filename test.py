import os
from src.predict import get_prediction

# Define your image location
target_image = "test_images/my_writing.jpg"

print(f"[INFO] Running prediction on user custom image: {target_image}")

# Call the function from src/predict.py
digit, accuracy = get_prediction(target_image)

if digit is None:
    print(f"[ERROR] Could not read image at {target_image}. Check the file name!")
else:
    # 1. Print result clearly to the screen
    print("\n==============================")
    print(f" SUCCESSFUL PREDICTION")
    print("==============================")
    print(f" Predicted Digit : {digit}")
    print(f" Confidence Score: {accuracy:.2f}%")
    print("==============================\n")

    # 2. Save the result into output.txt automatically
    with open("output.txt", "w") as f:
        f.write("=========================================\n")
        f.write("      AI HANDWRITING TEST RESULT         \n")
        f.write("=========================================\n")
        f.write(f" Tested Image    : {target_image}\n")
        f.write(f" Predicted Digit : {digit}\n")
        f.write(f" Model Confidence: {accuracy:.2f}%\n")
        f.write("=========================================\n")
    
    print("[SUCCESS] Results successfully written to output.txt!")


    #python test_my_handwriting.py