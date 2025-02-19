import tensorflow as tf
from PIL import Image
import numpy as np

model = None
interpreter = None
input_index = None
output_index = None
class_names = ["Early Blight", "Late Blight", "Healthy"]


def predict_using_tflite_model(image):
    test_image = np.expand_dims(image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, test_image)
    interpreter.invoke()
    output = interpreter.tensor(output_index)
    predictions = output()[0]
    print(predictions)

    predicted_class = class_names[np.argmax(predictions)]
    confidence = round(100 * (np.max(predictions)), 2)
    return predicted_class, confidence


def predict_lite(request):
    global interpreter
    global input_index
    global output_index

    interpreter = tf.lite.Interpreter(model_path="/tmp/potato-model.tflite")
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    image = request.files["file"]

    image = np.array(Image.open(image).convert("RGB").resize((256, 256)))[:, :, ::-1]
    predicted_class, confidence = predict_using_tflite_model(image)
    return {"class": predicted_class, "confidence": confidence}
