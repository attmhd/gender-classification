import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import asyncio

MODEL_PATH = './model/gender_model_vgg16.h5'
IMAGE_SIZE = (224, 224)


def load_gender_model(model_path=MODEL_PATH):
    """Load the gender classification model."""
    return tf.keras.models.load_model(model_path)


async def preprocess_image(image: Image.Image, target_size=IMAGE_SIZE) -> np.ndarray:
    """
    Preprocess the input image: convert to RGB, resize, convert to numpy array,
    add batch dimension, and normalize.
    This is now an async function to enable asynchronous processing.
    """
    if image is None:
        raise ValueError("No image provided")

    image = image.convert("RGB").resize(target_size)
    img_array = np.expand_dims(np.array(image), axis=0) / 255.0
    await asyncio.sleep(0)  # Yield control to event loop to simulate async behavior
    return img_array


async def predict_gender(image: Image.Image, model) -> tuple:
    """
    Predict gender from an input image asynchronously.

    Returns:
        A tuple containing (predicted_gender, confidence_score)
    """
    try:
        img_array = await preprocess_image(image)  # Await the image preprocessing
        prediction = model.predict(img_array)

        # Determine gender based on prediction probability
        prob = prediction[0][0]
        if prob > 0.5:
            return "Male", float(prob)
        else:
            return "Female", float(1 - prob)
    except ValueError as e:
        return "Error", str(e)
    except Exception as e:
        return "Error", f"An error occurred: {str(e)}"


def create_interface(model):
    """
    Create and return a Gradio interface for gender classification.
    """

    async def wrapper(image):
        return await predict_gender(image, model)

    return gr.Interface(
        fn=wrapper,
        inputs=gr.Image(type="pil", label="Upload Image"),
        outputs=[gr.Textbox(label="Predicted Gender"), gr.Textbox(label="Confidence Score")],
        live=True,
        allow_flagging="never",
        title="Gender Classification Model",
        description="Upload an image and the model will predict the gender."
    )


def main():
    model = load_gender_model()
    interface = create_interface(model)
    interface.launch(pwa=True, share=True)


if __name__ == "__main__":
    main()
