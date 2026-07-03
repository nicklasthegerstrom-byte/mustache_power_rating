import tensorflow as tf
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image

mtcnn = MTCNN(image_size=178, margin=40, post_process=False)
mtcnn_face = MTCNN(image_size=160, margin=0, post_process=True)
facenet = InceptionResnetV1(pretrained="vggface2").eval()

mustache_model = tf.keras.models.load_model("models/mustache_detector_3.keras")
epic_model = tf.keras.models.load_model("models/epic_detector_3class_test.keras")


def prepare_image(image: Image.Image):
    image = image.convert("RGB")
    face = mtcnn(image)
    if face is None:
        return None
    arr = face.permute(1, 2, 0).numpy().astype(np.uint8)
    return np.expand_dims(arr, axis=0)


def predict(image: Image.Image):
    img_array = prepare_image(image)
    if img_array is None:
        return None

    mustache_prob = float(mustache_model.predict(img_array, verbose=0)[0][0])
    if mustache_prob < 0.4:
        return {"mustache": False, "mustache_prob": mustache_prob}

    preds = epic_model.predict(img_array, verbose=0)[0]
    p_epic, p_medium, p_thin = float(preds[0]), float(preds[1]), float(preds[2])

    return {
        "mustache": True,
        "mustache_prob": mustache_prob,
        "p_epic": p_epic,
        "p_medium": p_medium,
        "p_thin": p_thin,
    }
