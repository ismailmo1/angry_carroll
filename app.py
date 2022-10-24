import os

import gradio as gr
from fastai.vision.all import PILImage, load_learner
from gradio.components import Image, Label

learn = load_learner("model.pkl")
labels = learn.dls.vocab
examples = ["examples/" + file for file in os.listdir("examples")]


def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


gradio_interface = gr.Interface(
    fn=predict,
    inputs=Image(shape=(512, 512)),
    outputs=Label(num_top_classes=2),
    allow_flagging="never",
    live=True,
    examples=examples,
    title="Angry Carroll",
    description="Can AI tell the difference between Carroll and Nunez? \
        Upload a photo or choose an example below to test it out!",
)
gradio_interface.launch(enable_queue=True)
