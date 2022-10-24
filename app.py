import gradio as gr
from fastai.vision.all import PILImage, load_learner

learn = load_learner("model.pkl")
labels = learn.dls.vocab


def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


gradio_interface = gr.Interface(
    fn=predict,
    inputs=gr.inputs.Image(shape=(512, 512)),
    outputs=gr.outputs.Label(num_top_classes=2),
    allow_flagging="never",
    live=True,
    title="Angry Carroll Detector",
    description="Is it Nunez or Carroll? Upload a picture of either and we'll \
        help you figure out which one it is!",
)
gradio_interface.launch(share=True)
