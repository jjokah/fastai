# %%
from fastai.vision.all import *
import gradio as gr

# %%
learn = load_learner('model.pkl')

# %%
im = PILImage.create('christmas.jpeg')
im.thumbnail((192, 192))
im

# %%
learn.predict(im)

# %%
categories = ('christmas', 'easter', 'ramadan')

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

# %%
classify_image(im)

# %%
examples = ['christmas.jpeg', 'easter.jpeg', 'ramadan.jpeg']

intf = gr.Interface(fn=classify_image, inputs="image", outputs="label", examples=examples)
intf.launch(inline=False)
