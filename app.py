from fastai.vision.all import *
import streamlit as st
import platform
import pathlib
import plotly.express as px

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

## title
st.title('Transportni klassifikatsiya qiluvchi model')

# rasmni joylash
file = st.file_uploader('Rasm yuklash', type=['png','jpeg','gif','svg','pkl'])
if file:
    st.image(file)
    # PIL konvert
    img = PILImage.create(file)
    # model
    model = load_learner('transport_model.pkl')

    #predict
    pred, prod_id , probs  = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f"Probability: {probs[prod_id]*100:0.1f}%")

    #plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)

