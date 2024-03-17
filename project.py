import numpy as np 
import matplotlib.pyplot as plt
import pickle 
import streamlit as st
from PIL import Image
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers


pickle_in=open('model.pkl','rb')
loaded_model=pickle.load(pickle_in)
img_height= 100
img_width = 100
batch_size = 32

dataset_url = "dataset/train"

train_ds = tf.keras.utils.image_dataset_from_directory( 
    dataset_url, 
    validation_split=0.2, 
    subset= 'training', 
    seed = 256, 
    image_size=(img_height,img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  dataset_url,
  validation_split=0.2,
  subset="validation",
  seed=256,
  image_size=(img_height,img_width),
  batch_size=batch_size
)

# Print labels

class_names = train_ds.class_names


def fruit(data):
    img = tf.keras.utils.load_img(data, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # create a batch

    predictions_apple = loaded_model.predict(img_array)
    score_apple = tf.nn.softmax(predictions_apple[0])
    plt.subplot(1, 2, 1)
    plt.imshow(img) 
    plt.axis("on")
    plt.show()
    if(class_names[np.argmax(score_apple)][:6]=="rotten"):
        return ("This",class_names[np.argmax(score_apple)][6:]," is {:.2f}".format(100-(100 * np.max(score_apple))),"% healthy")
    else:
        return ("This",class_names[np.argmax(score_apple)][5:]," is {:.2f}".format(100 * np.max(score_apple)),"% healthy") 
def main():
    ''' x=fruit("orange.jpg")
     st.title("Health of fruits")
     image=Image.open("orange.jpg")
     st.image(image,caption="Fruit Image")
     st.success('{}'.format(x))'''
    add_bg_from_url() 
    st.title("Health of  the fruits")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:   
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Fruit Image")
        result = fruit(uploaded_file)
        st.header(f"{result[0]} {result[1]} {result[2]}%healthy")
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://image.slidesdocs.com/responsive-images/background/cute-cartoon-with-fruits-and-flowers-powerpoint-background_bd0d8c9b12__960_540.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )



if __name__== '__main__':
    #st.markdown(background_html, unsafe_allow_html=True)
    main()
    # block is only executed when the script is run directly and not when it is imported as a module.