#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import cv2
from sklearn.utils.class_weight import compute_class_weight


# In[5]:


train_dir = "C:\\Users\\DELL\\Downloads\\archive\\Training"
test_dir = "C:\\Users\\DELL\\Downloads\\archive\\Testing"

# Augmentation and rescaling
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes)
class_weights = dict(enumerate(class_weights))


# In[7]:


# Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True
# Unfreeze the last 50 layers of ResNet50
for layer in base_model.layers[:-50]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(4, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

# Learning rate scheduler
def lr_schedule(epoch):
    lr = 1e-4
    if epoch > 5:
        lr = 1e-5
    elif epoch > 10:
        lr = 1e-6
    print("Learning rate: ", lr)
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Train and test generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

# Train the model
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10,
    callbacks=[lr_scheduler, early_stopping]
)

# Save the model
model.save("brain_tumor_model.h5")


# In[ ]:





# In[ ]:





# In[10]:


from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import os

# Load your model from epoch 10
model = load_model("brain_tumor_model.h5")

# Unfreeze more layers for fine-tuning (e.g., last 70 layers)
for layer in model.layers[-70:]:
    layer.trainable = True

# Recompile with lower learning rate
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Learning rate scheduler: tiny bump down mid-fine-tuning
def lr_schedule(epoch):
    return 1e-5 if epoch < 2 else 1e-6

# Callbacks
lr_scheduler = LearningRateScheduler(lr_schedule)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Data generators
train_dir = "C:\\Users\\DELL\\Downloads\\archive\\Training"
test_dir = "C:\\Users\\DELL\\Downloads\\archive\\Testing"

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224),
                                                    batch_size=32, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224),
                                                  batch_size=32, class_mode='categorical')

# Continue training (epochs 11–15)
model.fit(train_generator,
          validation_data=test_generator,
          epochs=15,
          initial_epoch=10,
          callbacks=[lr_scheduler, early_stop],
          verbose=1)

# Save updated model
model.save("brain_tumor_model_finetuned_epoch15.h5")


import os
import pandas as pd
import random
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import plotly.express as px

# ---------------------------------------
# 🚀 Streamlit Config
# ---------------------------------------
st.set_page_config(page_title="Brain Tumor Classifier 🧠", layout="wide")
st.sidebar.title("🚀 Navigation")

# Path to your dataset (for CSV generation)
train_path = r"C:\Users\DELL\Downloads\archive\Training"
test_path = r"C:\Users\DELL\Downloads\archive\Testing"

# Define tumor classes
tumor_classes = ['glioma', 'meningioma', 'pituitary', 'no_tumor']

# Gender and Region options
genders = ['Male', 'Female']
regions = ['North America', 'Europe', 'Asia', 'South America', 'Africa', 'Australia']

# Helper to generate random patient info
def generate_patient_info():
    gender = random.choice(genders)
    region = random.choice(regions)
    age = random.randint(5, 80)  # Age between 5 and 80
    return gender, region, age

# Assign unique Patient IDs
patient_counter = 1
data = []

# Scan both training and testing datasets for CSV generation
for base_path in [train_path, test_path]:
    for tumor_type in tumor_classes:
        folder_path = os.path.join(base_path, tumor_type)
        if not os.path.exists(folder_path):
            continue  # Skip missing folders
        
        for img_name in os.listdir(folder_path):
            if img_name.endswith(('.jpg', '.png', '.jpeg')):
                gender, region, age = generate_patient_info()
                patient_id = f"P{str(patient_counter).zfill(5)}"
                data.append([patient_id, gender, region, age, tumor_type, os.path.join(folder_path, img_name)])
                patient_counter += 1

# Create DataFrame for metadata
df = pd.DataFrame(data, columns=['PatientID', 'Gender', 'Region', 'Age', 'TumorType', 'ImagePath'])

# Save the dataframe to CSV
df.to_csv("brain_tumor_metadata.csv", index=False)
print("✅ Dummy metadata generated and saved to brain_tumor_metadata.csv")

# ---------------------------------------
# 🧠 Load Model ONCE
# ---------------------------------------
@st.cache_resource
def load_model_once():
    return load_model("brain_tumor_model_finetuned_epoch15.h5")

model = load_model_once()

# ---------------------------------------
# 🗺️ Page Selection
# ---------------------------------------
page = st.sidebar.selectbox("Select a page:", ["🏠 Home", "🔎 Image Classification", "📊 Dashboard"])

# ---------------------------------------
# 🏠 Home Page
# ---------------------------------------
if page == "🏠 Home":
    st.title("🧠 Brain Tumor Classification App")
    st.write("""
    Welcome to the Brain Tumor Detection App!  
    Upload an MRI scan and let the deep learning magic happen.  
    Stay curious, stay awesome. ✨
    """)

# ---------------------------------------
# 🔎 Image Classification Page
# ---------------------------------------
elif page == "🔎 Image Classification":
    st.title("🔎 Image Classification")

    uploaded_file = st.file_uploader("Upload an MRI Image 🖼️", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded MRI", use_column_width=True)
        except Exception as e:
            st.error(f"❌ Failed to load image: {e}")
            st.stop()

        if st.button("Predict 🚀"):
            with st.spinner('Loading model and predicting... ⏳'):
                # Preprocessing
                image_resized = image.resize((224, 224))
                img_array = np.expand_dims(np.array(image_resized) / 255.0, axis=0)

                prediction = model.predict(img_array)
                class_idx = np.argmax(prediction)

                class_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary Tumor"]
                predicted_label = class_labels[class_idx]

                st.success(f"🎯 Prediction: **{predicted_label}**")
                st.info(f"🔵 Confidence: `{np.max(prediction) * 100:.2f}%`")

# ---------------------------
# 📊 Dashboard Page
# ---------------------------
elif page == "📊 Dashboard":
    st.title("📊 Dashboard: Brain Tumor MRI Dataset")

    try:
        df = pd.read_csv("brain_tumor_metadata.csv")
    except FileNotFoundError:
        st.error("❌ Metadata CSV not found! Please make sure 'brain_tumor_metadata.csv' exists.")
        st.stop()

    # Sidebar filters
    st.sidebar.header("🔎 Filter the Data:")
    tumor_filter = st.sidebar.multiselect("Select Tumor Type", options=df['TumorType'].unique(), default=df['TumorType'].unique())
    gender_filter = st.sidebar.multiselect("Select Gender", options=df['Gender'].unique(), default=df['Gender'].unique())
    region_filter = st.sidebar.multiselect("Select Region", options=df['Region'].unique(), default=df['Region'].unique())

    # Apply filters
    filtered_df = df[
        (df['TumorType'].isin(tumor_filter)) &
        (df['Gender'].isin(gender_filter)) &
        (df['Region'].isin(region_filter))
    ]

    st.write(f"🔎 Showing `{filtered_df.shape[0]}` patients based on current filters.")

    # Layout for three graphs
    st.subheader("🔍 Visual Insights")

    # 1. Tumor Type vs Gender (Bar Chart)
    st.markdown("### 1️⃣ Tumor Type vs Gender")
    fig1 = px.histogram(filtered_df,
                        x='TumorType',
                        color='Gender',
                        barmode='group',
                        title="Tumor Type vs Gender Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    # 2. Tumor Type vs Age (Histogram - Frequency of Tumors in Age Groups)
    st.markdown("### 2️⃣ Tumor Type vs Age Distribution (Histogram)")

    fig2 = px.histogram(filtered_df,
                        x='Age',
                        color='TumorType',
                        nbins=30,
                        title="Tumor Type Distribution by Age",
                        labels={"Age": "Age", "TumorType": "Tumor Type"},
                        histfunc="count",
                        opacity=0.7)

    st.plotly_chart(fig2, use_container_width=True)

    # 3. Tumor Type across Region (Heatmap)
    st.markdown("### 3️⃣ Tumor Type Across Regions")

    region_tumor_count = (
        filtered_df
        .groupby(['TumorType', 'Region'])
        .size()
        .reset_index(name='Count')
    )

    pivot_df = (
        region_tumor_count
        .pivot(index='TumorType', columns='Region', values='Count')
        .fillna(0)
    )

    if pivot_df.shape[1] == 0:
        st.warning("No regions selected — nothing to display!")
    else:
        fig3 = px.imshow(
            pivot_df,
            labels={'x': 'Region', 'y': 'Tumor Type', 'color': 'Patient Count'},
            x=pivot_df.columns.tolist(),
            y=pivot_df.index.tolist(),
            title="Tumor Frequency Across Regions",
            aspect="auto",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig3, use_container_width=True)
