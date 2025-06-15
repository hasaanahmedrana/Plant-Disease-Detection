# app.py

import streamlit as st
from PIL import Image, UnidentifiedImageError
import numpy as np
import tensorflow as tf
import json
import os
import pandas as pd

# -------- Configuration --------
MODEL_PATH = "PlantDiseaseModel.h5"   
CLASS_NAMES_PATH = "class_names.json"  
IMG_HEIGHT = 180                       
IMG_WIDTH = 180                         
# ---------------------------------

st.set_page_config(page_title="🌿 Plant Disease Detection", layout="wide")

# ----------------- Caching -----------------
@st.cache_resource(show_spinner=False)
def load_model():
    """Load and return the Keras model. Cached so it's loaded only once."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}")
        st.stop()
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    return model

@st.cache_data(show_spinner=False)
def load_class_names():
    """Load and return the list of class names from JSON."""
    if not os.path.exists(CLASS_NAMES_PATH):
        st.error(f"Class names file not found at {CLASS_NAMES_PATH}")
        st.stop()
    try:
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
        if not isinstance(class_names, list):
            st.error(f"{CLASS_NAMES_PATH} does not contain a JSON list")
            st.stop()
    except Exception as e:
        st.error(f"Error loading class names: {e}")
        st.stop()
    return class_names

# Grad-CAM utility
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    """
    Generates a Grad-CAM heatmap for a single preprocessed image array.
    img_array: shape (1, H, W, 3), normalized [0,1].
    model: the tf.keras Model.
    last_conv_layer_name: optional string name of last conv layer; if None, auto-detect.
    Returns: heatmap array of shape (H, W) normalized to [0,1].
    """
    # Find last convolutional layer if not provided
    if last_conv_layer_name is None:
        # Heuristic: pick the last layer with 4D output
        for layer in reversed(model.layers):
            output_shape = getattr(layer.output, "shape", None)
            if output_shape is not None and len(output_shape) == 4:
                last_conv_layer_name = layer.name
                break
        if last_conv_layer_name is None:
            st.warning("No convolutional layer found for Grad-CAM.")
            return None

    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        st.warning(f"Layer {last_conv_layer_name} not found in model.")
        return None

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Compute gradients of the predicted class w.r.t. conv layer output
    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        st.warning("Gradients could not be computed for Grad-CAM.")
        return None
    # Compute guided gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]  # shape (H', W', channels)
    heatmap = tf.zeros(conv_outputs.shape[0:2], dtype=tf.float32)
    # Weighted sum of feature maps
    for i in range(pooled_grads.shape[-1]):
        heatmap += pooled_grads[i] * conv_outputs[:, :, i]
    # Relu
    heatmap = tf.nn.relu(heatmap)
    # Normalize to [0,1]
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return tf.zeros_like(heatmap).numpy()
    heatmap = heatmap / max_val
    return heatmap.numpy()

def overlay_heatmap_on_image(original_img: Image.Image, heatmap: np.ndarray, alpha=0.4):
    """
    Overlay the heatmap (2D array) onto the original PIL Image.
    """
    import cv2  # streamlit environment often has OpenCV
    # Convert PIL to OpenCV image (numpy BGR)
    img = np.array(original_img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # Convert to color map
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    # Overlay
    overlayed = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
    # Convert back to RGB PIL
    overlayed = cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)
    return Image.fromarray(overlayed)

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Given a PIL Image, resize to (IMG_WIDTH, IMG_HEIGHT), normalize to [0,1],
    and return a batch array shape (1, IMG_HEIGHT, IMG_WIDTH, 3).
    """
    image = image.convert('RGB')
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(image).astype('float32') / 255.0
    input_arr = np.expand_dims(img_array, axis=0)  # shape (1, H, W, 3)
    return input_arr

# Initialize session state for history
if 'pred_history' not in st.session_state:
    st.session_state.pred_history = []

def clear_history():
    st.session_state.pred_history = []

# ----------------- Main -----------------
def main():
    st.title("🌿 Plant Disease Detection")

    # Sidebar controls
    st.sidebar.header("Configuration & Controls")
    show_model_summary = st.sidebar.checkbox("Show model summary", value=False)
    min_confidence = st.sidebar.slider("Minimum confidence (%) to flag low-confidence", 0, 100, 50)
    top_k = st.sidebar.number_input("Top-K predictions to display", min_value=1, max_value=10, value=3, step=1)
    use_gradcam = st.sidebar.checkbox("Show Grad-CAM explanation", value=False)
    use_camera = st.sidebar.checkbox("Use camera input", value=False)
    batch_mode = st.sidebar.checkbox("Batch upload mode", value=False)
    if st.sidebar.button("Clear prediction history"):
        clear_history()
    st.sidebar.markdown("---")
    st.sidebar.subheader("About & FAQ")
    st.sidebar.info("""
- Upload or take a picture of a plant leaf.  
- The model will predict disease class and confidence.  
- Grad-CAM: visually highlights areas influencing the prediction.  
- Batch mode: upload multiple images at once; results downloadable as CSV.  
- Model summary: shows layer structure.  
- Low-confidence predictions (< threshold) will be flagged.  
Ensure that `plant_disease_model.h5` and `class_names.json` are in the same directory.
    """)
    st.sidebar.markdown("For more info, refer to your project documentation or contact your team.")

    # Load model and class names
    model = load_model()
    class_names = load_class_names()

    if show_model_summary:
        st.sidebar.text("Model architecture:")
        # Capture model summary as text
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        summary_str = "\n".join(stringlist)
        st.sidebar.text_area("Model Summary", summary_str, height=300)

    # Columns layout for main area
    col1, col2 = st.columns([1, 1])

    images_to_process = []
    filenames = []

    if use_camera:
        cam_img = st.camera_input("Take a picture")
        if cam_img is not None:
            try:
                image = Image.open(cam_img)
                images_to_process.append(image)
                filenames.append("camera_input.png")
            except Exception as e:
                st.error(f"Error reading camera image: {e}")
    else:
        if not batch_mode:
            uploaded_file = st.file_uploader("Upload a single image...", type=['jpg', 'jpeg', 'png'])
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file)
                    images_to_process.append(image)
                    filenames.append(uploaded_file.name)
                except UnidentifiedImageError:
                    st.error("Cannot open uploaded file as an image. Please upload a valid image.")
                except Exception as e:
                    st.error(f"Error reading image: {e}")
        else:
            uploaded_files = st.file_uploader(
                "Upload multiple images...", type=['jpg','jpeg','png'],
                accept_multiple_files=True
            )
            if uploaded_files:
                for uf in uploaded_files:
                    try:
                        img = Image.open(uf)
                        images_to_process.append(img)
                        filenames.append(uf.name)
                    except UnidentifiedImageError:
                        st.warning(f"File {uf.name} is not a valid image; skipped.")
                    except Exception as e:
                        st.warning(f"Error reading {uf.name}: {e}; skipped.")

    # Process images if any
    if images_to_process:
        results = []
        # Progress bar for batch
        progress_bar = None
        if len(images_to_process) > 1:
            progress_bar = st.progress(0)
        for idx, img in enumerate(images_to_process):
            with col1:
                st.image(img, caption=f"Input: {filenames[idx]}", use_column_width=True)

            # Preprocess
            try:
                input_arr = preprocess_image(img)
            except Exception as e:
                st.error(f"Error preprocessing image {filenames[idx]}: {e}")
                continue

            # Predict
            try:
                preds = model.predict(input_arr)
            except Exception as e:
                st.error(f"Inference error for {filenames[idx]}: {e}")
                continue

            # Interpret output
            if preds.ndim == 2 and preds.shape[0] == 1:
                pred_idx = int(np.argmax(preds, axis=1)[0])
                confidence = float(np.max(preds, axis=1)[0])  # in [0,1]
            else:
                st.error(f"Unexpected model output shape: {preds.shape}")
                continue

            # Map to class name
            if 0 <= pred_idx < len(class_names):
                pred_class = class_names[pred_idx]
            else:
                pred_class = "Unknown"

            # Low confidence flag
            low_conf_flag = confidence * 100 < min_confidence

            # Show results
            with col2:
                st.markdown(f"**Predicted class:** {pred_class}")
                st.markdown(f"**Confidence:** {confidence * 100:.2f}%")
                if low_conf_flag:
                    st.warning(f"Low confidence (< {min_confidence}%)")
                # Top-K
                if preds.shape[1] >= top_k:
                    top_indices = np.argsort(preds[0])[::-1][:top_k]
                    st.write(f"**Top {top_k} predictions:**")
                    for i in top_indices:
                        name = class_names[i] if i < len(class_names) else f"Index {i}"
                        conf = float(preds[0][i])
                        st.write(f"- {name}: {conf * 100:.2f}%")

                # Grad-CAM
                if use_gradcam:
                    heatmap = make_gradcam_heatmap(input_arr, model)
                    if heatmap is not None:
                        try:
                            overlayed = overlay_heatmap_on_image(img, heatmap, alpha=0.4)
                            st.image(overlayed, caption="Grad-CAM Overlay", use_column_width=True)
                        except Exception as e:
                            st.error(f"Error generating Grad-CAM overlay: {e}")

            # Record result in history
            results.append({
                "filename": filenames[idx],
                "predicted_class": pred_class,
                "confidence": confidence * 100
            })
            st.session_state.pred_history.append({
                "filename": filenames[idx],
                "predicted_class": pred_class,
                "confidence": confidence * 100
            })

            # Update progress
            if progress_bar:
                progress_bar.progress((idx + 1) / len(images_to_process))

        # After loop: show summary & download
        if results:
            df = pd.DataFrame(results)
            st.markdown("### Batch Prediction Summary")
            st.write(f"Processed {len(results)} image(s).")
            avg_conf = df['confidence'].mean()
            st.write(f"Average confidence: {avg_conf:.2f}%")
            st.dataframe(df)
            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name='prediction_results.csv',
                mime='text/csv'
            )

    else:
        st.info("Upload or capture an image to get started.")

    # Show session history in an expander
    if st.session_state.pred_history:
        with st.expander("Prediction history (current session)"):
            hist_df = pd.DataFrame(st.session_state.pred_history)
            st.write(hist_df)

if __name__ == "__main__":
    main()
