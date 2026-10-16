"""Optional Streamlit demo for known-class prediction and uncertainty rejection."""
from pathlib import Path
import tempfile

import streamlit as st
from PIL import Image

from src.predict import predict_image

st.set_page_config(page_title="The Model That Doesn't Know", page_icon="🔎", layout="centered")
st.title("The Model That Doesn't Know")
st.write(
    "Classify images from the model's 10 known CIFAR-10 classes, or reject a prediction "
    "when a confidence/energy score crosses a validation-derived threshold."
)
st.caption("A rejection is a warning, not proof that an image is truly unfamiliar.")

checkpoint_path = st.text_input("Checkpoint path", "artifacts/best_model.pt")
method = st.radio("Unknown-detection method", ["confidence", "energy"], horizontal=True)
upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp", "bmp"])

if upload is not None:
    image = Image.open(upload).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    if st.button("Classify image", type="primary"):
        if not Path(checkpoint_path).exists():
            st.error(f"Checkpoint not found: {checkpoint_path}. Train the model first.")
        else:
            # Reuse the normal CLI inference path and keep the temporary image short-lived.
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
                temp_path = Path(temp.name)
            try:
                image.save(temp_path)
                result = predict_image(temp_path, checkpoint_path, method)
            finally:
                temp_path.unlink(missing_ok=True)
            st.subheader(f"Decision: {result['decision']}")
            left, right = st.columns(2)
            left.metric("Max softmax confidence", f"{result['max_softmax_confidence']:.1%}")
            right.metric("Energy score", f"{result['energy_score']:.3f}")
            st.json(result)
            st.info(
                "The model is trained only on CIFAR-10 classes. An unknown result means "
                "the selected score crossed its validation threshold; a familiar-looking "
                "but misclassified image can still be accepted with high confidence."
            )
