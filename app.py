import streamlit as st
import cv2
import numpy as np
import pandas as pd
from processing import process_image, load_models

detector, classifier, class_names = load_models()

st.title("♻ Waste Detection & Classification")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    max_dim, min_dim = 800, 400
    height, width = image.shape[:2]
    if height > max_dim or width > max_dim or height < min_dim or width < min_dim:
        if height > width:
            new_height = min(max(height, min_dim), max_dim)
            new_width = int(width * (new_height / height))
        else:
            new_width = min(max(width, min_dim), max_dim)
            new_height = int(height * (new_width / width))
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns([6, 6])
    with col1:
        st.image(image_rgb, caption="Uploaded Image", use_container_width=True)

    image_with_boxes, total_objects, class_stats = process_image(image_rgb, detector, classifier, class_names)

    with col2:
        st.image(image_with_boxes, caption="Analysis Results", use_container_width=True)

    st.subheader(f"Found {total_objects} waste object{'s' if total_objects > 1 else ''}")

    summary_data = []
    for cls, stats in class_stats.items():
        if stats['count'] > 0:
            avg_conf = stats['confidence'] / stats['count']
            percentage = (stats['count'] / total_objects) * 100
            summary_data.append({
                'Class': cls,
                'Count': stats['count'],
                'Percentage': percentage,
                'Avg Confidence': avg_conf
            })

    if summary_data:
        df = pd.DataFrame(summary_data).sort_values('Count', ascending=False)

        col1, col2 = st.columns([3, 3])
        with col1:
            st.write("📊 Distribution of Waste Types")
            st.bar_chart(df.set_index('Class')['Count'])
        with col2:
            st.write("Summary:")
            st.dataframe(df[['Class', 'Count', 'Avg Confidence']].rename(
                columns={'Class': 'Type', 'Avg Confidence': 'Confidence'}
            ).assign(Confidence=lambda d: d['Confidence'].map(lambda x: f"{x:.1f}%")))
