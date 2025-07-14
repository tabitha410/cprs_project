import pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from lime.lime_tabular import LimeTabularExplainer

# Load trained Random Forest model and encoders
with open("cprs_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)
with open("target_encoder.pkl", "rb") as f:
    target_encoder = pickle.load(f)
with open("top_features.pkl", "rb") as f:
    top_features = pickle.load(f)
with open("X_train_resampled.pkl", "rb") as f:
    X_train_resampled = pickle.load(f)


# Configure the page
st.set_page_config(
    page_title="AI-Driven Career Path Recommendation System (CPRS)",
    layout="centered"
)

# Custom CSS styling with animated gradient background
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(270deg, white, #f1f1f1, white);
        background-size: 600% 600%;
        animation: gradientShift 20s ease infinite;
        padding: 3rem;
        color: black;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .big-title {
        font-size: 48px;
        font-weight: 700;
        color: #1f4e79;
        text-align: left;
    }

    .subtitle {
        font-size: 22px;
        color: black;
        text-align: left;
        margin-bottom: 30px;
    }

    .subtext {
        font-size: 22px;
        color: black;
        text-align: left;
        margin-bottom: 30px;
        font-style: italic;
    }

    .stButton>button {
        display: inline-block;
        background-color: #1f4e79;
        color: white;
        border-radius: 10px;
        padding: 10px 30px;
        font-size: 20px;
        font-weight: bold;
        transition: 0.3s;
        border: none;
    }

    .stButton>button:hover {
        background-color: #184061;
        color: white;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        border-right: 5px solid #1f4e79;
    }

    section[data-testid="stSidebar"] h1 {
        color: #1f4e79;
        font-weight: bold;
    }

    section[data-testid="stSidebar"] label > div[data-testid="stMarkdownContainer"] p {
        font-weight: bold;
    }
    section[data-testid="stSidebar"] label {
        font-weight: bold;
    }  
    section[data-testid="stSidebar"] {
        width: 350px !important; # Adjust as needed
    }
    </style>
""", unsafe_allow_html=True)

# Load dataset
df = pd.read_csv('cs_students.csv')

# Get domains
domain_counts = df['Interested Domain'].value_counts()
top_domains = domain_counts.head(15).index.tolist()

# Sidebar Content
st.sidebar.title("Input your Details")

# GPA scale 
scale = st.sidebar.radio("GPA Scale", ["5.0", "4.0"])
gpa_input = st.sidebar.slider("GPA", 0.0, float(scale), 3.0, step=0.01)

# Convert to 4.0 scale if needed
if scale == "5.0":
    gpa = (gpa_input / 5.0) * 4.0
else:
    gpa = gpa_input

python = st.sidebar.selectbox("Python Skill", ["Weak", "Average", "Strong"])
sql = st.sidebar.selectbox("SQL Skill", ["Weak", "Average", "Strong"])
java = st.sidebar.selectbox("Java Skill", ["Weak", "Average", "Strong"])

# Select interested domain
domain = st.sidebar.selectbox(
    "Interested Domain",
    top_domains
)

# Filter projects for chosen domain only
filtered_projects = df[df['Interested Domain'] == domain]['Projects'].dropna().unique().tolist()

# If no project is found
if not filtered_projects:
    filtered_projects = ["No projects found for this domain"]

# Select Project
project = st.sidebar.selectbox(
    f"Projects in {domain}",
    filtered_projects
)

# Determine if the button was clicked
recommend_button = st.sidebar.button("Get Recommendation")

if not recommend_button:
    # Landing page content when the button is not clicked
    st.markdown('<div class="big-title">AI-Driven Career Path Recommendation System (CPRS)</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">This tool uses your academic data, interests, and technical skills to suggest the most suitable career path in tech.</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtext">Discover yourself! Input your details to find the best-fit tech career for you!</div>', unsafe_allow_html=True)

if recommend_button:
    # Build new data
    new_data = pd.DataFrame([{
        "GPA": gpa,
        "Python": python,
        "SQL": sql,
        "Java": java,
        "Interested Domain": domain,
        "Projects": project
    }])

    # Encode categorical values
    for col in new_data.select_dtypes(include=['object']).columns:
        if col in label_encoders:
            new_data[col] = label_encoders[col].transform(new_data[col])

    # Match training column order
    new_data = new_data[top_features]

    # Predict 
    predicted_class = model.predict(new_data)[0]
    predicted_label = target_encoder.inverse_transform([predicted_class])[0]

    st.success(f"**Recommended Career Path:** {predicted_label}")

    # Explain prediction with LIME
    explainer = LimeTabularExplainer(
        training_data=np.array(X_train_resampled),
        feature_names=top_features,
        class_names=target_encoder.classes_,
        mode='classification'
    )

    explanation = explainer.explain_instance(
        data_row=new_data.iloc[0].values,
        predict_fn=model.predict_proba,
        labels=(predicted_class,)
    )

    # Get the explanation as list of tuples [(feature, weight), ...]
    exp_list = explanation.as_list(label=predicted_class)

    # Process for display
    exp_df = pd.DataFrame(exp_list, columns=['Feature', 'Contribution'])

    # Remove feature rule
    def clean_feature_name(feature_rule):
        for feat in top_features:
            if feat in feature_rule:
                return feat
        return feature_rule  # fallback if none matched


    exp_df['Feature'] = exp_df['Feature'].apply(clean_feature_name)

    exp_df = exp_df[abs(exp_df['Contribution']) >= 0.0001]  # Keep only meaningful contributions

    # Add Direction & Interpretation
    exp_df['Direction'] = exp_df['Contribution'].apply(
        lambda x: '✅ Positive' if x > 0 else '❌ Negative'
    )
    exp_df['Interpretation'] = exp_df.apply(
        lambda row: (
            f"Pushed the prediction towards {predicted_label}." if row['Contribution'] > 0
            else f"Pushed the prediction away from {predicted_label}."
        ),
        axis=1
    )

    exp_df['Contribution'] = exp_df['Contribution'].apply(lambda x: f"{x:.4f}")

    # Expander reveals explanation table
    with st.expander("🔍 Why this recommendation?"):
        st.write("#### Explanation of Features Influencing the Recommendation")
        st.dataframe(
            exp_df,
            column_config={
                "Feature": st.column_config.TextColumn(
                    width="medium"  
                ),
                "Contribution": st.column_config.NumberColumn(
                    width='200px',  
                    format="%.4f",   
                )
            },
            # use_container_width=False,
            hide_index=True,
            use_container_width=True
        )

    # Show top 3 class probabilities in a horizontal bar chart
    class_probs = model.predict_proba(new_data)[0]
    top3_idx = np.argsort(class_probs)[::-1][:3]
    top3_labels = target_encoder.inverse_transform(top3_idx)
    top3_probs = class_probs[top3_idx]

    # Put into DataFrame & sort in ascending order
    prob_df = pd.DataFrame({
        'Career Path': top3_labels,
        'Probability': top3_probs
    }).sort_values(by='Probability', ascending=True)

    # Plot with Plotly
    fig = px.bar(
        prob_df,
        x='Probability',
        y='Career Path',
        orientation='h',
        color_discrete_sequence=['#1f4e79']
    )

    fig.update_layout(
        title='Top 3 Recommended Career Paths',
        xaxis_title='Probability',
        yaxis_title='Career Path',
        yaxis=dict(categoryorder='total ascending'),
        plot_bgcolor='white'
    )

    st.plotly_chart(fig, use_container_width=True)

