import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_summary(model, feature_cols): #creates a bar chart
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_ #imp score given by rf model
    }).sort_values("Importance", ascending=False).head(10) #sorts & most imp feature is at the top

    fig, ax = plt.subplots(figsize=(6,4)) #xaxis- imp, yaxis- feature names
    ax.barh(importance_df["Feature"], importance_df["Importance"]) #creates horixontal bar chart
    ax.invert_yaxis()
    ax.set_title("Top 10 Most Important Features")
    ax.set_xlabel("Importance Score")

    return fig #streamlit will render with st.pyplot(fig)