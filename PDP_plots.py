import numpy as np
import pandas as pd
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import plot_partial_dependence
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import partial_dependence
from sklearn.ensemble import GradientBoostingRegressor  # You can use your specific model

import streamlit as st

def main():
    col1, col2 = st.columns(2)
    # Load your model from the pickle file
    with open("lgbm_model2.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    # Load your dataset (replace 'data_pdp_plot.csv' with the actual file path)
    df = pd.read_csv("data_pdp_plot2.csv")
#     df=df.dropna(axis=0)
#     df = df.mask(df.eq('None')).dropna()
#     st.write(df.shape)
    # Set the title for your Streamlit app

    # Specify the feature(s) for which you want to create PDPs
    features = ['None','SR_PARSED_PRICE',
 'COMP_TITLE_LENGTH',
 'COMP_DESCRIPTION_LENGTH',
 'COMP_IMG_COUNT',
 'RETAIL_SALES_VALUE',
#  'RETAIL_SALES_UNIT',
 'SEARCH_TERM_IN_NAME',
 'SEARCH_TERM_IN_DESC',
 'SEARCH_TERM_COUNT_IN_NAME',
 'SEARCH_TERM_COUNT_IN_DESC',
 'SEARCH_TERM_FW_POS_IN_TITLE',
 'Retailer_Margin',
 'LAST_1_WEEK_SALES',
 'LAST_2_WEEK_SALES',
 'LAST_4_WEEK_SALES',
 'LAST_8_WEEK_SALES',
 'LAST_16_WEEK_SALES',
 'HOLIDAY_FLAG',
 'FB_FEEDBACK_COMMENT_COUNT',
 'FB_FEEDBACK_AVG_RATING',
 'AV_OOS']
    with col1:
        
        st.subheader("Partial Dependence Plots")

        feature = st.sidebar.selectbox("Select a Feature", features,key=None)
        if st.sidebar.button("Show PDP Plot"):
            
            if feature=='None':
                fig, ax = plt.subplots()
                st.pyplot(fig)
            else:
#                 fig, ax = plt.subplots()  ## figsize=(10, 6)
                st.bar_chart(data=df,x=feature,y='RETAIL_SALES_UNIT')
#                 plot_partial_dependence(model, X=df, features=[feature], grid_resolution=100,ax=ax)
#                 pd_results = partial_dependence(model, X=df, features=[feature], grid_resolution=100)
#                 PartialDependenceDisplay([pd_results], features=[feature])
#                 ax.set_ylabel("RANK")
#                 ax.grid(True)
#                 st.pyplot(fig)


    # Custom CSS for styling
    custom_css = """
    <style>
    .custom-text {
        font-style: italic;
    }
    .custom-box {
        border: 2px solid orange;
        padding: 10px;
    }
    </style>
    """

    # Display the custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)

    # Add text with bullet points
    text = """
    We are using following search rank bin classifications for targeted digital shelf sales optimization
    <ul class="custom-text">
        <li>1 to 2</li>
        <li>3 to 6</li>
        <li>7 to 11</li>
        <li>12 to 15</li>
        <li>16 to 19</li>
        <li>20 to 22</li>
        <li>23 to 26</li>
    </ul>
    """

    # Add a box with an orange outline at the bottom
    st.sidebar.markdown(
        f'<div class="custom-box">{text}</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()


