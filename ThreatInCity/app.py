import streamlit as st
import pickle
import pandas as pd

# Load the model data
@st.cache_data
def load_model():
    with open('city_safety_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    return model_data

# Load data
model_data = load_model()
df = model_data['df']
max_crime = model_data['max_crime']
max_age_0_25 = model_data['max_age_0_25']
max_age_25_50 = model_data['max_age_25_50']
max_age_50_above = model_data['max_age_50_above']
max_accidents = model_data['max_accidents']

# Function to calculate safety score
def calculate_safety_score(city, age, gender):
    """
    Calculate safety score for a person based on city, age, and gender.
    """
    
    # Find city data
    city_data = df[df['City'] == city]
    
    if city_data.empty:
        return None, f"City '{city}' not found in dataset"
    
    city_data = city_data.iloc[0]
    
    # 1. Crime Risk based on gender
    if gender.upper() == 'M':
        person_crime = city_data['Male_Crime']
    elif gender.upper() == 'F':
        person_crime = city_data['Female_Crime']
    else:
        return None, "Invalid gender. Use 'Male' or 'Female'"
    
    crime_risk = person_crime / max_crime
    
    # 2. Age Risk based on age group
    if age < 25:
        age_group_crime = city_data['Age_0_25']
        max_age_group = max_age_0_25
        age_group_name = "0-25"
    elif age < 50:
        age_group_crime = city_data['Age_25_50']
        max_age_group = max_age_25_50
        age_group_name = "25-50"
    else:
        age_group_crime = city_data['Age_50_above']
        max_age_group = max_age_50_above
        age_group_name = "50+"
    
    age_risk = age_group_crime / max_age_group
    
    # 3. AQI Risk
    aqi_risk = city_data['AQI_Risk']
    
    # 4. Accident Risk
    accident_risk = city_data['Total_Accidents'] / max_accidents
    
    # Calculate Total Risk with weights
    total_risk = (0.40 * crime_risk) + (0.25 * age_risk) + (0.20 * aqi_risk) + (0.15 * accident_risk)
    
    # Calculate Safety Score
    safety_score = (1 - total_risk) * 100
    
    # Create result dictionary
    result = {
        'safety_score': safety_score,
        'total_risk': total_risk,
        'crime_risk': crime_risk,
        'age_risk': age_risk,
        'aqi_risk': aqi_risk,
        'accident_risk': accident_risk,
        'person_crime': person_crime,
        'age_group_crime': age_group_crime,
        'age_group_name': age_group_name,
        'aqi_category': city_data['AQI_Category'],
        'total_accidents': city_data['Total_Accidents']
    }
    
    return result, None

# Streamlit UI
st.set_page_config(page_title="City Safety Analyzer", page_icon="🏙️", layout="wide")

st.title("🏙️ City Safety Analyzer")
st.markdown("### Analyze the safety of Indian cities based on crime, age, air quality, and accidents")

# Sidebar for inputs
st.sidebar.header("Enter Your Details")

# Get list of cities
cities = sorted(df['City'].unique().tolist())

# User inputs
selected_city = st.sidebar.selectbox("Select City", cities)
age = st.sidebar.number_input("Enter Your Age", min_value=1, max_value=100, value=30)
gender = st.sidebar.radio("Select Gender", ["Male", "Female"])

# Convert gender to M/F
gender_code = 'M' if gender == "Male" else 'F'

# Calculate button
if st.sidebar.button("Analyze Safety", type="primary"):
    result, error = calculate_safety_score(selected_city, age, gender_code)
    
    if error:
        st.error(error)
    else:
        # Display results
        st.success(f"Analysis complete for **{selected_city}**!")
        
        # Main metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Safety Score", f"{result['safety_score']:.2f}%", 
                     delta=None)
        
        with col2:
            st.metric("Risk Level", f"{result['total_risk']:.4f}",
                     delta=None)
        
        with col3:
            # Recommendation
            if result['safety_score'] >= 60:
                recommendation = "✅ SAFE"
                rec_color = "green"
            elif result['safety_score'] >= 30:
                recommendation = "⚠️ MODERATE"
                rec_color = "orange"
            else:
                recommendation = "❌ HIGH RISK"
                rec_color = "red"
            
            st.markdown(f"<h3 style='color: {rec_color};'>{recommendation}</h3>", 
                       unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detailed breakdown
        st.subheader("📊 Risk Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Individual Risk Factors")
            st.markdown(f"""
            - **Crime Risk (40%)**: {result['crime_risk']:.4f}
                - {int(result['person_crime'])} {gender} crimes reported
            - **Age Risk (25%)**: {result['age_risk']:.4f}
                - {int(result['age_group_crime'])} crimes in age group {result['age_group_name']}
            - **AQI Risk (20%)**: {result['aqi_risk']:.4f}
                - Air Quality: {result['aqi_category']}
            - **Accident Risk (15%)**: {result['accident_risk']:.4f}
                - {result['total_accidents']} total accidents
            """)
        
        with col2:
            # Create a bar chart for risk factors
            risk_data = pd.DataFrame({
                'Risk Factor': ['Crime (40%)', 'Age (25%)', 'AQI (20%)', 'Accident (15%)'],
                'Risk Score': [result['crime_risk'], result['age_risk'], 
                              result['aqi_risk'], result['accident_risk']]
            })
            
            st.bar_chart(risk_data.set_index('Risk Factor'))
        
        st.markdown("---")
        
        # Recommendation text
        st.subheader("💡 Recommendation")
        if result['safety_score'] >= 70:
            st.success("This city is considered **SAFE** for you to live in based on the provided data. The overall risk factors are low.")
        elif result['safety_score'] >= 50:
            st.warning("This city has **MODERATE** risk. Consider taking appropriate safety precautions if you plan to live here.")
        else:
            st.error("This city has **HIGH RISK** levels. It is not recommended to live here based on current safety metrics.")

# Footer
st.markdown("---")
st.markdown("### About the Analysis")
st.info("""
**Risk Calculation Formula:**
- Crime Risk (40%): Based on gender-specific crime statistics
- Age Risk (25%): Based on crime in your age group
- AQI Risk (20%): Based on air quality index
- Accident Risk (15%): Based on total accidents

**Safety Score** = (1 - Total Risk) × 100
""")

# Show available cities
with st.expander("View All Available Cities"):
    st.write(", ".join(cities))
