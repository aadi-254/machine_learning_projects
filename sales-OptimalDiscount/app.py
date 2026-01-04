import streamlit as st
import pickle
import pandas as pd

# Load the model data
@st.cache_resource
def load_model():
    with open('discount_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    return model_data

# Prediction function
def predict_optimal_discount(product_type, actual_price, model_data):
    """
    Predict optimal discount based on product type and actual price
    """
    # Weights
    A = 0.5  # weight for demand
    B = 0.3  # weight for quality/rating
    C = 0.2  # weight for price
    
    # Get scores for the product type
    type_scores = model_data[model_data['type'] == product_type]
    
    if type_scores.empty:
        return None, None, None
    
    demand_score = type_scores['DemandScore'].values[0]
    quality_score = type_scores['QualityScore'].values[0]
    
    # Calculate PriceFactor
    price_factor = actual_price / (actual_price + 5000)
    
    # Calculate OptimalDiscount
    optimal_discount = (
        A * (1 - demand_score) +
        B * (1 - quality_score) +
        C * price_factor
    )
    
    # Convert to percentage
    optimal_discount_percent = optimal_discount * 100
    
    return round(optimal_discount_percent, 2), demand_score, quality_score

# Streamlit App
def main():
    # Page configuration
    st.set_page_config(
        page_title="Amazon Discount Optimizer",
        page_icon="🏷️",
        layout="wide"
    )
    
    # Load model
    model_data = load_model()
    
    # Header
    st.title("🏷️ Amazon Discount Optimization System")
    st.markdown("### Predict optimal discount percentages based on product type and price")
    st.divider()
    
    # Sidebar for information
    with st.sidebar:
        st.header("📊 Model Information")
        st.markdown("""
        **Model Formula:**
        
        OptimalDiscount = 
        - 0.5 × (1 - DemandScore)
        - 0.3 × (1 - QualityScore)  
        - 0.2 × PriceFactor
        
        **Factors:**
        - **DemandScore**: Sales relative to highest selling type
        - **QualityScore**: Rating relative to highest rated type
        - **PriceFactor**: Price/(Price + 5000)
        """)
        
        st.divider()
        
        st.header("📈 Product Types")
        for idx, row in model_data.iterrows():
            st.markdown(f"**{row['type']}**")
            st.caption(f"Demand: {row['DemandScore']:.2f} | Quality: {row['QualityScore']:.2f}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Discount Predictor")
        
        # Input form
        with st.form("prediction_form"):
            product_type = st.selectbox(
                "Select Product Type",
                options=model_data['type'].tolist(),
                help="Choose the category of your product"
            )
            
            actual_price = st.number_input(
                "Enter Actual Price (₹)",
                min_value=1.0,
                max_value=1000000.0,
                value=5000.0,
                step=100.0,
                help="Enter the original price in Indian Rupees"
            )
            
            submit_button = st.form_submit_button("🎯 Calculate Optimal Discount", use_container_width=True)
        
        if submit_button:
            # Make prediction
            discount, demand_score, quality_score = predict_optimal_discount(
                product_type, actual_price, model_data
            )
            
            if discount is not None:
                st.success("✅ Prediction Complete!")
                
                # Display results in columns
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    st.metric("Optimal Discount", f"{discount}%", delta=None)
                
                with res_col2:
                    discounted_price = actual_price * (1 - discount/100)
                    st.metric("Discounted Price", f"₹{discounted_price:,.2f}")
                
                with res_col3:
                    savings = actual_price - discounted_price
                    st.metric("Customer Savings", f"₹{savings:,.2f}")
                
                st.divider()
                
                # Additional insights
                st.subheader("📊 Discount Breakdown")
                
                insight_col1, insight_col2 = st.columns(2)
                
                with insight_col1:
                    st.markdown("**Product Scores:**")
                    st.write(f"- Demand Score: {demand_score:.3f}")
                    st.write(f"- Quality Score: {quality_score:.3f}")
                    
                    price_factor = actual_price / (actual_price + 5000)
                    st.write(f"- Price Factor: {price_factor:.3f}")
                
                with insight_col2:
                    st.markdown("**Interpretation:**")
                    if discount < 20:
                        st.info("🟢 Low discount recommended - High demand/quality product")
                    elif discount < 40:
                        st.info("🟡 Moderate discount recommended - Average market position")
                    else:
                        st.info("🔴 High discount recommended - Lower demand or premium pricing")
                
                # Visual representation
                st.divider()
                st.subheader("💰 Pricing Comparison")
                
                chart_data = pd.DataFrame({
                    'Price Type': ['Original Price', 'Discounted Price'],
                    'Amount (₹)': [actual_price, discounted_price]
                })
                
                st.bar_chart(chart_data.set_index('Price Type'))
            else:
                st.error("❌ Product type not found in model data!")
    
    with col2:
        st.subheader("🧮 Quick Calculator")
        st.markdown("**Test different scenarios:**")
        
        # Quick test scenarios
        st.markdown("#### Common Price Points")
        
        test_prices = [500, 1000, 5000, 10000, 20000, 50000]
        
        for price in test_prices:
            discount, _, _ = predict_optimal_discount(product_type, price, model_data)
            if discount:
                discounted = price * (1 - discount/100)
                st.write(f"₹{price:,} → **{discount}%** → ₹{discounted:,.0f}")
    
    # Footer
    st.divider()
    st.caption("🔬 Model trained on Amazon product data | Factors: Demand, Quality, Price")

if __name__ == "__main__":
    main()
