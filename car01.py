# ======================================================
# ULTRA ACCURATE CAR PRICE PREDICTION SYSTEM
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datetime import datetime
import io
import base64

# ========================================
# ULTRA ACCURATE CAR PRICE DATABASE
# ========================================

CAR_DATABASE = {
    'Maruti Suzuki': {
        'models': ['Alto', 'Alto K10', 'S-Presso', 'Celerio', 'Wagon R', 'Ignis', 'Swift', 'Baleno', 'Dzire', 'Ciaz', 
                  'Ertiga', 'XL6', 'Vitara Brezza', 'Jimny', 'Fronx', 'Grand Vitara', 'Eeco'],
        'base_prices': [300000, 400000, 450000, 550000, 600000, 650000, 800000, 900000, 850000, 950000,
                       1100000, 1300000, 1000000, 1250000, 950000, 1200000, 500000]
    },
    'Hyundai': {
        'models': ['i10', 'i20', 'Aura', 'Grand i10 Nios', 'Verna', 'Creta', 'Venue', 'Alcazar', 'Tucson', 'Kona Electric'],
        'base_prices': [500000, 700000, 650000, 600000, 1100000, 1400000, 950000, 2000000, 2800000, 2400000]
    },
    'Tata': {
        'models': ['Tiago', 'Tigor', 'Altroz', 'Nexon', 'Punch', 'Harrier', 'Safari', 'Nexon EV', 'Tigor EV'],
        'base_prices': [450000, 550000, 700000, 950000, 650000, 1800000, 2000000, 1600000, 1300000]
    },
    'Mahindra': {
        'models': ['Bolero', 'Scorpio', 'XUV300', 'XUV400', 'XUV700', 'Thar', 'Marazzo', 'Bolero Neo', 'Scorpio N'],
        'base_prices': [850000, 1500000, 1100000, 1700000, 1600000, 1500000, 1200000, 950000, 1700000]
    },
    'Toyota': {
        'models': ['Innova Crysta', 'Fortuner', 'Glanza', 'Urban Cruiser Hyryder', 'Camry', 'Vellfire', 'Hilux'],
        'base_prices': [2000000, 3500000, 750000, 1200000, 4500000, 9000000, 3800000]
    },
    'Honda': {
        'models': ['Amaze', 'City', 'Jazz', 'WR-V', 'Elevate', 'Civic', 'CR-V'],
        'base_prices': [750000, 1200000, 850000, 950000, 1200000, 2000000, 3200000]
    },
    'Kia': {
        'models': ['Seltos', 'Sonet', 'Carens', 'Carnival', 'EV6'],
        'base_prices': [1200000, 850000, 1300000, 3300000, 6500000]
    },
    'BMW': {
        'models': ['3 Series', '5 Series', 'X1', 'X3', 'X5'],
        'base_prices': [5000000, 6800000, 4700000, 6200000, 8500000]
    },
    'Mercedes-Benz': {
        'models': ['A-Class', 'C-Class', 'E-Class', 'GLA', 'GLC'],
        'base_prices': [4700000, 6000000, 7800000, 5200000, 6500000]
    },
    'Audi': {
        'models': ['A3', 'A4', 'A6', 'Q3', 'Q5'],
        'base_prices': [4500000, 5500000, 7000000, 5200000, 6800000]
    },
    'BYD': {
        'models': ['Atto 3', 'E6', 'Han'],
        'base_prices': [3400000, 2900000, 6500000]
    },
    'Volkswagen': {
        'models': ['Polo', 'Vento', 'Virtus', 'Taigun', 'Tiguan'],
        'base_prices': [700000, 900000, 1100000, 1300000, 3200000]
    },
    'Skoda': {
        'models': ['Rapid', 'Slavia', 'Kushaq', 'Kodiaq'],
        'base_prices': [800000, 1100000, 1200000, 3500000]
    },
    'MG': {
        'models': ['Hector', 'Astor', 'Gloster', 'ZS EV'],
        'base_prices': [1500000, 1300000, 3200000, 2200000]
    },
    'Renault': {
        'models': ['Kwid', 'Triber', 'Kiger', 'Duster'],
        'base_prices': [400000, 650000, 750000, 1100000]
    },
    'Nissan': {
        'models': ['Magnite', 'Kicks', 'Micra', 'Sunny'],
        'base_prices': [600000, 1100000, 700000, 800000]
    },
    'Ford': {
        'models': ['EcoSport', 'Endeavour', 'Figo', 'Aspire'],
        'base_prices': [850000, 3200000, 600000, 650000]
    }
}

FUEL_TYPES = ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"]
TRANSMISSIONS = ["Manual", "Automatic", "CVT", "DCT", "AMT"]
CAR_CONDITIONS = ["Excellent", "Very Good", "Good", "Fair", "Poor"]
OWNER_TYPES = ["First", "Second", "Third", "Fourth & Above"]
INSURANCE_STATUS = ["Comprehensive", "Third Party", "Expired", "No Insurance"]
COLORS = ["White", "Black", "Silver", "Grey", "Red", "Blue", "Brown", "Green", "Yellow", "Orange", "Purple", "Other"]
CITIES = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Pune", "Hyderabad", "Kolkata", "Ahmedabad", "Surat", "Jaipur", "Lucknow", "Chandigarh"]

# ========================================
# ULTRA ACCURATE PRICE PREDICTION ENGINE
# ========================================

class UltraAccurateCarPricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.encoders = {}
        self.is_trained = False
        
    def get_base_price(self, brand, model):
        """Get accurate base price from database"""
        try:
            if brand in CAR_DATABASE and model in CAR_DATABASE[brand]['models']:
                model_index = CAR_DATABASE[brand]['models'].index(model)
                return CAR_DATABASE[brand]['base_prices'][model_index]
            else:
                # Default base prices for unknown models
                default_prices = {
                    'Maruti Suzuki': 600000,
                    'Hyundai': 800000,
                    'Tata': 700000,
                    'Mahindra': 900000,
                    'Toyota': 1500000,
                    'Honda': 1000000,
                    'Kia': 900000,
                    'BMW': 5000000,
                    'Mercedes-Benz': 5500000,
                    'Audi': 5200000,
                    'BYD': 3000000,
                    'Volkswagen': 800000,
                    'Skoda': 900000,
                    'MG': 1200000,
                    'Renault': 500000,
                    'Nissan': 600000,
                    'Ford': 700000
                }
                return default_prices.get(brand, 500000)
        except:
            return 500000

    def calculate_accurate_price(self, input_data):
        """Calculate ultra accurate price using advanced formula"""
        try:
            # Get base price
            base_price = self.get_base_price(input_data['Brand'], input_data['Model'])
            
            # Fuel type adjustment
            fuel_multipliers = {
                "Petrol": 1.0,
                "Diesel": 1.12,
                "CNG": 0.92,
                "Electric": 1.65,
                "Hybrid": 1.35
            }
            base_price *= fuel_multipliers.get(input_data['Fuel_Type'], 1.0)
            
            # Transmission adjustment
            transmission_multipliers = {
                "Manual": 1.0,
                "Automatic": 1.18,
                "CVT": 1.15,
                "DCT": 1.22,
                "AMT": 1.08
            }
            base_price *= transmission_multipliers.get(input_data['Transmission'], 1.0)
            
            # Age depreciation (most important factor)
            current_year = datetime.now().year
            car_age = current_year - input_data['Year']
            
            if car_age == 0:
                depreciation = 0.10  # 10% in first year
            elif car_age == 1:
                depreciation = 0.25  # 25% after 1 year
            elif car_age == 2:
                depreciation = 0.35  # 35% after 2 years
            elif car_age == 3:
                depreciation = 0.45  # 45% after 3 years
            elif car_age == 4:
                depreciation = 0.53  # 53% after 4 years
            elif car_age == 5:
                depreciation = 0.60  # 60% after 5 years
            else:
                depreciation = min(0.75, 0.60 + (car_age - 5) * 0.05)  # 5% per year after 5 years
            
            # Mileage impact
            mileage = input_data['Mileage']
            if mileage <= 10000:
                mileage_impact = 0
            elif mileage <= 30000:
                mileage_impact = 0.03
            elif mileage <= 50000:
                mileage_impact = 0.07
            elif mileage <= 80000:
                mileage_impact = 0.12
            elif mileage <= 120000:
                mileage_impact = 0.18
            elif mileage <= 200000:
                mileage_impact = 0.25
            else:
                mileage_impact = 0.35
            
            total_depreciation = depreciation + mileage_impact
            
            # Condition multiplier
            condition_multipliers = {
                "Excellent": 0.92,  # Only 8% reduction from base
                "Very Good": 0.85,  # 15% reduction
                "Good": 0.75,       # 25% reduction
                "Fair": 0.60,       # 40% reduction
                "Poor": 0.45        # 55% reduction
            }
            
            # Owner type multiplier
            owner_multipliers = {
                "First": 1.0,
                "Second": 0.88,
                "Third": 0.75,
                "Fourth & Above": 0.60
            }
            
            # Calculate final price
            depreciated_price = base_price * (1 - total_depreciation)
            final_price = depreciated_price * condition_multipliers[input_data['Condition']] * owner_multipliers[input_data['Owner_Type']]
            
            # City adjustment
            city_premium = {
                "Delhi": 1.04, "Mumbai": 1.06, "Bangalore": 1.05, 
                "Chennai": 1.02, "Pune": 1.03, "Hyderabad": 1.03,
                "Kolkata": 1.01, "Ahmedabad": 1.02, "Surat": 1.01,
                "Jaipur": 1.01, "Lucknow": 1.00, "Chandigarh": 1.02
            }
            final_price *= city_premium.get(input_data['Registration_City'], 1.0)
            
            # Insurance adjustment
            if input_data['Insurance_Status'] == 'Comprehensive':
                final_price *= 1.03
            elif input_data['Insurance_Status'] == 'Expired':
                final_price *= 0.98
            
            return max(100000, int(final_price))
            
        except Exception as e:
            # Fallback to simple calculation
            return self.fallback_calculation(input_data)
    
    def fallback_calculation(self, input_data):
        """Simple fallback calculation"""
        base_price = self.get_base_price(input_data['Brand'], input_data['Model'])
        current_year = datetime.now().year
        age = current_year - input_data['Year']
        age_factor = max(0.3, 1 - (age * 0.15))
        
        condition_multipliers = {
            "Excellent": 1.0, "Very Good": 0.9, "Good": 0.8, "Fair": 0.7, "Poor": 0.5
        }
        
        price = base_price * age_factor * condition_multipliers[input_data['Condition']]
        return max(100000, int(price))

    def get_market_price_range(self, brand, model, year, condition):
        """Get accurate market price range"""
        try:
            base_price = self.get_base_price(brand, model)
            current_year = datetime.now().year
            age = current_year - year
            
            # Calculate typical depreciation for the age
            if age == 0:
                dep_factor = 0.85
            elif age == 1:
                dep_factor = 0.70
            elif age == 2:
                dep_factor = 0.60
            elif age == 3:
                dep_factor = 0.52
            elif age == 4:
                dep_factor = 0.45
            elif age == 5:
                dep_factor = 0.40
            else:
                dep_factor = max(0.25, 0.40 - (age - 5) * 0.03)
            
            avg_price = base_price * dep_factor
            
            # Adjust for condition
            condition_factors = {
                "Excellent": 1.1, "Very Good": 1.0, "Good": 0.9, "Fair": 0.75, "Poor": 0.6
            }
            avg_price *= condition_factors[condition]
            
            min_price = avg_price * 0.85
            max_price = avg_price * 1.15
            
            return [int(min_price), int(avg_price), int(max_price)]
            
        except:
            return [300000, 500000, 700000]

    def load_csv_data(self, uploaded_file):
        """Load CSV data for training"""
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(df)} records from CSV")
            return df
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
            return None

    def train_from_csv(self, df):
        """Train model from CSV data"""
        try:
            st.info("🔄 Training advanced model from CSV data...")
            
            # Create a copy
            df_processed = df.copy()
            
            # Handle Price_INR
            if 'Price_INR' in df_processed.columns and 'Price' not in df_processed.columns:
                df_processed['Price'] = df_processed['Price_INR']
                st.success("✅ Mapped 'Price_INR' → 'Price'")
            
            # Required columns
            required_columns = ['Brand', 'Model', 'Year', 'Fuel_Type', 'Transmission', 
                              'Mileage', 'Condition', 'Price']
            
            # Check columns
            missing_columns = [col for col in required_columns if col not in df_processed.columns]
            if missing_columns:
                st.error(f"Missing columns: {missing_columns}")
                return False
            
            # Clean data
            df_clean = df_processed.dropna()
            if len(df_clean) < 5:
                st.error("Not enough data after cleaning")
                return False
            
            # Prepare features
            features = ['Brand', 'Model', 'Year', 'Fuel_Type', 'Transmission', 'Mileage', 'Condition']
            X = df_clean[features]
            y = df_clean['Price']
            
            # Encode categorical variables
            categorical_features = ['Brand', 'Model', 'Fuel_Type', 'Transmission', 'Condition']
            for feature in categorical_features:
                self.encoders[feature] = LabelEncoder()
                X[feature] = self.encoders[feature].fit_transform(X[feature].astype(str))
            
            # Train model
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X, y)
            self.is_trained = True
            
            # Evaluate
            y_pred = self.model.predict(X)
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            
            st.success(f"✅ Model trained! R²: {r2:.3f}, MAE: ₹{mae:,.0f}")
            return True
            
        except Exception as e:
            st.error(f"Training error: {str(e)}")
            return False

    def predict_price(self, input_data):
        """Main prediction function"""
        if self.is_trained:
            try:
                # Use trained model
                features = ['Brand', 'Model', 'Year', 'Fuel_Type', 'Transmission', 'Mileage', 'Condition']
                input_df = pd.DataFrame([input_data])
                
                for feature in ['Brand', 'Model', 'Fuel_Type', 'Transmission', 'Condition']:
                    if feature in self.encoders:
                        input_df[feature] = self.encoders[feature].transform([input_data[feature]])[0]
                
                prediction = self.model.predict(input_df[features])[0]
                return max(100000, int(prediction))
            except:
                return self.calculate_accurate_price(input_data)
        else:
            return self.calculate_accurate_price(input_data)

# ========================================
# STREAMLIT UI COMPONENTS
# ========================================

def main():
    # Initialize predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = UltraAccurateCarPricePredictor()
    
    st.set_page_config(
        page_title="Ultra Accurate Car Price Predictor", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )
    
    st.title("🚗 Ultra Accurate Car Price Prediction System")
    st.markdown("### **Real Market Prices with Advanced Depreciation Analysis**")
    
    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to", [
            "🎯 Price Prediction", 
            "📊 Market Analysis",
            "📁 CSV Training"
        ])
        
        st.markdown("---")
        st.subheader("Database Info")
        total_brands = len(CAR_DATABASE)
        total_models = sum(len(CAR_DATABASE[brand]['models']) for brand in CAR_DATABASE)
        st.info(f"**Brands:** {total_brands}\n**Models:** {total_models}")
    
    # Page routing
    if page == "🎯 Price Prediction":
        show_prediction_interface()
    elif page == "📊 Market Analysis":
        show_market_analysis()
    elif page == "📁 CSV Training":
        show_csv_training()

def show_prediction_interface():
    st.subheader("🎯 Ultra Accurate Price Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Brand and Model Selection
        brand = st.selectbox("Select Brand", list(CAR_DATABASE.keys()))
        
        if brand in CAR_DATABASE:
            model = st.selectbox("Select Model", CAR_DATABASE[brand]['models'])
            
            # Show base price
            base_price = st.session_state.predictor.get_base_price(brand, model)
            st.info(f"**Base New Price:** ₹{base_price:,}")
        
        # Car details
        current_year = datetime.now().year
        year = st.slider("Manufacturing Year", 2010, current_year, current_year - 3)
        
        fuel_type = st.selectbox("Fuel Type", FUEL_TYPES)
        transmission = st.selectbox("Transmission", TRANSMISSIONS)
    
    with col2:
        mileage = st.number_input("Mileage (km)", min_value=0, max_value=300000, value=30000, step=5000)
        condition = st.selectbox("Condition", CAR_CONDITIONS)
        owner_type = st.selectbox("Owner Type", OWNER_TYPES)
        insurance_status = st.selectbox("Insurance Status", INSURANCE_STATUS)
        registration_city = st.selectbox("Registration City", CITIES)
    
    # Prediction button
    if st.button("🎯 Get Ultra Accurate Price", type="primary", use_container_width=True):
        with st.spinner('Calculating ultra accurate price...'):
            # Prepare input data
            input_data = {
                'Brand': brand,
                'Model': model,
                'Year': year,
                'Fuel_Type': fuel_type,
                'Transmission': transmission,
                'Mileage': mileage,
                'Condition': condition,
                'Owner_Type': owner_type,
                'Insurance_Status': insurance_status,
                'Registration_City': registration_city
            }
            
            # Get prediction
            predicted_price = st.session_state.predictor.predict_price(input_data)
            
            # Get market range
            market_prices = st.session_state.predictor.get_market_price_range(brand, model, year, condition)
            
            # Display results
            st.success(f"## 🎯 Predicted Price: ₹{predicted_price:,}")
            
            # Market comparison
            st.subheader("📊 Market Comparison")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Market Low", f"₹{market_prices[0]:,}")
            with col2:
                st.metric("Market Average", f"₹{market_prices[1]:,}")
            with col3:
                st.metric("Market High", f"₹{market_prices[2]:,}")
            
            # Price analysis
            st.subheader("📈 Price Analysis")
            base_price = st.session_state.predictor.get_base_price(brand, model)
            depreciation = base_price - predicted_price
            depreciation_percent = (depreciation / base_price) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Original Price", f"₹{base_price:,}")
            with col2:
                st.metric("Total Depreciation", f"₹{depreciation:,}", f"-{depreciation_percent:.1f}%")
            
            # Confidence indicator
            confidence = calculate_confidence(input_data)
            st.progress(confidence/100, f"Confidence Level: {confidence}%")

def calculate_confidence(input_data):
    """Calculate prediction confidence"""
    confidence = 85
    
    # Age factor
    current_year = datetime.now().year
    age = current_year - input_data['Year']
    if age <= 2:
        confidence += 10
    elif age <= 5:
        confidence += 5
    
    # Mileage factor
    if input_data['Mileage'] <= 30000:
        confidence += 5
    elif input_data['Mileage'] > 100000:
        confidence -= 10
    
    # Condition factor
    condition_scores = {"Excellent": 5, "Very Good": 3, "Good": 0, "Fair": -5, "Poor": -10}
    confidence += condition_scores[input_data['Condition']]
    
    return min(95, max(70, confidence))

def show_market_analysis():
    st.subheader("📊 Car Market Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        brand = st.selectbox("Select Brand for Analysis", list(CAR_DATABASE.keys()))
        
        if brand in CAR_DATABASE:
            model = st.selectbox("Select Model", CAR_DATABASE[brand]['models'])
            
            # Show price analysis for different years
            st.subheader("💰 Price Analysis Over Years")
            
            base_price = st.session_state.predictor.get_base_price(brand, model)
            current_year = datetime.now().year
            
            price_data = []
            for years_old in range(0, 8):
                year = current_year - years_old
                input_data = {
                    'Brand': brand, 'Model': model, 'Year': year,
                    'Fuel_Type': 'Petrol', 'Transmission': 'Manual',
                    'Mileage': years_old * 15000, 'Condition': 'Very Good',
                    'Owner_Type': 'First', 'Insurance_Status': 'Comprehensive',
                    'Registration_City': 'Delhi'
                }
                price = st.session_state.predictor.predict_price(input_data)
                price_data.append({'Year': year, 'Price': price, 'Age': years_old})
            
            price_df = pd.DataFrame(price_data)
            
            fig = px.line(price_df, x='Age', y='Price', 
                         title=f'{brand} {model} - Price Depreciation',
                         labels={'Age': 'Years Old', 'Price': 'Price (₹)'})
            st.plotly_chart(fig)
    
    with col2:
        st.subheader("🏷️ Current Base Prices")
        
        price_data = []
        for brand_name in list(CAR_DATABASE.keys())[:10]:  # Show first 10 brands
            models = CAR_DATABASE[brand_name]['models'][:3]  # Show first 3 models per brand
            for model_name in models:
                base_price = st.session_state.predictor.get_base_price(brand_name, model_name)
                price_data.append({
                    'Brand': brand_name,
                    'Model': model_name,
                    'Base Price': base_price
                })
        
        price_df = pd.DataFrame(price_data)
        st.dataframe(price_df.style.format({'Base Price': '₹{:,.0f}'}))

def show_csv_training():
    st.subheader("📁 CSV Data Training")
    
    st.info("Upload your CSV file with car data to train the AI model")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = st.session_state.predictor.load_csv_data(uploaded_file)
        
        if df is not None:
            st.write("### Dataset Preview")
            st.dataframe(df.head())
            
            if st.button("🚀 Train Model from CSV", type="primary"):
                success = st.session_state.predictor.train_from_csv(df)
                if success:
                    st.balloons()
                    st.success("Model trained successfully! Now using AI for predictions.")

if __name__ == "__main__":
    main()
