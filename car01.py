# ======================================================
# SMART CAR PRICING SYSTEM - COMPLETE CAR DATABASE
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
# COMPREHENSIVE CAR DATABASE FOR MANUAL INPUT
# ========================================

CAR_DATABASE = {
    'Maruti Suzuki': {
        'models': ['Alto', 'Alto K10', 'S-Presso', 'Celerio', 'Wagon R', 'Ignis', 'Swift', 'Baleno', 'Dzire', 'Ciaz', 
                  'Ertiga', 'XL6', 'Vitara Brezza', 'Jimny', 'Fronx', 'Grand Vitara', 'Eeco', 'Omni', 'Celerio X'],
        'car_types': ['Hatchback', 'Hatchback', 'Hatchback', 'Hatchback', 'Hatchback', 'Hatchback', 'Hatchback', 'Hatchback', 'Sedan', 'Sedan',
                     'MUV', 'MUV', 'SUV', 'SUV', 'SUV', 'SUV', 'Van', 'Van', 'Hatchback'],
        'engine_cc': [796, 998, 998, 998, 998, 1197, 1197, 1197, 1197, 1462,
                     1462, 1462, 1462, 1462, 1197, 1462, 1196, 796, 998],
        'power_hp': [48, 67, 67, 67, 67, 83, 90, 90, 90, 103,
                    103, 103, 103, 103, 90, 103, 73, 35, 67],
        'seats': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                 7, 6, 5, 5, 5, 5, 5, 8, 5]
    },
    'Hyundai': {
        'models': ['i10', 'i20', 'Aura', 'Grand i10 Nios', 'Verna', 'Creta', 'Venue', 'Alcazar', 'Tucson', 'Kona Electric',
                  'Santro', 'Xcent', 'Elantra', 'Ioniq 5'],
        'car_types': ['Hatchback', 'Hatchback', 'Sedan', 'Hatchback', 'Sedan', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV',
                     'Hatchback', 'Sedan', 'Sedan', 'SUV'],
        'engine_cc': [1086, 1197, 1197, 1197, 1493, 1493, 1197, 2199, 2199, 0,
                     1086, 1197, 1999, 0],
        'power_hp': [69, 83, 83, 83, 115, 115, 83, 148, 148, 136,
                    69, 83, 152, 217],
        'seats': [5, 5, 5, 5, 5, 5, 5, 6, 5, 5,
                 5, 5, 5, 5]
    },
    'Tata': {
        'models': ['Tiago', 'Tigor', 'Altroz', 'Nexon', 'Punch', 'Harrier', 'Safari', 'Nexon EV', 'Tigor EV', 'Tiago EV',
                  'Indica', 'Indigo', 'Sumo', 'Hexa'],
        'car_types': ['Hatchback', 'Sedan', 'Hatchback', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'Sedan', 'Hatchback',
                     'Hatchback', 'Sedan', 'SUV', 'SUV'],
        'engine_cc': [1199, 1199, 1199, 1199, 1199, 1956, 1956, 0, 0, 0,
                     1405, 1405, 2179, 2179],
        'power_hp': [85, 85, 85, 120, 120, 170, 170, 129, 75, 75,
                    70, 70, 120, 156],
        'seats': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                 5, 5, 8, 7]
    },
    'Mahindra': {
        'models': ['Bolero', 'Scorpio', 'XUV300', 'XUV400', 'XUV700', 'Thar', 'Marazzo', 'KUV100', 'TUV300', 'Alturas G4',
                  'Bolero Neo', 'Scorpio N', 'Verito', 'Xylo'],
        'car_types': ['SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'MUV', 'Hatchback', 'SUV', 'SUV',
                     'SUV', 'SUV', 'Sedan', 'MUV'],
        'engine_cc': [1493, 2179, 1197, 0, 1997, 1997, 1497, 1198, 1493, 2157,
                     1493, 1997, 1461, 2179],
        'power_hp': [75, 140, 110, 150, 200, 150, 123, 83, 100, 178,
                    100, 200, 65, 120],
        'seats': [7, 7, 5, 5, 7, 4, 8, 5, 7, 7,
                 7, 7, 5, 8]
    },
    'Toyota': {
        'models': ['Innova Crysta', 'Fortuner', 'Glanza', 'Urban Cruiser Hyryder', 'Camry', 'Vellfire', 'Hilux', 'Etios', 
                  'Etios Liva', 'Yaris', 'Corolla Altis', 'Innova Hycross'],
        'car_types': ['MUV', 'SUV', 'Hatchback', 'SUV', 'Sedan', 'MUV', 'Pickup', 'Sedan',
                     'Hatchback', 'Sedan', 'Sedan', 'MUV'],
        'engine_cc': [2393, 2694, 1197, 1462, 2487, 2494, 2755, 1496,
                     1496, 1496, 1798, 1987],
        'power_hp': [150, 204, 90, 103, 177, 197, 204, 90,
                    90, 107, 140, 186],
        'seats': [7, 7, 5, 5, 5, 7, 5, 5,
                 5, 5, 5, 7]
    },
    'Honda': {
        'models': ['Amaze', 'City', 'Jazz', 'WR-V', 'Elevate', 'Civic', 'CR-V', 'Brio'],
        'car_types': ['Sedan', 'Sedan', 'Hatchback', 'SUV', 'SUV', 'Sedan', 'SUV', 'Hatchback'],
        'engine_cc': [1199, 1498, 1199, 1199, 1498, 1799, 1997, 1198],
        'power_hp': [90, 121, 90, 90, 121, 141, 158, 88],
        'seats': [5, 5, 5, 5, 5, 5, 5, 5]
    },
    'Kia': {
        'models': ['Seltos', 'Sonet', 'Carens', 'Carnival', 'EV6', 'Rio'],
        'car_types': ['SUV', 'SUV', 'MUV', 'MUV', 'SUV', 'Hatchback'],
        'engine_cc': [1353, 1197, 1493, 2199, 0, 1248],
        'power_hp': [115, 83, 115, 200, 229, 83],
        'seats': [5, 5, 6, 7, 5, 5]
    },
    'Volkswagen': {
        'models': ['Polo', 'Vento', 'Virtus', 'Taigun', 'Tiguan', 'T-Roc'],
        'car_types': ['Hatchback', 'Sedan', 'Sedan', 'SUV', 'SUV', 'SUV'],
        'engine_cc': [999, 999, 999, 999, 1984, 1498],
        'power_hp': [110, 110, 110, 115, 190, 150],
        'seats': [5, 5, 5, 5, 5, 5]
    },
    'Skoda': {
        'models': ['Rapid', 'Slavia', 'Kushaq', 'Kodiaq', 'Superb', 'Octavia'],
        'car_types': ['Sedan', 'Sedan', 'SUV', 'SUV', 'Sedan', 'Sedan'],
        'engine_cc': [999, 999, 999, 1984, 1984, 1984],
        'power_hp': [110, 115, 115, 190, 190, 190],
        'seats': [5, 5, 5, 7, 5, 5]
    },
    'MG': {
        'models': ['Hector', 'Astor', 'Gloster', 'ZS EV', 'Comet EV'],
        'car_types': ['SUV', 'SUV', 'SUV', 'SUV', 'Hatchback'],
        'engine_cc': [1451, 1349, 1996, 0, 0],
        'power_hp': [143, 134, 218, 177, 42],
        'seats': [5, 5, 7, 5, 4]
    },
    'Renault': {
        'models': ['Kwid', 'Triber', 'Kiger', 'Duster', 'Captur'],
        'car_types': ['Hatchback', 'MUV', 'SUV', 'SUV', 'SUV'],
        'engine_cc': [999, 999, 999, 1498, 1498],
        'power_hp': [68, 72, 100, 106, 106],
        'seats': [5, 7, 5, 5, 5]
    },
    'Nissan': {
        'models': ['Magnite', 'Kicks', 'Micra', 'Sunny', 'Terrano'],
        'car_types': ['SUV', 'SUV', 'Hatchback', 'Sedan', 'SUV'],
        'engine_cc': [999, 1498, 1198, 1498, 1461],
        'power_hp': [100, 106, 77, 99, 110],
        'seats': [5, 5, 5, 5, 5]
    },
    'Ford': {
        'models': ['EcoSport', 'Endeavour', 'Figo', 'Aspire', 'Freestyle'],
        'car_types': ['SUV', 'SUV', 'Hatchback', 'Sedan', 'Hatchback'],
        'engine_cc': [1498, 1996, 1194, 1194, 1194],
        'power_hp': [123, 170, 96, 96, 96],
        'seats': [5, 7, 5, 5, 5]
    },
    'BMW': {
        'models': ['3 Series', '5 Series', '7 Series', 'X1', 'X3', 'X5', 'X7', 'i4', 'iX'],
        'car_types': ['Sedan', 'Sedan', 'Sedan', 'SUV', 'SUV', 'SUV', 'SUV', 'Sedan', 'SUV'],
        'engine_cc': [1998, 1998, 2998, 1499, 1998, 2998, 2998, 0, 0],
        'power_hp': [255, 248, 335, 140, 248, 335, 335, 340, 523],
        'seats': [5, 5, 5, 5, 5, 5, 7, 5, 5]
    },
    'Mercedes-Benz': {
        'models': ['A-Class', 'C-Class', 'E-Class', 'S-Class', 'GLA', 'GLC', 'GLE', 'GLS', 'EQC'],
        'car_types': ['Sedan', 'Sedan', 'Sedan', 'Sedan', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV'],
        'engine_cc': [1332, 1496, 1991, 2999, 1332, 1991, 1991, 2999, 0],
        'power_hp': [163, 204, 258, 435, 163, 258, 258, 325, 408],
        'seats': [5, 5, 5, 5, 5, 5, 5, 7, 5]
    },
    'Audi': {
        'models': ['A3', 'A4', 'A6', 'A8', 'Q3', 'Q5', 'Q7', 'Q8', 'e-tron'],
        'car_types': ['Sedan', 'Sedan', 'Sedan', 'Sedan', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV'],
        'engine_cc': [1395, 1984, 1984, 2995, 1395, 1984, 2995, 2995, 0],
        'power_hp': [150, 190, 245, 335, 150, 245, 335, 335, 355],
        'seats': [5, 5, 5, 5, 5, 5, 7, 5, 5]
    },
    'Lexus': {
        'models': ['ES', 'LS', 'RX', 'NX', 'UX', 'LC'],
        'car_types': ['Sedan', 'Sedan', 'SUV', 'SUV', 'SUV', 'Coupe'],
        'engine_cc': [2487, 3445, 3456, 2487, 1987, 4969],
        'power_hp': [215, 416, 295, 235, 169, 471],
        'seats': [5, 5, 5, 5, 5, 4]
    },
    'Jaguar': {
        'models': ['XE', 'XF', 'XJ', 'F-PACE', 'E-PACE', 'I-PACE'],
        'car_types': ['Sedan', 'Sedan', 'Sedan', 'SUV', 'SUV', 'SUV'],
        'engine_cc': [1997, 1997, 2993, 1997, 1997, 0],
        'power_hp': [247, 247, 335, 247, 247, 400],
        'seats': [5, 5, 5, 5, 5, 5]
    },
    'Land Rover': {
        'models': ['Range Rover', 'Range Rover Sport', 'Range Rover Velar', 'Range Rover Evoque', 'Discovery', 'Defender'],
        'car_types': ['SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV'],
        'engine_cc': [2996, 2996, 1997, 1997, 2996, 2996],
        'power_hp': [355, 355, 247, 247, 355, 400],
        'seats': [5, 5, 5, 5, 7, 7]
    },
    'Porsche': {
        'models': ['911', 'Panamera', 'Cayenne', 'Macan', 'Taycan'],
        'car_types': ['Coupe', 'Sedan', 'SUV', 'SUV', 'Sedan'],
        'engine_cc': [2981, 2894, 2995, 1984, 0],
        'power_hp': [385, 325, 340, 261, 402],
        'seats': [4, 5, 5, 5, 5]
    },
    'Volvo': {
        'models': ['S60', 'S90', 'XC40', 'XC60', 'XC90'],
        'car_types': ['Sedan', 'Sedan', 'SUV', 'SUV', 'SUV'],
        'engine_cc': [1969, 1969, 1969, 1969, 1969],
        'power_hp': [197, 197, 197, 197, 197],
        'seats': [5, 5, 5, 5, 7]
    },
    'Jeep': {
        'models': ['Compass', 'Wrangler', 'Grand Cherokee', 'Meridian'],
        'car_types': ['SUV', 'SUV', 'SUV', 'SUV'],
        'engine_cc': [1368, 1998, 1998, 1998],
        'power_hp': [170, 268, 268, 170],
        'seats': [5, 5, 5, 7]
    },
    'Citroen': {
        'models': ['C3', 'C5 Aircross', 'e-C3'],
        'car_types': ['Hatchback', 'SUV', 'Hatchback'],
        'engine_cc': [1199, 1499, 0],
        'power_hp': [82, 130, 57],
        'seats': [5, 5, 5]
    },
    'Mitsubishi': {
        'models': ['Outlander', 'Pajero Sport', 'Eclipse Cross'],
        'car_types': ['SUV', 'SUV', 'SUV'],
        'engine_cc': [2360, 2477, 1468],
        'power_hp': [166, 178, 163],
        'seats': [7, 7, 5]
    },
    'Isuzu': {
        'models': ['D-Max', 'MU-X'],
        'car_types': ['Pickup', 'SUV'],
        'engine_cc': [1898, 1898],
        'power_hp': [164, 164],
        'seats': [5, 7]
    },
    'Force': {
        'models': ['Gurkha', 'Gurkha Xtreme', 'Traveller'],
        'car_types': ['SUV', 'SUV', 'Van'],
        'engine_cc': [2596, 2596, 2596],
        'power_hp': [80, 80, 80],
        'seats': [5, 5, 13]
    },
    'Mini': {
        'models': ['Cooper', 'Countryman', 'Clubman'],
        'car_types': ['Hatchback', 'SUV', 'Hatchback'],
        'engine_cc': [1499, 1499, 1499],
        'power_hp': [136, 136, 136],
        'seats': [4, 5, 5]
    },
    'BYD': {
        'models': ['Atto 3', 'E6', 'Han'],
        'car_types': ['SUV', 'MPV', 'Sedan'],
        'engine_cc': [0, 0, 0],
        'power_hp': [201, 95, 218],
        'seats': [5, 5, 5]
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
# SIMPLIFIED PRICE PREDICTION ENGINE
# ========================================

class CarPricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_importance = {}
        self.is_trained = False
        self.training_data = None
        self.training_records_count = 0
        
    def get_live_prices(self, brand, model):
        """Get live prices for car models with proper error handling"""
        try:
            # Enhanced price database with more brands
            price_database = {
                'Maruti Suzuki': {
                    'Alto': [150000, 250000, 350000],
                    'Swift': [300000, 450000, 600000],
                    'Baleno': [350000, 500000, 700000],
                    'Dzire': [320000, 480000, 650000],
                    'Vitara Brezza': [500000, 700000, 900000],
                    'Ertiga': [450000, 650000, 850000],
                    'Wagon R': [200000, 300000, 400000],
                    'Celerio': [250000, 350000, 450000]
                },
                'Hyundai': {
                    'i10': [250000, 350000, 450000],
                    'i20': [350000, 500000, 650000],
                    'Creta': [600000, 850000, 1100000],
                    'Verna': [450000, 650000, 850000],
                    'Venue': [450000, 600000, 800000]
                },
                'Tata': {
                    'Tiago': [250000, 350000, 450000],
                    'Nexon': [450000, 650000, 850000],
                    'Altroz': [350000, 500000, 650000],
                    'Harrier': [800000, 1100000, 1400000],
                    'Safari': [900000, 1200000, 1500000]
                },
                'Mahindra': {
                    'Scorpio': [500000, 700000, 900000],
                    'XUV300': [450000, 600000, 800000],
                    'XUV700': [900000, 1200000, 1500000],
                    'Thar': [600000, 850000, 1100000]
                },
                'Toyota': {
                    'Innova Crysta': [1000000, 1400000, 1800000],
                    'Fortuner': [1500000, 2000000, 2500000],
                    'Glanza': [350000, 500000, 650000]
                },
                'Honda': {
                    'City': [450000, 650000, 850000],
                    'Amaze': [350000, 500000, 650000]
                },
                'Kia': {
                    'Seltos': [600000, 850000, 1100000],
                    'Sonet': [450000, 650000, 850000],
                    'Carens': [700000, 950000, 1200000]
                },
                'Volkswagen': {
                    'Polo': [350000, 500000, 650000],
                    'Vento': [400000, 550000, 700000],
                    'Taigun': [700000, 900000, 1100000]
                },
                'BMW': {
                    '3 Series': [1800000, 2500000, 3200000],
                    'X1': [2200000, 3000000, 3800000],
                    'X3': [3500000, 4500000, 5500000]
                },
                'Mercedes-Benz': {
                    'C-Class': [2500000, 3500000, 4500000],
                    'GLA': [2800000, 3800000, 4800000],
                    'E-Class': [4500000, 6000000, 7500000]
                },
                'Audi': {
                    'A4': [2200000, 3200000, 4200000],
                    'Q3': [2500000, 3500000, 4500000],
                    'Q5': [4000000, 5500000, 7000000]
                }
            }
            
            # Check if brand exists in database
            if brand not in price_database:
                default_prices = [300000, 500000, 800000]
                return default_prices, ["Market Estimate - Unknown Brand"]
            
            # Check if model exists for the brand
            if model not in price_database[brand]:
                default_prices = [300000, 500000, 800000]
                return default_prices, ["Market Estimate - Unknown Model"]
            
            # Return the actual prices
            prices = price_database[brand][model]
            sources = ["Market Database"]
            return prices, sources
            
        except Exception as e:
            default_prices = [300000, 500000, 800000]
            return default_prices, ["General Market Average"]

    def load_csv_data(self, uploaded_file):
        """Load and process CSV data for training"""
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(df)} records from CSV")
            
            # Display dataset info
            st.subheader("📊 Dataset Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            # Show sample data
            with st.expander("View Sample Data"):
                st.dataframe(df.head(10))
            
            return df
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
            return None

    def train_from_csv(self, df):
        """Train model from CSV data with flexible column mapping"""
        try:
            st.info("🔄 Training model from CSV data...")
            
            # Show available columns in the uploaded CSV
            st.write("📋 Columns found in your CSV:")
            st.write(list(df.columns))
            
            # Flexible column mapping for different CSV formats
            column_mapping = {
                'brand': 'Brand',
                'car_brand': 'Brand',
                'model': 'Model', 
                'car_model': 'Model',
                'year': 'Year',
                'manufacture_year': 'Year',
                'car_year': 'Year',
                'fuel': 'Fuel_Type',
                'fuel_type': 'Fuel_Type',
                'fueltype': 'Fuel_Type',
                'transmission': 'Transmission',
                'mileage': 'Mileage',
                'km_driven': 'Mileage',
                'kmdriven': 'Mileage',
                'engine': 'Engine_cc',
                'engine_cc': 'Engine_cc',
                'engine_capacity': 'Engine_cc',
                'engine_cap': 'Engine_cc',
                'power': 'Power_HP',
                'power_hp': 'Power_HP',
                'horsepower': 'Power_HP',
                'hp': 'Power_HP',
                'condition': 'Condition',
                'car_condition': 'Condition',
                'price': 'Price',
                'selling_price': 'Price',
                'car_price': 'Price',
                'exshowroom_price': 'Price',
                'price_inr': 'Price',
                'price_inr': 'Price',
                'price_inr': 'Price'  # This will handle Price_INR
            }
            
            # Create a copy to avoid modifying original
            df_processed = df.copy()
            
            # Rename columns if they exist with alternative names
            columns_renamed = False
            for old_col, new_col in column_mapping.items():
                if old_col.lower() in [col.lower() for col in df_processed.columns]:
                    # Find the actual column name (case insensitive)
                    actual_col = [col for col in df_processed.columns if col.lower() == old_col.lower()][0]
                    if new_col not in df_processed.columns:
                        df_processed[new_col] = df_processed[actual_col]
                        st.success(f"✅ Mapped '{actual_col}' → '{new_col}'")
                        columns_renamed = True
            
            # Special handling for Price_INR
            if 'Price_INR' in df_processed.columns and 'Price' not in df_processed.columns:
                df_processed['Price'] = df_processed['Price_INR']
                st.success("✅ Mapped 'Price_INR' → 'Price'")
                columns_renamed = True
            
            if columns_renamed:
                st.write("📋 Updated columns after mapping:")
                st.write(list(df_processed.columns))
            
            # Required columns
            required_columns = ['Brand', 'Model', 'Year', 'Fuel_Type', 'Transmission', 
                              'Mileage', 'Engine_cc', 'Power_HP', 'Condition', 'Price']
            
            # Check if required columns exist
            missing_columns = [col for col in required_columns if col not in df_processed.columns]
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                
                # Show helpful information
                st.info("""
                **📋 Required Column Format:**
                
                | Brand | Model | Year | Fuel_Type | Transmission | Mileage | Engine_cc | Power_HP | Condition | Price |
                |-------|-------|------|-----------|--------------|---------|-----------|----------|-----------|-------|
                | Maruti Suzuki | Swift | 2020 | Petrol | Manual | 25000 | 1197 | 90 | Very Good | 450000 |
                
                **🔄 Alternative Column Names Accepted:**
                - **Brand:** brand, car_brand
                - **Model:** model, car_model  
                - **Year:** year, manufacture_year, car_year
                - **Fuel_Type:** fuel, fuel_type, fueltype
                - **Transmission:** transmission
                - **Mileage:** mileage, km_driven, kmdriven
                - **Engine_cc:** engine, engine_cc, engine_capacity, engine_cap
                - **Power_HP:** power, power_hp, horsepower, hp
                - **Condition:** condition, car_condition
                - **Price:** price, selling_price, car_price, exshowroom_price, price_inr, Price_INR
                """)
                
                # Show sample of current data
                with st.expander("View Your CSV Data Sample"):
                    st.dataframe(df.head())
                
                return False
            
            # Clean data
            df_clean = df_processed.dropna()
            if len(df_clean) < 10:
                st.error("Not enough data after cleaning. Need at least 10 records.")
                return False
            
            st.success(f"✅ Data validated! Using {len(df_clean)} records for training.")
            
            # Prepare features
            features = ['Brand', 'Model', 'Year', 'Fuel_Type', 'Transmission',
                       'Mileage', 'Engine_cc', 'Power_HP', 'Condition']
            
            X = df_clean[features]
            y = df_clean['Price']
            
            # Show data summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training Records", len(X))
            with col2:
                st.metric("Price Range", f"₹{y.min():,} - ₹{y.max():,}")
            with col3:
                st.metric("Average Price", f"₹{y.mean():,.0f}")
            
            # Encode categorical variables
            categorical_features = ['Brand', 'Model', 'Fuel_Type', 'Transmission', 'Condition']
            for feature in categorical_features:
                self.encoders[feature] = LabelEncoder()
                X[feature] = self.encoders[feature].fit_transform(X[feature].astype(str))
            
            # Scale numerical features
            numerical_features = ['Year', 'Mileage', 'Engine_cc', 'Power_HP']
            X[numerical_features] = self.scaler.fit_transform(X[numerical_features])
            
            # Train model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42
            )
            
            self.model.fit(X, y)
            self.is_trained = True
            self.training_data = df_clean
            self.training_records_count = len(df_clean)
            
            # Store feature importance
            self.feature_importance = dict(zip(features, self.model.feature_importances_))
            
            # Evaluate model
            y_pred = self.model.predict(X)
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            
            st.success(f"✅ Model trained successfully!")
            
            # Show performance metrics
            st.subheader("📊 Model Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{r2:.3f}")
            with col2:
                st.metric("MAE", f"₹{mae:,.0f}")
            with col3:
                st.metric("RMSE", f"₹{rmse:,.0f}")
            
            # Show feature importance
            st.subheader("📈 Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': list(self.feature_importance.keys()),
                'Importance': list(self.feature_importance.values())
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        title='Feature Importance (Trained from CSV)')
            st.plotly_chart(fig, use_container_width=True)
            
            return True
            
        except Exception as e:
            st.error(f"Error training from CSV: {str(e)}")
            return False

    def predict_price(self, input_data):
        """Predict car price"""
        if not self.is_trained:
            # Use fallback if no model trained
            return self.fallback_prediction(input_data)
        
        try:
            # Prepare input features
            features = ['Brand', 'Model', 'Year', 'Fuel_Type', 'Transmission',
                       'Mileage', 'Engine_cc', 'Power_HP', 'Condition']
            
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical variables
            for feature in ['Brand', 'Model', 'Fuel_Type', 'Transmission', 'Condition']:
                if feature in self.encoders:
                    try:
                        input_df[feature] = self.encoders[feature].transform([input_data[feature]])[0]
                    except:
                        input_df[feature] = 0
            
            # Scale numerical features
            numerical_features = ['Year', 'Mileage', 'Engine_cc', 'Power_HP']
            input_df[numerical_features] = self.scaler.transform(input_df[numerical_features])
            
            # Ensure all features are present
            for feature in features:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            
            input_df = input_df[features]
            
            # Get prediction
            prediction = self.model.predict(input_df)[0]
            return max(100000, int(prediction))
            
        except Exception as e:
            st.warning(f"Using fallback prediction: {str(e)}")
            return self.fallback_prediction(input_data)
    
    def fallback_prediction(self, input_data):
        """Fallback price prediction when model fails"""
        base_prices, _ = self.get_live_prices(input_data['Brand'], input_data['Model'])
        base_price = base_prices[1]
        
        # Simple calculation based on age and condition
        current_year = datetime.now().year
        age = current_year - input_data['Year']
        age_factor = max(0.3, 1 - (age * 0.1))
        
        condition_multipliers = {
            "Excellent": 1.1, "Very Good": 1.0, "Good": 0.9, "Fair": 0.8, "Poor": 0.6
        }
        
        price = base_price * age_factor * condition_multipliers[input_data['Condition']]
        return max(100000, int(price))

# ========================================
# UTILITY FUNCTIONS
# ========================================

def show_brand_statistics():
    """Show statistics about available car brands"""
    st.sidebar.subheader("📈 Brand Statistics")
    
    total_brands = len(CAR_DATABASE)
    total_models = sum(len(CAR_DATABASE[brand]['models']) for brand in CAR_DATABASE)
    
    st.sidebar.info(f"""
    **Database Overview:**
    - 🚗 **Brands:** {total_brands}
    - 🎯 **Models:** {total_models}
    - 📊 **Coverage:** Comprehensive
    """)

def show_manual_input_form():
    """Show comprehensive manual input form for car details"""
    st.subheader("🔧 Complete Car Details Entry")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Brand selection
        brand = st.selectbox("Brand", list(CAR_DATABASE.keys()))
        
        if brand in CAR_DATABASE:
            model = st.selectbox("Model", CAR_DATABASE[brand]['models'])
            
            # Auto-fill technical specifications
            if model in CAR_DATABASE[brand]['models']:
                model_index = CAR_DATABASE[brand]['models'].index(model)
                car_type = CAR_DATABASE[brand]['car_types'][model_index]
                engine_cc = CAR_DATABASE[brand]['engine_cc'][model_index]
                power_hp = CAR_DATABASE[brand]['power_hp'][model_index]
                seats = CAR_DATABASE[brand]['seats'][model_index]
                
                st.text_input("Car Type", value=car_type, disabled=True)
                st.text_input("Engine Capacity", value=f"{engine_cc} cc", disabled=True)
                st.text_input("Power", value=f"{power_hp} HP", disabled=True)
                st.text_input("Seating Capacity", value=f"{seats} seats", disabled=True)
            else:
                car_type = "Sedan"
                engine_cc = 1200
                power_hp = 80
                seats = 5
        
        current_year = datetime.now().year
        year = st.number_input("Manufacturing Year", min_value=1990, max_value=current_year, value=current_year-3)
        
        fuel_type = st.selectbox("Fuel Type", FUEL_TYPES)
        transmission = st.selectbox("Transmission", TRANSMISSIONS)
    
    with col2:
        mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000, value=30000, step=1000)
        color = st.selectbox("Color", COLORS)
        condition = st.selectbox("Car Condition", CAR_CONDITIONS)
        owner_type = st.selectbox("Owner Type", OWNER_TYPES)
        insurance_status = st.selectbox("Insurance Status", INSURANCE_STATUS)
        registration_city = st.selectbox("Registration City", CITIES)
    
    # Return input data
    input_data = {
        'Brand': brand,
        'Model': model,
        'Car_Type': car_type,
        'Year': year,
        'Fuel_Type': fuel_type,
        'Transmission': transmission,
        'Mileage': mileage,
        'Engine_cc': engine_cc,
        'Power_HP': power_hp,
        'Seats': seats,
        'Color': color,
        'Condition': condition,
        'Owner_Type': owner_type,
        'Insurance_Status': insurance_status,
        'Registration_City': registration_city
    }
    
    return input_data

def calculate_confidence(input_data):
    """Calculate prediction confidence"""
    confidence = 80
    
    # Increase confidence for newer cars
    current_year = datetime.now().year
    if current_year - input_data['Year'] <= 5:
        confidence += 10
    
    # Decrease confidence for high mileage
    if input_data['Mileage'] > 100000:
        confidence -= 10
    
    return min(95, max(60, confidence))

def add_to_prediction_history(input_data, predicted_price, confidence):
    """Add prediction to history"""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    history_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'brand': input_data['Brand'],
        'model': input_data['Model'],
        'year': input_data['Year'],
        'mileage': input_data['Mileage'],
        'condition': input_data['Condition'],
        'predicted_price': predicted_price,
        'confidence': confidence
    }
    
    st.session_state.prediction_history.append(history_entry)
    
    # Keep only last 100 predictions
    if len(st.session_state.prediction_history) > 100:
        st.session_state.prediction_history = st.session_state.prediction_history[-100:]

def show_prediction_history():
    """Show prediction history"""
    st.subheader("📋 Prediction History")
    
    if 'prediction_history' not in st.session_state or not st.session_state.prediction_history:
        st.info("No prediction history yet. Make some predictions to see them here!")
        return
    
    # Convert to DataFrame for better display
    history_df = pd.DataFrame(st.session_state.prediction_history[::-1])  # Show newest first
    
    # Display history
    st.dataframe(history_df, use_container_width=True)
    
    # Show statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        total_predictions = len(st.session_state.prediction_history)
        st.metric("Total Predictions", total_predictions)
    with col2:
        avg_confidence = history_df['confidence'].mean()
        st.metric("Average Confidence", f"{avg_confidence:.1f}%")
    with col3:
        if st.button("Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    
    # Show chart of recent predictions
    if len(history_df) > 1:
        st.subheader("📈 Recent Prediction Trends")
        recent_history = history_df.head(10)
        fig = px.line(recent_history, x='timestamp', y='predicted_price', 
                      title='Recent Price Predictions', markers=True)
        st.plotly_chart(fig, use_container_width=True)

# ========================================
# CSV UPLOAD INTERFACE
# ========================================

def show_csv_upload_interface():
    """Show CSV upload interface for dataset learning"""
    st.subheader("📁 Upload Car Dataset CSV")
    
    st.info("""
    **Upload a CSV file with car data to train the AI model.**
    The system automatically recognizes different column names including Price_INR.
    """)
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Load and process CSV data
        df = st.session_state.predictor.load_csv_data(uploaded_file)
        
        if df is not None:
            # Train model button
            if st.button("🚀 Train Model from CSV Data", type="primary"):
                success = st.session_state.predictor.train_from_csv(df)
                if success:
                    st.balloons()

# ========================================
# CAR COMPARISON INTERFACE
# ========================================

def show_car_comparison_interface():
    """Show car comparison interface"""
    st.subheader("🔍 Compare Multiple Cars")
    
    st.info("Compare up to 3 cars side by side to make informed decisions")
    
    # Initialize comparison list in session state
    if 'cars_to_compare' not in st.session_state:
        st.session_state.cars_to_compare = []
    
    # Car selection forms
    cols = st.columns(3)
    
    for i in range(3):
        with cols[i]:
            st.write(f"**Car {i+1}**")
            brand = st.selectbox(f"Brand {i+1}", list(CAR_DATABASE.keys()), key=f"brand_{i}")
            
            if brand in CAR_DATABASE:
                model = st.selectbox(f"Model {i+1}", CAR_DATABASE[brand]['models'], key=f"model_{i}")
                year = st.number_input(f"Year {i+1}", min_value=1990, max_value=datetime.now().year, 
                                     value=datetime.now().year-3, key=f"year_{i}")
                condition = st.selectbox(f"Condition {i+1}", CAR_CONDITIONS, key=f"condition_{i}")
                
                if st.button(f"Add Car {i+1}", key=f"add_{i}"):
                    car_data = {
                        'brand': brand,
                        'model': model,
                        'year': year,
                        'condition': condition
                    }
                    st.session_state.cars_to_compare.append(car_data)
                    st.success(f"Added {brand} {model} to comparison!")
    
    # Show comparison button
    if st.session_state.cars_to_compare and st.button("🔄 Compare Cars", type="primary"):
        compare_cars(st.session_state.cars_to_compare)
    
    # Clear comparison button
    if st.session_state.cars_to_compare:
        if st.button("Clear Comparison"):
            st.session_state.cars_to_compare = []
            st.rerun()

def compare_cars(cars_to_compare):
    """Compare multiple cars"""
    comparison_data = []
    
    with st.spinner("Comparing cars..."):
        for i, car in enumerate(cars_to_compare):
            # Prepare input data for prediction
            input_data = {
                'Brand': car['brand'],
                'Model': car['model'],
                'Year': car['year'],
                'Fuel_Type': 'Petrol',
                'Transmission': 'Manual',
                'Mileage': 30000,
                'Engine_cc': 1200,
                'Power_HP': 80,
                'Condition': car['condition'],
                'Seats': 5
            }
            
            # Get predicted price
            predicted_price = st.session_state.predictor.predict_price(input_data)
            
            # Get market prices
            market_prices, _ = st.session_state.predictor.get_live_prices(car['brand'], car['model'])
            
            # Get car specifications
            if car['brand'] in CAR_DATABASE and car['model'] in CAR_DATABASE[car['brand']]['models']:
                model_index = CAR_DATABASE[car['brand']]['models'].index(car['model'])
                car_type = CAR_DATABASE[car['brand']]['car_types'][model_index]
                engine_cc = CAR_DATABASE[car['brand']]['engine_cc'][model_index]
                power_hp = CAR_DATABASE[car['brand']]['power_hp'][model_index]
                seats = CAR_DATABASE[car['brand']]['seats'][model_index]
            else:
                car_type = "Unknown"
                engine_cc = 0
                power_hp = 0
                seats = 5
            
            comparison_data.append({
                'Car': f"Car {i+1}",
                'Brand': car['brand'],
                'Model': car['model'],
                'Year': car['year'],
                'Type': car_type,
                'Engine (cc)': engine_cc,
                'Power (HP)': power_hp,
                'Seats': seats,
                'Condition': car['condition'],
                'Predicted Price': predicted_price,
                'Market Low': market_prices[0],
                'Market Average': market_prices[1],
                'Market High': market_prices[2],
                'Value Score': (predicted_price / market_prices[1]) * 100 if market_prices[1] > 0 else 0
            })
    
    if comparison_data:
        # Create comparison dataframe
        comparison_df = pd.DataFrame(comparison_data)
        
        st.subheader("📊 Car Comparison Results")
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visual comparison
        st.subheader("📈 Visual Comparison")
        
        # Price comparison chart
        fig1 = px.bar(comparison_df, 
                     x='Car', 
                     y=['Predicted Price', 'Market Average'],
                     title='Price Comparison',
                     barmode='group')
        st.plotly_chart(fig1, use_container_width=True)
        
        # Value score comparison
        fig2 = px.bar(comparison_df,
                     x='Car',
                     y='Value Score',
                     title='Value Score (Higher is Better)',
                     color='Value Score')
        st.plotly_chart(fig2, use_container_width=True)

# ========================================
# SAMPLE CSV GENERATOR
# ========================================

def generate_sample_csv():
    """Generate sample CSV for users to download"""
    st.subheader("📝 Sample CSV Format")
    
    st.info("""
    **Download this sample CSV file to understand the required format.**
    You can then modify it with your own car data and upload for training.
    The system automatically recognizes different column names including Price_INR.
    """)
    
    # Create sample data
    sample_data = {
        'Brand': ['Maruti Suzuki', 'Hyundai', 'Tata', 'Mahindra', 'Toyota', 'Honda', 'Kia', 'Volkswagen'],
        'Model': ['Swift', 'i20', 'Nexon', 'Scorpio', 'Innova Crysta', 'City', 'Seltos', 'Polo'],
        'Year': [2020, 2019, 2021, 2018, 2020, 2022, 2021, 2019],
        'Fuel_Type': ['Petrol', 'Petrol', 'Diesel', 'Diesel', 'Diesel', 'Petrol', 'Petrol', 'Petrol'],
        'Transmission': ['Manual', 'Manual', 'Manual', 'Automatic', 'Manual', 'Automatic', 'Automatic', 'Manual'],
        'Mileage': [25000, 35000, 15000, 60000, 45000, 10000, 20000, 40000],
        'Engine_cc': [1197, 1197, 1199, 2179, 2393, 1498, 1353, 999],
        'Power_HP': [90, 83, 120, 140, 150, 121, 115, 110],
        'Condition': ['Very Good', 'Good', 'Excellent', 'Good', 'Very Good', 'Excellent', 'Very Good', 'Good'],
        'Price_INR': [450000, 350000, 650000, 700000, 1400000, 650000, 850000, 500000]
    }
    
    sample_df = pd.DataFrame(sample_data)
    
    # Convert to CSV
    csv = sample_df.to_csv(index=False)
    
    # Download button
    st.download_button(
        label="📥 Download Sample CSV",
        data=csv,
        file_name="sample_car_data.csv",
        mime="text/csv",
        help="Download sample CSV file to understand the required format"
    )
    
    # Show sample data
    with st.expander("View Sample Data Format"):
        st.dataframe(sample_df)

# ========================================
# MAIN PREDICTION INTERFACE
# ========================================

def show_prediction_interface():
    """Show the price prediction interface"""
    st.subheader("🎯 Car Price Prediction")
    
    # Manual input form
    input_data = show_manual_input_form()
    
    if input_data:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Show market prices
            brand = input_data['Brand']
            model = input_data['Model']
            
            if brand and model:
                with st.spinner('🔍 Analyzing market trends...'):
                    prices, sources = st.session_state.predictor.get_live_prices(brand, model)
                    min_price, avg_price, max_price = prices
                
                st.subheader("📊 Market Intelligence")
                
                market_col1, market_col2, market_col3 = st.columns(3)
                
                with market_col1:
                    st.metric("Market Low", f"₹{min_price:,.0f}")
                with market_col2:
                    st.metric("Market Average", f"₹{avg_price:,.0f}")
                with market_col3:
                    st.metric("Market High", f"₹{max_price:,.0f}")
                
                st.info(f"**Data Sources:** {', '.join(sources)}")
        
        with col2:
            st.subheader("🤖 AI Prediction")
            
            if st.button("🎯 Get Price Prediction", type="primary", use_container_width=True):
                with st.spinner('🤖 Calculating price...'):
                    # Get AI prediction
                    predicted_price = st.session_state.predictor.predict_price(input_data)
                    
                    # Show confidence
                    confidence = calculate_confidence(input_data)
                    
                    # Display result
                    st.success(f"**Predicted Price: ₹{predicted_price:,.0f}**")
                    st.metric("Confidence Level", f"{confidence}%")
                    
                    # Add to prediction history
                    add_to_prediction_history(input_data, predicted_price, confidence)
                    
                    st.balloons()

# ========================================
# MAIN APPLICATION
# ========================================

def main():
    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = CarPricePredictor()
    
    st.set_page_config(
        page_title="Advanced Car Price Predictor", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )
    
    st.title("🚗 Advanced Car Price Prediction System")
    st.markdown("### **AI-Powered Price Estimation with Dataset Learning**")
    
    # Show brand statistics in sidebar
    show_brand_statistics()
    
    # Navigation
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/48/car.png")
        st.title("Navigation")
        page = st.radio("Go to", [
            "🎯 Price Prediction", 
            "📁 CSV Upload & Learning", 
            "🔍 Car Comparison", 
            "📋 Prediction History",
            "📝 Sample CSV"
        ])
        
        st.markdown("---")
        st.subheader("AI Features")
        st.success("✅ Machine Learning Model")
        st.success("✅ CSV Dataset Learning")
        st.success("✅ Real-Time Market Data")
        st.success("✅ Car Comparison")
        st.success("✅ Prediction History")
        
        st.markdown("---")
        st.subheader("Model Status")
        if st.session_state.predictor.is_trained:
            st.success("✅ Model Trained")
            # Safe check for training_data
            if hasattr(st.session_state.predictor, 'training_data') and st.session_state.predictor.training_data is not None:
                st.info(f"📊 Trained on {st.session_state.predictor.training_records_count} records")
        else:
            st.warning("⚠️ Using Fallback Model")
    
    # Page routing
    if page == "🎯 Price Prediction":
        show_prediction_interface()
    
    elif page == "📁 CSV Upload & Learning":
        show_csv_upload_interface()
        st.markdown("---")
        generate_sample_csv()
    
    elif page == "🔍 Car Comparison":
        show_car_comparison_interface()
    
    elif page == "📋 Prediction History":
        show_prediction_history()
    
    elif page == "📝 Sample CSV":
        generate_sample_csv()

if __name__ == "__main__":
    main()
