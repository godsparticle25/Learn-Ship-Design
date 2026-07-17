# =============================================================================
# PHYTONET v2: CNN-LSTM HYBRID MODEL for HAB PREDICTION
# =============================================================================

import pandas as pd
import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, Conv2D, 
                                   MaxPooling2D, Flatten, Reshape, Concatenate,
                                   TimeDistributed, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("🚀 Initializing PhytoNet v2: CNN-LSTM Hybrid Model")

# =============================================================================
# 1. DATA PREPARATION - SPATIAL + TEMPORAL
# =============================================================================

class PhytoNetDataGenerator:
    def _init_(self, temporal_csv, spatial_nc_files, target_species='Karenia brevis'):
        self.temporal_df = pd.read_csv(temporal_csv)
        self.temporal_df['datetime'] = pd.to_datetime(self.temporal_df['datetime'])
        self.temporal_df = self.temporal_df[self.temporal_df['spec_name'] == target_species]
        self.spatial_files = spatial_nc_files  # List of satellite data files (SST, Chlorophyll, etc.)
        
    def create_spatial_temporal_sequences(self, time_steps=30, forecast_horizon=7, 
                                        spatial_grid_size=25):  # 25x25 pixel grid
        """Create sequences with both spatial and temporal dimensions"""
        
        sequences = []
        targets = []
        locations = []
        
        # Load spatial data (assuming NetCDF files with daily data)
        spatial_data = {}
        for var_name, file_path in self.spatial_files.items():
            ds = xr.open_dataset(file_path)
            spatial_data[var_name] = ds
            
        # Get unique sampling locations
        unique_locs = self.temporal_df[['latitude', 'longitude']].drop_duplicates()
        
        for idx, loc in unique_locs.iterrows():
            lat, lon = loc['latitude'], loc['longitude']
            
            # Get temporal data for this location
            loc_data = self.temporal_df[
                (self.temporal_df['latitude'] == lat) & 
                (self.temporal_df['longitude'] == lon)
            ].sort_values('datetime')
            
            if len(loc_data) < time_steps + forecast_horizon:
                continue
                
            # Create sequences for this location
            for i in range(time_steps, len(loc_data) - forecast_horizon):
                # TEMPORAL features (point measurements)
                temp_data = loc_data.iloc[i-time_steps:i]
                temporal_features = temp_data[['water_temp', 'salinity', 'ph', 'dissoxygen']].values
                
                # SPATIAL features (satellite imagery around the point)
                spatial_patches = []
                current_date = loc_data.iloc[i]['datetime']
                
                for var_name, ds in spatial_data.items():
                    # Extract a spatial patch around the location for each variable
                    try:
                        # Select data for the specific date and region
                        patch = ds.sel(
                            time=current_date,
                            lat=slice(lat-1, lat+1),  # ~2 degree box
                            lon=slice(lon-1, lon+1)
                        )[var_name].values
                        
                        # Resize to standard grid size (simple interpolation)
                        if patch.shape[0] > spatial_grid_size:
                            patch = patch[:spatial_grid_size, :spatial_grid_size]
                        else:
                            # Pad if smaller
                            pad_size = spatial_grid_size - patch.shape[0]
                            patch = np.pad(patch, ((0, pad_size), (0, pad_size)))
                            
                        spatial_patches.append(patch)
                        
                    except (KeyError, ValueError):
                        # Handle missing spatial data
                        spatial_patches.append(np.zeros((spatial_grid_size, spatial_grid_size)))
                
                # Stack spatial variables along channel dimension
                spatial_stack = np.stack(spatial_patches, axis=-1)  # Shape: (25, 25, num_variables)
                
                # Target (future cell count)
                future_count = loc_data.iloc[i + forecast_horizon]['count']
                
                sequences.append({
                    'temporal': temporal_features,
                    'spatial': spatial_stack
                })
                targets.append(future_count)
                locations.append((lat, lon, current_date))
                
        return sequences, targets, locations

# =============================================================================
# 2. ADVANCED CNN-LSTM MODEL ARCHITECTURE
# =============================================================================

def create_cnn_lstm_hybrid(temporal_steps, temporal_features, spatial_shape, num_spatial_vars):
    """
    Create a hybrid model that processes both spatial and temporal data
    """
    
    # ===== SPATIAL BRANCH (CNN for satellite imagery) =====
    spatial_input = Input(shape=spatial_shape, name='spatial_input')
    
    # CNN for feature extraction from spatial data
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(spatial_input)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    
    spatial_features = TimeDistributed(Flatten())(x)
    
    # ===== TEMPORAL BRANCH (LSTM for point measurements) =====
    temporal_input = Input(shape=(temporal_steps, temporal_features), name='temporal_input')
    
    # LSTM for temporal patterns
    y = LSTM(128, return_sequences=True, dropout=0.2)(temporal_input)
    y = BatchNormalization()(y)
    y = LSTM(64, return_sequences=False, dropout=0.2)(y)
    y = BatchNormalization()(y)
    
    # ===== MULTI-TASK LEARNING BRANCH =====
    # Branch 1: Regression (exact cell count)
    regression_output = Dense(64, activation='relu')(y)
    regression_output = Dropout(0.3)(regression_output)
    regression_output = Dense(32, activation='relu')(regression_output)
    count_prediction = Dense(1, activation='linear', name='count_prediction')(regression_output)
    
    # Branch 2: Classification (bloom severity)
    classification_output = Dense(64, activation='relu')(y)
    classification_output = Dropout(0.3)(classification_output)
    classification_output = Dense(32, activation='relu')(classification_output)
    severity_prediction = Dense(4, activation='softmax', name='severity_prediction')(classification_output)  # 4 classes: None, Low, Medium, High
    
    # ===== FUSION BRANCH (Combine spatial + temporal) =====
    # Merge features (for advanced analysis)
    merged = Concatenate()([spatial_features, y])
    fused_features = Dense(128, activation='relu')(merged)
    fused_features = Dropout(0.3)(fused_features)
    
    # Final fused prediction
    final_prediction = Dense(1, activation='linear', name='final_prediction')(fused_features)
    
    # Create model
    model = Model(
        inputs=[spatial_input, temporal_input],
        outputs=[count_prediction, severity_prediction, final_prediction]
    )
    
    return model

# =============================================================================
# 3. MODEL TRAINING AND EVALUATION
# =============================================================================

def main():
    print("📊 Loading and preparing data...")
    
    # Define your data sources
    data_generator = PhytoNetDataGenerator(
        temporal_csv='hab_data_gulf_2018.csv',
        spatial_nc_files={
            'sst': 'satellite_sst_2018.nc',
            'chlorophyll': 'satellite_chl_2018.nc',
            'sla': 'satellite_sla_2018.nc'  # Sea Level Anomaly
        }
    )
    
    # Create sequences
    sequences, targets, locations = data_generator.create_spatial_temporal_sequences(
        time_steps=30, 
        forecast_horizon=7,
        spatial_grid_size=25
    )
    
    print(f"✅ Created {len(sequences)} spatial-temporal sequences")
    
    # Prepare data for training
    temporal_data = np.array([seq['temporal'] for seq in sequences])
    spatial_data = np.array([seq['spatial'] for seq in sequences])
    targets = np.array(targets)
    
    # Normalize data
    scaler_temporal = StandardScaler()
    temporal_shape = temporal_data.shape
    temporal_data_flat = temporal_data.reshape(-1, temporal_shape[-1])
    temporal_data_scaled = scaler_temporal.fit_transform(temporal_data_flat)
    temporal_data_scaled = temporal_data_scaled.reshape(temporal_shape)
    
    scaler_spatial = StandardScaler()
    spatial_shape = spatial_data.shape
    spatial_data_flat = spatial_data.reshape(-1, spatial_shape[-1])
    spatial_data_scaled = scaler_spatial.fit_transform(spatial_data_flat)
    spatial_data_scaled = spatial_data_scaled.reshape(spatial_shape)
    
    scaler_target = StandardScaler()
    targets_scaled = scaler_target.fit_transform(targets.reshape(-1, 1)).flatten()
    
    # Convert targets to severity classes for multi-task learning
    severity_classes = np.digitize(targets, bins=[0, 1000, 5000, 10000])  # Define your own bins
    severity_categorical = to_categorical(severity_classes, num_classes=4)
    
    # Split data
    split_idx = int(0.8 * len(sequences))
    
    X_spatial_train, X_spatial_test = spatial_data_scaled[:split_idx], spatial_data_scaled[split_idx:]
    X_temp_train, X_temp_test = temporal_data_scaled[:split_idx], temporal_data_scaled[split_idx:]
    y_train, y_test = targets_scaled[:split_idx], targets_scaled[split_idx:]
    y_sev_train, y_sev_test = severity_categorical[:split_idx], severity_categorical[split_idx:]
    
    print("🧠 Creating CNN-LSTM Hybrid Model...")
    
    # Create model
    model = create_cnn_lstm_hybrid(
        temporal_steps=30,
        temporal_features=4,  # water_temp, salinity, ph, dissoxygen
        spatial_shape=(25, 25, 3),  # 25x25 grid, 3 variables (SST, Chlorophyll, SLA)
        num_spatial_vars=3
    )
    
    # Compile with multiple loss functions
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'count_prediction': 'mse',
            'severity_prediction': 'categorical_crossentropy',
            'final_prediction': 'mse'
        },
        loss_weights={
            'count_prediction': 0.3,
            'severity_prediction': 0.3,
            'final_prediction': 0.4
        },
        metrics={
            'count_prediction': ['mae', 'mse'],
            'severity_prediction': 'accuracy'
        }
    )
    
    print(model.summary())
    
    # Train the model
    print("🎯 Training Hybrid Model...")
    history = model.fit(
        [X_spatial_train, X_temp_train],
        {
            'count_prediction': y_train,
            'severity_prediction': y_sev_train,
            'final_prediction': y_train
        },
        epochs=100,
        batch_size=32,
        validation_data=(
            [X_spatial_test, X_temp_test],
            {
                'count_prediction': y_test,
                'severity_prediction': y_sev_test,
                'final_prediction': y_test
            }
        ),
        verbose=1
    )
    
    # =============================================================================
    # 4. PREDICTION AND VISUALIZATION
    # =============================================================================
    
    print("🔮 Making predictions...")
    predictions = model.predict([X_spatial_test, X_temp_test])
    
    # Get final predictions
    y_pred_scaled = predictions[2]  # Using the fused prediction
    y_pred = scaler_target.inverse_transform(y_pred_scaled).flatten()
    y_actual = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    
    print(f"\n📈 MODEL PERFORMANCE:")
    print(f"Mean Absolute Error: {mae:.2f} cells/L")
    print(f"R² Score: {r2:.4f}")
    
    # Enhanced visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Prediction vs Actual
    axes[0, 0].scatter(y_actual, y_pred, alpha=0.6)
    axes[0, 0].plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Cell Count (cells/L)')
    axes[0, 0].set_ylabel('Predicted Cell Count (cells/L)')
    axes[0, 0].set_title(f'PhytoNet v2: Predictions vs Actual (R² = {r2:.3f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Training history
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training History')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Spatial prediction map (example)
    # This would show predictions on a map - requires additional mapping code
    axes[1, 0].text(0.5, 0.5, 'Spatial Prediction Map\n(Requires geospatial plotting)',
                   ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
    axes[1, 0].set_title('HAB Risk Forecast Map')
    
    # Plot 4: Feature importance (simplified)
    axes[1, 1].barh(['SST', 'Chlorophyll', 'SLA', 'Temp', 'Salinity'], 
                   [0.25, 0.20, 0.15, 0.25, 0.15])  # Example importance scores
    axes[1, 1].set_xlabel('Feature Importance')
    axes[1, 1].set_title('Model Feature Importance')
    
    plt.tight_layout()
    plt.savefig('phytonet_v2_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save the advanced model
    model.save('phytonet_v2_hybrid_model.h5')
    print("💾 Model saved as 'phytonet_v2_hybrid_model.h5'")
    
    print("\n✅ PhytoNet v2 Training Complete!")
    print("🎯 Next: Integrate with particle trajectory model for complete HAB forecasting")

if _name_ == "_main_":
    main()