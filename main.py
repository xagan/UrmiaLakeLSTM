import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load the data
df = pd.read_excel('dam_data.xlsx')
print("Data shape after loading:", df.shape)
print("Columns:", df.columns)

# Function to clean and convert columns to numeric
def clean_and_convert(val):
    if isinstance(val, str):
        val = val.strip()
        if val in ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']:
            return val
    try:
        return float(val)
    except ValueError:
        return np.nan

# Apply the cleaning function to all columns
for col in df.columns:
    df[col] = df[col].apply(clean_and_convert)

print("Data types after cleaning:", df.dtypes)
print("Number of null values:", df.isnull().sum())

# Separate year and month columns
year_col = df.columns[0]  # Assuming the first column is the year
month_col = df.columns[1]  # Assuming the second column is the month

# Create a date column
df['Date'] = pd.to_datetime(df[year_col].astype(int).astype(str) + '-' + df[month_col].astype(str), format='%Y-%B')
df = df.sort_values('Date')

# Select features for prediction (excluding year and month columns)
features = df.columns[2:-1]  # All columns except year, month, and Date
print("Selected features:", features)

# Handle missing values
df[features] = df[features].interpolate()

# Check again for any NaNs
if df[features].isnull().sum().sum() > 0:
    print("Shit, still got NaNs after interpolation. Dropping them.")
    df.dropna(inplace=True)
    print("New data shape after dropping NaNs:", df.shape)

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])
print("Scaled data shape:", scaled_data.shape)

# Prepare data for LSTM (use past 12 months to predict next month)
X, y = [], []
for i in range(12, len(scaled_data)):
    X.append(scaled_data[i - 12:i])
    y.append(scaled_data[i, 0])  # Predict the first feature

X, y = np.array(X), np.array(y)
print("X shape:", X.shape)
print("y shape:", y.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
    LSTM(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train the model and add a callback to stop if it goes haywire
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping])

# Evaluate the model
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Train Loss: {train_loss:.4f}')
print(f'Test Loss: {test_loss:.4f}')

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Function to create future predictions
def predict_future(model, last_sequence, num_months=120):
    future_predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(num_months):
        next_pred = model.predict(current_sequence.reshape(1, 12, X.shape[2]))[0, 0]
        future_predictions.append(next_pred)
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = [next_pred] * X.shape[2]  # Assume all features have the same value for simplicity

    return np.array(future_predictions)

# Get the last 12 months of data
last_sequence = scaled_data[-12:]
print("Last sequence shape:", last_sequence.shape)

# Predict for the next 10 years (120 months)
future_scaled = predict_future(model, last_sequence, num_months=120)
print("Future scaled predictions shape:", future_scaled.shape)

# Inverse transform the predictions
future_predictions = scaler.inverse_transform(np.column_stack([future_scaled] * len(features)))[:, 0]
print("Future predictions shape:", future_predictions.shape)
print("First few predictions:", future_predictions[:5])

# Create future dates
last_date = df['Date'].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=120, freq='ME')

# Plot the predictions
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df[features[0]], label='Historical Data')
plt.plot(future_dates, future_predictions, label='Predictions', color='red')
plt.title('Dam Surface Area: Historical Data and Future Predictions')
plt.xlabel('Date')
plt.ylabel('Surface Area')
plt.legend()
plt.show()

# Save predictions to CSV
future_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Surface_Area': future_predictions
})
future_df.to_csv('future_predictions.csv', index=False)
print("Future predictions saved to 'future_predictions.csv'")
print("First few rows of saved predictions:")
print(future_df.head())
