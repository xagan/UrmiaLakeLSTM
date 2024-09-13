# Dam Surface Area Prediction (Early Development Stage)

**Note: This project is currently in its first stage of development. The code is basic and under active development. Use with caution and expect significant changes and improvements in future iterations.**

This project aims to predict the surface area of a dam using historical data and LSTM (Long Short-Term Memory) neural networks. It's designed to forecast dam surface area for the next 10 years based on past trends.

## Current Features

- Data loading from Excel file
- Basic data cleaning and preprocessing
- LSTM model for time series prediction
- Future prediction for 10 years (120 months)
- Visualization of historical data and predictions
- Saving predictions to a CSV file

## Dependencies

- pandas
- numpy
- tensorflow
- scikit-learn
- matplotlib

## Usage

1. Ensure all required dependencies are installed.
2. Place your dam data in an Excel file named `dam_data.xlsx` in the same directory as the script.
3. Run the script:
   ```
   python main.py
   ```
4. The script will generate plots and save future predictions to `future_predictions.csv`.

## Current Limitations and Future Work

As this project is in its early stages, there are several limitations and areas for improvement:

1. Limited error handling and data validation
2. Basic LSTM model architecture that may not capture complex patterns
3. Assumes all features have the same value in future predictions
4. Limited hyperparameter tuning
5. No cross-validation or extensive model evaluation

Future development will focus on addressing these limitations and adding more sophisticated features.

## Output

The script currently provides:
- Plots of model loss during training
- A plot of historical data and future predictions
- A CSV file with future predictions

## Caution

This code is experimental and not yet suitable for production use. Results should be interpreted with caution and validated against other forecasting methods.

## Note

This project is designed for educational and research purposes.
