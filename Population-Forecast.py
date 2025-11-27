# ============================================================
# Global Population Forecast (Corrected Version - No Prophet)
# Uses World Bank API + Holt-Winters Exponential Smoothing
# ============================================================

!pip install --quiet pandas-datareader statsmodels matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import wb
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error

# -------- PARAMETERS --------
country = "IN"   # ISO2 code
indicator = "SP.POP.TOTL"
years_back = 40
forecast_years = 10

# -------- FETCH DATA --------
current_year = pd.Timestamp.now().year
start = current_year - years_back
end = current_year

df = wb.download(indicator=indicator, country=country, start=start, end=end)
df = df.reset_index().rename(columns={'country':'country', 'year':'year', indicator:'population'})
df = df.sort_values('year')
df['ds'] = pd.to_datetime(df['year'].astype(str) + "-01-01")
df = df[['ds','population']]

print(f"Fetched {len(df)} rows for {country}")
display(df.tail())

# -------- PLOT HISTORICAL SERIES --------
plt.figure(figsize=(10,5))
plt.plot(df['ds'], df['population'], marker='o', label="Historical Population")
plt.title(f"Population History – {country}")
plt.xlabel("Year")
plt.ylabel("Population")
plt.legend()
plt.grid(True)
plt.show()

# -------- PREPARE FOR FORECAST --------
series = df.set_index('ds')['population'].astype(float)

# Holt-Winters model (trend only, yearly data has no monthly seasonality)
model = ExponentialSmoothing(series, trend='add', seasonal=None, initialization_method="estimated")
fit = model.fit()

# Forecast future years
future_index = pd.date_range(start=series.index[-1] + pd.DateOffset(years=1),
                             periods=forecast_years, freq='YS')
forecast = pd.Series(fit.forecast(forecast_years), index=future_index)

# -------- VISUALIZATION WITH LEGENDS --------
plt.figure(figsize=(12,6))
plt.plot(series.index, series.values, marker='o', label="Historical Population", linewidth=2)
plt.plot(forecast.index, forecast.values, marker='o', linestyle='--',
         color='orange', label="Forecast Population (Holt-Winters)", linewidth=2)
plt.title(f"Population Forecast for {country} (Next {forecast_years} Years)")
plt.xlabel("Year")
plt.ylabel("Population")
plt.legend()
plt.grid(True)
plt.show()

# -------- FORECAST TABLE --------
forecast_table = pd.DataFrame({
    "year": forecast.index.year,
    "predicted_population": forecast.values.astype(int)
})
display(forecast_table)

# -------- BACKTEST MAE (last 5 years) --------
if len(series) > 10:
    train = series.iloc[:-5]
    test = series.iloc[-5:]

    model_bt = ExponentialSmoothing(train, trend='add', seasonal=None)
    fit_bt = model_bt.fit()
    pred_bt = fit_bt.forecast(5)

    mae = mean_absolute_error(test, pred_bt)
    print(f"Backtest MAE (last 5 years): {mae:,.0f} people")
