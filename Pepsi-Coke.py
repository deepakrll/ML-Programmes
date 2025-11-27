# Corrected Notebook 2: Brand comparison & short-term forecast using Google Trends (pytrends)
# Replaced Prophet with Statsmodels' ExponentialSmoothing to avoid Prophet Stan backend issues.
!pip install --quiet pytrends statsmodels

from pytrends.request import TrendReq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from google.colab import files
import time, random

# PARAMETERS
kw_list = ["Coca-Cola", "Pepsi"]   # change to any two brand names (or local brands)
geo = "IN"       # country code for Google Trends ('IN' for India). Use '' for worldwide.
timeframe = "today 5-y"  # last 5 years
forecast_steps = 52  # 52 weeks forecast

# Helper: safe pytrends fetch with retries
def fetch_trends(kw_list, geo="IN", timeframe="today 5-y", retries=4):
    pytrends = TrendReq(hl='en-US', tz=330)
    attempt = 0
    while attempt < retries:
        try:
            pytrends.build_payload(kw_list, timeframe=timeframe, geo=geo)
            df = pytrends.interest_over_time()
            if df is None or df.empty:
                raise ValueError("Empty response from pytrends")
            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])
            df = df.reset_index().rename(columns={"date":"ds"})
            return df
        except Exception as e:
            attempt += 1
            wait = (2 ** attempt) + random.random()
            print(f"pytrends attempt {attempt} failed: {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    return None

print("Fetching Google Trends...")
df_trends = fetch_trends(kw_list, geo=geo, timeframe=timeframe, retries=5)

# Fallback: if pytrends failed, synthesize reasonable weekly data (keeps demo running)
if df_trends is None or df_trends.empty:
    print("Google Trends unavailable or rate-limited. Generating synthetic weekly series for demo.")
    # create a weekly date index for last 5 years
    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=5)
    idx = pd.date_range(start=start, end=end, freq='W')
    # produce two synthetic—but realistic—series with seasonality + noise
    def synth_series(base, length):
        t = np.arange(length)
        seasonal = 10 * np.sin(2 * np.pi * t / 52)   # yearly weekly seasonality
        trend = np.linspace(0, 10, length)
        noise = np.random.normal(0, 3, length)
        series = base + seasonal + trend + noise
        # scale 0-100
        s = (series - series.min()) / (series.max() - series.min()) * 100
        return s
    df_trends = pd.DataFrame({"ds": idx})
    df_trends[kw_list[0]] = synth_series(40, len(idx))
    df_trends[kw_list[1]] = synth_series(35, len(idx))
    df_trends = df_trends.reset_index(drop=True)

# Inspect
print("Trends rows:", len(df_trends))
display(df_trends.tail())

# Ensure datetime index and weekly frequency
df_trends['ds'] = pd.to_datetime(df_trends['ds'])
df_trends = df_trends.set_index('ds')
# If data is daily, resample to weekly (mean); if already weekly, this keeps same
df_trends = df_trends.resample('W').mean().interpolate()

# Plot historical series
plt.figure(figsize=(12,5))
for kw in kw_list:
    plt.plot(df_trends.index, df_trends[kw], label=kw, linewidth=2)
plt.legend()
plt.title("Google Trends Interest Over Time (proxy for brand interest)")
plt.xlabel("Date")
plt.ylabel("Relative Interest (0-100)")
plt.grid(True)
plt.show()

# Forecast both using ExponentialSmoothing (Holt-Winters)
forecasts = {}
for kw in kw_list:
    series = df_trends[kw].astype(float)
    # choose seasonal_periods=52 for weekly seasonality; if series shorter, set seasonal_periods to None
    seasonal_periods = 52 if len(series) >= 2*52 else None
    try:
        if seasonal_periods:
            model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=seasonal_periods, initialization_method="estimated")
        else:
            model = ExponentialSmoothing(series, trend='add', seasonal=None, initialization_method="estimated")
        fit = model.fit(optimized=True, use_boxcox=False, remove_bias=False)
        pred = fit.forecast(forecast_steps)
    except Exception as e:
        # fallback simpler exponential smoothing if the above fails
        print(f"ExponentialSmoothing failed for {kw} with error {e}. Using simple trend extrapolation.")
        last = series[-max(12, int(len(series)/10)):]  # use window
        slope = (last[-1] - last[0]) / max(1, (len(last)-1))
        pred = pd.Series([last.iloc[-1] + slope*(i+1) for i in range(forecast_steps)], index=pd.date_range(start=series.index[-1] + pd.Timedelta(weeks=1), periods=forecast_steps, freq='W'))
    forecasts[kw] = pred

# Merge forecasts into DataFrame for comparison
pred_df = pd.DataFrame({f'{kw}_forecast': forecasts[kw] for kw in kw_list})
pred_df.index.name = 'ds'
display(pred_df.head())

# Plot last N weeks of combined history + forecast
history_weeks = 52  # show last 52 weeks of history
history_start = df_trends.index[-history_weeks] if len(df_trends) > history_weeks else df_trends.index[0]
plt.figure(figsize=(12,5))
for kw in kw_list:
    # plot history
    plt.plot(df_trends.loc[history_start:, kw].index, df_trends.loc[history_start:, kw].values, label=f'{kw} (history)')
    # plot forecast
    plt.plot(pred_df.index, pred_df[f'{kw}_forecast'].values, linestyle='--', label=f'{kw} (forecast)')
plt.legend()
plt.title("Last 52 weeks history and 52-week forecast (weekly resolution)")
plt.xlabel("Date")
plt.ylabel("Relative interest")
plt.grid(True)
plt.show()

# Compute % change from start of forecast to end of forecast for each brand
pct_changes = {}
for kw in kw_list:
    start_val = pred_df[f'{kw}_forecast'].iloc[0]
    end_val = pred_df[f'{kw}_forecast'].iloc[-1]
    pct = (end_val - start_val) / (start_val if start_val != 0 else np.nan) * 100
    pct_changes[kw] = pct

print("Percentage change over next 52 weeks (approx):")
for kw, val in pct_changes.items():
    print(f"{kw}: {val:.2f}%")
