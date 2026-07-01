import sdmx
import pandas as pd

ecb = sdmx.Client('ECB')
dataflow_id = 'BSI'
series_key = 'M.U2.Y.V.M30.X.1.U2.2300.Z01.E'
parameters = {'startPeriod': '2015-01'}

print("Fetching M3 data from the ECB data portal...")
response = ecb.data(resource_id=dataflow_id, key=series_key, params=parameters)
df = sdmx.to_pandas(response, datetime='TIME_PERIOD')
df = df.reset_index()

print("\nData successfully retrieved!")
print(f"Columns: {df.columns.tolist()}")
print(f"Shape: {df.shape}")
print(f"\nFirst 3 rows:")
print(df.head(3))
print(f"\nLast 5 rows:")
print(df.tail(5))

# Find the value column
value_cols = [c for c in df.columns if 'value' in str(c).lower() or c == 0]
print(f"\nValue columns found: {value_cols}")
