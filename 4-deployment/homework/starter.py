import pickle
import pandas as pd
import argparse

def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model

def read_data(filename,categorical):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict duration for given year and month.')
    parser.add_argument('--year', type=int, required=True, help='Year for data (e.g., 2023)')
    parser.add_argument('--month', type=int, required=True, help='Month for data (e.g., 4)')
    args = parser.parse_args()
    year = args.year
    month = args.month
    categorical = ['PULocationID', 'DOLocationID']
    print(f'Downloading data for {year}-{month}')
    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet',categorical)
    print('Data downloaded')
    print('Predicting the duration')
    dicts = df[categorical].to_dict(orient='records')
    dv, model = load_model()
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    print('Mean of Duration:', y_pred.mean())
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df['prediction'] = y_pred
    df_result = df[['ride_id', 'prediction']].copy()
    print('Saving the predictions')
    output_file = f'yellow_tripdata_{year:04d}-{month:02d}_predictions.parquet'
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

