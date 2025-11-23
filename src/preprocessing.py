import pandas as pd
import tensorflow as tf

def load_and_preprocess(path, batch_size=32, shuffle=False):
    df = pd.read_csv(path)
    # basic cleaning + normalization (fit locally)
    df = df.dropna()
    features = df[["heart_rate","steps","pm25","temp","humidity"]].astype(float)
    labels = df["risk"].astype(int)
    # local normalization (mean/std computed per node)
    mean = features.mean()
    std = features.std().replace(0,1)
    features = (features - mean) / std

    ds = tf.data.Dataset.from_tensor_slices((
        dict(features.to_dict(orient="list")), # tff likes dict-of-arrays or tensors
        labels.values
    ))
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, mean.to_dict(), std.to_dict()

