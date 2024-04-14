import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
import time
from matplotlib import pyplot as plt

# Load the CSV file and create a Dataframe
file_path = 'mirai3.csv'
data = pd.read_csv(file_path)
X = data.values

# Split the data into training and testing sets, preserving the order
train_size = 55000
X_train, X_test = X[:train_size], X[train_size:]

# Normalize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the VAE model
latent_dim = 2  # Latent dimensionality of the latent space

class VAE(Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(X_train.shape[1],)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(latent_dim + latent_dim),  # Two outputs: mean and log-variance
        ])
        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(X_train.shape[1]),  # Output should have same dimension as input
        ])

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return epsilon * tf.exp(logvar * .5) + mean

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        reconstructed = self.decode(z)
        return reconstructed

# Define the loss function
def vae_loss(x, reconstructed, mean, logvar):
    reconstruction_loss = losses.mean_squared_error(x, reconstructed)
    reconstruction_loss *= X_train.shape[1]
    kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=1)
    return tf.reduce_mean(reconstruction_loss + kl_loss)

start=time.time()
# Compile and train the VAE model
vae = VAE(latent_dim)
vae.compile(optimizer='adam',
            loss=lambda x, reconstructed: vae_loss(x, reconstructed, mean=vae.encode(x)[0], logvar=vae.encode(x)[1]))
vae.fit(X_train, X_train, epochs=10, batch_size=32, validation_data=(X_test, X_test))

# Evaluate the model
reconstructed_X_test = vae.predict(X_test)

# Compute RMSE
RMSE = np.sqrt(np.mean((X_test - reconstructed_X_test)**2, axis=1))
stop=time.time()
print("time elapsed..."+str(stop-start))

# Compute log probabilities
from scipy.stats import norm
benignSample = np.log(RMSE[1:16000])
logProbs = norm.logsf(np.log(RMSE), np.mean(benignSample), np.std(benignSample))

print("Plotting results")
plt.figure(figsize=(10, 5))
fig = plt.scatter(range(55001, 100000), RMSE, s=0.1, c=logProbs, cmap='RdYlGn')
plt.yscale("log")
plt.title("Anomaly Scores from Kitsune's Execution Phase")
plt.ylabel("RMSE (log scaled)")
plt.xlabel("Number of samples")
figbar = plt.colorbar()
figbar.ax.set_ylabel('Log Probability\n ', rotation=270)
plt.show()
