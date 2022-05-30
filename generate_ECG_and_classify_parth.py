
from tensorflow.keras.models import load_model
from numpy import asarray
from matplotlib import pyplot
from numpy.random import randn
from Evaluation_parth import *
from sklearn.model_selection import train_test_split
from CNN_model_parth import get_1DCNN
from tensorflow.keras.utils import to_categorical
# load model
model = load_model("./save_model/gen_49999.h5")

def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

def scale_to_recover(Scale, latent_dim, n_samples):
    latent_points = generate_latent_points(latent_dim, n_samples)
    X = model.predict(latent_points)
    return X*2


def InputPreprocess(x_train, y_train, x_test, y_test):
    # reshape x
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    y_true = y_test
    # reshape y
    y_test = to_categorical(y_test, 2, dtype='int8')
    y_train = to_categorical(y_train, 2, dtype='int8')

    return x_train, y_train, x_test, y_test, y_true

X = scale_to_recover(Scale=2,latent_dim=100, n_samples = 100000)
X_temp = np.squeeze(X, axis = -1) # just reshape as of original features and labels

# import original heartbeat features and labels
import pickle
print('Loading original features and Labels')
original_data = pickle.load(open(r"C:\Users\Parth Modi\Desktop\AI_in_Med\GAN_ecg\X_train_af.pkl", 'rb'))
original_label = pickle.load(open(r"C:\Users\Parth Modi\Desktop\AI_in_Med\GAN_ecg\y_af.pkl", 'rb'))
print("Combining Original and Generated data for 'A' class for data balancing")
concated_X  = np.concatenate((original_data, X_temp), axis=0) # combine original and generated for data balance
# This is also called data augmentation
y_temp = np.ones(shape=(100000,), dtype = 'int64')
concated_y = np.concatenate((original_label, y_temp), axis = 0)

print('Generating 80:20 split')

X_train, X_test, y_train, y_test = train_test_split(concated_X,concated_y,test_size=0.2, stratify=concated_y)
x_train, y_train, x_test, y_test, y_true = InputPreprocess(X_train, y_train, X_test, y_test)
history, model = get_1DCNN(x_train, y_train, x_test, y_test)
model.summary()
get_summary(x_test, y_true, model)
Plot_Acc_and_Loss(history, x_test, y_test, model)