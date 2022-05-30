import pickle
from data_loader_parth import DataLoader
AF_DATASET_DIR = r'C:\Users\Parth Modi\Desktop\training\\'
LABEL_PATH = r'C:\Users\Parth Modi\Desktop\training\REFERENCE.csv'


dataloader = DataLoader()

ECG_AF = dataloader.load_af_challenge_db(AF_DATASET_DIR, LABEL_PATH, save=True)
AF_hrbt = dataloader.process_signals(signals=ECG_AF, sampling_rate=300, save=True, save_name='AF_heartbeat.pkl')
X_af, y_af = dataloader.prepare_input_challenge(AF_hrbt, save=True)