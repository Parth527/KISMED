# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author: Christoph Hoog Antink, Maurice Rohr
"""

import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import os
from typing import List, Tuple
from data_loader_parth import DataLoader

###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(ecg_leads : List[np.ndarray], fs : float, ecg_names : List[str], model_name : str='model.npy',is_binary_classifier : bool=False) -> List[Tuple[str,str]]:
    '''
    Parameters
    ----------
    model_name : str
        Dateiname des Models. In Code-Pfad
    ecg_leads : list of numpy-Arrays
        EKG-Signale.
    fs : float
        Sampling-Frequenz der Signale.
    ecg_names : list of str
        eindeutige Bezeichnung für jedes EKG-Signal.
    model_name : str
        Name des Models, kann verwendet werden um korrektes Model aus Ordner zu laden
    is_binary_classifier : bool
        Falls getrennte Modelle für F1 und Multi-Score trainiert werden, wird hier übergeben, 
        welches benutzt werden soll
    Returns
    -------
    predictions : list of tuples
        ecg_name und eure Diagnose
    '''

# Euer Code ab hier  
    model = load_model('1DCNN_best_model.h5')
    predictions = list()
    dataloader = DataLoader()
    ECG_Heartbeats = dataloader.process_signals(signals=ecg_leads, ecg_names=ecg_names, sampling_rate=fs, save=False, save_name = 'ECG_heartbeats.pkl', )
    X, ecg_name = dataloader.prepare_input_challenge(ECG_Heartbeats)
    y_pred_prob = model.predict(X)
    y_pred_classes = y_pred_prob.argmax(axis=1)
    for ecg,y in zip(ecg_name, y_pred_classes):
            if y == 1:
                predictions.append((ecg, 'A'))
            else:
                predictions.append((ecg, 'N'))
    pred_set = list(set(predictions))
    visited_data = set()
    predictions = []
    for a, b in sorted(pred_set):
        if not a in visited_data:
            visited_data.add(a)
            predictions.append((a, b))
#------------------------------------------------------------------------------    
    return predictions # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!
                               
                               
        
