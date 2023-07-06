from __future__ import annotations
import pandas as pd
import threading


class CustomThread(threading.Thread):
    
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        threading.Thread.join(self, *args)
        
    
    def value(self):
        return self._return



def get_df_filtered(path, separation, frequency):
    df = pd.read_csv(path, sep=separation)
    fact = 10**6 if df['timestamp'][0]>9*10**9 else 1
    df['timestamp'] = df['timestamp']/fact 
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    return df

