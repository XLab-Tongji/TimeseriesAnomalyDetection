import pandas as pd
import numpy as np
import os

class DataLoader():
    def __init__(self):
        self.batch_size = 1
        self.data_file_path = "aiopsdata"
        self.padding = 1                # the gap between the windows
        self.num_classes = 2
        self.data_window = 1       # the length of every window
       
        self.kpi_id = 'da403e4e3f87c9e0'
        self.kpi_file = 'test'
        self.data = pd.DataFrame()
        self.data_stream = []
        self.labels = []
        self.sequences = []
        self.x = []
        self.y = []
        self.num_batches = 0
        self.load_data()

   

    def load_aiops_data(self):
        pathname = os.path.join(self.data_file_path, self.kpi_file+ ".csv")
        self.data = pd.read_csv(pathname)
       

        self.data = self.data.reset_index()
        self.data = self.data.drop("timestamp", axis=1)
        self.data = self.data[self.data['KPI ID'] == self.kpi_id]
        print(self.data)
        self.data = self.data.drop("KPI ID", axis=1)
      
        # normorize the raw data
        minx = self.data["value"].min()
        maxx = self.data["value"].max()
        norm = self.data["value"].apply(lambda x: float(x - minx) / (maxx - minx))
        self.data = self.data.drop("value", axis=1)
        self.data["value"] = norm
        self.data = self.data.reset_index()
        for i in range(self.data.shape[0]):
            features = []
            features.append(self.data["value"][i])
            self.data_stream.append(features)
            self.labels.append(self.data["label"][i])
        print(self.data_stream)



    def load_data(self):
        print('loading datafile: ' + self.data_file_path)
        self.load_aiops_data()
        print(self.data.shape[0]-self.data_window)
        for i in range(0, self.data.shape[0]-self.data_window, self.padding):
            sequence = []
            anom = 0
            for j in range(self.data_window):
                sequence.append(self.data_stream[i + j]) #the original version is self.data_stream[j]
                if (self.labels[i  + j] == 1):
                    anom = 1
            print(sequence)
            self.x.append(sequence)
            self.y.append([anom])
       
        self.x = np.array(self.x)
        self.y = np.array(self.y)

        
        print('=====size of the training samples =====')
        print(self.x.shape)
        print('===============END=====================')
        print(self.x)
       
        self.x_test = self.x
        self.y_test = self.y

        self.num_batches_test = int(self.x_test.shape[0]/self.batch_size)
        self.x_test = self.x_test[:self.num_batches_test * self.batch_size]
        self.y_test = self.y_test[:self.num_batches_test * self.batch_size]
        self.batch_size=self.num_batches_test
        print(self.num_batches_test)
        print('===============MMMM=====================')
        print(self.x_test)
      

    def get_test_data(self):
       
        print('===============PPPP====================')
        print(self.y_test)
        return self.x_test, self.y_test

    def reset_batch(self):
        self.pointer = 0

if __name__=="__main__":
    dataloader = DataLoader()
    hhh = dataloader.get_test_data()
    print (hhh[0].shape)
    print (hhh[1].shape)
    print (dataloader.num_batches)

    a = dataloader.y_test
    print (a[a==1].shape)

