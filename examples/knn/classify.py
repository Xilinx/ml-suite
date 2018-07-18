##################################################
#Copyright (c) 2018, Xilinx, Inc.
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without modification,
#are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice,
#this list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
#this list of conditions and the following disclaimer in the documentation
#and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its contributors
#may be used to endorse or promote products derived from this software
#without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
#IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
#EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
##################################################

import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import gemx
from gemx_knn import GemxKNN
from gemx_rt import GemxRT

args, xclbin_opt = gemx.processCommandLine()
gemx.createGEMMHandle(args, xclbin_opt)

# define column names
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
num_neighbor = 3
# loading training data
iris = datasets.load_iris()
X = iris.data[:, :4]
y = iris.target

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Quantization of floating point input should be applied for better accuracy
#Cast and round input data to int16 for brevity
X_train_int = np.ascontiguousarray(np.rint(X_train), dtype=np.int16)
X_test_int = np.ascontiguousarray(np.rint(X_test), dtype=np.int16)

knn = KNeighborsClassifier(n_neighbors=num_neighbor)
# fitting the model
knn.fit(X_train_int, y_train)
# predict the response
pred_sklearn = knn.predict(X_test)

pred_cpu = []
pred_fpga = []
knnInst = GemxKNN(X_train_int, y_train , X_test.shape, xclbin_opt)
pred_cpu = knnInst.predict_cpu(X_test_int, num_neighbor)
pred_fpga = knnInst.predict_fpga(X_test_int, num_neighbor)

# evaluate accuracy
acc_sklearn = accuracy_score(y_test, pred_sklearn) * 100
print('\nsklearn classifier accuracy: %d%%' % acc_sklearn)

acc_cpu = accuracy_score(y_test, pred_cpu) * 100
print('\nCPU classifier accuracy: %d%%' % acc_cpu)

acc_fpga = accuracy_score(y_test, pred_fpga) * 100
print('\nFPGA classifier accuracy: %d%%' % acc_fpga)
