import os
import sys
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap
import matplotlib.pyplot as plt
import seaborn as sn
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score
from PyQt5.QtWidgets import QTableWidgetItem, QMessageBox, QFileDialog
from keras import models, optimizers, regularizers
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization,GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import Model
import keras.utils as us
from keras.saving.legacy.model_config import model_from_json
from keras.utils import load_img,img_to_array
from keras.utils import to_categorical
from keras.applications import DenseNet201,VGG16,ResNet50
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
import cv2

from sklearn.utils import shuffle
from tasarim import Ui_MainWindow

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms
from torch.nn import functional as F
from fastai.vision import *

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def tablolariDoldur(self):
        try:
            X_train = np.load("./Train_Test_Val/X_train.npy")
            y_train = np.load("./Train_Test_Val/y_train.npy")
            X_test = np.load("./Train_Test_Val/X_test.npy")
            y_test = np.load("./Train_Test_Val/y_test.npy")
            X_val = np.load("./Train_Test_Val/X_val.npy")
            y_val = np.load("./Train_Test_Val/y_val.npy")
            data = np.load("./Train_Test_Val/data.npy")
            labels = np.load("./Train_Test_Val/labels.npy")
            self.tblX.setRowCount(data.shape[0])
            self.tblX.setColumnCount(data.shape[1])
            self.tblY.setRowCount(labels.shape[0])
            self.tblY.setColumnCount(1)
            self.tblXTrain.setRowCount(X_train.shape[0])
            self.tblXTrain.setColumnCount(X_train.shape[1])
            self.tblYTrain.setRowCount(y_train.shape[0])
            self.tblYTrain.setColumnCount(1)
            self.tblXTest.setRowCount(X_test.shape[0])
            self.tblXTest.setColumnCount(X_test.shape[1])
            self.tblYTest.setRowCount(y_test.shape[0])
            self.tblYTest.setColumnCount(1)
            self.tblXVal.setRowCount(X_val.shape[0])
            self.tblXVal.setColumnCount(X_val.shape[1])
            self.tblYVal.setRowCount(y_val.shape[0])
            self.tblYVal.setColumnCount(1)
            for i, dt in enumerate(data):
                for j in range(0, data.shape[1]):
                    self.tblX.setItem(i, j, QTableWidgetItem(str(data[i][j][0])))
                self.tblY.setItem(i, 0, QTableWidgetItem(str(labels[i])))
            for i, dt in enumerate(X_train):
                for j in range(0, X_train.shape[1]):
                    self.tblXTrain.setItem(i, j, QTableWidgetItem(str(X_train[i][j][0])))
                self.tblYTrain.setItem(i, 0, QTableWidgetItem(str(y_train[i])))
            for i, dt in enumerate(X_test):
                for j in range(0, X_test.shape[1]):
                    self.tblXTest.setItem(i, j, QTableWidgetItem(str(X_test[i][j][0])))
                self.tblYTest.setItem(i, 0, QTableWidgetItem(str(y_test[i])))
            for i, dt in enumerate(X_val):
                for j in range(0, X_val.shape[1]):
                    self.tblXVal.setItem(i, j, QTableWidgetItem(str(X_val[i][j][0])))
                self.tblYVal.setItem(i, 0, QTableWidgetItem(str(y_val[i])))
        except:
            QMessageBox.about(self, "Hata", "Veri setini doğru bir şekilde oluşturduğunuzdan emin olun!")
            QMessageBox.setStyleSheet(self, " ")
    def readImage(self):
        directory = './dataset/'
        dataset = []
        #mapping = {"paper": 0, "rock": 1, "scissors": 2}
        count = 0
        for file in os.listdir(directory):
            if file == 'README_rpc-cv-images.txt' or file == 'rps-cv-images':
                continue
            path = os.path.join(directory, file)
            if not os.listdir(path):
                continue
            for im in os.listdir(path):
                if im.startswith('.'):
                    continue
                image = load_img(os.path.join(path, im), target_size=(200, 200))
                image = img_to_array(image)
                image = image / 255.0
                dataset.append([image, count])
            count = count + 1
        data, labels = zip(*dataset)
        print("Toplam resim sayısı {}".format(len(labels)))
        c = 0
        c1 = 0
        c2 = 0
        for i in labels:
            if i == 0:
                c += 1
            elif i == 1:
                c1 += 1
            elif i == 2:
                c2 += 1
        print("Taş resim sayısı: {}\nKağıt resim sayısı: {}\nMakas resim sayısı: {}".format(c1,c,c2))
        labels = to_categorical(labels)
        data = np.array(data)
        labels = np.array(labels)
        sizeValueTest = float(self.hsVeriSetiBoyutuTest.value())
        sizeValueTest = sizeValueTest / 100
        sizeValueVal = float(self.hsVeriSetiBoyutuVal.value())
        sizeValueVal = sizeValueVal / 100
        print("Veriler:{}\nEtiketler: {}".format(data.shape, labels.shape))
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=44,shuffle=True)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=44,shuffle=True)
        print("Shape of x_train: {}".format(X_train.shape))
        print("Shape of x_val:   {}".format(X_val.shape))
        print("Shape of x_test:  {}".format(X_test.shape))
        print("Shape of y_train: {}".format(y_train.shape))
        print("Shape of y_val:   {}".format(y_val.shape))
        print("Shape of y_test:  {}".format(y_test.shape))
        n = np.random.randint(0, np.shape(X_test)[0])  # Generating Random Number
        np.save("./Train_Test_Val/X_train.npy",X_train)
        np.save("./Train_Test_Val/X_test.npy",X_test)
        np.save("./Train_Test_Val/X_val.npy",X_val)
        np.save("./Train_Test_Val/y_train.npy",y_train)
        np.save("./Train_Test_Val/y_test.npy",y_test)
        np.save("./Train_Test_Val/y_val.npy",y_val)
        np.save("./Train_Test_Val/data",data)
        np.save("./Train_Test_Val/labels",labels)
        self.tblX.setRowCount(data.shape[0])
        self.tblX.setColumnCount(data.shape[1])
        self.tblY.setRowCount(labels.shape[0])
        self.tblY.setColumnCount(1)
        self.tblXTrain.setRowCount(X_train.shape[0])
        self.tblXTrain.setColumnCount(X_train.shape[1])
        self.tblYTrain.setRowCount(y_train.shape[0])
        self.tblYTrain.setColumnCount(1)
        self.tblXTest.setRowCount(X_test.shape[0])
        self.tblXTest.setColumnCount(X_test.shape[1])
        self.tblYTest.setRowCount(y_test.shape[0])
        self.tblYTest.setColumnCount(1)
        self.tblXVal.setRowCount(X_val.shape[0])
        self.tblXVal.setColumnCount(X_val.shape[1])
        self.tblYVal.setRowCount(y_val.shape[0])
        self.tblYVal.setColumnCount(1)

        for i,dt in enumerate(data):
            for j in range(0, data.shape[1]):
                    self.tblX.setItem(i, j, QTableWidgetItem(str(data[i][j][0])))
            self.tblY.setItem(i, 0, QTableWidgetItem(str(labels[i])))
        for i,dt in enumerate(X_train):
            for j in range(0, X_train.shape[1]):
                    self.tblXTrain.setItem(i, j, QTableWidgetItem(str(X_train[i][j][0])))
            self.tblYTrain.setItem(i, 0, QTableWidgetItem(str(y_train[i])))
        for i,dt in enumerate(X_test):
            for j in range(0, X_test.shape[1]):
                    self.tblXTest.setItem(i, j, QTableWidgetItem(str(X_test[i][j][0])))
            self.tblYTest.setItem(i, 0, QTableWidgetItem(str(y_test[i])))
        for i,dt in enumerate(X_val):
            for j in range(0, X_val.shape[1]):
                    self.tblXVal.setItem(i, j, QTableWidgetItem(str(X_val[i][j][0])))
            self.tblYVal.setItem(i, 0, QTableWidgetItem(str(y_val[i])))
    def confusionMatrix(self):
        if len(self.confusion)!=0:
            df_cm = pd.DataFrame(self.confusion, index=[i for i in "210"],
                                 columns=[i for i in "210"])
            plt.figure(figsize=(10, 7))
            sn.heatmap(df_cm, annot=True,fmt='d')
            plt.show()
        else:
            QMessageBox.about(self, "Hata", "Önce model eğitimini tamamlamalısınız!")
            QMessageBox.setStyleSheet(self, "")
    def confusionMatrixKFold1(self):
        if len(self.confusionFold1) != 0:
            df_cm = pd.DataFrame(self.confusionFold1, index=[i for i in "210"],
                                 columns=[i for i in "210"])
            plt.figure(figsize=(10, 7))
            sn.heatmap(df_cm, annot=True,fmt='d')
            plt.show()
        else:
            QMessageBox.about(self, "Hata", "Önce model eğitimini tamamlamalısınız!")
            QMessageBox.setStyleSheet(self, "")
    def confusionMatrixKFold2(self):
        if len(self.confusionFold2) != 0:
            df_cm = pd.DataFrame(self.confusionFold2, index=[i for i in "210"],
                                 columns=[i for i in "210"])
            plt.figure(figsize=(10, 7))
            sn.heatmap(df_cm, annot=True,fmt='d')
            plt.show()
        else:
            QMessageBox.about(self, "Hata", "Önce model eğitimini tamamlamalısınız!")
            QMessageBox.setStyleSheet(self, "")
    def confusionMatrixKFold3(self):
        if len(self.confusionFold3) != 0:
            df_cm = pd.DataFrame(self.confusionFold3, index=[i for i in "210"],
                                 columns=[i for i in "210"])
            plt.figure(figsize=(10, 7))
            sn.heatmap(df_cm, annot=True,fmt='d')
            plt.show()
        else:
            QMessageBox.about(self, "Hata", "Önce model eğitimini tamamlamalısınız!")
            QMessageBox.setStyleSheet(self, "")
    def confusionMatrixKFold4(self):
        if len(self.confusionFold3) != 0:
            df_cm = pd.DataFrame(self.confusionFold4, index=[i for i in "210"],
                                 columns=[i for i in "210"])
            plt.figure(figsize=(10, 7))
            sn.heatmap(df_cm, annot=True,fmt='d')
            plt.show()
        else:
            QMessageBox.about(self, "Hata", "Önce model eğitimini tamamlamalısınız!")
            QMessageBox.setStyleSheet(self, "")
    def hs_valueChangedTest(self):
        val = "% " + str(self.hsVeriSetiBoyutuTest.value())
        self.lblBoyutTest.setText(val)

    def hs_valueChangedVal(self):
        val = "% " + str(self.hsVeriSetiBoyutuVal.value())
        self.lblBoyutVal.setText(val)
    def getModel(self,X_train):
        base_filtros = 32
        w_regularizers = 1e-5
        model = Sequential()
        model.add(Conv2D(base_filtros, (3, 3), padding='same', strides=(1, 1),kernel_regularizer=regularizers.l2(w_regularizers),
                         input_shape=(200, 200, 3)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(4, 4)))

        model.add(Conv2D(2 * base_filtros, (3, 3), padding='same', strides=(1, 1),
                         kernel_regularizer=regularizers.l2(w_regularizers)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(3, activation='softmax'))
        model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.Adam(),
                      metrics=['accuracy'])
        return model
    def getModelResnet50(self):
        model = ResNet50(include_top=False, weights='imagenet', input_shape=(200, 200, 3))
        av1 = GlobalAveragePooling2D()(model.output)
        fc1 = Dense(256, activation='relu')(av1)
        d1 = Dropout(0.5)(fc1)
        fc2 = Dense(3, activation='softmax')(d1)
        model_new = Model(inputs=model.input, outputs=fc2)
        for ix in range(169):
            model_new.layers[ix].trainable = False
        adam = Adam(learning_rate=0.00003)
        model_new.compile(
            loss='categorical_crossentropy',
            optimizer=adam,
            metrics=['accuracy']
        )
        return model_new

    def getModelVGG16(self):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
        base_model.trainable = False

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(3, activation='softmax')
        ])

        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    def getIndex(self):
        index = self.cmbModel.currentIndex()
        return  index
    def func_performansHesapla(self,cm):
        FP=cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        print("TP:", TP)
        print("TN:", TN)
        print("FP:", FP)
        print("FN:", FN)
        accuracy=0
        sensitivity=0
        specificity=0
        for i in range(0,len(TP)):
            accuracy += (TP[i] + TN[i]) / (TP[i] + FP[i] + TN[i] + FN[i]) * 100
            sensitivity += TP[i] / (TP[i] + FN[i]) * 100
            specificity += TN[i] / (TN[i] + FP[i]) * 100
        accuracy=accuracy/len(TP)
        sensitivity=sensitivity/len(TP)
        specificity=specificity/len(TP)
        print("Accuracy:", accuracy, " Sen:", sensitivity, " Spe", specificity)
        return accuracy,sensitivity,specificity
    def modelEgit(self):
        epochDegeri = self.leEpoch.text()
        batchSize = self.leBatchSize.text()
        if epochDegeri != "" and batchSize != "":
            epochDegeri=int(epochDegeri)
            batchSize=int(batchSize)
            X_train=np.load("./Train_Test_Val/X_train.npy")
            y_train=np.load("./Train_Test_Val/y_train.npy")
            X_test=np.load("./Train_Test_Val/X_test.npy")
            y_test=np.load("./Train_Test_Val/y_test.npy")
            X_val = np.load("./Train_Test_Val/X_val.npy")
            y_val = np.load("./Train_Test_Val/y_val.npy")
            checkpoint = ModelCheckpoint('Checkpoint.hdf5',
                                                 verbose=1,
                                                 save_best_only=True,
                                                 monitor='val_accuracy'
                                                 )
            earlystopping = EarlyStopping(monitor='val_loss',
                                                    mode='min',
                                                    verbose=1,
                                                    patience=5)
            callback=checkpoint
            if self.rbModelCheckpoint.isChecked()==True:
                callback=checkpoint
            elif self.rbErkenDurdurma.isChecked()==True:
                callback=earlystopping

            datagen = ImageDataGenerator(rotation_range=15,
                                                 width_shift_range=0.1,
                                                 height_shift_range=0.1,
                                                 horizontal_flip=True,
                                                 vertical_flip=True
                                                 )

            model=self.getModel(X_train)
            cb=self.getIndex()
            if cb == 0:
                model=self.getModel(X_train)
                self.history = model.fit(X_train, y_train,
                                         callbacks=[callback],
                                         batch_size=batchSize,
                                         epochs=epochDegeri,
                                         verbose=2,
                                         validation_data=(X_val, y_val)
                                         )
            if cb == 1:
                model=self.getModelResnet50()
                self.history = model.fit(X_train,y_train,callbacks=[callback],batch_size=batchSize
                                         ,epochs=epochDegeri,validation_data=(X_val,y_val))
            if cb == 2:
                model=self.getModelVGG16()
                self.history = model.fit(X_train, y_train,callbacks=[callback], batch_size=batchSize
                                         , epochs=epochDegeri, validation_data=(X_val, y_val))
            y_pred = model.predict(X_test)
            pred = np.argmax(y_pred, axis=1)
            ground = np.argmax(y_test, axis=1)
            cm = confusion_matrix(ground,pred)
            self.confusion=cm
            print(cm)
            acc,sens,spe=self.func_performansHesapla(cm)
            self.lblACC.setText("Accuracy: {0}".format(acc))
            self.lblSen.setText("Sensitivity: {0}".format(sens))
            self.lblSpe.setText("Specificity: {0}".format(spe))
            print(classification_report(ground, pred))
            if cb ==0:
                model_json = model.to_json()
                with open("./Models/benimCNN.json", "w") as json_file:
                            json_file.write(model_json)
                # model Ağarlığı
                model.save_weights("./Models/benimCNN.h5")
            elif cb == 1:
                model_json = model.to_json()
                with open("./Models/ResNet50.json", "w") as json_file:
                    json_file.write(model_json)
                # model Ağarlığı
                model.save_weights("./Models/ResNet50.h5")
            elif cb == 2:
                model_json = model.to_json()
                with open("./Models/VGG16.json", "w") as json_file:
                    json_file.write(model_json)
                # model Ağarlığı
                model.save_weights("./Models/VGG16.h5")
            print("Saved model to disk")
            self.egitimOldu=True
        else:
            QMessageBox.about(self, "Hata", "Epoch ve Batch Size değerlerinizi kontrol ediniz !")
            QMessageBox.setStyleSheet(self, " ")
    def getIndexKFold(self):
        index = self.cmbModelKFold.currentIndex()
        return  index
    def KFoldModelEgit(self):
        epochDegeri = self.leEpochKFold.text()
        batchSize = self.leBatchSizeKFold.text()
        if epochDegeri != "" and batchSize != "":
            epochDegeri = int(epochDegeri)
            batchSize = int(batchSize)
            X_train = np.load("./Train_Test_Val/X_train.npy")
            y_train = np.load("./Train_Test_Val/y_train.npy")
            X_test = np.load("./Train_Test_Val/X_test.npy")
            y_test = np.load("./Train_Test_Val/y_test.npy")
            X_val = np.load("./Train_Test_Val/X_val.npy")
            y_val = np.load("./Train_Test_Val/y_val.npy")
            data = np.load("./Train_Test_Val/data.npy")
            checkpoint = ModelCheckpoint('Checkpoint.hdf5',
                                         verbose=1,
                                         save_best_only=True,
                                         monitor='val_accuracy'
                                         )
            earlystopping = EarlyStopping(monitor='val_loss',
                                          mode='min',
                                          verbose=1,
                                          patience=10)
            callback = checkpoint
            if self.rbModelCheckpointKFold.isChecked() == True:
                callback = checkpoint
            elif self.rbErkenDurdurmaKFold.isChecked() == True:
                callback = earlystopping
            datagen = ImageDataGenerator(rotation_range=15,
                                         width_shift_range=0.1,
                                         height_shift_range=0.1,
                                         horizontal_flip=True,
                                         vertical_flip=True
                                         )
            cb = self.getIndexKFold()
            model = self.getModel(X_train)
            if cb == 0:
                model = self.getModel(X_train)
            if cb == 1:
                model = self.getModelDenseNet201()
            if cb == 2:
                model = self.getModelVGG16()

            model.compile(loss='categorical_crossentropy',
                          optimizer=optimizers.Adam(),
                          metrics=['accuracy'])
            X_train1 = []
            X_test1 = []
            y_train1 = []
            y_test1 = []
            X_train2 = []
            X_test2 = []
            y_train2 = []
            y_test2 = []
            X_train3 = []
            X_test3 = []
            y_train3 = []
            y_test3 = []
            X_train4 = []
            X_test4 = []
            y_train4 = []
            y_test4 = []
            score_Model = []
            KFold1XTrain = []
            KFold1XTest = []
            KFold1YTrain = []
            KFold1YTest = []
            KFold2XTrain = []
            KFold2XTest = []
            KFold2YTrain = []
            KFold2YTest = []
            KFold3XTrain = []
            KFold3XTest = []
            KFold3YTrain = []
            KFold3YTest = []
            KFold4XTrain = []
            KFold4XTest = []
            KFold4YTrain = []
            KFold4YTest = []
            acc_per_fold = []
            loss_per_fold = []
            kf = KFold(n_splits=4)
            kf.get_n_splits(data)
            inputs = np.concatenate((X_train, X_test), axis=0)
            targets = np.concatenate((y_train, y_test), axis=0)
            fold_no=1
            for i,(train_index, test_index) in enumerate(kf.split(inputs)):
                if fold_no == 5:
                    break
                print('------------------------------------------------------------------------')
                print(f'Training for fold {fold_no} ...')

                history = model.fit(inputs[train_index], targets[train_index],
                                    batch_size=batchSize,
                                    epochs=epochDegeri,
                                    verbose=1,callbacks=[callback],validation_data=(X_val, y_val))

                scores = model.evaluate(inputs[test_index], targets[test_index], verbose=0)
                print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
                acc_per_fold.append(scores[1] * 100)
                loss_per_fold.append(scores[0])
                if fold_no==1:
                    KFold1XTrain.append(inputs[train_index])
                    KFold1XTest.append(inputs[test_index])
                    KFold1YTrain.append(targets[train_index])
                    KFold1YTest.append(targets[test_index])
                    X_train1, X_test1 = inputs[train_index], inputs[test_index]
                    y_train1, y_test1 = targets[train_index], targets[test_index]
                    y_pred = model.predict(X_test1)
                    pred = np.argmax(y_pred, axis=1)
                    ground = np.argmax(y_test1, axis=1)
                    cm = confusion_matrix(ground, pred)
                    acc = accuracy_score(ground,pred)
                    self.lblFold1.setText(str(acc))
                    self.confusionFold1 = cm
                    self.historyFold1=history
                if fold_no==2:
                    KFold2XTrain.append(inputs[train_index])
                    KFold2XTest.append(inputs[test_index])
                    KFold2YTrain.append(targets[train_index])
                    KFold2YTest.append(targets[test_index])
                    X_train2, X_test2 = inputs[train_index], inputs[test_index]
                    y_train2, y_test2 = targets[train_index], targets[test_index]
                    y_pred = model.predict(X_test2)
                    pred = np.argmax(y_pred, axis=1)
                    ground = np.argmax(y_test2, axis=1)
                    cm = confusion_matrix(ground, pred)
                    acc = accuracy_score(ground, pred)
                    self.lblFold2.setText(str(acc))
                    self.confusionFold2 = cm
                    self.historyFold2 = history
                if fold_no==3:
                    KFold3XTrain.append(inputs[train_index])
                    KFold3XTest.append(inputs[test_index])
                    KFold3YTrain.append(targets[train_index])
                    KFold3YTest.append(targets[test_index])
                    X_train3, X_test3 = inputs[train_index], inputs[test_index]
                    y_train3, y_test3 = targets[train_index], targets[test_index]
                    y_pred = model.predict(X_test3)
                    pred = np.argmax(y_pred, axis=1)
                    ground = np.argmax(y_test3, axis=1)
                    cm = confusion_matrix(ground, pred)
                    acc = accuracy_score(ground, pred)
                    self.lblFold3.setText(str(acc))
                    self.confusionFold3 = cm
                    self.historyFold3 = history
                if fold_no==4:
                    KFold4XTrain.append(inputs[train_index])
                    KFold4XTest.append(inputs[test_index])
                    KFold4YTrain.append(targets[train_index])
                    KFold4YTest.append(targets[test_index])
                    X_train4, X_test4 = inputs[train_index], inputs[test_index]
                    y_train4, y_test4 = targets[train_index], targets[test_index]
                    y_pred = model.predict(X_test4)
                    pred = np.argmax(y_pred, axis=1)
                    ground = np.argmax(y_test4, axis=1)
                    cm = confusion_matrix(ground, pred)
                    acc = accuracy_score(ground, pred)
                    self.lblFold4.setText(str(acc))
                    self.confusionFold4 = cm
                    self.historyFold4 = history
                # Increase fold number
                fold_no = fold_no + 1

            KFold1XTrain = np.array(KFold1XTrain)
            KFold1XTest = np.array(KFold1XTest)
            KFold1YTrain = np.array(KFold1YTrain)
            KFold1YTest = np.array(KFold1YTest)
            KFold2XTrain = np.array(KFold2XTrain)
            KFold2XTest = np.array(KFold2XTest)
            KFold2YTrain = np.array(KFold2YTrain)
            KFold2YTest = np.array(KFold2YTest)
            KFold3XTrain = np.array(KFold3XTrain)
            KFold3XTest = np.array(KFold3XTest)
            KFold3YTrain = np.array(KFold3YTrain)
            KFold3YTest = np.array(KFold3YTest)
            KFold4XTrain = np.array(KFold4XTrain)
            KFold4XTest = np.array(KFold4XTest)
            KFold4YTrain = np.array(KFold4YTrain)
            KFold4YTest = np.array(KFold4YTest)
            column_number = X_train1.shape[1]
            self.tblKFold1XTrain.setRowCount(KFold1XTrain.shape[1])
            self.tblKFold1XTrain.setColumnCount(column_number)
            self.tblKFold1YTrain.setRowCount(KFold1YTrain.shape[1])
            self.tblKFold1YTrain.setColumnCount(1)
            self.tblKFold1XTest.setRowCount(KFold1XTest.shape[1])
            self.tblKFold1XTest.setColumnCount(column_number)
            self.tblKFold1YTest.setRowCount(KFold1YTest.shape[1])
            self.tblKFold1YTest.setColumnCount(1)
            self.tblKFold2XTrain.setRowCount(KFold2XTrain.shape[1])
            self.tblKFold2XTrain.setColumnCount(column_number)
            self.tblKFold2XTest.setRowCount(KFold2XTest.shape[1])
            self.tblKFold2XTest.setColumnCount(column_number)
            self.tblKFold2YTest.setRowCount(KFold2YTest.shape[1])
            self.tblKFold2YTest.setColumnCount(column_number)
            self.tblKFold2YTrain.setRowCount(KFold2YTrain.shape[1])
            self.tblKFold2YTrain.setColumnCount(column_number)
            self.tblKFold3XTrain.setRowCount(KFold3XTrain.shape[1])
            self.tblKFold3XTrain.setColumnCount(column_number)
            self.tblKFold3XTest.setRowCount(KFold3XTest.shape[1])
            self.tblKFold3XTest.setColumnCount(column_number)
            self.tblKFold3YTest.setRowCount(KFold3YTest.shape[1])
            self.tblKFold3YTest.setColumnCount(column_number)
            self.tblKFold3YTrain.setRowCount(KFold3YTrain.shape[1])
            self.tblKFold3YTrain.setColumnCount(column_number)
            self.tblKFold4XTrain.setRowCount(KFold4XTrain.shape[1])
            self.tblKFold4XTrain.setColumnCount(column_number)
            self.tblKFold4XTest.setRowCount(KFold4XTest.shape[1])
            self.tblKFold4XTest.setColumnCount(column_number)
            self.tblKFold4YTest.setRowCount(KFold4YTest.shape[1])
            self.tblKFold4YTest.setColumnCount(column_number)
            self.tblKFold4YTrain.setRowCount(KFold4YTrain.shape[1])
            self.tblKFold4YTrain.setColumnCount(column_number)
            for i, dt in enumerate(X_train1):
                for j in range(0, X_train1.shape[1]):
                    self.tblKFold1XTrain.setItem(i, j, QTableWidgetItem(str(X_train1[i][j][0])))
                self.tblKFold1YTrain.setItem(i, 0, QTableWidgetItem(str(y_train1[i])))
            for i, dt in enumerate(X_test1):
                for j in range(0, X_test1.shape[1]):
                    self.tblKFold1XTest.setItem(i, j, QTableWidgetItem(str(X_test1[i][j][0])))
                self.tblKFold1YTest.setItem(i, 0, QTableWidgetItem(str(y_test1[i])))
            for i, dt in enumerate(X_train2):
                for j in range(0, X_train2.shape[1]):
                    self.tblKFold2XTrain.setItem(i, j, QTableWidgetItem(str(X_train2[i][j][0])))
                self.tblKFold2YTrain.setItem(i, 0, QTableWidgetItem(str(y_train2[i])))
            for i, dt in enumerate(X_test2):
                for j in range(0, X_test2.shape[1]):
                    self.tblKFold2XTest.setItem(i, j, QTableWidgetItem(str(X_test2[i][j][0])))
                self.tblKFold2YTest.setItem(i, 0, QTableWidgetItem(str(y_test2[i])))
            for i, dt in enumerate(X_train3):
                for j in range(0, X_train3.shape[1]):
                    self.tblKFold3XTrain.setItem(i, j, QTableWidgetItem(str(X_train3[i][j][0])))
                self.tblKFold3YTrain.setItem(i, 0, QTableWidgetItem(str(y_train3[i])))
            for i, dt in enumerate(X_test3):
                for j in range(0, X_test3.shape[1]):
                    self.tblKFold3XTest.setItem(i, j, QTableWidgetItem(str(X_test3[i][j][0])))
                self.tblKFold3YTest.setItem(i, 0, QTableWidgetItem(str(y_test3[i])))
            for i, dt in enumerate(X_train4):
                for j in range(0, X_train4.shape[1]):
                    self.tblKFold4XTrain.setItem(i, j, QTableWidgetItem(str(X_train4[i][j][0])))
                self.tblKFold4YTrain.setItem(i, 0, QTableWidgetItem(str(y_train4[i])))
            for i, dt in enumerate(X_test4):
                for j in range(0, X_test4.shape[1]):
                    self.tblKFold4XTest.setItem(i, j, QTableWidgetItem(str(X_test4[i][j][0])))
                self.tblKFold4YTest.setItem(i, 0, QTableWidgetItem(str(y_test4[i])))
            print('------------------------------------------------------------------------')
            print('Score per fold')
            for i in range(0, len(acc_per_fold)):
                print('------------------------------------------------------------------------')
                print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
            print('------------------------------------------------------------------------')
            print('Average scores for all folds:')
            print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
            print(f'> Loss: {np.mean(loss_per_fold)}')
            print('------------------------------------------------------------------------')

            acc1, sens1, spe1 = self.func_performansHesapla(self.confusionFold1)
            acc2, sens2, spe2 = self.func_performansHesapla(self.confusionFold2)
            acc3, sens3, spe3 = self.func_performansHesapla(self.confusionFold3)
            acc4, sens4, spe4 = self.func_performansHesapla(self.confusionFold4)
            acc = (acc1 + acc2 + acc3 + acc4) / 4
            sens = (sens1 + sens2 + sens3 + sens4) / 4
            spe = (spe1 + spe2 + spe3 + spe4) / 4
            self.lblACCKFold.setText("Ortalama Accuracy: {0}".format(acc))
            self.lblSenKFold.setText("Ortalama Sensitivity: {0}".format(sens))
            self.lblSpeKFold.setText("Ortalama Specificity: {0}".format(spe))

            if cb == 0:
                model_json = model.to_json()
                with open("./Models/benimCNN.json", "w") as json_file:
                    json_file.write(model_json)
                # model Ağarlığı
                model.save_weights("./Models/benimCNN.h5")
            elif cb == 1:
                model_json = model.to_json()
                with open("./Models/ResNet50.json", "w") as json_file:
                    json_file.write(model_json)
                # model Ağarlığı
                model.save_weights("./Models/ResNet50.h5")
            elif cb == 2:
                model_json = model.to_json()
                with open("./Models/VGG16.json", "w") as json_file:
                    json_file.write(model_json)
                # model Ağarlığı
                model.save_weights("./Models/VGG16.h5")
            print("Saved model to disk")
            self.egitimOlduFold=True
        else:
            QMessageBox.about(self, "Hata", "Epoch ve Batch Size değerlerinizi kontrol ediniz!")
            QMessageBox.setStyleSheet(self, " ")
    def gorselSec(self):
        dosya_yolu = QFileDialog.getOpenFileName(parent=self, caption="Dosya seç",directory="C:\\Users\\kalbi\\OneDrive\\Masaüstü\\DerinOgrenme\\Tahmin_Verisi")
        dosyaUzanti = dosya_yolu[0].split("/")
        resimAdi = dosyaUzanti[len(dosyaUzanti) - 1]
        dosyaAdi = dosyaUzanti[len(dosyaUzanti) - 2]
        qpixmap = QPixmap("./Tahmin_Verisi/"+dosyaAdi+"/"+resimAdi)
        self.lblGorsel.setPixmap(qpixmap)
        self.resimAdi = resimAdi
        self.dosyaAdi = dosyaAdi
    def getIndexTahmin(self):
        index = self.cmbModelTahmin.currentIndex()
        return  index
    def modelTahmin(self):
        cb=self.getIndexTahmin()
        if cb ==0:
            json_file = open('./Models/benimCNN.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("./Models/benimCNN.h5")
            loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
            anaklasor="./Tahmin_Verisi/"+self.dosyaAdi+"/"
            if self.resimAdi!="":
                uploaded = os.path.join(anaklasor, self.resimAdi)
                img = us.load_img(uploaded, target_size=(200, 200))
                x = us.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                images = np.vstack([x])
                classes = loaded_model.predict(images, batch_size=60)
                print(classes)
                if int(classes[0, 0]) == 1:
                    print('Paper')
                    self.lblSonuc.setText("Kağıt")
                elif int(classes[0, 1]) == 1:
                    print('Rock')
                    self.lblSonuc.setText("Taş")
                elif int(classes[0, 2]) == 1:
                    print('Scissors')
                    self.lblSonuc.setText("Makas")
            else:
                QMessageBox.about(self, "Hata", "Önce tahmin edilecek görseli seçmelisiniz!")
                QMessageBox.setStyleSheet(self, " ")
        elif cb == 1:
            json_file = open('./Models/ResNet50.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("./Models/ResNet50.h5")
            loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
            anaklasor = "./Tahmin_Verisi/" + self.dosyaAdi + "/"
            if self.resimAdi != "":
                uploaded = os.path.join(anaklasor, self.resimAdi)
                img = us.load_img(uploaded, target_size=(200, 200))
                x = us.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                images = np.vstack([x])
                classes = loaded_model.predict(images, batch_size=60)
                print(classes)

                if round(classes[0,0]) == 1:
                    print(round(classes[0,0]))
                    print('Paper')
                    self.lblSonuc.setText("Kağıt")
                elif round(classes[0,1]) == 1:
                    print('Rock')
                    self.lblSonuc.setText("Taş")
                elif round(classes[0,2]) == 1:
                    print('Scissors')
                    self.lblSonuc.setText("Makas")
            else:
                QMessageBox.about(self, "Hata", "Önce tahmin edilecek görseli seçmelisiniz!")
                QMessageBox.setStyleSheet(self, " ")
        elif cb == 2:
            json_file = open('./Models/VGG16.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("./Models/VGG16.h5")
            loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
            anaklasor = "./Tahmin_Verisi/" + self.dosyaAdi + "/"
            if self.resimAdi != "":
                uploaded = os.path.join(anaklasor, self.resimAdi)
                img = us.load_img(uploaded, target_size=(200, 200))
                x = us.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                images = np.vstack([x])
                classes = loaded_model.predict(images, batch_size=60)
                print(classes)
                if round(classes[0,0]) == 1:
                    print(round(classes[0,0]))
                    print('Paper')
                    self.lblSonuc.setText("Kağıt")
                elif round(classes[0,1]) == 1:
                    print('Rock')
                    self.lblSonuc.setText("Taş")
                elif round(classes[0,2]) == 1:
                    print('Scissors')
                    self.lblSonuc.setText("Makas")
            else:
                QMessageBox.about(self, "Hata", "Önce tahmin edilecek görseli seçmelisiniz!")
                QMessageBox.setStyleSheet(self, " ")
    def epochGrafikACC(self):
        if self.egitimOldu==True:
            plt.plot(self.history.history['accuracy'])
            plt.plot(self.history.history['val_accuracy'])
            plt.legend(['train', 'test'], loc='lower right')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.show()
        else:
            QMessageBox.about(self, "Hata", "Önce model eğitimini gerçekleştirmelisiniz!")
            QMessageBox.setStyleSheet(self, " ")
    def epochGrafikLoss(self):
        if self.egitimOldu == True:
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.legend(['train', 'test'], loc='lower right')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.show()
        else:
            QMessageBox.about(self, "Hata", "Önce model eğitimini gerçekleştirmelisiniz!")
            QMessageBox.setStyleSheet(self, " ")
    def epochGrafikACCFold1(self):
        if self.egitimOlduFold == True:
            plt.plot(self.historyFold1.history['accuracy'])
            plt.plot(self.historyFold1.history['val_accuracy'])
            plt.legend(['train', 'test'], loc='lower right')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.show()
        else:
            QMessageBox.about(self, "Hata", "Önce model eğitimini gerçekleştirmelisiniz!")
            QMessageBox.setStyleSheet(self, " ")
    def epochGrafikLossFold1(self):
        if self.egitimOlduFold == True:
            plt.plot(self.historyFold1.history['loss'])
            plt.plot(self.historyFold1.history['val_loss'])
            plt.legend(['train', 'test'], loc='lower right')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.show()
        else:
            QMessageBox.about(self, "Hata", "Önce model eğitimini gerçekleştirmelisiniz!")
            QMessageBox.setStyleSheet(self, " ")
    def epochGrafikACCFold2(self):
        if self.egitimOlduFold == True:
            plt.plot(self.historyFold2.history['accuracy'])
            plt.plot(self.historyFold2.history['val_accuracy'])
            plt.legend(['train', 'test'], loc='lower right')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.show()
        else:
            QMessageBox.about(self, "Hata", "Önce model eğitimini gerçekleştirmelisiniz!")
            QMessageBox.setStyleSheet(self, " ")
    def epochGrafikLossFold2(self):
        if self.egitimOlduFold == True:
            plt.plot(self.historyFold2.history['loss'])
            plt.plot(self.historyFold2.history['val_loss'])
            plt.legend(['train', 'test'], loc='lower right')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.show()
        else:
            QMessageBox.about(self, "Hata", "Önce model eğitimini gerçekleştirmelisiniz!")
            QMessageBox.setStyleSheet(self, " ")
    def epochGrafikACCFold3(self):
        if self.egitimOlduFold == True:
            plt.plot(self.historyFold3.history['accuracy'])
            plt.plot(self.historyFold3.history['val_accuracy'])
            plt.legend(['train', 'test'], loc='lower right')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.show()
        else:
            QMessageBox.about(self, "Hata", "Önce model eğitimini gerçekleştirmelisiniz!")
            QMessageBox.setStyleSheet(self, " ")
    def epochGrafikLossFold3(self):
        if self.egitimOlduFold == True:
            plt.plot(self.historyFold3.history['loss'])
            plt.plot(self.historyFold3.history['val_loss'])
            plt.legend(['train', 'test'], loc='lower right')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.show()
        else:
            QMessageBox.about(self, "Hata", "Önce model eğitimini gerçekleştirmelisiniz!")
            QMessageBox.setStyleSheet(self, " ")
    def epochGrafikACCFold4(self):
        if self.egitimOlduFold == True:
            plt.plot(self.historyFold4.history['accuracy'])
            plt.plot(self.historyFold4.history['val_accuracy'])
            plt.legend(['train', 'test'], loc='lower right')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.show()
        else:
            QMessageBox.about(self, "Hata", "Önce model eğitimini gerçekleştirmelisiniz!")
            QMessageBox.setStyleSheet(self, " ")
    def epochGrafikLossFold4(self):
        if self.egitimOlduFold == True:
            plt.plot(self.historyFold4.history['loss'])
            plt.plot(self.historyFold4.history['val_loss'])
            plt.legend(['train', 'test'], loc='lower right')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.show()
        else:
            QMessageBox.about(self, "Hata", "Önce model eğitimini gerçekleştirmelisiniz!")
            QMessageBox.setStyleSheet(self, " ")
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.history=None
        self.historyFold1=None
        self.historyFold2=None
        self.historyFold3=None
        self.historyFold4=None
        self.resimAdi=""
        self.dosyaAdi=""
        self.btnveri.clicked.connect(self.readImage)
        self.btnGorselSec.clicked.connect(self.gorselSec)
        self.btnModelEgit.clicked.connect(self.modelEgit)
        self.btnModelEgitKFold.clicked.connect(self.KFoldModelEgit)
        self.btnTahminEt.clicked.connect(self.modelTahmin)
        self.btnGrafikEgitim.clicked.connect(self.epochGrafikACC)
        self.btnGrafikEgitimLoss.clicked.connect(self.epochGrafikLoss)
        self.hsVeriSetiBoyutuTest.valueChanged.connect(self.hs_valueChangedTest)
        self.hsVeriSetiBoyutuVal.valueChanged.connect(self.hs_valueChangedVal)
        self.btnTabloDoldur.clicked.connect(self.tablolariDoldur)
        self.confusion=[]
        self.btnConfusion.clicked.connect(self.confusionMatrix)
        self.confusionFold1 = []
        self.confusionFold2 = []
        self.confusionFold3 = []
        self.confusionFold4 = []
        self.btnConfusionKFold1.clicked.connect(self.confusionMatrixKFold1)
        self.btnConfusionKFold2.clicked.connect(self.confusionMatrixKFold2)
        self.btnConfusionKFold3.clicked.connect(self.confusionMatrixKFold3)
        self.btnConfusionKFold4.clicked.connect(self.confusionMatrixKFold4)
        self.btnACCKFold1.clicked.connect(self.epochGrafikACCFold1)
        self.btnACCKFold2.clicked.connect(self.epochGrafikACCFold2)
        self.btnACCKFold3.clicked.connect(self.epochGrafikACCFold3)
        self.btnACCKFold4.clicked.connect(self.epochGrafikACCFold4)
        self.btnLossKFold1.clicked.connect(self.epochGrafikLossFold1)
        self.btnLossKFold2.clicked.connect(self.epochGrafikLossFold2)
        self.btnLossKFold3.clicked.connect(self.epochGrafikLossFold3)
        self.btnLossKFold4.clicked.connect(self.epochGrafikLossFold4)
        self.egitimOldu=False
        self.egitimOlduFold=False
app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
