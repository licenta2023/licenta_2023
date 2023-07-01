
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, inputSize):
        self.inputSize = inputSize
        self.layers = []
        
        self.LossTraining = np.empty(0)
        self.AccuracyTraining = np.empty(0)

        self.LossValidation = np.empty(0)
        self.AccuracyValidation = np.empty(0)
        
    def addLayer(self, newLayerSize, activationFunction, batchNormalization=False, alpha_activationFunction=0.01, beta_activationFunction=1.0, beta_1=0.9, beta_2=0.999, eps=1e-8):
        try:
            # se salveaza dimensiunea ultimului layer daca exista
            lastLayerSize = self.layers[-1].layerSize
        except:
            # daca nu exista atunci se pune dimensiunea intrarii
            lastLayerSize = self.inputSize
        
        self.layers.append(Layer(newLayerSize, lastLayerSize, activationFunction, batchNormalization, alpha_activationFunction, beta_activationFunction, beta_1, beta_2, eps))
    
    def fit(self, x_train, y_train, x_val, y_val, batchSize=1, epochs=1, learningRate=0.001, decayRate=0):
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.decayRate = decayRate
        
        # Intai transpun toate matricile pentru ca eu lucrez cu vectori coloana
        self.x_train = x_train.T
        self.y_train = y_train.T
        self.x_val = x_val.T
        self.y_val = y_val.T
        
        # Apoi fac antrenarea
        # parcurg pe epoci si batch-uri
        nrOfBatchsTraining = int(np.floor(self.x_train.shape[1] / self.batchSize))
        nrOfBatchsValidation = int(np.floor(self.x_val.shape[1] / self.batchSize))
        for epoch in range(epochs):
            # Verificare ce se intampla cu reteaua dupa fiecare epoca
            #for layer in self.layers:
            #    print(np.max(layer.W))
            #    print(np.max(layer.b))
            #    print(np.max(layer.x))
            #    print()
            
            # updatez learning-rate-ul
            self.learningRate /= (1 + self.decayRate * epoch)
            
            # antrenarea
            for batch in range(nrOfBatchsTraining):
                # selectez batch-ul curent
                self.x = self.x_train[:, batch*self.batchSize:(batch+1)*self.batchSize]
                self.y = self.y_train[:, batch*self.batchSize:(batch+1)*self.batchSize]
                
                # forward
                self.y_pred = self.forward(training=True)
                
                # backward
                self.backward()
                
                # update weights
                self.updateWeights(learningRate, epoch*nrOfBatchsTraining+batch+1)
                
            # loss si acuratete pe lotul de antrenare
            self.LossTraining = np.append(self.LossTraining, self.cross_entropy_loss())
            self.AccuracyTraining = np.append(self.AccuracyTraining, self.accuracy())
            
            # dupa fiecare epoca se face validarea
            for batch in range(nrOfBatchsValidation):
                # selectez batch-ul curent
                self.x = self.x_train[:, batch*self.batchSize:(batch+1)*self.batchSize]
                self.y = self.y_train[:, batch*self.batchSize:(batch+1)*self.batchSize]
                
                # forward
                self.y_pred = self.forward()
                
            # loss si acuratete pe lotul de validare
            self.LossValidation = np.append(self.LossValidation, self.cross_entropy_loss())
            self.AccuracyValidation = np.append(self.AccuracyValidation, self.accuracy())
            
            # afisez in consola rezultatele
            print('epoch =', epoch, ' loss_train =', self.LossTraining[-1], 'acc_train =', self.AccuracyTraining[-1], ' loss_val =', self.LossValidation[-1], 'acc_val =', self.AccuracyValidation[-1])
            #input("Press any key to continue...")
    
    def forward(self, training=False):
        a = self.x
        for layer in self.layers:
            a = layer.forward(a, training)
        
        # evit impartirea la zero
        eps = 1e-8
        a = np.clip(a, eps, 1 - eps)
        return a
    
    def backward(self):
        dLoss_da = self.y
        for layer in reversed(self.layers):
            dLoss_da = layer.backward(dLoss_da)
    
    def updateWeights(self, learningRate, t):
        for layer in self.layers:
            #layer.updateWeightsGradientDescent(learningRate)
            layer.updateWeightsAdam(learningRate, t)
    
    def cross_entropy_loss(self):
        loss = np.sum(-self.y * np.log(self.y_pred)) / self.batchSize
        return loss
    
    def cross_entropy_loss_derivative(self):
        dLoss_dy_pred = -self.y / self.y_pred
        return dLoss_dy_pred
    
    def accuracy(self):
        return np.sum(np.argmax(self.y, axis=0) == np.argmax(self.y_pred, axis=0)) / self.y.shape[1]

    def lossAndAccuracyPlot(self):
        print()
        print('Loss Training =', self.LossTraining[-1])
        print('Accuracy Training =', self.AccuracyTraining[-1])
        
        fig, ax1 = plt.subplots()

        ax1.plot(self.LossTraining, color='red', label='loss')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss', color='red')
        ax1.tick_params(axis='y', labelcolor='red')

        ax2 = ax1.twinx()
        ax2.plot(self.AccuracyTraining, color='blue', label='accuracy')
        ax2.set_ylabel('accuracy', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        ax1.set_title('Loss and Accuracy Training')

        plt.show()


        print()
        print('Loss Validation =', self.LossValidation[-1])
        print('Accuracy Validation =', self.AccuracyValidation[-1])
        fig, ax1 = plt.subplots()

        ax1.plot(self.LossValidation, color='red', label='loss')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss', color='red')
        ax1.tick_params(axis='y', labelcolor='red')

        ax2 = ax1.twinx()
        ax2.plot(self.AccuracyValidation, color='blue', label='accuracy')
        ax2.set_ylabel('accuracy', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        ax1.set_title('Loss and Accuracy Validation')

        plt.show()

class Layer:
    def __init__(self, newLayerSize, lastLayerSize, activationFunction, batchNorm, alpha_activationFunction, beta_activationFunction, beta_1, beta_2, eps):
        self.layerSize = newLayerSize
        self.lastLayerSize = lastLayerSize
        self.batchNorm = batchNorm
        self.activationFunctionName = activationFunction
        
        self.x = None
        self.z = None
        self.a = None
        self.W = None
        self.b = None
        
        self.dLoss_da = None
        self.dLoss_dW = None
        self.dLoss_db = None
        
        # Voi implementa Adam ca algoritm de optimizare
        # acesta e compus din alti doi algoritmi
        # GRADIENT DESCENT WITH MOMENTUM  si  RMS PROPAGATION
        
        # initializarea variabilelor folosite pentru implementarea algoritmului Gradient descent with momentum
        self.v_dW = np.zeros((self.layerSize, self.lastLayerSize))
        self.v_db = np.zeros((self.layerSize, 1))
        self.beta_1 = beta_1
        
        # initializarea variabilelor folosite pentru implementarea algoritmului RMS propagation
        self.s_dW = np.zeros((self.layerSize, self.lastLayerSize))
        self.s_db = np.zeros((self.layerSize, 1))
        self.beta_2 = beta_2
        self.eps = eps
        
        # initializarea variabilelor folosite pentru Batch Normalization
        self.mean = np.zeros((self.layerSize, 1))
        self.sigma = np.zeros((self.layerSize, 1))
        self.ews = 0.9
        
        # stabilirea functiei de activare si a initializarilor
        if self.activationFunctionName == 'sigmoid':
            self.activationFunction = self.sigmoid
            self.activationFunctionDerivative = self.sigmoidDerivative
            self.XavierInitialization()
        elif self.activationFunctionName == 'tanh':
            self.activationFunction = self.tanh
            self.activationFunctionDerivative = self.tanhDerivative
            self.XavierInitialization()
        elif self.activationFunctionName == 'relu':
            self.activationFunction = self.relu
            self.activationFunctionDerivative = self.reluDerivative
            self.HeInitialization()
        elif self.activationFunctionName == 'leakyrelu':
            self.activationFunction = lambda x: self.leakyrelu(x, alpha=alpha_activationFunction)
            self.activationFunctionDerivative = lambda x: self.leakyreluDerivative(x, alpha=alpha_activationFunction)
            self.HeInitialization()
        elif self.activationFunctionName == 'elu':
            self.activationFunction = lambda x: self.elu(x, alpha=alpha_activationFunction)
            self.activationFunctionDerivative = lambda x: self.eluDerivative(x, alpha=alpha_activationFunction)
            self.HeInitialization()
        elif self.activationFunctionName == 'swish':
            self.activationFunction = lambda x: self.swish(x, beta=beta_activationFunction)
            self.activationFunctionDerivative = lambda x: self.swishDerivative(x, beta=beta_activationFunction)
            self.HeInitialization()
        elif self.activationFunctionName == 'softmax':
            self.activationFunction = self.softmax
            self.activationFunctionDerivative = self.softmaxDerivative
            self.XavierInitialization()
    
    def forward(self, a, training=False):
        self.x = a        
        self.z = np.dot(self.W, self.x) + self.b
        
        if self.batchNorm == True:
            if training == True:
                self.BatchNormalizationUpdate()
            self.BatchNormalization()
        
        self.a = self.activationFunction(self.z)
        return self.a
    
    def backward(self, dLoss_da):
        if self.activationFunctionName == 'softmax':
            # prima data dLoss_da este chiar y pentru ca voi calcula direct dLoss_dz
            self.dLoss_dz = self.activationFunctionDerivative(self.a, dLoss_da)
        else:
            self.dLoss_dz = dLoss_da * self.activationFunctionDerivative(self.z)
        
        if self.batchNorm == True:
            self.dLoss_dW = np.dot(self.dLoss_dz / (self.sigma+1e-15), self.x.T) / self.x.shape[1]
            self.dLoss_db = np.sum(self.dLoss_dz / (self.sigma+1e-15), axis=1, keepdims=True) / self.x.shape[1]
            self.dLoss_dx = np.dot(self.W.T, self.dLoss_dz / (self.sigma+1e-15))
        else:
            self.dLoss_dW = np.dot(self.dLoss_dz, self.x.T) / self.x.shape[1]
            self.dLoss_db = np.sum(self.dLoss_dz, axis=1, keepdims=True) / self.x.shape[1]
            self.dLoss_dx = np.dot(self.W.T, self.dLoss_dz)
        
        return self.dLoss_dx
    
    def updateWeightsGradientDescent(self, learningRate):
        self.W -= learningRate * self.dLoss_dW
        self.b -= learningRate * self.dLoss_db
    
    def updateWeightsAdam(self, learningRate, t):
        # Adam este compus din alti doi algoritmi:
        #  - Gradient negativ cu moment
        #  - Propagare RMS
        # Fiecare dintre ei se implementeaza ca suma ponderata exponential
        # Dupa se face corectie de bias
        
        # Gradient negativ cu moment
        self.v_dW = self.beta_1 * self.v_dW + (1 - self.beta_1) * self.dLoss_dW
        self.v_db = self.beta_1 * self.v_db + (1 - self.beta_1) * self.dLoss_db
        v_dW_corrected = self.v_dW / (1 - self.beta_1 ** t)
        v_db_corrected = self.v_db / (1 - self.beta_1 ** t)
        
        # Propagare RMS
        self.s_dW = self.beta_1 * self.s_dW + (1 - self.beta_2) * (self.dLoss_dW ** 2)
        self.s_db = self.beta_1 * self.s_db + (1 - self.beta_2) * (self.dLoss_db ** 2)
        s_dW_corrected = self.s_dW / (1 - self.beta_2 ** t)
        s_db_corrected = self.s_db / (1 - self.beta_2 ** t)
        
        # update parametrii
        self.W -= learningRate * (v_dW_corrected / (np.sqrt(s_dW_corrected) - self.eps))
        self.b -= learningRate * (v_db_corrected / (np.sqrt(s_db_corrected) - self.eps))
        
        #print(self.v_dW)
        #print(self.v_db)
        #print(self.s_dW)
        #print(self.s_db)
        #print()
        #print(v_dW_corrected)
        #print(v_db_corrected)
        #print(s_dW_corrected)
        #print(s_db_corrected)
        #input("Press any key to continue...")
    
    def BatchNormalization(self):
        self.z = (self.z - self.mean) / (self.sigma + self.eps)
    
    def BatchNormalizationUpdate(self):
        new_mean = np.mean(self.z, axis=1)
        new_mean = new_mean.reshape((new_mean.shape[0], 1))
        self.mean = self.ews * self.mean + (1 - self.ews) * new_mean
        
        new_std = np.std(self.z, axis=1)
        new_std = new_std.reshape((new_std.shape[0], 1))
        self.sigma = self.ews * self.sigma + (1 - self.ews) * new_std
    
    def XavierInitialization(self):
        self.W = np.sqrt(1/self.lastLayerSize) * np.random.randn(self.layerSize, self.lastLayerSize)
        self.b = np.zeros((self.layerSize, 1))
    
    def HeInitialization(self):
        self.W = np.sqrt(2/self.lastLayerSize) * np.random.randn(self.layerSize, self.lastLayerSize)
        self.b = np.zeros((self.layerSize, 1))
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def leakyrelu(self, x, alpha):
        return np.maximum(alpha * x, x)
    
    def elu(self, x, alpha):
        return np.where(x < 0, alpha * (np.exp(x) - 1), x)
    
    def swish(self, x, beta):
        return x * self.sigmoid(-beta * x)
    
    def softmax(self, x):
        #print(np.max(x))
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=0)
    
    def sigmoidDerivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def tanhDerivative(self, x):
        return 1.0 - np.tanh(x) ** 2
    
    def reluDerivative(self, x):
        return np.where(x < 0, 0, 1)
    
    def leakyreluDerivative(self, x, alpha):
        return np.where(x < 0, alpha, 1)
    
    def eluDerivative(self, x, alpha):
        return np.where(x < 0, alpha * np.exp(x), 1)
    
    def swishDerivative(self, x, beta):
        s = self.sigmoid(beta * x)
        return s + beta * x * s * (1 - s)
    
    def softmaxDerivative(self, s, y):
        dLoss_dz = s - y
        return dLoss_dz
