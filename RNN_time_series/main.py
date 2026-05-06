import numpy as np 
import matplotlib.pyplot as plt

#**************
#cargar datos 
#**************

data_path = "control_charts_data/synthetic_control.data"

#cada fila del dataset es una serie temporal de 60 puntos
X= np.loadtxt(data_path)

print("Tamaño del dataset: ", X.shape) #600,60

#**************
#Crear etiquetas
#**************
# 6 clases, y cada clase tiene 100 ejemplos de series temporales

y= np.zeros(600, dtype=int) #inicializamos un vector de ceros de tamaño 600

for i in range (6):
    y[i*100:(i+1)*100] =i #asignamos la etiqueta i a cada bloque de 100 ejemplos

print("Tamaño de las etiquetas: ", y.shape) #600
print("Clases únicas:", np.unique(y)) #0,1,2,3,4,5

#**************
#NORMALIZACIÓN 
#**************

#normalizamos cada punto x= (x- media) / desviación estándar
# axis=1 indica que queremos calcular la media y la desviación estándar a lo largo de las filas (es decir, para cada serie temporal individualmente)
#keepdims=True mantiene las dimensiones originales del array para que la operación de resta y división se realice correctamente

X_norm= (X- X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

print("Ejemplo de la serie normalizada", X_norm[0][:5]) #imprime los primeros 5 puntos de la primera serie temporal normalizada

#**************
#division de datos de entrenamiento y prueba
#***************

np.random.seed(42) #para reproducibilidad

indices= np.arange(600) #creamos un array con los índices de las muestras
np.random.shuffle(indices) #mezclamos los índices de forma aleatoria para obtener una distribución aleatoria de las muestras

X_norm= X_norm[indices] #reordenamos las series temporales según los índices mezclados
y= y[indices] #reordenamos las etiquetas según los índices mezclados    

# 80% train, 20% test
split= int(0.8 * 600)

X_train= X_norm[:split] #las primeras 480 muestras para entrenamiento
y_train= y[:split] #etiquetas correspondientes a las muestras de entrenamiento

X_test= X_norm[split:] #las últimas 120 muestras para prueba
y_test= y[split:] #etiquetas correspondientes a las muestras de prueba

print("Train:", X_train.shape)
print("Test:", X_test.shape)

  
#----------------------------------
#funciones auxiliares
#----------------------------------

def softmax(z):
    exp_z = np.exp(z-np.max(z))
    return exp_z/ np.sum(exp_z)

def onehot(label,num_classes=6):
    vec= np.zeros(num_classes)
    vec[label]=1
    return vec 


def tanh_derivative(h):
    return 1 - h**2

#----------------------
# inicialización Red de Jordan
#----------------------


np.random.seed(0)

input_size= 1
context_size = 6
hidden_size= 12
output_size = 6

#capa oculta
W1 = np.random.randn(hidden_size, input_size + context_size)*0.1
b1 = np.zeros(hidden_size)

#Capa de salida 
W2 = np.random.randn(output_size, hidden_size)* 0.1
b2 = np.zeros(output_size)



mu= 0.3
lr= 0.005
epochs= 100

# ============================
# ENTRENAMIENTO JORDAN 
# ============================

for epoch in range(epochs):

    total_loss = 0

    for x_seq, label in zip(X_train, y_train):

        # contexto inicial
        c = np.zeros(context_size)

        # recorrer cada punto de la serie temporal
        for xt in x_seq:

            # entrada  aumentada
            z = np.concatenate(([xt], c))

            #capa oculta
            n1= W1 @ z + b1
            h= np.tanh(n1)

            # salida
            n2 = W2 @ h +b2
            y_pred = softmax(n2)

            # actualización contexto (Jordan)
            c = mu * c + y_pred # es un vector de tamaño de context_size

        # target one-hot
        t = onehot(label)

        #loss final
        loss = -np.sum(t * np.log(y_pred + 1e-8)) #es un valor escalar
        total_loss += loss

        #error de salida
        e2 = y_pred - t # es un vector de tamaño output_size

        #gradientes de capa de salida
        dW2= np.outer(e2, h) # matriz de tamaño output_size x hidden_size
        db2 = e2 # vector de tamaño output_size

        #error de capa oculta
        e1 = (W2.T @ e2) * tanh_derivative(h) # vector de tamaño hidden_size 

        #gradientes de capa oculta
        dW1= np.outer(e1,z)
        db1= e1


        # actualización de pesos y sesgo

        W2 -= lr * dW2
        b2 -= lr * db2

        W1 -= lr * dW1
        b1 -= lr * db1
        

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.2f}")


#--------------------------
#Red de Elman 
#--------------------------

np.random.seed(1)
input_size= 1
hidden_size = 12
context_size_elman = hidden_size
output_size = 6

W1_e = np.random.randn(hidden_size, input_size + context_size_elman)*0.1
b1_e = np.zeros(hidden_size)

W2_e = np.random.randn(output_size, hidden_size)*0.1 
b2_e = np.zeros(output_size)

mu_e= 0.3
lr_e= 0.005
epochs_e = 100

loss_elman =[]

#------------------
#Comienza entrenamiento de Elman
#------------------
for epoch in range(epochs_e):
    total_loss = 0

    for x_seq,label in zip(X_train,y_train):
        c= np.zeros(context_size_elman)

        for xt in x_seq:
            z= np.concatenate(([xt],c))
            n1= W1_e @ z + b1_e
            h=np.tanh(n1)

            n2= W2_e @ h + b2_e
            y_pred = softmax(n2)

            #contexto Elman: guarda la activación oculta
            c= mu_e*c + h
        
        t= onehot(label)

        loss = -np.sum(t * np.log(y_pred +  1e-8))
        total_loss += loss

        e2= y_pred - t

        dW2= np.outer(e2,h)
        db2= e2

        e1=(W2_e.T @ e2) * tanh_derivative(h)

        dW1 = np.outer(e1,z)
        db1= e1

        W2_e -= lr_e * dW2
        b2_e -= lr_e * db2

        W1_e -= lr_e *dW1
        b1_e -= lr_e * db1

    loss_elman.append(total_loss)
    print(f"Elman Epoch {epoch+1}/{epochs_e}, Loss: {total_loss:.2f}")



#*_*_*_*_*_*_*_*_*_*_*_*_*
#Predicción Jordan
#*_*_*_*_*_*_*_*_*_*_*_*_*
 

def predict_jordan(X):
    preds= []

    for x_seq in X:

        c = np.zeros(context_size)

        for xt in x_seq:

            z= np.concatenate(([xt],c))

            n1= W1 @ z + b1
            h= np.tanh(n1)

            n2= W2 @ h + b2
            y_pred= softmax(n2)

            c = mu * c + y_pred
        preds.append(np.argmax(y_pred))
    return np.array(preds)


#-----------------------------
#PREDICCIÓN ELMAN
#-----------------------------



#**********************
#metricas
#**********************


y_pred = predict_jordan(X_test)

accuracy_jordan = np.mean(y_pred == y_test) # ejemplo np.mean([True, False, True]) -> (1 + 0 + 1) / 3 = 0.6667
print(f"Accuracy: {accuracy_jordan:.3f}") 

#matriz de confusión 
conf_matrix = np.zeros((6,6), dtype=int)
for yt,yp in zip(y_test, y_pred):
    conf_matrix[yt,yp] +=1


print("Matriz de confusión:")
print(conf_matrix)
