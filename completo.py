import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

# Cargar el conjunto de datos desde un archivo CSV
data = pd.read_csv('C:/Users/USER/Documents/Practica2U2/DATA.csv.csv')

# Ajustar las dimensiones eliminando las columnas que no necesitas
# Por ejemplo, si deseas eliminar las columnas "STUDENT ID" y "COURSE ID" puedes hacer lo siguiente:
data = data.drop(['STUDENT ID', 'COURSE ID'], axis=1)

# También puedes ajustar las dimensiones seleccionando un subconjunto de columnas si lo prefieres
# Por ejemplo, si solo deseas mantener las columnas "1" a "30" y "GRADE", puedes hacer lo siguiente:
data = data[['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', 'GRADE']]


class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = np.zeros(3)  # 3 pesos: 2 para características y 1 para sesgo



    def fit(self, X, y):
        
        self.w_ = np.zeros(1 + X.shape[1])

        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
    
        return np.dot(X, self.w_[1:]) + self.w_[0]


    def predict(self, X):
    
        return np.where(self.net_input(X) >= 0.0, 1, -1)



#############################################################################
## MUESTRA LOS DATOS DE ENTRENAMIENTO CON PERCEPTRON ##
print(50 * '=')
print('Section: Training a perceptron model on the Iris dataset')
print(50 * '-')
## AQUÍ VAN A TOMAR SUS DATOS DEL PROGRAMA ANTERIOR ##
df = pd.read_csv('C:/Users/USER/Documents/Practica2U2/DATA.csv.csv')
print(df.tail())

#############################################################################
print(50 * '=')
print('Plotting the Iris data')
print(50 * '-')

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(df['GRADE'] == 'Aprobado', 1, -1)


# extract sepal length and petal length
X = df.iloc[:, 1:31].values  # Ajusta las columnas adecuadas

# plot data   --- CLASIFICA POR COLORES
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

plt.show()

#############################################################################
print(50 * '=')
print('Training the perceptron model')
print(50 * '-')

ppn = Perceptron(eta=0.1, n_iter=10)

ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')


plt.show()

#############################################################################
print(50 * '=')
print('A function for plotting decision regions')
print(50 * '-')

classifiers = [ppn]  # o [ppn, ada1] si también quieres trazar las regiones de decisión de Adaline

def plot_decision_regions(X, y, classifiers, resolution=0.02):
    classifiers = [ppn]  # Agrega todos los clasificadores que desees trazar en esta lista

# Luego, pasa la lista de clasificadores a la función plot_decision_regions
plot_decision_regions(X, y, classifiers=classifiers)

    # setup marker generator and color map
markers = ('s', 'x', 'o', '^', 'v')
    
cmap = plt.get_cmap('magma')


resolution = 0.02  # Puedes ajustar este valor según tus necesidades

x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

Z = np.zeros(xx1.shape)

for i, classifiers in enumerate(classifiers):
        Z_i = classifiers.predict(np.c_[xx1.ravel(), xx2.ravel()])
        Z_i = Z_i.reshape(xx1.shape)
        Z = Z + Z_i * (i + 1)

plt.contourf(xx1, xx2, Z, cmap=cmap, alpha=0.4)
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())

    # plot class samples
for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


plot_decision_regions(X, y, classifiers=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

plt.show()


#############################################################################
## IMPLEMENTACIÓN DE ADAPTIVE ##
print(50 * '=')
print('Implementación de rendimiento escolar')
print(50 * '-')


class AdalineGD(object):
    classifiers = [ppn]  # Agrega todos los clasificadores que desees trazar en esta lista

# Luego, pasa la lista de clasificadores a la función plot_decision_regions
plot_decision_regions(X, y, classifiers=classifiers)

def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

def fit(self, X, y):
        
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')

# plt.tight_layout()
# plt.savefig('./adaline_1.png', dpi=300)
plt.show()


print('standardize features')
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada = AdalineGD(n_iter=15, eta=0.01)
ada.fit(X_std, y)

plot_decision_regions(X_std, y, classifiers=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')

plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')


plt.show()


#############################################################################
print(50 * '=')
print('Large scale machine learning and stochastic gradient descent')
print(50 * '-')


class AdalineSGD(object):
    
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            np.random.seed(random_state)

    def fit(self, X, y):
        
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights to zeros"""
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)


ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)

plot_decision_regions(X_std, y, classifiers=ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')


plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')


plt.show()

ada = ada.partial_fit(X_std[0, :], y[0])