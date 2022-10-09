"""goal is to create a ML model for a quadratic"""

import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(start=-10, stop=10, num=1000).reshape(-1, 1)
y = np.squeeze(X**2 + 1)

training_indices = np.random.randint(low=len(y),size=10)
X_train, y_train = X[training_indices], y[training_indices]

plt.plot(X, y, label=r"$f(x) = x^{2}$", linestyle="dotted")
plt.scatter(X_train,y_train, label=r"$Samples$", marker="x")
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.show()