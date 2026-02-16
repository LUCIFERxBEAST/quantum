import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit.primitives import StatevectorSampler

from sklearn.svm import SVC

class QuantumSpamClassifier:
    def __init__(self, n_qubits=2, use_quantum=True):
        self.n_qubits = n_qubits
        self.use_quantum = use_quantum
        
        if self.use_quantum:
            print(f"Initializing Quantum Kernel with {n_qubits} qubits...")
            # ZZFeatureMap is a common choice for quantum machine learning on text data
            self.feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement='linear')
            self.sampler = StatevectorSampler()

            self.fidelity = ComputeUncompute(sampler=self.sampler)
            self.quantum_kernel = FidelityQuantumKernel(fidelity=self.fidelity, feature_map=self.feature_map)
            self.model = QSVC(quantum_kernel=self.quantum_kernel)
        else:
            print("Initializing Classical SVM...")
            self.model = SVC(kernel='rbf')

    def train(self, X_train, y_train):
        print(f"Training model on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def predict(self, X_test):
        print(f"Predicting on {len(X_test)} samples...")
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        return self.model.score(X_test, y_test)

if __name__ == "__main__":
    # Test with dummy data
    X_train = np.array([[0.1, 0.2], [0.9, 0.8], [0.1, 0.3], [0.8, 0.9]])
    y_train = np.array([0, 1, 0, 1])
    
    X_test = np.array([[0.2, 0.2], [0.8, 0.8]])
    y_test = np.array([0, 1])
    
    qsc = QuantumSpamClassifier(n_qubits=2)
    qsc.train(X_train, y_train)
    acc = qsc.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc}")
