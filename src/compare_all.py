import time
import math
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from qiskit.circuit.library import ZZFeatureMap
from qiskit.compiler import transpile
from qiskit import QuantumCircuit
import sys
import os

warnings.filterwarnings("ignore")

from preprocess import load_data, clean_text
from features import FeatureExtractor
from quantum_model import QuantumSpamClassifier
from grover_classifier import GroverSpamClassifier

def main():
    print("================================================================")
    print("      COMPREHENSIVE ALGORITHM COMPARISON (Accuracy, Speed, Steps)")
    print("================================================================\\n")
    
    # 1. Load Data (Tiny subset to ensure Quantum SVM actually finishes)
    print("Loading data for benchmark...")
    df = load_data("data/SMSSpamCollection")
    if df is None:
        if os.path.exists("../data/SMSSpamCollection"):
            df = load_data("../data/SMSSpamCollection")
        else:
            print("Could not load data. Exiting.")
            sys.exit(1)
            
    df['clean_message'] = df['message'].apply(clean_text)
    
    # Use 20 samples to keep simulation time reasonable for QSVM
    n_samples = 500
    spam = df[df['label'] == 1]
    ham = df[df['label'] == 0]
    n_spam = int(n_samples / 2)
    n_ham = n_samples - n_spam
    df = pd.concat([spam.sample(n=min(len(spam), n_spam), random_state=42), 
                    ham.sample(n=min(len(ham), n_ham), random_state=42)]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    X_text = df['clean_message'].values
    y = df['label'].values
    print(f"Dataset Size: {len(X_text)} (Balanced Spam/Ham)\\n")
    
    # TF-IDF -> PCA just for SVMs
    print("Extracting classical features for SVMs (TF-IDF -> PCA -> 2 dims)...")
    fe = FeatureExtractor(n_components=2)
    X = fe.fit_transform(X_text)
    
    # Train/Test Split (70/30) -> 14 train, 6 test
    X_train, X_test, y_train, y_test, X_text_train, X_text_test = train_test_split(
        X, y, X_text, test_size=0.3, random_state=42
    )
    
    results = {}

    # --- 1. Classical SVM ---
    print("\\n[1] Running Classical SVM...")
    start_time = time.time()
    csvm = QuantumSpamClassifier(n_qubits=2, use_quantum=False)
    csvm.train(X_train, y_train)
    y_pred_csvm = csvm.predict(X_test)
    csvm_time = time.time() - start_time
    csvm_acc = accuracy_score(y_test, y_pred_csvm)
    
    # Steps: N_Train * N_Test * N_Features array operations
    csvm_steps = f"~{len(X_train)*len(X_test)*2} Math Ops"
    results['Classical SVM'] = {'acc': csvm_acc, 'time': csvm_time, 'steps': csvm_steps}

    # --- 2. Quantum SVM ---
    print("\\n[2] Running Quantum SVM (This may take a minute depending on CPU)...")
    start_time = time.time()
    qsvm = QuantumSpamClassifier(n_qubits=2, use_quantum=True)
    qsvm.train(X_train, y_train)
    y_pred_qsvm = qsvm.predict(X_test)
    qsvm_time = time.time() - start_time
    qsvm_acc = accuracy_score(y_test, y_pred_qsvm)
    
    # Steps: Calculate gate depth of ZZFeatureMap * pairwise evaluations
    feat_map = ZZFeatureMap(feature_dimension=2, reps=2, entanglement='linear')
    try:
        t_feat = transpile(feat_map, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=1)
        depth = t_feat.depth()
        qsvm_steps = f"{len(X_train)*len(X_test) * depth} Gate Depth"  # Rough estimate of pairwise evaluating
    except:
        qsvm_steps = "Compilation Error"
    
    results['Quantum SVM'] = {'acc': qsvm_acc, 'time': qsvm_time, 'steps': qsvm_steps}

    # --- 3. Grover's Database Search ---
    print("\\n[3] Running Grover's Search Classifier...")
    # Grover's doesn't need to "train", but we will evaluate it on the same X_test to be fair.
    keywords = ['act', 'now', 'immediate', 'limited', 'time', 'prize', 'claims', 'cash', 'bonus', 'buy', 'clearance', 'discount', 'offer', 'promo', 'subscribe', 'trial', 'verify', 'suspend', 'security', 'unauthorized', 'password', 'login', 'earn', 'save', 'guaranteed', 'exclusive', 'congratulations', 'xxx', 'win', 'free', 'urgent', 'call', 'winner', 'selected', 'mobile', 'text', 'stop', 'reply']
    
    start_time = time.time()
    grover = GroverSpamClassifier(suspicious_keywords=keywords)
    # Don't print loop updates in compare output
    import os, sys
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    y_pred_grover = grover.predict(X_text_test)
    sys.stdout = old_stdout
    grover_time = time.time() - start_time
    grover_acc = accuracy_score(y_test, y_pred_grover)
    
    # Calculate Grover Steps dynamically based on the max words in test set
    max_words = max([len(t.split()) for t in X_text_test])
    max_words = max(1, min(max_words, 20)) # Cap at 20
    addr_qubits = max(1, math.ceil(math.log2(max_words)))
    data_qubits = 6 # Vocab limit from previous optimization
    total_qubits = addr_qubits + data_qubits
    
    # Provide rough approximation for steps based on QRAM build (length * mcx logic)
    qram = QuantumCircuit(total_qubits)
    addr_regs = list(range(addr_qubits))
    data_regs = list(range(addr_qubits, total_qubits))
    for _ in range(max_words):
        qram.mcx(addr_regs, data_regs[0])
    try:
        M_iters = max(1, math.floor(math.pi / 4 * math.sqrt(max_words)))
        t_qram = transpile(qram, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=1)
        grover_exact = M_iters * (t_qram.depth() * 2 + 50) 
        grover_steps = f"{grover_exact} Gate Depth (Avg/Msg)"
    except:
        grover_steps = "Compilation Error"
        
    results['Grover Search'] = {'acc': grover_acc, 'time': grover_time, 'steps': grover_steps}

    # --- SUMMARY REPORT ---
    report_lines = []
    report_lines.append("\n================================================================")
    report_lines.append("                     FINAL COMPARISON REPORT                    ")
    report_lines.append("================================================================")
    report_lines.append(f"{'Algorithm':<20} | {'Accuracy':<10} | {'Speed (Total)':<15} | {'Complexity/Steps':<25}")
    report_lines.append("-" * 78)
    
    for algo, res in results.items():
        time_str = f"{res['time']:.2f} s"
        acc_str = f"{res['acc'] * 100:.1f}%"
        steps_str = str(res['steps'])
        report_lines.append(f"{algo:<20} | {acc_str:<10} | {time_str:<15} | {steps_str:<25}")
        
    report_lines.append("================================================================")
    report_lines.append("\nARCHITECTURAL INSIGHT: Zero-Shot Capability")
    report_lines.append("-" * 43)
    report_lines.append("Note: Unlike SVMs (Classical & Quantum) which REQUIRE training labels (y_train)")
    report_lines.append("to learn, Grover's Search is 'Zero-Shot'. It functions purely based on")
    report_lines.append("mathematical Oracle definitions (keywords) without ever seeing a label.")
    
    # Print to console
    for line in report_lines:
        print(line)
        
    # Save to file
    with open("comparison_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print("\nComparison report saved to comparison_report.txt")

if __name__ == "__main__":
    main()
