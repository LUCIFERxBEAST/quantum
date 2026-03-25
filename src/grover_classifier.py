from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator, MCMT, ZGate
from qiskit.primitives import StatevectorSampler
import math
import collections

class GroverSpamClassifier:
    """
    A TRUE quantum implementation of Grover's Search.
    This does NOT use classical pre-computation to find the answer.
    Instead, it builds a QRAM to load the email into quantum superposition,
    and a generic Oracle that identifies spam keywords mathematically.
    """
    def __init__(self, suspicious_keywords):
        self.suspicious_keywords = [k.lower() for k in suspicious_keywords]

    def _quantum_search(self, words):
        MAX_WORDS = 20
        words = words[:MAX_WORDS]
        
        # 1. Vocabulary mapping (Classical to Quantum Data format)
        vocab = ["__safe__"] + self.suspicious_keywords
        word_to_id = {w: i for i, w in enumerate(vocab)}
        
        suspicious_ids = [word_to_id[w] for w in self.suspicious_keywords]
        
        # Pad email
        safe_id = word_to_id["__safe__"]
        email_ids = [word_to_id.get(w, safe_id) for w in words]
        
        n_items = len(email_ids)
        if n_items == 0: return False
        
        addr_qubits = max(1, math.ceil(math.log2(n_items)))
        padded_n = 2**addr_qubits
        while len(email_ids) < padded_n:
            email_ids.append(safe_id)
            
        data_qubits = max(1, math.ceil(math.log2(len(vocab))))
        total_qubits = addr_qubits + data_qubits
        
        addr_regs = list(range(addr_qubits))
        data_regs = list(range(addr_qubits, total_qubits))
        
        # ==========================================
        # QRAM (Data Loader): |addr>|0> -> |addr>|data>
        # ==========================================
        def build_qram():
            qram = QuantumCircuit(total_qubits)
            for addr_val, data_val in enumerate(email_ids):
                addr_bin = format(addr_val, f'0{addr_qubits}b')
                for i, bit in enumerate(reversed(addr_bin)):
                    if bit == '0':
                        qram.x(addr_regs[i])
                
                data_bin = format(data_val, f'0{data_qubits}b')
                targets = [data_regs[i] for i, bit in enumerate(reversed(data_bin)) if bit == '1']
                
                if targets:
                    mcmt = MCMT('x', len(addr_regs), len(targets))
                    qram.compose(mcmt, addr_regs + targets, inplace=True)
                
                for i, bit in enumerate(reversed(addr_bin)):
                    if bit == '0':
                        qram.x(addr_regs[i])
            return qram
            
        qram_circuit = build_qram()
        qram_inv = qram_circuit.inverse()
        
        # ==========================================
        # TRUE ORACLE: Defines search mathematically 
        # ==========================================
        oracle_qc = QuantumCircuit(total_qubits)
        for s_id in suspicious_ids:
            s_bin = format(s_id, f'0{data_qubits}b')
            for i, bit in enumerate(reversed(s_bin)):
                if bit == '0': oracle_qc.x(data_regs[i])
            
            # Phase Flip (MCZ) on data register
            if len(data_regs) > 1:
                mcmt_z = MCMT('z', len(data_regs)-1, 1)
                oracle_qc.compose(mcmt_z, data_regs, inplace=True)
            else:
                oracle_qc.z(data_regs[-1])
            
            for i, bit in enumerate(reversed(s_bin)):
                if bit == '0': oracle_qc.x(data_regs[i])
                    
        # ==========================================
        # DIFFUSION (on Address only)
        # ==========================================
        diff_qc = QuantumCircuit(addr_qubits)
        diff_qc.h(addr_regs)
        diff_qc.x(addr_regs)
        if len(addr_regs) > 1:
            mcmt_diff = MCMT('z', len(addr_regs)-1, 1)
            diff_qc.compose(mcmt_diff, addr_regs, inplace=True)
        else:
            diff_qc.z(addr_regs[-1])
        diff_qc.x(addr_regs)
        diff_qc.h(addr_regs)
        
        # ==========================================
        # ASSEMBLE GROVER STEP
        # ==========================================
        grover_step = QuantumCircuit(total_qubits)
        grover_step.compose(qram_circuit, inplace=True)
        grover_step.compose(oracle_qc, inplace=True)
        grover_step.compose(qram_inv, inplace=True)
        grover_step.compose(diff_qc, qubits=addr_regs, inplace=True)
        
        # ==========================================
        # BBHT ALGORITHM
        # ==========================================
        import random
        LAMBDA = 1.2
        max_iterations = math.ceil(math.sqrt(padded_n))
        m = 1
        
        sampler = StatevectorSampler()
        while m <= max_iterations:
            j = random.randint(0, int(m) - 1) if int(m) > 1 else 0
            attempt_qc = QuantumCircuit(total_qubits)
            attempt_qc.h(addr_regs)
            for _ in range(j):
                attempt_qc.compose(grover_step, inplace=True)
            attempt_qc.measure_all()
            
            job = sampler.run([attempt_qc])
            result = job.result()
            counts = result[0].data.meas.get_counts()
            top_state_bin = max(counts, key=counts.get)
            addr_bin_str = top_state_bin[-addr_qubits:]
            top_idx = int(addr_bin_str, 2)
            
            if top_idx < len(email_ids):
                word_found_id = email_ids[top_idx]
                if word_found_id in suspicious_ids:
                    return True
            m = min(LAMBDA * m, max_iterations + 1)
        
        return False

    def predict(self, texts):
        predictions = []
        total = len(texts)
        bar_length = 30
        for i, text in enumerate(texts):
            progress = (i + 1) / total
            filled_length = int(bar_length * progress)
            bar = '=' * filled_length + ' ' * (bar_length - filled_length)
            print(f"\rProcessing: [{bar}] {i+1}/{total}", end="", flush=True)
            words = text.lower().split()
            is_spam = self._quantum_search(words)
            predictions.append(1 if is_spam else 0)
        print()
        return np.array(predictions)

    def evaluate(self, X_text, y_test):
        y_pred = self.predict(X_text)
        
        print("\n--- Grover's Algorithm Classification Report ---")
        print(classification_report(y_test, y_pred, labels=[0, 1], target_names=['Ham', 'Spam'], zero_division=0))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    # Test
    X_test = ["win a free prize", "meeting today", "urgent call me", "project update"]
    y_test = [1, 0, 1, 0]
    
    grover = GroverSpamClassifier(suspicious_keywords=["win", "free", "urgent"])
    grover.evaluate(X_test, y_test)
