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
    WARNING: Simulating QRAM and true Oracles is EXTREMELY computationally expensive.
    """
    def __init__(self, suspicious_keywords):
        self.suspicious_keywords = [k.lower() for k in suspicious_keywords]

    def _quantum_search(self, words):
        # Limit email length to avoid statevector simulation size exploding
        MAX_WORDS = 20
        words = words[:MAX_WORDS]
        
        # 1. Vocabulary mapping (Classical to Quantum Data format)
        vocab = list(set(words + self.suspicious_keywords + ["__safe__"]))
        word_to_id = {w: i for i, w in enumerate(vocab)}
        
        suspicious_ids = [word_to_id[w] for w in self.suspicious_keywords]
        
        # Pad email
        email_ids = [word_to_id[w] for w in words]
        n_items = len(email_ids)
        if n_items == 0: return False
        
        addr_qubits = max(1, math.ceil(math.log2(n_items)))
        padded_n = 2**addr_qubits
        safe_id = word_to_id["__safe__"]
        while len(email_ids) < padded_n:
            email_ids.append(safe_id)
            
        data_qubits = max(1, math.ceil(math.log2(len(vocab))))
        total_qubits = addr_qubits + data_qubits
        
        qc = QuantumCircuit(total_qubits)
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
                for i, bit in enumerate(reversed(data_bin)):
                    if bit == '1':
                        qram.mcx(addr_regs, data_regs[i])
                
                for i, bit in enumerate(reversed(addr_bin)):
                    if bit == '0':
                        qram.x(addr_regs[i])
            return qram
            
        qram_circuit = build_qram()
        qram_inv = qram_circuit.inverse()
        
        # ==========================================
        # TRUE ORACLE: Defines search mathematically 
        # independent of where the words actually are.
        # ==========================================
        oracle_qc = QuantumCircuit(total_qubits)
        for s_id in suspicious_ids:
            s_bin = format(s_id, f'0{data_qubits}b')
            for i, bit in enumerate(reversed(s_bin)):
                if bit == '0':
                    oracle_qc.x(data_regs[i])
            
            # Phase Flip (MCZ) on data register
            if len(data_regs) > 1:
                oracle_qc.h(data_regs[-1])
                oracle_qc.mcx(data_regs[:-1], data_regs[-1])
                oracle_qc.h(data_regs[-1])
            else:
                oracle_qc.z(data_regs[-1])
            
            for i, bit in enumerate(reversed(s_bin)):
                if bit == '0':
                    oracle_qc.x(data_regs[i])
                    
        # ==========================================
        # DIFFUSION (on Address only)
        # ==========================================
        diff_qc = QuantumCircuit(addr_qubits)
        diff_qc.h(addr_regs)
        diff_qc.x(addr_regs)
        if len(addr_regs) > 1:
            diff_qc.h(addr_regs[-1])
            diff_qc.mcx(addr_regs[:-1], addr_regs[-1])
            diff_qc.h(addr_regs[-1])
        else:
            diff_qc.z(addr_regs[-1])
        diff_qc.x(addr_regs)
        diff_qc.h(addr_regs)
        
        # ==========================================
        # ASSEMBLE GROVER STEP (With visual separators)
        # ==========================================
        grover_step = QuantumCircuit(total_qubits)
        
        # 1. QRAM
        grover_step.compose(qram_circuit, inplace=True)
        grover_step.barrier()
        
        # 2. Oracle
        grover_step.compose(oracle_qc, inplace=True)
        grover_step.barrier()
        
        # 3. Inverse QRAM
        grover_step.compose(qram_inv, inplace=True)
        grover_step.barrier()
        
        # 4. Diffusion
        grover_step.compose(diff_qc, qubits=addr_regs, inplace=True)
        
        # Export Clean Diagram for the paper
        try:
            qram_circuit.draw(output='mpl', fold=30).savefig("1_qram.png", bbox_inches="tight")
            oracle_qc.draw(output='mpl', fold=30).savefig("2_oracle.png", bbox_inches="tight")
            qram_inv.draw(output='mpl', fold=30).savefig("3_inverse_qram.png", bbox_inches="tight")
            diff_qc.draw(output='mpl', fold=30).savefig("4_diffusion.png", bbox_inches="tight")
            
            fig = grover_step.draw(output='mpl', fold=30)
            fig.savefig("final_grover.png", bbox_inches="tight")
        except Exception as e:
            pass
        
        # ==========================================
        # BBHT ALGORITHM (Boyer-Brassard-Høyer-Tapp)
        # Industry standard for Grover with unknown M.
        # Instead of guessing M and risking overshoot,
        # we randomly pick iteration counts from a
        # geometrically growing upper bound.
        # ==========================================
        import random
        
        LAMBDA = 6 / 5  # Growth factor (standard BBHT parameter)
        max_iterations = math.ceil(math.sqrt(padded_n))  # Upper ceiling
        m = 1  # Starting upper bound
        
        while m <= max_iterations:
            # Pick a random number of Grover iterations in [0, m)
            j = random.randint(0, int(m) - 1) if int(m) > 1 else 0
            
            # Build the circuit fresh for this attempt
            attempt_qc = QuantumCircuit(total_qubits)
            attempt_qc.h(addr_regs)
            
            for _ in range(j):
                attempt_qc.compose(grover_step, inplace=True)
                
            attempt_qc.measure_all()
            
            # Simulate
            sampler = StatevectorSampler()
            job = sampler.run([attempt_qc])
            result = job.result()
            counts = result[0].data.meas.get_counts()
            
            # Get most probable state
            top_state_bin = max(counts, key=counts.get)
            
            # Extract address from measurement (little-endian)
            addr_bin_str = top_state_bin[-addr_qubits:]
            top_idx = int(addr_bin_str, 2)
            
            # Check if the measured address holds a spam word
            if top_idx < len(email_ids):
                word_found_id = email_ids[top_idx]
                if word_found_id in suspicious_ids:
                    return True  # Found a match!
            
            # No match found this round — grow the bound
            m = min(LAMBDA * m, max_iterations + 1)
        
        # Exhausted all BBHT rounds without finding a spam word
        return False

    def predict(self, texts):
        predictions = []
        for text in texts:
            words = text.lower().split()
            is_spam = self._quantum_search(words)
            predictions.append(1 if is_spam else 0)
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
