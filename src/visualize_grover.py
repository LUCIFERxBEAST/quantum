from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMT
import math
import sys
import os

def export_grover_diagram(output_dir="circuit_viz"):
    print(f"Generating Detailed Grover Visualization Package in '{output_dir}'...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # --- DATA CONTEXT ---
    # Representative 4-word sample email: "win free prize now"
    # Suspicious keywords: "win", "free", "prize", "urgent"
    # Mapping: "win"->1, "free"->2, "prize"->3, "urgent"->4, "__safe__"->0
    
    words = ["win", "free", "prize", "now"]
    suspicious_keywords = ["win", "free", "prize", "urgent"]
    vocab = ["__safe__"] + suspicious_keywords
    word_to_id = {w: i for i, w in enumerate(vocab)}
    safe_id = word_to_id["__safe__"]
    
    email_ids = [word_to_id.get(w, safe_id) for w in words] # [1, 2, 3, 0]
    n_items = len(email_ids)
    addr_qubits = max(1, math.ceil(math.log2(n_items))) # 2 qubits
    data_qubits = max(1, math.ceil(math.log2(len(vocab)))) # 3 qubits
    total_qubits = addr_qubits + data_qubits
    
    addr_regs = list(range(addr_qubits))
    data_regs = list(range(addr_qubits, total_qubits))
    
    # 1. QRAM Component
    qram = QuantumCircuit(total_qubits, name="QRAM")
    for addr_val, data_val in enumerate(email_ids):
        addr_bin = format(addr_val, f'0{addr_qubits}b')
        for i, bit in enumerate(reversed(addr_bin)):
            if bit == '0': qram.x(addr_regs[i])
        data_bin = format(data_val, f'0{data_qubits}b')
        targets = [data_regs[i] for i, bit in enumerate(reversed(data_bin)) if bit == '1']
        if targets:
            mcmt = MCMT('x', len(addr_regs), len(targets))
            qram.compose(mcmt, addr_regs + targets, inplace=True)
        for i, bit in enumerate(reversed(addr_bin)):
            if bit == '0': qram.x(addr_regs[i])
    
    # 2. Oracle Component
    oracle = QuantumCircuit(total_qubits, name="Oracle")
    suspicious_ids = [word_to_id[w] for w in suspicious_keywords]
    for s_id in suspicious_ids:
        s_bin = format(s_id, f'0{data_qubits}b')
        for i, bit in enumerate(reversed(s_bin)):
            if bit == '0': oracle.x(data_regs[i])
        mcmt_z = MCMT('z', len(data_regs)-1, 1)
        oracle.compose(mcmt_z, data_regs, inplace=True)
        for i, bit in enumerate(reversed(s_bin)):
            if bit == '0': oracle.x(data_regs[i])

    # 3. Diffusion Component
    diff = QuantumCircuit(addr_qubits, name="Diffusion")
    diff.h(range(addr_qubits))
    diff.x(range(addr_qubits))
    mcmt_diff = MCMT('z', addr_qubits-1, 1)
    diff.compose(mcmt_diff, range(addr_qubits), inplace=True)
    diff.x(range(addr_qubits))
    diff.h(range(addr_qubits))

    # 4. Full Grover Step
    full_step = QuantumCircuit(total_qubits, name="Grover_Step")
    full_step.append(qram.to_gate(), range(total_qubits))
    full_step.barrier()
    full_step.append(oracle.to_gate(), range(total_qubits))
    full_step.barrier()
    qram_inv = qram.inverse()
    full_step.append(qram_inv.to_gate(), range(total_qubits))
    full_step.barrier()
    full_step.append(diff.to_gate(), addr_regs)
    
    # --- EXPORT LOGIC ---
    try:
        import matplotlib.pyplot as plt
        print("Matplotlib detected. Exporting component PNGs (DECOMPOSED & SYNCED)...")
        
        # 1. Decompose individual components FIRST for exact mapping
        qram_dec = qram.decompose().decompose()
        oracle_dec = oracle.decompose().decompose()
        diff_dec = diff.decompose().decompose()
        qram_inv_dec = qram_inv.decompose().decompose()
        
        # 2. Build the FULL STEP using these already-decomposed circuits
        # This ensures the full_step diagram is a literal concatenation of the parts
        synced_full_step = QuantumCircuit(total_qubits, name="Synced_Grover_Step")
        synced_full_step.compose(qram_dec, inplace=True)
        synced_full_step.barrier()
        synced_full_step.compose(oracle_dec, inplace=True)
        synced_full_step.barrier()
        synced_full_step.compose(qram_inv_dec, inplace=True)
        synced_full_step.barrier()
        synced_full_step.compose(diff_dec, qubits=addr_regs, inplace=True)
        
        # 3. Save Standalone Components
        qram_dec.draw(output='mpl', style='iqp').savefig(os.path.join(output_dir, "1_qram.png"))
        oracle_dec.draw(output='mpl', style='iqp').savefig(os.path.join(output_dir, "2_oracle.png"))
        qram_inv_dec.draw(output='mpl', style='iqp').savefig(os.path.join(output_dir, "3_qram_inverse.png"))
        diff_dec.draw(output='mpl', style='iqp').savefig(os.path.join(output_dir, "4_diffusion.png"))
        
        # 4. Save the SYNCED Full Step
        synced_full_step.draw(output='mpl', style='iqp').savefig(os.path.join(output_dir, "full_step_decomposed.png"))
        
        print(f"Success! Synced gate-level PNGs saved to {output_dir}/")
        
        # Update ASCII with the same synced circuit
        with open(os.path.join(output_dir, "exact_architecture.txt"), "w") as f:
            f.write("EXACT GATE-LEVEL ARCHITECTURE (SYNCED WITH PNGS)\n")
            f.write("==============================================\n\n")
            f.write(str(synced_full_step.draw(output='text')))
            
    except Exception as e:
        print(f"PNG Export failed: {e}")

if __name__ == "__main__":
    export_grover_diagram("circuit_viz")
