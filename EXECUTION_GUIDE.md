# Mixtral Pressure Half-Life Execution Guide

**Status**: Infrastructure validated ✓, awaiting GPU execution
**Current environment**: MacBook (CPU only)

---

## What's Ready

✓ **Complete infrastructure:**
- External Mixtral adapter (HuggingFace integration)
- Pressure half-life measurement framework
- Session controller with pressure computation
- Comparison and analysis tools

✓ **Validation passed:**
- All pipeline tests successful
- Integration verified with GPT-2 proxy
- Ready for Mixtral-8x7B

✓ **Toy baseline:**
- Infinite half-life measured
- Full trajectory data collected
- Ready for comparison

---

## Execution Options

### Option 1: Local GPU (If Available)

**Requirements:**
- NVIDIA GPU with 12GB+ VRAM (RTX 3090, A100, A6000, etc.)
- CUDA 11.8+ installed
- ~50GB disk space for model

**Setup:**
```bash
# Install GPU dependencies
pip install transformers accelerate bitsandbytes

# Run measurement (8-bit quantization)
PYTHONPATH=src python tests/measure_mixtral_halflife.py --8bit

# Or full precision (requires 24GB VRAM)
PYTHONPATH=src python tests/measure_mixtral_halflife.py
```

**Runtime:** ~20-30 minutes (50 chunks × 20 tokens)

---

### Option 2: Google Colab Pro (Recommended for Quick Test)

**Cost:** $10/month subscription
**GPU:** A100 40GB
**Advantages:** No setup, pre-installed libraries, easy sharing

**Steps:**

1. **Create new Colab notebook**
   - Go to https://colab.research.google.com
   - New notebook
   - Runtime → Change runtime type → A100 GPU

2. **Mount Drive and clone repo:**
```python
from google.colab import drive
drive.mount('/content/drive')

# Clone or upload Chronovisor repo
!git clone https://github.com/your-repo/Chronovisor.git
%cd Chronovisor
```

3. **Install dependencies:**
```python
!pip install transformers accelerate bitsandbytes matplotlib numpy
```

4. **Run measurement:**
```python
!PYTHONPATH=src python tests/measure_mixtral_halflife.py --8bit --chunks 50
```

5. **Download results:**
```python
# Results will be in test_results/
from google.colab import files
files.download('test_results/mixtral_pressure_halflife.json')
files.download('test_results/mixtral_vs_toy_comparison.png')
```

---

### Option 3: Lambda Labs (Best for Serious Runs)

**Cost:** ~$1.10/hour (A100 40GB)
**GPU:** A100 40GB or 80GB
**Advantages:** Persistent storage, SSH access, Jupyter

**Steps:**

1. **Create account:** https://lambdalabs.com/service/gpu-cloud

2. **Launch instance:**
   - Select: 1x A100 (40GB)
   - Image: PyTorch 2.0
   - SSH key: Upload your public key

3. **SSH into instance:**
```bash
ssh ubuntu@<instance-ip>
```

4. **Setup:**
```bash
# Clone repo
git clone <your-repo-url>
cd Chronovisor

# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install transformers accelerate bitsandbytes
```

5. **Run measurement:**
```bash
PYTHONPATH=src python tests/measure_mixtral_halflife.py --8bit
```

6. **Download results:**
```bash
# On your local machine
scp ubuntu@<instance-ip>:~/Chronovisor/test_results/* ./test_results/
```

**Don't forget to STOP instance when done!**

---

### Option 4: RunPod (Most Economical)

**Cost:** ~$0.79/hour (A100 40GB), ~$1.39/hour (A100 80GB)
**GPU:** Various A100 options
**Advantages:** Cheapest, flexible, community templates

**Steps:**

1. **Create account:** https://www.runpod.io/

2. **Deploy pod:**
   - GPU: A100 40GB PCIe
   - Template: PyTorch 2.0
   - Storage: 50GB

3. **Connect via Jupyter or SSH**

4. **Run measurement** (same as Lambda Labs)

---

## What Gets Generated

After execution completes, you'll have:

### Results Files
```
test_results/
├── mixtral_pressure_halflife.json      (full trajectory data)
├── mixtral_vs_toy_comparison.png       (4-panel comparison)
└── MIXTRAL_HALFLIFE_FINDING.md         (auto-generated analysis)
```

### Key Metrics
- **Half-life:** Chunks until pressure drops to 50%
- **Decay shape:** Exponential/linear/cliff characterization
- **Entropy evolution:** Router convergence pattern
- **Generated text:** Full output with pressure annotations

### Comparison to Toy
- Half-life: Mixtral vs Toy (expected: 15-25 vs ∞)
- Decay rate: Slope comparison
- Entropy collapse: Magnitude and pattern
- Interpretation: Natural vs artificial behavior

---

## Expected Findings

### Likely Scenario: Natural Decay

**Prediction:**
```
Half-life: 15-25 chunks
Decay shape: Exponential with rhetorical cliffs
Pressure trajectory: 0.65 → 0.30 over 50 chunks
```

**Interpretation:**
- Full Mixtral shows natural completion pressure
- Toy model infinite half-life was artifact
- Validates pressure as control surface

### Plot Will Show

**Panel 1:** Mid-pressure decay
- Mixtral: Smooth exponential decay
- Toy: Flat line (infinite half-life)

**Panel 2:** Entropy evolution
- Both show collapse
- Mixtral: Semantic convergence
- Toy: Statistical convergence

**Panel 3:** Decay rate comparison
- Mixtral: -0.01 to -0.02 per chunk
- Toy: ~0.000004 per chunk

**Panel 4:** Half-life bars
- Mixtral: 15-25 chunks
- Toy: 50+ (not reached)

---

## After Execution

### 1. Analyze Results

Review generated files:
```bash
# View trajectory data
cat test_results/mixtral_pressure_halflife.json | jq '.analysis'

# Open comparison plot
open test_results/mixtral_vs_toy_comparison.png

# Read generated text
cat test_results/mixtral_pressure_halflife.json | jq -r '.generated_text'
```

### 2. Document Findings

Create `test_results/MIXTRAL_HALFLIFE_FINDING.md`:
- What is the measured half-life?
- Is decay exponential, linear, or cliff-like?
- How does it compare to toy baseline?
- What does this tell us about pressure dynamics?

### 3. Next Step: Neutral Perturbation

Per Halcyon's protocol:
```python
# Modify measure_mixtral_halflife.py
# After natural pause, add minimal continuation:

continuation_prompt = " Please continue:"
perturbed_results = measure_with_perturbation(prompt, continuation_prompt)

# Compare delta
external_energy = perturbed_pressure - baseline_pressure
```

### 4. Final Step: DeepSeek Comparison

**After Mixtral baseline:**
```bash
# Same protocol, different model
python tests/measure_mixtral_halflife.py \
    --model deepseek-ai/DeepSeek-V2 \
    --8bit

# Compare results
python tests/compare_models.py \
    --mixtral test_results/mixtral_pressure_halflife.json \
    --deepseek test_results/deepseek_pressure_halflife.json
```

**Expected (Halcyon's hypothesis):**
- DeepSeek: Longer half-life, smoother decay
- Mixtral: Shorter half-life, sharper cliffs

**This is where novelty emerges** (from measurements, not claims)

---

## Troubleshooting

### Out of Memory

**Error:** `CUDA out of memory`

**Solutions:**
1. Use 8-bit quantization: `--8bit`
2. Reduce chunk size: `--chunk-size 10`
3. Reduce max chunks: `--chunks 25`
4. Use smaller model: `--model mistralai/Mixtral-8x7B-Instruct-v0.1`

### Model Download Fails

**Error:** `Connection timeout` or `403 Forbidden`

**Solutions:**
1. Login to HuggingFace: `huggingface-cli login`
2. Accept model license on HuggingFace website
3. Use mirror: `HF_ENDPOINT=https://hf-mirror.com python ...`

### Slow Generation

**Expected:** ~30 seconds per chunk on A100
**If slower:** Check GPU utilization (`nvidia-smi`)

---

## Cost Estimates

### One-time Measurement (50 chunks)

**Google Colab Pro:**
- Cost: $10/month subscription
- Runtime: ~30 minutes
- **Effective cost:** ~$0 (if doing multiple experiments)

**Lambda Labs:**
- Cost: $1.10/hour × 0.5 hours = **$0.55**
- Plus: Persistent storage option

**RunPod:**
- Cost: $0.79/hour × 0.5 hours = **$0.40**
- Cheapest option

### Full Experimental Suite

**Baseline + Perturbation + DeepSeek:**
- 3 measurements × 30 minutes = 90 minutes
- Lambda: ~$1.65
- RunPod: ~$1.20

**Recommendation:** Use RunPod for cost, Colab Pro for convenience

---

## Summary

**Infrastructure:** ✓ Ready
**Validation:** ✓ Passed
**Toy baseline:** ✓ Measured
**GPU access:** Required for Mixtral execution

**Recommended path:**
1. Use Google Colab Pro (easiest) or RunPod (cheapest)
2. Run baseline measurement (30 min)
3. Analyze results
4. Run perturbation test (optional)
5. Compare to DeepSeek (final step)

**Expected timeline:**
- Setup: 10 minutes
- Execution: 30 minutes
- Analysis: 30 minutes
- **Total:** ~1.5 hours of work, 30 minutes of GPU time

**Ready to execute when you have GPU access.**
