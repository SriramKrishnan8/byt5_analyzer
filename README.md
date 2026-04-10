# ByT5 Sanskrit Interface

A lightweight, standalone Python interface and CLI for ByT5-based Sanskrit morphological analysis and word segmentation.

## Acknowledgements & Citation
The core inference models and sequence-to-sequence approaches utilized in this interface are based on the work at [Dharmamitra](https://dharmamitra.org/) and the team behind the ByT5 Sanskrit Analyzers.

* **Original Repository:** [sebastian-nehrdich/byt5-sanskrit-analyzers](https://github.com/sebastian-nehrdich/byt5-sanskrit-analyzers)
* **Model:** `chronbmm/sanskrit5-multitask`

## Setup
1. Clone the repository.
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
The interface can be run via the command line using the `cli.py` script or the `run_byt5.sh` wrapper.

**Single String Inference:**
```bash
# Word Segmentation (Devanagari)
python3 cli.py DN deva -m ws -t "अग्निमीळेपुरोहितम्"
```

**Batch File Processing:**
```bash
# Segmentation and Morphological Analysis on a file of sentences (WX)
python3 cli.py WX WX -m mp -i inputs/sentences.tsv -o outputs/results.json
```

**Running the Tests:**
```bash
bash run_byt5.sh --test
```