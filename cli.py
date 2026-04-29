#!/usr/bin/env python3

import os
import sys
import argparse
import json
from tqdm import tqdm
from aksharamukha import transliterate
import re

# Import the Byt5Analyzer class (from the previous setup)
from byt5_analyzer import *

# Encoding mappings for Aksharamukha
enc_map_in = {"DN": "Devanagari", "KH": "HK", "RN": "IAST", "SL": "SLP1", "VH": "Velthuis", "WX": "WX"}
enc_map_out = {"deva": "Devanagari", "roma": "IAST", "WX": "WX"}

def to_iast(text, input_enc):
    src = enc_map_in.get(input_enc, "Devanagari")
    if src == "IAST" or not text.strip():
        return text
    return transliterate.process(src, 'IAST', text)

def from_iast(text, output_enc):
    tgt = enc_map_out.get(output_enc, "IAST")
    if tgt == "IAST" or not text.strip():
        return text
    return transliterate.process('IAST', tgt, text)

def format_output(iast_input, raw_result, output_encoding, mode):
    """Selectively transliterates only the Sanskrit components of the output."""
    raw_result = raw_result.replace("/", "।")
    raw_result = re.sub(r' +', ' ', raw_result)
    
    if output_encoding in ["roma", "IAST"]:
        return raw_result # No transliteration needed
        
    if mode == "ws":
        # Mode ws: The entire result is just words, safe to transliterate all at once
        return from_iast(raw_result, output_encoding)
        
    formatted_tokens = []
    for token in raw_result.split(" "):
        parts = token.split("_")
        
        if mode == "wsmp" and len(parts) == 3:
            # Mode wsmp: unsandhied_lemma_morph
            word, lemma, morph = parts
            word_trans = from_iast(word, output_encoding)
            lemma_trans = from_iast(lemma, output_encoding)
            formatted_tokens.append(f"{word_trans}_{lemma_trans}_{morph}")
            
        elif mode == "mp" and len(parts) == 2:
            # Mode mp: lemma_morph
            lemma, morph = parts
            lemma_trans = from_iast(lemma, output_encoding)
            word_trans = from_iast(iast_input, output_encoding)
            formatted_tokens.append(f"{word_trans}_{lemma_trans}_{morph}")
            
        else:
            # Fallback for unexpected formats to prevent crashing
            formatted_tokens.append(token)
            
    return " ".join(formatted_tokens)

def run_byt5_text(analyzer, input_text, input_encoding, output_encoding, mode):
    """ Handles transliteration, runs inference, and formats the output. """
    input_text = input_text.strip()
    if not input_text:
        return {}

    try:
        # 1. Transliterate to IAST for ByT5
        iast_input = to_iast(input_text, input_encoding)
        
        # 2. Run Inference
        raw_result = analyzer.analyze(iast_input, mode=mode)
        
        # 3. Smart Transliteration (Applies only to word/lemma, protects morph tags)
        formatted_result = format_output(iast_input, raw_result, output_encoding, mode)
        
        # 4. Construct SH-style response
        morph_analysis = {
            "input": from_iast(iast_input, output_encoding),
            "status": "success",
            "source": "ByT5",
            "raw_result": formatted_result
        }
            
        return morph_analysis

    except Exception as e:
        return {
            "input": input_text,
            "status": "failed",
            "error": str(e),
            "source": "ByT5"
        }

def run_byt5_file(analyzer, input_file, output_file, input_encoding, output_encoding, mode):
    """ Handles batch processing for a file. """
    try:
        with open(input_file, 'r', encoding='utf-8') as ifile:
            input_list = [line.strip() for line in ifile if line.strip()]
    except OSError as e:
        print(f"Unable to open {input_file}: {e}", file=sys.stderr)
        sys.exit(1)

    # 1. Transliterate all inputs to IAST
    iast_inputs = [to_iast(w, input_encoding) for w in input_list]

    # 2. Run batched inference natively (handles words, compounds, or sentences the exact same way)
    raw_results = analyzer.analyze_batch(iast_inputs, mode)
    
    # 3. Format and reconstruct JSON
    output_list = []
    for original_text, res in zip(input_list, raw_results):
        formatted_res = format_output(res, output_encoding, mode)
        output_list.append({
            "input": from_iast(to_iast(original_text, input_encoding), output_encoding),
            "status": "success",
            "raw_result": formatted_res,
            "source": "ByT5"
        })
    
    with open(output_file, 'w', encoding='utf-8') as out_file:
        output_contents = [json.dumps(item, ensure_ascii=False) for item in output_list]
        out_file.write("\n".join(output_contents) + "\n")

def main():
    parser = argparse.ArgumentParser(description="ByT5 Sanskrit Analyzer CLI")
    
    parser.add_argument("input_enc", default="WX", choices=enc_map_in.keys(), help="input encoding")
    parser.add_argument("output_enc", default="roma", choices=enc_map_out.keys(), help="output encoding")
    
    parser.add_argument("-m", "--model_mode", default="wsmp", choices=['ws', 'wsmp', 'mp'], help="ByT5 specific mode")
    parser.add_argument("-t", "--input_text", type=str, help="input string")
    parser.add_argument("-i", "--input_file", type=str, help="reads from file if specified")
    parser.add_argument("-o", "--output_file", type=str, help="for writing to file")
    
    args = parser.parse_args()
    
    if args.input_file and args.input_text:
        print("Please specify either input text ('-t') or input file ('-i, -o')")
        sys.exit(1)
        
    analyzer = Byt5Analyzer()

    if args.input_file:
        o_file = args.output_file if args.output_file else "output.txt"
        run_byt5_file(analyzer, args.input_file, o_file, args.input_enc, args.output_enc, args.model_mode)
    elif args.input_text:
        res = run_byt5_text(analyzer, args.input_text, args.input_enc, args.output_enc, args.model_mode)
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as o_file:
                json.dump(res, o_file, ensure_ascii=False)
        else:
            print(json.dumps(res, ensure_ascii=False))
    else:
        print("Please specify one of text ('-t') or file ('-i & -o')")
        sys.exit(1)

if __name__ == "__main__":
    main()