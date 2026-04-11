import os, sys
import pandas as pd
import re
import ast
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from tqdm import tqdm


class Byt5Analyzer:
    def __init__(self, model_name="chronbmm/sanskrit5-multitask", device=None, tags_file=None, batch_size=32):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = 512
        self.batch_size = batch_size
        
        # Load Model and Tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Mode to Prefix Mapping
        self.mode_map = {
            'ws': "S ",        # Word Segmentation
            'wsmp': "SLM ",    # Segmentation + Lemma + Morphosyntax
            'mp': "LM "        # Lemma + Morphosyntax
        }
        
        if tags_file is None:
            # Gets the absolute path of the directory containing byt5_analyzer.py
            current_dir = os.path.dirname(os.path.abspath(__file__))
            tags_file = os.path.join(current_dir, "data", "sanskrit_tags.tsv")
        
        # Load Tag Expansions
        self.sanskrit_tags = self._load_tags(tags_file)

    def _load_tags(self, tags_file):
        tags = {}
        try:
            with open(tags_file, "r") as f:
                for line in f:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        tags[parts[0]] = parts[1].strip()
        except FileNotFoundError:
            print(f"Warning: {tags_file} not found. Returning raw tags.")
        return tags

    def _postprocess(self, sentence, mode):
        if mode in ["wsmp", "mp"]:        
            result = ""                
            for item in sentence.split(" "):
                parts = item.split("_")
                if mode == "wsmp" and len(parts) == 3:
                    unsandhied, lemma, short_tag = parts
                    expanded_tag = self.sanskrit_tags.get(short_tag, short_tag)
                    if "Cpd" in expanded_tag:
                        unsandhied += "-"
                    result += f"{unsandhied}_{lemma}_{expanded_tag} "
                
                elif mode == "mp" and len(parts) == 2:
                    lemma, short_tag = parts
                    expanded_tag = self.sanskrit_tags.get(short_tag, short_tag)
                    result += f"{lemma}_{expanded_tag} "
                else:
                    result += f"{item} " # Fallback
            return result.strip()
            
        elif mode == "ws":
            # The model outputs underscores for segmentation, replace with spaces
            return sentence.replace("_", " ")
            
        return sentence

    def process_batch(self, batch, mode):
        prefix = self.mode_map[mode]
        input_texts = [f"{prefix}{text}" for text in batch]
        inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=self.max_length)
        
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def analyze_batch(self, input_list, mode):
        """Runs batch inference and postprocessing over a given list of strings."""
        if mode not in self.mode_map:
            raise ValueError(f"Invalid mode. Choose from {list(self.mode_map.keys())}")

        results = []
        for i in tqdm(range(0, len(input_list), self.batch_size), desc="Byt5 Inference"):
            batch = input_list[i:i+self.batch_size]
            raw_outputs = self.process_batch(batch, mode)
            results.extend([self._postprocess(res, mode=mode) for res in raw_outputs])
        return results

    def analyze(self, text, mode):
        """Single string inference for API calls."""
        return self.analyze_batch([text], mode)[0]
