#!/usr/bin/env python3

import os
import sys
import json
import unittest
import warnings

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from byt5_analyzer import Byt5Analyzer
from cli import run_byt5_text, run_byt5_file

class TestByt5Interface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """This runs ONCE before any tests start. Perfect for loading the model and preparing files."""
        # Suppress HuggingFace noise for clean test outputs
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        warnings.filterwarnings("ignore")
        
        print("\n" + "="*50)
        print("Loading ByT5 Model for Testing. This will take a moment...")
        cls.analyzer = Byt5Analyzer()
        print("Model loaded successfully. Starting tests...")
        print("="*50 + "\n")
        
        # Setup paths for test files
        cls.tests_dir = os.path.dirname(__file__)
        cls.inputs_dir = os.path.join(cls.tests_dir, "inputs")
        cls.outputs_dir = os.path.join(cls.tests_dir, "outputs")
        
        os.makedirs(cls.inputs_dir, exist_ok=True)
        os.makedirs(cls.outputs_dir, exist_ok=True)
        
        # Define standard input files
        cls.word_file = os.path.join(cls.inputs_dir, "words.tsv")
        cls.sent_file = os.path.join(cls.inputs_dir, "sentences.tsv")
        
        # Auto-generate standard input files for testing
        with open(cls.word_file, "w", encoding="utf-8") as f:
            f.write("gacCawi\nnIlowpalam\nagnim\n")
            
        with open(cls.sent_file, "w", encoding="utf-8") as f:
            f.write("rAmaH vanaM gacCawi\nagnimIde purohiwaM yajFasya xevamqwvijam . howAraM rawnaXawamam ..\n")

    # --- Helper methods for validation ---
    def _validate_single_result(self, res):
        self.assertEqual(res.get("status"), "success", f"Failed: {res.get('error')}")
        self.assertEqual(res.get("source"), "ByT5")
        self.assertIn("raw_result", res)
        self.assertTrue(len(res["raw_result"]) > 0)

        print(f"  --> Output: {res['raw_result']}")

    def _validate_file_result(self, output_path, expected_lines):
        self.assertTrue(os.path.exists(output_path), f"File {output_path} was not created.")
        
        with open(output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), expected_lines, "Output line count does not match input.")
            
            # Verify the first line is valid JSON and successful
            first_entry = json.loads(lines[0])
            self._validate_single_result(first_entry)

            print(f"  --> Successfully generated {expected_lines} lines in {os.path.basename(output_path)}")

    # ==========================================
    # SCENARIOS 1-6: Single Text Inferences
    # ==========================================
    
    def test_s01_word_morph(self):
        print("\nRunning Scenario 1: Word -> Morphological Analysis (WX -> WX) [Input: 'gacCawi']")
        res = run_byt5_text(self.analyzer, "gacCawi", "WX", "WX", "mp")
        self._validate_single_result(res)

    def test_s02_sentence_morph(self):
        print("\nRunning Scenario 2: Sentence -> Morphological Analysis (DN -> deva) [Input: 'रामः वनं गच्छति']")
        res = run_byt5_text(self.analyzer, "रामः वनं गच्छति", "DN", "deva", "mp")
        self._validate_single_result(res)

    def test_s03_word_seg(self):
        print("\nRunning Scenario 3: Word -> Segmentation (DN -> deva) [Input: 'नीलोत्पलम्']")
        res = run_byt5_text(self.analyzer, "नीलोत्पलम्", "DN", "deva", "ws")
        self._validate_single_result(res)

    def test_s04_sentence_seg(self):
        print("\nRunning Scenario 4: Sentence -> Segmentation (DN -> deva) [Input: 'अग्निमीळेपुरोहितम्']")
        res = run_byt5_text(self.analyzer, "अग्निमीळेपुरोहितम्", "DN", "deva", "ws")
        self._validate_single_result(res)

    def test_s05_word_wsmp(self):
        print("\nRunning Scenario 5: Word -> Seg + Morph (WX -> roma) [Input: 'nIlowpalam']")
        res = run_byt5_text(self.analyzer, "nIlowpalam", "WX", "roma", "wsmp")
        self._validate_single_result(res)

    def test_s06_sentence_wsmp(self):
        print("\nRunning Scenario 6: Sentence -> Seg + Morph (WX -> roma) [Input: 'agnimIde purohiwaM yajFasya xevamqwvijam . howAraM rawnaXawamam ..']")
        res = run_byt5_text(self.analyzer, "agnimIde purohiwaM yajFasya xevamqwvijam . howAraM rawnaXawamam ..", "WX", "roma", "wsmp")
        self._validate_single_result(res)

    # ==========================================
    # SCENARIOS 7-12: File Batching
    # ==========================================
    
    def test_s07_file_word_morph(self):
        print("\nRunning Scenario 7: File of Words -> Morphological Analysis (WX -> WX)")
        out_file = os.path.join(self.outputs_dir, "s7_out_words_mp.json")
        run_byt5_file(self.analyzer, self.word_file, out_file, "WX", "WX", "mp")
        self._validate_file_result(out_file, expected_lines=3)

    def test_s08_file_word_seg(self):
        print("\nRunning Scenario 8: File of Words -> Segmentation (WX -> WX)")
        out_file = os.path.join(self.outputs_dir, "s8_out_words_ws.json")
        run_byt5_file(self.analyzer, self.word_file, out_file, "WX", "WX", "ws")
        self._validate_file_result(out_file, expected_lines=3)

    def test_s09_file_word_wsmp(self):
        print("\nRunning Scenario 9: File of Words -> Seg + Morph (WX -> WX)")
        out_file = os.path.join(self.outputs_dir, "s9_out_words_wsmp.json")
        run_byt5_file(self.analyzer, self.word_file, out_file, "WX", "WX", "wsmp")
        self._validate_file_result(out_file, expected_lines=3)

    def test_s10_file_sent_morph(self):
        print("\nRunning Scenario 10: File of Sentences -> Morphological Analysis (WX -> WX)")
        out_file = os.path.join(self.outputs_dir, "s10_out_sents_mp.json")
        run_byt5_file(self.analyzer, self.sent_file, out_file, "WX", "WX", "mp")
        self._validate_file_result(out_file, expected_lines=2)

    def test_s11_file_sent_seg(self):
        print("\nRunning Scenario 11: File of Sentences -> Segmentation (WX -> WX)")
        out_file = os.path.join(self.outputs_dir, "s11_out_sents_ws.json")
        run_byt5_file(self.analyzer, self.sent_file, out_file, "WX", "WX", "ws")
        self._validate_file_result(out_file, expected_lines=2)

    def test_s12_file_sent_wsmp(self):
        print("\nRunning Scenario 12: File of Sentences -> Seg + Morph (WX -> WX)")
        out_file = os.path.join(self.outputs_dir, "s12_out_sents_wsmp.json")
        run_byt5_file(self.analyzer, self.sent_file, out_file, "WX", "WX", "wsmp")
        self._validate_file_result(out_file, expected_lines=2)


if __name__ == "__main__":
    # verbosity=1 prevents unittest from printing its own duplicate test names
    unittest.main(verbosity=1)