
import pandas as pd
import re
from collections import defaultdict

def main():
    print("Forensic Analysis of English Translations...")
    
    # Load Data
    train_df = pd.read_csv('train_log.csv')
    
    # Dictionary to store stats: word -> [total_count, tde_count]
    stats = defaultdict(lambda: [0, 0])
    
    # Regex to split by +, ,, and cleanup whitespace
    # Example: "hewn, log + wolf + hound of chase" -> ["hewn", "log", "wolf", "hound of chase"]
    # Actually, let's split by '+' first to get segments, then maybe split by ','?
    # Or just tokenize everything into words. "hound of chase" -> "hound", "of", "chase".
    # Let's try both: phrase-level and word-level.
    
    phrase_stats = defaultdict(lambda: [0, 0])
    word_stats = defaultdict(lambda: [0, 0])
    
    stop_words = {'of', 'the', 'and', 'a', 'in', 'to', 'for', 'with', 'on', 'at', 'from', 'by'}
    
    for idx, row in train_df.iterrows():
        is_tde = row['target']
        text = str(row['English Translation']).lower()
        
        # Split by '+' first (major components)
        major_parts = [p.strip() for p in text.split('+')]
        
        for part in major_parts:
            # Clean part (remove punctuation like parens)
            # "Trawn Folk (Dwarfs)" -> "Trawn Folk" and "Dwarfs" maybe?
            # Let's keep it simple first
            
            # Phrase Level (clean string)
            clean_phrase = re.sub(r'[^\w\s]', '', part).strip()
            if clean_phrase:
                phrase_stats[clean_phrase][0] += 1
                if is_tde: phrase_stats[clean_phrase][1] += 1
            
            # Word Level
            words = re.split(r'\W+', part)
            for w in words:
                if w and w not in stop_words and len(w) > 2:
                    word_stats[w][0] += 1
                    if is_tde: word_stats[w][1] += 1

    print("\n--- MAGIC PHRASES (Prob > 0.5, Count > 2) ---")
    magic_phrases = []
    for p, (tot, tdes) in phrase_stats.items():
        if tot > 2 and (tdes / tot) > 0.5:
            prob = tdes / tot
            print(f"Phrase: '{p}' | Count: {tot} | TDEs: {tdes} | Prob: {prob:.2f}")
            magic_phrases.append((p, prob))
            
    print("\n--- MAGIC WORDS (Prob > 0.5, Count > 5) ---")
    magic_words = []
    for w, (tot, tdes) in word_stats.items():
        if tot > 5 and (tdes / tot) > 0.5:
            prob = tdes / tot
            print(f"Word: '{w}' | Count: {tot} | TDEs: {tdes} | Prob: {prob:.2f}")
            magic_words.append((w, prob))
            
    print("\n--- TOP TDE ASSOCIATIONS ---")
    # Sort purely by TDE count to see common themes
    sorted_words = sorted(word_stats.items(), key=lambda x: x[1][1], reverse=True)
    for w, (tot, tdes) in sorted_words[:20]:
         print(f"Word: '{w}' | Count: {tot} | TDEs: {tdes} | Prob: {tdes/tot:.2f}")

    # Inspect 'Magic' coverage on Train
    # If we use the magic list, how many TDEs do we catch?
    # Let's formulate a rule.

if __name__ == "__main__":
    main()
