
import csv
import os
import glob

# Mapping for number words
NUMBER_WORDS = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
    'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
    'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80,
    'ninety': 90, 'hundred': 100
}

def parse_number_word(text):
    """Parse a number word (e.g., 'seventeen') to integer."""
    text = text.lower().strip()
    
    # Direct lookup
    if text in NUMBER_WORDS:
        return NUMBER_WORDS[text]
    
    # Handle compound numbers like "twenty-one", "thirty-two", etc.
    # Pattern: tens-word + hyphen + ones-word
    parts = text.split('-')
    if len(parts) == 2:
        tens = NUMBER_WORDS.get(parts[0], 0)
        ones = NUMBER_WORDS.get(parts[1], 0)
        if tens > 0 and ones > 0:
            return tens + ones
    
    return None

def parse_temperature(temp_str):
    """Parse temperature, handling numeric strings and number words."""
    # Try direct integer conversion first
    try:
        return int(temp_str)
    except ValueError:
        pass
    
    # Try parsing as number word
    num = parse_number_word(temp_str)
    if num is not None:
        return num
    
    return None

def solve_input_file(input_file):
    """Solve a single input file and return sorted BOP IDs."""
    bops = []
    
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        # Skip header
        next(reader)
        
        for row in reader:
            if len(row) < 3:
                continue
            
            bop_id = row[0].strip()
            temp_str = row[1].strip()
            humidity_str = row[2].strip()
            
            # Parse temperature
            temp = parse_temperature(temp_str)
            if temp is None:
                # Skip invalid entries
                continue
            
            # Parse humidity (also handle number words)
            try:
                humidity = int(humidity_str)
            except ValueError:
                # Try parsing as number word
                humidity = parse_number_word(humidity_str)
                if humidity is None:
                    # Skip invalid humidity
                    continue
            
            bops.append({
                'id': int(bop_id),
                'temperature': temp,
                'humidity': humidity
            })
    
    # Sort: temperature (desc), humidity (asc), BOP id (asc)
    bops.sort(key=lambda x: (-x['temperature'], x['humidity'], x['id']))
    
    # Return sorted BOP IDs as space-separated string
    return ' '.join(str(bop['id']) for bop in bops)

def verify_solution(input_file, result):
    """Verify solution against expected output if it exists."""
    expected_file = input_file.replace('.in', '.txt')
    if os.path.exists(expected_file):
        with open(expected_file, 'r') as f:
            expected = f.read().strip()
        if result == expected:
            return True, "✓ MATCHES"
        else:
            return False, f"✗ MISMATCH\n    Expected: {expected}\n    Got:      {result}"
    return None, "No expected file to compare"

def main():
    # Find all input files
    input_files = sorted(glob.glob('level_1_*.in'))
    
    if not input_files:
        print("No input files found!")
        return
    
    # Solve each input file
    all_correct = True
    for input_file in input_files:
        output_file = input_file.replace('.in', '.txt')
        result = solve_input_file(input_file)
        
        # Verify before writing (read expected if it exists)
        status, message = verify_solution(input_file, result)
        if status is False:
            all_correct = False
        
        # Write output
        with open(output_file, 'w') as f:
            f.write(result + '\n')
        
        print(f"{input_file} -> {output_file}")
        print(f"  {message}")
        if status is None:
            print(f"  Result: {result}")
        print()
    
    if all_correct:
        print("✓ All solutions verified!")
    elif not all_correct:
        print("✗ Some solutions don't match expected outputs!")

if __name__ == '__main__':
    main()

