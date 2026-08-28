import json
import random
import os

def main():
    input_file = "data/final_dataset.jsonl"
    temp_file = "data/final_dataset_temp.jsonl"
    
    # 1. Read all data
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data.append(json.loads(line))
            
    # 2. Collect a pool of all unique tool schemas across the dataset
    all_tools_pool = []
    seen_tool_names = set()
    
    for row in data:
        # Collect from candidates
        for t in row.get("candidate_tool_schemas", []):
            t_name = t.get("tool_name", "")
            if t_name and t_name not in seen_tool_names:
                seen_tool_names.add(t_name)
                all_tools_pool.append(t)
        # Collect from distractors
        for t in row.get("distractor_tool_schemas", []):
            t_name = t.get("tool_name", "")
            if t_name and t_name not in seen_tool_names:
                seen_tool_names.add(t_name)
                all_tools_pool.append(t)
                
    print(f"Total unique tools in pool: {len(all_tools_pool)}")
    
    # 3. Add random distractors to each row
    for row in data:
        current_candidates = {t.get("tool_name", "") for t in row.get("candidate_tool_schemas", [])}
        current_distractors = {t.get("tool_name", "") for t in row.get("distractor_tool_schemas", [])}
        
        # Determine how many distractors to add to ensure there are at least some (e.g., add 3 to 5 random ones)
        # Let's add exactly 4 distractors from the pool to every row's distractor list
        num_to_add = 4
        added = 0
        
        # Shuffle the pool for this row
        row_pool = list(all_tools_pool)
        random.shuffle(row_pool)
        
        for pool_tool in row_pool:
            p_name = pool_tool.get("tool_name", "")
            # Ensure it is not a candidate tool for this row, and not already a distractor
            if p_name not in current_candidates and p_name not in current_distractors:
                row.setdefault("distractor_tool_schemas", []).append(pool_tool)
                current_distractors.add(p_name)
                added += 1
                if added >= num_to_add:
                    break
                    
    # 4. Save the updated dataset
    with open(temp_file, 'w', encoding='utf-8') as f:
        for row in data:
            f.write(json.dumps(row) + "\n")
            
    # Replace old file with new file
    os.replace(temp_file, input_file)
    print("Distractors successfully added to the dataset.")

if __name__ == "__main__":
    # Ensure reproducibility
    random.seed(42)
    main()
