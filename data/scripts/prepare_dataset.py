import json
import os
import glob
from huggingface_hub import hf_hub_download

def process_taskbench():
    results = []
    try:
        tb_file = hf_hub_download(repo_id="microsoft/Taskbench", filename="data_dailylifeapis/data.json", repo_type="dataset")
        with open(tb_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    row = json.loads(line)
                    
                    task_query = row.get("user_request", "")
                    decomposed_steps = row.get("task_steps", [])
                    
                    candidate_tool_schemas = []
                    for node in row.get("sampled_nodes", []):
                        tool_name = node.get("task", "")
                        candidate_tool_schemas.append({
                            "tool_name": tool_name,
                            "raw_definition": json.dumps(node)
                        })
                        
                    results.append({
                        "task_query": task_query,
                        "decomposed_steps": decomposed_steps,
                        "candidate_tool_schemas": candidate_tool_schemas,
                        "distractor_tool_schemas": []
                    })
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error processing Taskbench: {e}")
    return results

def process_ultratool():
    results = []
    base_dir = "data/raw/UltraTool/data/English-dataset/test_set"
    
    # We will zip planning.json (for query & full steps) and tool_usage.json (for toolset & used tools)
    planning_path = os.path.join(base_dir, "planning.json")
    usage_path = os.path.join(base_dir, "tool_usage.json")
    
    try:
        with open(planning_path, "r", encoding="utf-8") as fp, \
             open(usage_path, "r", encoding="utf-8") as fu:
            
            for p_line, u_line in zip(fp, fu):
                if not p_line.strip() or not u_line.strip():
                    continue
                
                try:
                    p_row = json.loads(p_line)
                    u_row = json.loads(u_line)
                    
                    # 1. task_query
                    task_query = u_row.get("question", p_row.get("input", ""))
                    
                    # 2. decomposed_steps (from planning)
                    decomposed_steps = []
                    p_ref = p_row.get("reference", "")
                    if isinstance(p_ref, str):
                        decomposed_steps = p_ref.split("\n")
                    
                    # 3. Collect used tools (from tool_usage)
                    used_tools_refs = set()
                    u_ref = u_row.get("reference", [])
                    if isinstance(u_ref, list):
                        for item in u_ref:
                            if isinstance(item, dict) and "tool" in item:
                                t = item["tool"]
                                if isinstance(t, dict) and "name" in t:
                                    used_tools_refs.add(str(t["name"]))
                                else:
                                    used_tools_refs.add(str(t))
                                    
                    # 4. Candidate & distractor tools
                    candidate_tool_schemas = []
                    distractor_tool_schemas = []
                    
                    toolset = u_row.get("toolset", [])
                    for idx, tool in enumerate(toolset):
                        if not isinstance(tool, dict):
                            continue
                        tool_name = tool.get("name", "")
                        tool_schema = {
                            "tool_name": tool_name,
                            "raw_definition": json.dumps(tool)
                        }
                        
                        if tool_name in used_tools_refs or str(idx) in used_tools_refs:
                            candidate_tool_schemas.append(tool_schema)
                        else:
                            distractor_tool_schemas.append(tool_schema)
                            
                    results.append({
                        "task_query": task_query,
                        "decomposed_steps": decomposed_steps,
                        "candidate_tool_schemas": candidate_tool_schemas,
                        "distractor_tool_schemas": distractor_tool_schemas
                    })
                    
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error processing UltraTool: {e}")
        
    return results

def main():
    print("Processing TaskBench...")
    tb_data = process_taskbench()
    print(f"Processed {len(tb_data)} samples from TaskBench.")
    
    print("Processing UltraTool...")
    ut_data = process_ultratool()
    print(f"Processed {len(ut_data)} samples from UltraTool.")
    
    combined = tb_data + ut_data
    
    out_path = "data/final_dataset.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out_f:
        for item in combined:
            out_f.write(json.dumps(item) + "\n")
            
    print(f"Saved {len(combined)} total samples to {out_path}.")

if __name__ == "__main__":
    main()
