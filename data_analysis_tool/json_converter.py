# -*- coding: utf-8 -*-

"""
Convert JSON data file to Excel format
"""
import json
import pandas as pd
import sys
import os
from pathlib import Path
from typing import List, Dict, Any


def load_json_data(json_file: str) -> List[Dict[str, Any]]:
    """Load data from JSON file"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        raise


def flatten_json_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten JSON data for Excel export"""
    flattened = []
    
    for item in data:
        row = {}
        
        # Basic fields
        row['序号'] = item.get('序号', '')
        row['用户问题'] = item.get('用户问题', item.get('问题', ''))  # 支持"用户问题"或"问题"字段
        row['正确溯源'] = item.get('正确溯源', item.get('参考知识', ''))
        row['正确答案'] = item.get('正确答案', item.get('参考答案', ''))
        row['taskid'] = item.get('taskid', '')
        row['sessionid'] = item.get('sessionid', '')
        row['requestid'] = item.get('requestid', '')
        row['检索结果'] = item.get('检索结果', '')
        row['模型回复'] = item.get('模型输出结果', item.get('回复结果', ''))
        
        # Extract all source fields (溯源1, 溯源2, etc.)
        source_fields = {}
        for key, value in item.items():
            if key.startswith('溯源'):
                source_fields[key] = value
        
        # Add source fields to row (up to 10 sources)
        for i in range(1, 11):
            source_key = f'溯源{i}'
            row[source_key] = source_fields.get(source_key, '')
        
        flattened.append(row)
    
    return flattened


def convert_json_to_excel(json_file: str, output_file: str = None):
    """Convert JSON file to Excel format"""
    json_path = Path(json_file)
    
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_file}")
        return
    
    # Determine output file path
    if output_file is None:
        output_file = str(json_path.parent / f"{json_path.stem}.xlsx")
    else:
        output_file = str(Path(output_file))
    
    print(f"Loading JSON file: {json_file}")
    data = load_json_data(json_file)
    print(f"Loaded {len(data)} records")
    
    print("Flattening data...")
    flattened_data = flatten_json_data(data)
    
    print(f"Creating Excel file: {output_file}")
    df = pd.DataFrame(flattened_data)
    
    # Reorder columns for better readability
    column_order = ['序号', '用户问题', '正确溯源', '正确答案', 'taskid', 'sessionid', 'requestid', 
                    '检索结果', '模型回复']
    # Add source columns
    for i in range(1, 11):
        column_order.append(f'溯源{i}')
    
    # Only include columns that exist in the dataframe
    column_order = [col for col in column_order if col in df.columns]
    
    # Reorder dataframe
    df = df[column_order]
    
    # Write to Excel
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"Successfully converted to Excel: {output_file}")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python json_to_excel.py <json_file> [output_file]")
        print("Example: python json_to_excel.py batch_1764828830_data.json")
        return
    
    json_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        convert_json_to_excel(json_file, output_file)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


