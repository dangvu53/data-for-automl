import pandas as pd
from sklearn.model_selection import train_test_split
from meta_learning_pipeline import run_pipeline_for_dataset
import os
from datasets import load_dataset

# --- Dataset Loading Functions ---
# Each function should return train_df, val_df, test_df

def get_casehold_data():
    """Loader for the Casehold dataset."""
    print("Loading Casehold dataset from Hugging Face...")
    dataset = load_dataset("MothMalone/SLMS-KD-Benchmarks", "casehold")
    
    def prepare_casehold_data(split_data):
        data = []
        for item in split_data:
            # Combine citing prompt with all holdings for context
            text_features = item['citing_prompt']
            # The holdings are in separate features: holding_0, holding_1, etc.
            for i in range(5):
                text_features += f" [HOLDING_{i}] " + item[f'holding_{i}']
            
            data.append({
                'text': text_features,
                'label': item['label']
            })
        return pd.DataFrame(data)

    train_df = prepare_casehold_data(dataset['train'])
    val_df = prepare_casehold_data(dataset['validation'])
    test_df = prepare_casehold_data(dataset['test'])
    print(f"CaseHold loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df

def get_anli_r1_data():
    """Loader for the ANLI R1 dataset."""
    print("Loading ANLI R1 dataset from Hugging Face...")
    dataset = load_dataset("facebook/anli")
    
    def prepare_anli_data(split_data):
        data = []
        for item in split_data:
            # Combine premise and hypothesis for NLI
            text = f"[PREMISE] {item['premise']} [HYPOTHESIS] {item['hypothesis']}"
            data.append({'text': text, 'label': item['label']})
        return pd.DataFrame(data)

    train_df = prepare_anli_data(dataset['train_r1'])
    val_df = prepare_anli_data(dataset['dev_r1'])
    test_df = prepare_anli_data(dataset['test_r1'])
    print(f"ANLI R1 loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df

def get_scienceqa_data():
    """Loader for the ScienceQA dataset (text-only)."""
    print("Loading ScienceQA dataset from Hugging Face...")
    dataset = load_dataset("MothMalone/SLMS-KD-Benchmarks", "scienceqa")

    def prepare_scienceqa_data(split_data):
        data = []
        for item in split_data:
            # Prepare text features
            text_features = item['question']
            if item['choices']:
                text_features += " [CHOICES] " + " | ".join(item['choices'])
            if item['hint']:
                text_features += " [HINT] " + item['hint']
            if item.get('lecture'): # lecture might not be present
                text_features += " [LECTURE] " + item['lecture']
            
            data.append({'text': text_features, 'label': item['answer']})
        return pd.DataFrame(data)

    train_df = prepare_scienceqa_data(dataset['train'])
    val_df = prepare_scienceqa_data(dataset['validation'])
    test_df = prepare_scienceqa_data(dataset['test'])
    print(f"ScienceQA loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df

# --- Main Execution Logic ---

def main():
    """
    Main function to run the meta-learning pipeline on multiple datasets.
    """
    # Define the datasets to process
    datasets = {
        'casehold': {
            'loader': get_casehold_data,
            'text_column': 'text',
            'label_column': 'label'
        },
        'anli_r1': {
            'loader': get_anli_r1_data,
            'text_column': 'text',
            'label_column': 'label'
        },
        'scienceqa': {
            'loader': get_scienceqa_data,
            'text_column': 'text',
            'label_column': 'label'
        },
    }
    
    output_base_dir = 'cleaned_datasets'
    os.makedirs(output_base_dir, exist_ok=True)
    
    for name, config in datasets.items():
        run_pipeline_for_dataset(
            dataset_name=name,
            get_data_function=config['loader'],
            text_column=config['text_column'],
            label_column=config['label_column'],
            output_base_dir=output_base_dir
        )
        
    print("--- All dataset processing complete. ---")
    print(f"Cleaned data and pipelines saved in '{output_base_dir}' directory.")

if __name__ == '__main__':
    main()
