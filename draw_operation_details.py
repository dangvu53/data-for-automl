#!/usr/bin/env python3
"""
Detailed Operation Flow Diagram for Meta-Learning Framework

This script creates detailed diagrams showing:
1. Operation pool and parameter spaces
2. Data transformation pipeline
3. Evaluation process
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

def create_operation_details_diagram():
    """Create detailed operation flow diagrams"""
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(24, 16))
    
    # Operation pool diagram
    ax_pool = plt.subplot2grid((3, 4), (0, 0), colspan=2)
    
    # Data transformation pipeline
    ax_pipeline = plt.subplot2grid((3, 4), (0, 2), colspan=2)
    
    # Evaluation process
    ax_eval = plt.subplot2grid((3, 4), (1, 0), colspan=4)
    
    # Parameter spaces
    ax_params = plt.subplot2grid((3, 4), (2, 0), colspan=4)
    
    # Draw each component
    draw_operation_pool(ax_pool)
    draw_data_pipeline(ax_pipeline)
    draw_evaluation_process(ax_eval)
    draw_parameter_spaces(ax_params)
    
    plt.tight_layout()
    plt.savefig('meta_learning_operation_details.png', dpi=300, bbox_inches='tight')
    plt.savefig('meta_learning_operation_details.pdf', bbox_inches='tight')
    print("Operation details diagram saved as 'meta_learning_operation_details.png' and '.pdf'")
    plt.show()

def draw_operation_pool(ax):
    """Draw the complete operation pool"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.set_title('Complete Operation Pool (15+ Operations)', fontsize=14, fontweight='bold')
    
    # Operation categories
    categories = [
        {
            'name': 'CLEANING',
            'ops': ['lowercase', 'remove_punctuation', 'normalize_whitespace', 'remove_numbers'],
            'color': '#FFE6CC',
            'pos': (2, 10)
        },
        {
            'name': 'FILTERING', 
            'ops': ['length_filter', 'quality_filter'],
            'color': '#E6F3FF',
            'pos': (8, 10)
        },
        {
            'name': 'SELECTION',
            'ops': ['difficulty_select_easy', 'difficulty_select_hard', 'difficulty_select_balanced', 'random_select'],
            'color': '#E8F5E8',
            'pos': (2, 6)
        },
        {
            'name': 'AUGMENTATION',
            'ops': ['synonym_augment', 'insertion_augment', 'deletion_augment', 'mixed_augment', 'class_balance_augment'],
            'color': '#F0E6FF',
            'pos': (8, 6)
        }
    ]
    
    for category in categories:
        # Category header
        header_box = FancyBboxPatch((category['pos'][0]-1.5, category['pos'][1]), 3, 0.8,
                                   boxstyle="round,pad=0.1", facecolor=category['color'],
                                   edgecolor='black', linewidth=2)
        ax.add_patch(header_box)
        ax.text(category['pos'][0], category['pos'][1]+0.4, category['name'], 
                ha='center', va='center', fontweight='bold', fontsize=11)
        
        # Operations
        for i, op in enumerate(category['ops']):
            y_pos = category['pos'][1] - 0.8 - (i * 0.6)
            op_box = Rectangle((category['pos'][0]-1.4, y_pos-0.2), 2.8, 0.4,
                              facecolor='white', edgecolor='gray', linewidth=1)
            ax.add_patch(op_box)
            ax.text(category['pos'][0], y_pos, op, ha='center', va='center', fontsize=9)
    
    # Arrows showing flow
    ax.text(5, 2, 'Operations can be combined in ANY order\nwith flexible parameters', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def draw_data_pipeline(ax):
    """Draw data transformation pipeline"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.set_title('Data Transformation Pipeline Example', fontsize=14, fontweight='bold')
    
    # Pipeline steps
    steps = [
        ('Original Data', '16,946 samples', '#E8F4FD', 11),
        ('quality_filter', '14,200 samples\n(remove low quality)', '#E6F3FF', 9.5),
        ('difficulty_select_hard', '9,800 samples\n(keep challenging)', '#E8F5E8', 8),
        ('mixed_augment', '12,500 samples\n(add variations)', '#F0E6FF', 6.5),
        ('lowercase', '12,500 samples\n(normalize text)', '#FFE6CC', 5),
        ('Final Training Data', '12,500 samples\nready for AutoGluon', '#FFE6E6', 3.5)
    ]
    
    for i, (name, desc, color, y_pos) in enumerate(steps):
        # Step box
        box = FancyBboxPatch((2, y_pos-0.5), 6, 1, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        
        # Step name
        ax.text(3, y_pos, name, ha='left', va='center', fontweight='bold', fontsize=10)
        
        # Description
        ax.text(7, y_pos, desc, ha='right', va='center', fontsize=9)
        
        # Arrow to next step
        if i < len(steps) - 1:
            ax.arrow(5, y_pos-0.5, 0, -0.5, head_width=0.2, head_length=0.1, 
                    fc='black', ec='black', lw=1.5)
    
    # Side annotations
    ax.text(0.5, 8, 'Data Size\nChanges', ha='center', va='center', fontweight='bold', rotation=90)
    ax.text(9.5, 8, 'Quality\nImproves', ha='center', va='center', fontweight='bold', rotation=90)
    
    # Validation/Test data note
    ax.text(5, 1.5, '⚠️ Validation & Test Data: COMPLETELY UNTOUCHED\nVal: 1,000 → 1,000 (100%)\nTest: 1,000 → 1,000 (100%)', 
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.8))
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def draw_evaluation_process(ax):
    """Draw the evaluation process"""
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.set_title('Pipeline Evaluation Process (Training Data Only)', fontsize=14, fontweight='bold')
    
    # Evaluation steps
    eval_steps = [
        ('Training\nSubset\n(2,000)', (2, 4), '#E8F4FD'),
        ('Apply\nPipeline', (4.5, 4), '#FFE6CC'),
        ('Processed\nData', (7, 4), '#E8F5E8'),
        ('Train/Val\nSplit', (9.5, 4), '#E6F3FF'),
        ('Quick\nEvaluation', (12, 4), '#F0E6FF'),
        ('Fitness\nScore', (14.5, 4), '#FFE6E6')
    ]
    
    for name, pos, color in eval_steps:
        box = FancyBboxPatch((pos[0]-0.7, pos[1]-0.7), 1.4, 1.4, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(pos[0], pos[1], name, ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Arrows between steps
    for i in range(len(eval_steps) - 1):
        start_pos = eval_steps[i][1]
        end_pos = eval_steps[i+1][1]
        ax.arrow(start_pos[0]+0.7, start_pos[1], end_pos[0]-start_pos[0]-1.4, 0,
                head_width=0.15, head_length=0.2, fc='black', ec='black', lw=1.5)
    
    # Details below
    details = [
        'Original training\ndata subset for\nfaster evaluation',
        'Apply discovered\noperation sequence\nto data',
        'Transformed data\nafter all operations\napplied',
        '70% train\n30% validation\n(internal split)',
        'RandomForest\nwith simple\nfeatures',
        'Accuracy score\n0.0 - 1.0\n(higher = better)'
    ]
    
    for i, detail in enumerate(details):
        x_pos = eval_steps[i][1][0]
        ax.text(x_pos, 1.5, detail, ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgray', alpha=0.7))
    
    # Key principle
    ax.text(8, 0.3, '🔑 KEY: No validation/test data used in meta-learning evaluation!', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='gold', alpha=0.9))
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def draw_parameter_spaces(ax):
    """Draw parameter spaces for operations"""
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 8)
    ax.set_title('Operation Parameter Spaces (Examples)', fontsize=14, fontweight='bold')
    
    # Parameter examples
    param_examples = [
        {
            'operation': 'length_filter',
            'params': {
                'min_percentile': '[5, 10, 15, 20]',
                'max_percentile': '[80, 85, 90, 95]'
            },
            'pos': (3, 6),
            'color': '#E6F3FF'
        },
        {
            'operation': 'difficulty_select_hard',
            'params': {
                'threshold_percentile': '[50, 60, 70]',
                'ratio': '[0.5, 0.6, 0.7, 0.8]'
            },
            'pos': (8, 6),
            'color': '#E8F5E8'
        },
        {
            'operation': 'mixed_augment',
            'params': {
                'ratio': '[0.1, 0.15, 0.2, 0.25]',
                'minority_boost': '[1.5, 2.0, 2.5, 3.0]'
            },
            'pos': (13, 6),
            'color': '#F0E6FF'
        },
        {
            'operation': 'quality_filter',
            'params': {
                'threshold_percentile': '[10, 20, 30, 40]'
            },
            'pos': (18, 6),
            'color': '#FFE6CC'
        }
    ]
    
    for example in param_examples:
        # Operation box
        op_box = FancyBboxPatch((example['pos'][0]-2, example['pos'][1]), 4, 1.5,
                               boxstyle="round,pad=0.1", facecolor=example['color'],
                               edgecolor='black', linewidth=1.5)
        ax.add_patch(op_box)
        
        # Operation name
        ax.text(example['pos'][0], example['pos'][1]+1, example['operation'], 
                ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Parameters
        param_text = '\n'.join([f"{k}: {v}" for k, v in example['params'].items()])
        ax.text(example['pos'][0], example['pos'][1]+0.3, param_text, 
                ha='center', va='center', fontsize=8)
    
    # Exploration note
    ax.text(10, 3, 'Meta-Learning Explores ALL Parameter Combinations\n\n' +
                   '• length_filter(min=10, max=90) + difficulty_select_hard(threshold=60, ratio=0.7)\n' +
                   '• quality_filter(threshold=20) + mixed_augment(ratio=0.15, boost=2.0)\n' +
                   '• Any combination of 15+ operations with their parameter spaces\n' +
                   '• Discovers optimal combinations automatically!', 
            ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", facecolor='lightyellow', alpha=0.9))
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

if __name__ == "__main__":
    create_operation_details_diagram()
