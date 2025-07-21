#!/usr/bin/env python3
"""
Flow Diagram Generator for Hybrid Exploratory Meta-Learning Framework

This script creates a comprehensive flow diagram showing how the meta-learning
framework works, including all generation strategies and the evolution process.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_meta_learning_flow_diagram():
    """Create comprehensive flow diagram of the meta-learning framework"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Main flow diagram
    ax_main = plt.subplot2grid((4, 3), (0, 0), colspan=3, rowspan=2)
    
    # Generation strategies detail
    ax_strategies = plt.subplot2grid((4, 3), (2, 0), colspan=2)
    
    # Evolution process detail  
    ax_evolution = plt.subplot2grid((4, 3), (2, 2))
    
    # Example sequences
    ax_examples = plt.subplot2grid((4, 3), (3, 0), colspan=3)
    
    # Draw main flow
    draw_main_flow(ax_main)
    
    # Draw generation strategies
    draw_generation_strategies(ax_strategies)
    
    # Draw evolution process
    draw_evolution_process(ax_evolution)
    
    # Draw example sequences
    draw_example_sequences(ax_examples)
    
    plt.tight_layout()
    plt.savefig('meta_learning_framework_flow.png', dpi=300, bbox_inches='tight')
    plt.savefig('meta_learning_framework_flow.pdf', bbox_inches='tight')
    print("Flow diagram saved as 'meta_learning_framework_flow.png' and '.pdf'")
    plt.show()

def draw_main_flow(ax):
    """Draw the main meta-learning flow"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_title('Hybrid Exploratory Meta-Learning Framework - Main Flow', fontsize=16, fontweight='bold')
    
    # Define colors
    colors = {
        'data': '#E8F4FD',
        'meta': '#FFE6CC', 
        'eval': '#E8F5E8',
        'apply': '#F0E6FF',
        'autogluon': '#FFE6E6'
    }
    
    # Step 1: Load Data
    box1 = FancyBboxPatch((0.5, 6.5), 1.5, 1, boxstyle="round,pad=0.1", 
                          facecolor=colors['data'], edgecolor='black', linewidth=2)
    ax.add_patch(box1)
    ax.text(1.25, 7, 'Load ANLI R1\nDataset', ha='center', va='center', fontweight='bold')
    
    # Step 2: Meta-Learning
    box2 = FancyBboxPatch((3, 6.5), 2, 1, boxstyle="round,pad=0.1",
                          facecolor=colors['meta'], edgecolor='black', linewidth=2)
    ax.add_patch(box2)
    ax.text(4, 7, 'Hybrid Meta-Learning\n(Training Data Only)', ha='center', va='center', fontweight='bold')
    
    # Step 3: Best Pipeline
    box3 = FancyBboxPatch((6, 6.5), 1.5, 1, boxstyle="round,pad=0.1",
                          facecolor=colors['eval'], edgecolor='black', linewidth=2)
    ax.add_patch(box3)
    ax.text(6.75, 7, 'Best Pipeline\nDiscovered', ha='center', va='center', fontweight='bold')
    
    # Step 4: Apply Pipeline
    box4 = FancyBboxPatch((3, 4.5), 2, 1, boxstyle="round,pad=0.1",
                          facecolor=colors['apply'], edgecolor='black', linewidth=2)
    ax.add_patch(box4)
    ax.text(4, 5, 'Apply Pipeline\nto Training Data', ha='center', va='center', fontweight='bold')
    
    # Step 5: AutoGluon Training
    box5 = FancyBboxPatch((6, 4.5), 2, 1, boxstyle="round,pad=0.1",
                          facecolor=colors['autogluon'], edgecolor='black', linewidth=2)
    ax.add_patch(box5)
    ax.text(7, 5, 'AutoGluon Training\n(Your Exact Settings)', ha='center', va='center', fontweight='bold')
    
    # Step 6: Evaluation
    box6 = FancyBboxPatch((4, 2.5), 2, 1, boxstyle="round,pad=0.1",
                          facecolor=colors['eval'], edgecolor='black', linewidth=2)
    ax.add_patch(box6)
    ax.text(5, 3, 'Evaluate on\nOriginal Test Data', ha='center', va='center', fontweight='bold')
    
    # Data splits
    ax.text(0.5, 5.5, 'Train: 16,946\nVal: 1,000\nTest: 1,000', ha='left', va='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    ax.text(8.5, 5.5, 'Processed Train: ~11,000\nVal: 1,000 (untouched)\nTest: 1,000 (untouched)', 
            ha='left', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    # Arrows
    arrows = [
        ((2, 7), (3, 7)),      # Load -> Meta-Learning
        ((5, 7), (6, 7)),      # Meta-Learning -> Best Pipeline
        ((4, 6.5), (4, 5.5)),  # Meta-Learning -> Apply Pipeline
        ((5, 5), (6, 5)),      # Apply -> AutoGluon
        ((7, 4.5), (5.5, 3.5)), # AutoGluon -> Evaluation
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data", 
                              arrowstyle="->", shrinkA=5, shrinkB=5, 
                              mutation_scale=20, fc="black", lw=2)
        ax.add_patch(arrow)
    
    # Meta-learning detail box
    detail_box = FancyBboxPatch((0.5, 0.5), 4, 1.5, boxstyle="round,pad=0.1",
                                facecolor='lightyellow', edgecolor='orange', linewidth=2)
    ax.add_patch(detail_box)
    ax.text(2.5, 1.25, 'Meta-Learning Details:\n• 8 pipelines per generation\n• 5 generations\n• 4 generation strategies\n• Evolutionary optimization', 
            ha='center', va='center', fontsize=10)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def draw_generation_strategies(ax):
    """Draw the four generation strategies"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_title('Generation Strategies (Population Diversity)', fontsize=14, fontweight='bold')
    
    strategies = [
        ('Staged\n(40%)', 'Clean → Filter → Select → Augment', '#FFE6CC', (1, 4.5)),
        ('Random\n(30%)', 'Any operations in any order', '#E6F3FF', (3.5, 4.5)),
        ('Objective\n(20%)', 'Goal-driven combinations', '#E8F5E8', (6, 4.5)),
        ('Conditional\n(10%)', 'Adaptive based on data state', '#F0E6FF', (8.5, 4.5))
    ]
    
    for name, desc, color, pos in strategies:
        # Strategy box
        box = FancyBboxPatch((pos[0]-0.7, pos[1]-0.7), 1.4, 1.4, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(pos[0], pos[1], name, ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Description
        ax.text(pos[0], pos[1]-1.5, desc, ha='center', va='center', fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Examples
    examples = [
        'Clean(punct=F) → Filter(10-90%) → Select(hard) → Augment(mixed)',
        'QualityFilter → SynonymAug → DifficultySelect → Lowercase',
        'MaxAccuracy: QualityFilter + BalancedSelect + SynonymAug',
        'If(size>0.7): AddSelection; If(balance<0.7): AddAugment'
    ]
    
    for i, example in enumerate(examples):
        x_pos = 1 + i * 2.5
        ax.text(x_pos, 1.5, f'Example:\n{example}', ha='center', va='center', fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgray', alpha=0.7))
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def draw_evolution_process(ax):
    """Draw the evolution process"""
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 8)
    ax.set_title('Evolution Process', fontsize=14, fontweight='bold')
    
    # Generation cycle
    steps = [
        ('Generate\n8 Pipelines', (3, 7), '#FFE6CC'),
        ('Evaluate\nFitness', (3, 5.5), '#E8F5E8'),
        ('Select\nBest 4', (3, 4), '#E6F3FF'),
        ('Mutate &\nCrossover', (3, 2.5), '#F0E6FF'),
        ('Next\nGeneration', (3, 1), '#FFE6E6')
    ]
    
    for i, (name, pos, color) in enumerate(steps):
        box = FancyBboxPatch((pos[0]-0.8, pos[1]-0.4), 1.6, 0.8, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(pos[0], pos[1], name, ha='center', va='center', fontweight='bold', fontsize=9)
        
        # Arrow to next step
        if i < len(steps) - 1:
            arrow = ConnectionPatch((pos[0], pos[1]-0.4), (steps[i+1][1][0], steps[i+1][1][1]+0.4), 
                                  "data", "data", arrowstyle="->", shrinkA=5, shrinkB=5, 
                                  mutation_scale=15, fc="black", lw=1.5)
            ax.add_patch(arrow)
    
    # Cycle arrow
    cycle_arrow = ConnectionPatch((3.8, 1), (3.8, 6.6), "data", "data", 
                                arrowstyle="->", shrinkA=5, shrinkB=5, 
                                mutation_scale=15, fc="red", lw=2,
                                connectionstyle="arc3,rad=0.3")
    ax.add_patch(cycle_arrow)
    ax.text(5, 4, '5 Generations', ha='center', va='center', fontweight='bold', 
            color='red', rotation=90)
    
    # Fitness improvement
    ax.text(0.5, 6, 'Fitness\nImprovement', ha='center', va='center', fontweight='bold')
    ax.arrow(0.5, 5.5, 0, -4, head_width=0.1, head_length=0.2, fc='green', ec='green', lw=2)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def draw_example_sequences(ax):
    """Draw example discovered sequences"""
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.set_title('Example Discovered Sequences', fontsize=14, fontweight='bold')
    
    examples = [
        {
            'title': 'Staged Approach',
            'sequence': ['Clean(punct=F)', 'Filter(10-90%)', 'Select(hard)', 'Augment(mixed)'],
            'fitness': '0.742',
            'pos': (2, 3),
            'color': '#FFE6CC'
        },
        {
            'title': 'Random Discovery', 
            'sequence': ['QualityFilter', 'SynonymAug', 'DiffSelect', 'Lowercase'],
            'fitness': '0.758',
            'pos': (6, 3),
            'color': '#E6F3FF'
        },
        {
            'title': 'Objective-Driven',
            'sequence': ['MixedAug', 'HardSelect', 'InsertAug'],
            'fitness': '0.751',
            'pos': (10, 3),
            'color': '#E8F5E8'
        }
    ]
    
    for example in examples:
        # Main box
        box = FancyBboxPatch((example['pos'][0]-1.5, example['pos'][1]-1), 3, 2, 
                            boxstyle="round,pad=0.1", facecolor=example['color'], 
                            edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        
        # Title
        ax.text(example['pos'][0], example['pos'][1]+0.7, example['title'], 
                ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Sequence
        sequence_text = ' → '.join(example['sequence'])
        ax.text(example['pos'][0], example['pos'][1], sequence_text, 
                ha='center', va='center', fontsize=8, wrap=True)
        
        # Fitness
        ax.text(example['pos'][0], example['pos'][1]-0.7, f"Fitness: {example['fitness']}", 
                ha='center', va='center', fontweight='bold', color='red')
    
    # Best sequence indicator
    ax.text(6, 0.5, '🏆 Best Sequence Applied to Training Data', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='gold', alpha=0.8))
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

if __name__ == "__main__":
    create_meta_learning_flow_diagram()
