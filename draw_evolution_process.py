#!/usr/bin/env python3
"""
Evolution Process Visualization for Meta-Learning Framework

This script creates detailed visualizations of:
1. Generation-by-generation evolution
2. Mutation and crossover operations
3. Fitness improvement over time
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

def create_evolution_visualization():
    """Create comprehensive evolution process visualization"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 14))
    
    # Evolution timeline
    ax_timeline = plt.subplot2grid((3, 3), (0, 0), colspan=3)
    
    # Mutation examples
    ax_mutation = plt.subplot2grid((3, 3), (1, 0))
    
    # Crossover examples  
    ax_crossover = plt.subplot2grid((3, 3), (1, 1))
    
    # Fitness progression
    ax_fitness = plt.subplot2grid((3, 3), (1, 2))
    
    # Population diversity
    ax_diversity = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    
    # Draw each component
    draw_evolution_timeline(ax_timeline)
    draw_mutation_examples(ax_mutation)
    draw_crossover_examples(ax_crossover)
    draw_fitness_progression(ax_fitness)
    draw_population_diversity(ax_diversity)
    
    plt.tight_layout()
    plt.savefig('meta_learning_evolution_process.png', dpi=300, bbox_inches='tight')
    plt.savefig('meta_learning_evolution_process.pdf', bbox_inches='tight')
    print("Evolution process diagram saved as 'meta_learning_evolution_process.png' and '.pdf'")
    plt.show()

def draw_evolution_timeline(ax):
    """Draw the evolution timeline across generations"""
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.set_title('Evolution Timeline: 5 Generations of Pipeline Discovery', fontsize=16, fontweight='bold')
    
    generations = [
        {
            'gen': 1,
            'x': 2,
            'population': ['Staged', 'Random', 'Objective', 'Conditional', 'Staged', 'Random', 'Objective', 'Random'],
            'best_fitness': 0.65,
            'color': '#FFE6CC'
        },
        {
            'gen': 2,
            'x': 5,
            'population': ['Best4', 'Mutated', 'Mutated', 'Mutated', 'Mutated', 'Crossover', 'Crossover', 'New'],
            'best_fitness': 0.71,
            'color': '#E6F3FF'
        },
        {
            'gen': 3,
            'x': 8,
            'population': ['Best4', 'Mutated', 'Mutated', 'Mutated', 'Mutated', 'Crossover', 'Crossover', 'New'],
            'best_fitness': 0.74,
            'color': '#E8F5E8'
        },
        {
            'gen': 4,
            'x': 11,
            'population': ['Best4', 'Mutated', 'Mutated', 'Mutated', 'Mutated', 'Crossover', 'Crossover', 'New'],
            'best_fitness': 0.76,
            'color': '#F0E6FF'
        },
        {
            'gen': 5,
            'x': 14,
            'population': ['Best4', 'Mutated', 'Mutated', 'Mutated', 'Mutated', 'Crossover', 'Crossover', 'New'],
            'best_fitness': 0.78,
            'color': '#FFE6E6'
        }
    ]
    
    for gen_data in generations:
        # Generation header
        header_box = FancyBboxPatch((gen_data['x']-1, 7), 2, 0.8, boxstyle="round,pad=0.1",
                                   facecolor=gen_data['color'], edgecolor='black', linewidth=2)
        ax.add_patch(header_box)
        ax.text(gen_data['x'], 7.4, f"Gen {gen_data['gen']}", ha='center', va='center', 
                fontweight='bold', fontsize=12)
        
        # Population
        for i, pipeline_type in enumerate(gen_data['population']):
            y_pos = 6 - (i * 0.6)
            
            # Color code by type
            if 'Best' in pipeline_type:
                color = 'gold'
            elif 'Mutated' in pipeline_type:
                color = 'lightblue'
            elif 'Crossover' in pipeline_type:
                color = 'lightgreen'
            else:
                color = 'lightgray'
            
            pipeline_box = FancyBboxPatch((gen_data['x']-0.9, y_pos-0.15), 1.8, 0.3,
                                         boxstyle="round,pad=0.05", facecolor=color,
                                         edgecolor='black', linewidth=1)
            ax.add_patch(pipeline_box)
            ax.text(gen_data['x'], y_pos, pipeline_type, ha='center', va='center', fontsize=8)
        
        # Best fitness
        ax.text(gen_data['x'], 0.5, f"Best: {gen_data['best_fitness']}", ha='center', va='center',
                fontweight='bold', fontsize=10, color='red')
        
        # Arrow to next generation
        if gen_data['gen'] < 5:
            ax.arrow(gen_data['x']+1, 4, 1, 0, head_width=0.2, head_length=0.2,
                    fc='black', ec='black', lw=2)
    
    # Legend
    legend_elements = [
        ('Best4: Elite selection', 'gold'),
        ('Mutated: Modified parameters', 'lightblue'),
        ('Crossover: Combined pipelines', 'lightgreen'),
        ('New: Fresh random pipeline', 'lightgray')
    ]
    
    for i, (desc, color) in enumerate(legend_elements):
        ax.text(1, 3.5 - i*0.4, desc, ha='left', va='center', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7))
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def draw_mutation_examples(ax):
    """Draw mutation operation examples"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title('Mutation Examples', fontsize=12, fontweight='bold')
    
    # Original pipeline
    ax.text(5, 9, 'Original Pipeline', ha='center', va='center', fontweight='bold', fontsize=11)
    original = ['quality_filter(20%)', 'hard_select(0.7)', 'mixed_augment(0.15)']
    
    for i, op in enumerate(original):
        box = FancyBboxPatch((1, 8-i*0.8), 8, 0.6, boxstyle="round,pad=0.1",
                            facecolor='lightblue', edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(5, 8.3-i*0.8, op, ha='center', va='center', fontsize=9)
    
    # Mutation arrow
    ax.arrow(5, 5.5, 0, -1, head_width=0.3, head_length=0.2, fc='red', ec='red', lw=2)
    ax.text(6, 5, 'MUTATE', ha='left', va='center', fontweight='bold', color='red')
    
    # Mutated pipeline
    ax.text(5, 4, 'Mutated Pipeline', ha='center', va='center', fontweight='bold', fontsize=11)
    mutated = ['quality_filter(30%)', 'hard_select(0.7)', 'mixed_augment(0.15)', 'lowercase(True)']
    
    for i, op in enumerate(mutated):
        color = 'lightcoral' if i == 0 or i == 3 else 'lightblue'  # Highlight changes
        box = FancyBboxPatch((0.5, 3-i*0.6), 9, 0.5, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(5, 3.25-i*0.6, op, ha='center', va='center', fontsize=8)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def draw_crossover_examples(ax):
    """Draw crossover operation examples"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title('Crossover Examples', fontsize=12, fontweight='bold')
    
    # Parent 1
    ax.text(2.5, 9.5, 'Parent 1', ha='center', va='center', fontweight='bold')
    parent1 = ['quality_filter', 'hard_select', 'mixed_augment']
    for i, op in enumerate(parent1):
        box = FancyBboxPatch((0.5, 8.5-i*0.6), 4, 0.5, boxstyle="round,pad=0.1",
                            facecolor='lightblue', edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(2.5, 8.75-i*0.6, op, ha='center', va='center', fontsize=8)
    
    # Parent 2
    ax.text(7.5, 9.5, 'Parent 2', ha='center', va='center', fontweight='bold')
    parent2 = ['lowercase', 'length_filter', 'synonym_augment']
    for i, op in enumerate(parent2):
        box = FancyBboxPatch((5.5, 8.5-i*0.6), 4, 0.5, boxstyle="round,pad=0.1",
                            facecolor='lightgreen', edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(7.5, 8.75-i*0.6, op, ha='center', va='center', fontsize=8)
    
    # Crossover arrows
    ax.arrow(2.5, 6.5, 2, -1.5, head_width=0.2, head_length=0.1, fc='purple', ec='purple', lw=2)
    ax.arrow(7.5, 6.5, -2, -1.5, head_width=0.2, head_length=0.1, fc='purple', ec='purple', lw=2)
    ax.text(5, 6, 'CROSSOVER', ha='center', va='center', fontweight='bold', color='purple')
    
    # Child
    ax.text(5, 4.5, 'Child Pipeline', ha='center', va='center', fontweight='bold')
    child = ['quality_filter', 'length_filter', 'mixed_augment', 'lowercase']
    for i, op in enumerate(child):
        # Color based on parent
        if op in parent1:
            color = 'lightblue'
        elif op in parent2:
            color = 'lightgreen'
        else:
            color = 'lightyellow'
        
        box = FancyBboxPatch((1, 3.5-i*0.6), 8, 0.5, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(5, 3.75-i*0.6, op, ha='center', va='center', fontsize=8)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def draw_fitness_progression(ax):
    """Draw fitness improvement over generations"""
    ax.set_xlim(0, 6)
    ax.set_ylim(0.6, 0.8)
    ax.set_title('Fitness Progression', fontsize=12, fontweight='bold')
    
    # Example fitness progression
    generations = [1, 2, 3, 4, 5]
    best_fitness = [0.65, 0.71, 0.74, 0.76, 0.78]
    avg_fitness = [0.62, 0.67, 0.70, 0.72, 0.74]
    
    # Plot lines
    ax.plot(generations, best_fitness, 'ro-', linewidth=3, markersize=8, label='Best Fitness')
    ax.plot(generations, avg_fitness, 'bo-', linewidth=2, markersize=6, label='Average Fitness')
    
    # Fill area between
    ax.fill_between(generations, best_fitness, avg_fitness, alpha=0.3, color='green')
    
    # Annotations
    for i, (gen, best, avg) in enumerate(zip(generations, best_fitness, avg_fitness)):
        ax.annotate(f'{best:.3f}', (gen, best), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontweight='bold', color='red')
        ax.annotate(f'{avg:.3f}', (gen, avg), textcoords="offset points", 
                   xytext=(0,-15), ha='center', fontweight='bold', color='blue')
    
    ax.set_xlabel('Generation', fontweight='bold')
    ax.set_ylabel('Fitness Score', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

def draw_population_diversity(ax):
    """Draw population diversity across approaches"""
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.set_title('Population Diversity: Multiple Approaches in Each Generation', fontsize=14, fontweight='bold')
    
    # Show distribution of approaches
    approaches = ['Staged (40%)', 'Random (30%)', 'Objective (20%)', 'Conditional (10%)']
    colors = ['#FFE6CC', '#E6F3FF', '#E8F5E8', '#F0E6FF']
    
    # Population distribution
    for i, (approach, color) in enumerate(zip(approaches, colors)):
        x_start = i * 3
        
        # Approach header
        header_box = FancyBboxPatch((x_start, 5), 2.5, 0.8, boxstyle="round,pad=0.1",
                                   facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(header_box)
        ax.text(x_start + 1.25, 5.4, approach, ha='center', va='center', fontweight='bold')
        
        # Example pipelines for this approach
        if i == 0:  # Staged
            examples = ['Clean→Filter→Select', 'Clean→Filter→Augment', 'Clean→Select→Augment']
        elif i == 1:  # Random
            examples = ['Quality→Synonym→Hard', 'Lower→Length→Mixed', 'Random→Insert→Easy']
        elif i == 2:  # Objective
            examples = ['MaxAccuracy combo', 'MaxRobustness combo', 'MinSize combo']
        else:  # Conditional
            examples = ['Adaptive sequence', 'State-driven ops', 'Context-aware']
        
        for j, example in enumerate(examples):
            y_pos = 4 - j * 0.8
            example_box = FancyBboxPatch((x_start + 0.1, y_pos - 0.3), 2.3, 0.6,
                                        boxstyle="round,pad=0.05", facecolor='white',
                                        edgecolor='gray', linewidth=1)
            ax.add_patch(example_box)
            ax.text(x_start + 1.25, y_pos, example, ha='center', va='center', fontsize=8)
    
    # Diversity benefit
    ax.text(6, 0.5, '🎯 Diversity Benefits:\n• Explores different strategies\n• Avoids local optima\n• Discovers unexpected solutions\n• Balances exploration vs exploitation', 
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.9))
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

if __name__ == "__main__":
    create_evolution_visualization()
