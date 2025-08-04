import subprocess
import logging
import sys
import os
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='autogluon_before_only.log',
    filemode='w'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
logger = logging.getLogger(__name__)

def run_experiment(dataset):
    """Run fast_fixed_meta_learning.py on a specific dataset"""
    logger.info(f"="*80)
    logger.info(f"Starting experiment for {dataset}")
    logger.info(f"="*80)
    
    start_time = time.time()
    
    try:
        # Run the script with the dataset parameter
        cmd = [sys.executable, "fast_fixed_meta_learning.py", dataset]
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run process and capture output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Stream output to log
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logger.info(output.strip())
                
        # Get any errors
        stderr = process.communicate()[1]
        if stderr:
            logger.error(f"Errors: {stderr}")
            
        if process.returncode != 0:
            logger.error(f"Process exited with code {process.returncode}")
        else:
            logger.info(f"Process completed successfully")
            
    except Exception as e:
        logger.error(f"Failed to run experiment: {e}")
        
    duration = time.time() - start_time
    logger.info(f"Experiment for {dataset} completed in {duration/60:.2f} minutes")

def main():
    """Run experiments for all datasets"""
    logger.info(f"Starting experiments at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # List of datasets to process
    datasets = ['anli_r1_noisy', 'casehold_imbalanced']
    
    for dataset in datasets:
        run_experiment(dataset)
        
    logger.info(f"All experiments completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
