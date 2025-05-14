from experiments.dataset_runners.run_brain_tumor_experiments import run_experiments_for_brain_tumor_dataset
from experiments.dataset_runners.run_chest_xray_experiments import run_experiments_for_chest_xray_dataset
from experiments.dataset_runners.run_lung_cancer_experiments import run_experiments_for_lung_cancer_dataset

if __name__ == "__main__":
    run_experiments_for_brain_tumor_dataset()
    print(f"\n--- Brain Tumor dataset processing finished ---")
    run_experiments_for_chest_xray_dataset()
    print(f"\n--- Chest Xray dataset processing finished ---")
    run_experiments_for_lung_cancer_dataset()
    print(f"\n--- Lung cancer dataset processing finished ---")
