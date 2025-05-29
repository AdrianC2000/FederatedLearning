from common.fml_utils import DataSplitStrategy
from experiments.dataset_runners.run_brain_tumor_experiments import run_experiments_for_brain_tumor_dataset
from experiments.dataset_runners.run_chest_xray_experiments import run_experiments_for_chest_xray_dataset
from experiments.dataset_runners.run_lung_cancer_experiments import run_experiments_for_lung_cancer_dataset

if __name__ == "__main__":
    print("DataSplitStrategy: STRATIFIED_EQUAL")
    run_experiments_for_brain_tumor_dataset(DataSplitStrategy.STRATIFIED_EQUAL)
    print(f"\n--- Brain Tumor dataset processing finished ---")
    run_experiments_for_chest_xray_dataset(DataSplitStrategy.STRATIFIED_EQUAL)
    print(f"\n--- Chest Xray dataset processing finished ---")
    run_experiments_for_lung_cancer_dataset(DataSplitStrategy.STRATIFIED_EQUAL)
    print(f"\n--- Lung cancer dataset processing finished ---")

    print("DataSplitStrategy: STRATIFIED_IMBALANCED")
    run_experiments_for_brain_tumor_dataset(DataSplitStrategy.STRATIFIED_IMBALANCED)
    print(f"\n--- Brain Tumor dataset processing finished ---")
    run_experiments_for_chest_xray_dataset(DataSplitStrategy.STRATIFIED_IMBALANCED)
    print(f"\n--- Chest Xray dataset processing finished ---")
    run_experiments_for_lung_cancer_dataset(DataSplitStrategy.STRATIFIED_IMBALANCED)
    print(f"\n--- Lung cancer dataset processing finished ---")

    print("DataSplitStrategy: NON_IID_EQUAL")
    run_experiments_for_brain_tumor_dataset(DataSplitStrategy.NON_IID_EQUAL)
    print(f"\n--- Brain Tumor dataset processing finished ---")
    run_experiments_for_chest_xray_dataset(DataSplitStrategy.NON_IID_EQUAL)
    print(f"\n--- Chest Xray dataset processing finished ---")
    run_experiments_for_lung_cancer_dataset(DataSplitStrategy.NON_IID_EQUAL)
    print(f"\n--- Lung cancer dataset processing finished ---")
