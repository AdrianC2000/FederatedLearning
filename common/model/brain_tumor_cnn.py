from common.model.general_cnn import GeneralCNN

class BrainTumorCNN(GeneralCNN):
    def __init__(self) -> None:
        super().__init__(input_size=32, num_classes=4)
