from common.model.general_cnn import GeneralCNN

class LungCancerCNN(GeneralCNN):
    def __init__(self) -> None:
        super().__init__(input_size=24, num_classes=5)
