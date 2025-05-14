from enum import Enum


class AggregationMethod(Enum):
    FED_AVG = "fed_avg"
    FED_SGD = "fed_sgd"
    FED_PROX = "fed_prox"
    FED_ADAM = "fed_adam"
    FED_ADAGRAD = "fed_adagrad"
    FED_YOGI = "fed_yogi"
