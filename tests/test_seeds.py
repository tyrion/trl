from trl import experiment

import numpy as np


class DummyExperiment(Experiment):

    def get_dataset(self):
        pass

    def get_q(self):
        np.random()


    def get_algorithm(self):
        pass

    def evaluate(self):
        pass


