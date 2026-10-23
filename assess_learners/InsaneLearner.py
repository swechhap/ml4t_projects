import numpy as np
import LinRegLearner as lrl
import BagLearner as bl


def author(self):
    return "SPANDEY343"

def study_group(self):
    return "SPANDEY343"

class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.learners = [bl.BagLearner(lrl.LinRegLearner, {}, bags=20, boost=False, verbose=verbose) for _ in range(20)]
    def author(self):
        return "SPANDEY343"
    def add_evidence(self, X, Y):
        for learner in self.learners: learner.add_evidence(X, Y)
    def query(self, Xtest):
        return np.mean([learner.query(Xtest) for learner in self.learners], axis=0)