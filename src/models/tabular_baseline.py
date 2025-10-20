from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

def make_tabular_classifier():
    base = GradientBoostingClassifier(random_state=7)
    clf = CalibratedClassifierCV(base, method='isotonic')
    return clf
