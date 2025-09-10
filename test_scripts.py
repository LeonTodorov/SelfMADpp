print("___AUGMENTATIONS___")
from data.augmentation_pipeline import test
test()

print("___EVALUATION DATSETS___")
from data.dataset import test
test()

print("___DETECTOR___")
from model.detector import test
test()

print("___NETS___")
from model.nets import test
test()