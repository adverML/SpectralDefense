#!/usr/bin/env python3

from sklearn.svm import SVC 
from cuml.svm import SVC as SVC_gpu



clf_svc = SVC(kernel='poly', degree=2, gamma='auto', C=1)
sklearn_time_svc = %timeit -o train_data(clf_svc)

clf_svc = SVC_gpu(kernel='poly', degree=2, gamma='auto', C=1)
cuml_time_svc = %timeit -o train_data(clf_svc)

print(f"""Average time of sklearn's {clf_svc.__class__.__name__}""", sklearn_time_svc.average, 's')
print(f"""Average time of cuml's {clf_svc.__class__.__name__}""", cuml_time_svc.average, 's')

print('Ratio between sklearn and cuml is', sklearn_time_svc.average/cuml_time_svc.average)