import numpy as np

y = np.array([0, 2, 1, 1, 0])
score = np.random.random((5, 3))
print(score)
N = score.shape[0]
score -= score[range(y.shape[0]), y].repeat(score.shape[1]).reshape(score.shape)
score = (score > 0) * score
print(score)
print(score[range(score.shape[0]),:].sum())