import numpy as np

def viterbi(problem):
  e_mat = problem['emission_mat']
  t_mat = problem['transition_mat']
  features = problem['features']

  tokens = problem['txt'].split(" ") + problem['features'][-1:]
  costs = np.zeros((len(features), 1+len(tokens)))
  label = []
  for i in range(1, len(tokens)+1):
    idx = problem['codes'][i-1]
    prev_cost = costs[:, i-1]
    emit_cost = e_mat[:, idx]

    if i == 1:
      acc_cost = (t_mat + emit_cost)[0, :]
    elif i == len(tokens):
      acc_cost = costs[:, i-1] + t_mat[:, -1]
      idx = np.argmax(acc_cost)
      label.append(features[idx])

    else:
      acc_cost = (costs[:, i-1] + t_mat.T) + emit_cost.reshape(-1, 1)

      acc_cost = np.max(acc_cost, axis=1)
      max_global_idx = np.argmax(acc_cost)
      idx = np.argmax(t_mat[:, max_global_idx] + costs[:, i-1])
      label.append(features[idx])

    costs[:, i] = acc_cost

  return costs, label


problems = [
    dict(
        # from https://youtu.be/AGl1a1NzY-c?t=588
        txt="they can fish",
        codes=[0, 1, 2, 3],
        features = ['START', 'N', 'V', 'END'],
        emission_mat = np.array([
            [-100, -100, -100, 0], # <START> -> W
            [-2, -3, -3, 0], # N -> W
            [-10, -1, -3, 0], # V -> W
            [-100, -100, -100, 0] # END -> W
        ]),
        transition_mat = np.array([
            [-1000, -1, -2, -1000],
            [-1000, -3, -1, -2],
            [-1000, -1, -3, -2],
            [-1000, -1000, -1000, -1000],
        ]),
        optimal_cost=-11,
        labels="N-V-N"
    ),
    dict(
        # from https://youtu.be/zoXe0DFduNA?t=518
        txt = "they can fish",
        codes = [0, 1, 2, 3],
        features = ['START', 'N', 'V', 'M', 'END'],
        emission_mat = np.array([
        [-100, -100, -100, 0], # <START> -> W
        [-2,     -3,   -3, 0], # N -> W
        [-10,    -2,   -3, 0], # V -> W
        [-10,    -1,  -10, 0], # M -> W
        [-100, -100, -100, 0] # END -> W
        ]),
        transition_mat = np.array([
        [-1000, -1,    -2,     -3, -1000],
        [-1000, -3,    -1,     -2, -2],
        [-1000, -1,    -2,     -5, -2],
        [-1000, -10,    0,     -3, -10],
        [-1000, -1000, -1000, -1000, -1000],
        ]),
        optimal_cost=-11,
        labels="N-M-V"
    )
]

def test_problems():
    for i, p in enumerate(problems):
        print("problem %d" % (i + 1))
        costs, label = viterbi(p)

        c = float(np.max(costs[:, -1]))
        assert p["optimal_cost"] == c
        assert p["labels"] == '-'.join(label)
