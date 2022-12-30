import os
import heapq
import pickle
import torch


class CheckpointManager():
    def __init__(self, path_to_checkpoints, score_heap_filename='score_heap.pkl', n_best=-1):
        self.path_to_checkpoints = path_to_checkpoints
        self.score_heap_filename = score_heap_filename
        self.n_best = n_best
        self.score_heap = []
        self.score_heap_filepath = os.path.join(self.path_to_checkpoints, score_heap_filename)

    def add_checkpoint(self, state, score, checkpoint_name):
        heapq.heappush(self.score_heap, (score, checkpoint_name))
        torch.save(state, os.path.join(self.path_to_checkpoints, checkpoint_name))
        if self.n_best != -1 and len(self.score_heap) > self.n_best:
            _, worst_checkpoint_name = heapq.heappop(self.score_heap)
            os.remove(os.path.join(self.path_to_checkpoints, worst_checkpoint_name))
        with open(self.score_heap_filepath, 'wb') as handle:
            pickle.dump(list(self.score_heap), handle, protocol=pickle.HIGHEST_PROTOCOL)
        return