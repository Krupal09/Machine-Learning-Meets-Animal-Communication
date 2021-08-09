class EarlyStoppingCriterion(object):
    """
    Early stopping criterion as a regularization in order to stop the training after no changes with respect to a chosen
    validation metric to avoid overfitting.

    Code from https://github.com/pytorch/pytorch/pull/7661
    Access Data: 12.09.2018, Last Access Date: 08.12.2019
    """

    def __init__(self, patience, mode, min_delta=0.0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving; cumulative number of times the validation loss does not improve during the whole training process
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        :param mode: whether the objective of the chosen metric is to increase (maximize or ‘max‘)
               or to decrease (minimize or ‘min‘).
        """
        assert patience >= 0
        assert mode in {"min", "max"}
        assert min_delta >= 0.0
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self._count = 0 # keeps count of the number of times the current validation loss does not improve
        self._best_score = None
        self.is_improved = None

    def step(self, cur_score):
        if self._best_score is None:
            self._best_score = cur_score # At the beginning of training, we make the current loss as the best loss
            return False
        else:
            if self.mode == "max":
                self.is_improved = cur_score >= self._best_score + self.min_delta
            else:
                self.is_improved = cur_score <= self._best_score - self.min_delta

            if self.is_improved:
                self._count = 0
                self._best_score = cur_score
            else:
                self._count += 1
            return self._count > self.patience

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
