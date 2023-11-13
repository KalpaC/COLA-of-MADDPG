from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .cola_learner import COLALearner
from .q_learner_for_v2x import QLearner as QLearner_v2x

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["cola_learner"] = COLALearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["q_learner_for_v2x"] = QLearner_v2x
