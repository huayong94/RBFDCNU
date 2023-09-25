from .BendingEnergy import (BendingEnergyMetric, RBFBendingEnergyLoss,
                            RBFBendingEnergyLossA)
from .CrossCorrelation import (LocalCrossCorrelation2D,
                               WeightedLocalCrossCorrelation2D)
from .DiceCoefficient import DiceCoefficient, DiceCoefficientAll
from .Distance import MaxMinPointDist, SurfaceDistanceFromSeg
from .Jacobian import JacobianDeterminantLoss, JacobianDeterminantMetric
from .MeanSquareError import MeanSquareError

LOSSDICT = {
    'LCC': LocalCrossCorrelation2D,
    'WLCC': WeightedLocalCrossCorrelation2D,
    'MSE': MeanSquareError
}
