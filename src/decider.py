import numpy as np
from scipy.optimize import minimize


class Decider():
    def __init__(self, QP: int = 3, r: float =0.85):
        delta = 2**((QP - 4)/6)
        self.lagrange_mult = r * delta

    def __call__(self,q_step: int, *args):
        self.q_step = q_step
        Coeffs_list = list(args)
        self._RDO(Coeffs_list)

    def _quantize(self, Coeffs):
        Coeffs_quant = np.round(Coeffs/self.q_step)
        return Coeffs_quant

    def _qError(self, Coeffs: np.ndarray, Coeffs_quant: np.ndarray):
        N = Coeffs_quant[:,0].shape[0]
        Coeff_dequant = Coeffs_quant*self.q_step
        norm_value = np.linalg.norm(Coeffs[:,0] - Coeff_dequant[:,0])
        psnr_Y = -10 * np.log10((norm_value ** 2) / (N * 255 ** 2))
        return psnr_Y
    
    def _RDcost(self, Coeffs: np.ndarray):
        """
        Rate-Distorsion cost
        """
        Coeffs_quant = self._quantize(Coeffs)
        qerror = self._qError(Coeffs, Coeffs_quant)
        sparsity = self._zeroNorm(Coeffs_quant)
        return qerror + self.lagrange_mult*sparsity 
    
    def _RDO(self, Coeffs_list: list[np.ndarray]):
        res = minimize(self._RDcost, Coeffs_list, method='Nelder-Mead', tol=1e-6)
        return res

    def _zeroNorm(self, Coeffs_quant: np.ndarray):
        zeroNorm = np.linalg.norm(Coeffs_quant, 0)
        return zeroNorm