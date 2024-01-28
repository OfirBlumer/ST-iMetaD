import numpy as np
import pandas as pd
from scipy import stats
from scipy import optimize

class STiMetaD:

    def __init__(self, minSampleSize = 5):
        """
        Initate the STiMetaD class, used to estimate the unbiased 
        kinetics from enhanced molecular dynamics simulations.

        minSampleSize: The minimum number of sampled first-passage times
                       to use for kinetics inference. (int) 
        """
        
        self.minSampleSize = minSampleSize
    
    def _comulative(self, t, a):
        return 1-np.exp(-t / a)
    
    def obtainEstimationsDataFrame(self, samples, minSampleSize = None):
        """
        Estimates the kinetic rate for different choices of Tstar.

        samples: Samples of rescaled first-passage times. (numpy array)
        minSampleSize: The minimum number of sampled first-passage times
                       to use for kinetics inference. (int) 
        """

        samples.sort()
        minSampleSize = self.minSampleSize if minSampleSize is None else minSampleSize
        survival = np.array([(len(samples) - i) / len(samples) for i in range(len(samples))])
        predictions = []
        R2s = []

        for limit in range(minSampleSize, len(samples)):
            k = -sum(samples[:limit] * np.log(survival[:limit])) / sum(samples[:limit] ** 2)
            predictions.append(k)
            R2s.append(1 - sum((np.log(survival[:limit]) + k * samples[:limit]) ** 2) / sum((np.log(survival[:limit]) - np.log(survival[:limit]).mean()) ** 2))

        return pd.DataFrame({"time": samples[minSampleSize:], "prediction": predictions,"R2": R2s})

    def estimateMFPT(self, samples, minSampleSize = None):
        """
        Estimates the mean first-passage time.

        samples: Samples of rescaled first-passage times. (numpy array)
        minSampleSize: The minimum number of sampled first-passage times
                       to use for kinetics inference. (int) 
        """

        data = self.obtainEstimationsDataFrame(samples = samples, minSampleSize = minSampleSize)
        return float(1 / data.loc[data.R2 == data.R2.max()].prediction)
       
    def estimateRate(self, samples, minSampleSize = None):
        """
        Estimates the kinetic rate.

        samples: Samples of rescaled first-passage times. (numpy array)
        minSampleSize: The minimum number of sampled first-passage times
                       to use for kinetics inference. (int) 
        """

        data = self.obtainEstimationsDataFrame(samples = samples, minSampleSize = minSampleSize)
        return float(data.loc[data.R2 == data.R2.max()].prediction)
    
    def estimateTstar(self, samples, minSampleSize = None):
        """
        Estimates Tstar.

        samples: Samples of rescaled first-passage times. (numpy array)
        minSampleSize: The minimum number of sampled first-passage times
                       to use for kinetics inference. (int) 
        """

        data = self.obtainEstimationsDataFrame(samples = samples, minSampleSize = minSampleSize)
        return float(data.loc[data.R2 == data.R2.max()].time)
    
    def iMetaDMFPT(self, samples, KStest = False, fitSamples = 1000000):
        """
        Estimates the kinetic rate through standard iMetaD.

        samples: Samples of rescaled first-passage times. (numpy array)
        KStest: Whether to provide the p-value of a KS test. (boolian)
        fitSamples: The number of samples from the exponential fit, for the KS test. (int)
        """

        CDF = np.array([i / len(samples) for i in range(1,len(samples)+1)])
        fit = optimize.curve_fit(self._comulative, samples, CDF, p0 = (samples.mean()))[0]

        if KStest:
            fSamples = np.random.exponential(fit[0],size=fitSamples)
            pvalue = stats.kstest(samples,fSamples)[1]
            ret = fit[0], pvalue
        else:
            ret = fit[0]

        return ret
    
    def iMetaDrate(self, samples, KStest = False, fitSamples = 1000000):
        """
        Estimates the kinetic rate through standard iMetaD.

        samples: Samples of rescaled first-passage times. (numpy array)
        KStest: Whether to provide the p-value of a KS test. (boolian)
        fitSamples: The number of samples from the exponential fit, for the KS test. (int)
        """

        CDF = np.array([i / len(samples) for i in range(1,len(samples)+1)])
        fit = optimize.curve_fit(self._comulative, samples, CDF, p0 = (samples.mean()))[0]

        if KStest:
            fSamples = np.random.exponential(fit[0],size=fitSamples)
            pvalue = stats.kstest(samples,fSamples)[1]
            ret = 1 / fit[0], pvalue
        else:
            ret = 1 / fit[0]

        return ret
