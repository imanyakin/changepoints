import numpy as np 
import matplotlib.pyplot as plt 

'''Changepoint detection based on:

Optimal Detection of Changepoints With a Linear Computational Cost
R.Killick, P. Fearnhead & I.A. Eckley 

Cost functions taken from changepy that also implements PELT algorithm:

https://github.com/ruipgil/changepy/blob/master/changepy/pelt.py

'''

import numpy as np

def normal_mean(data, variance):
    """ Creates a segment cost function for a time series with a
        Normal distribution with changing mean

    Args:
        data (:obj:`list` of float): 1D time series data
        variance (float): variance
    Returns:
        function: Function with signature
            (int, int) -> float
            where the first arg is the starting index, and the second
            is the last arg. Returns the cost of that segment
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    i_variance_2 = 1 / (variance ** 2)
    cmm = [0.0]
    cmm.extend(np.cumsum(data))

    cmm2 = [0.0]
    cmm2.extend(np.cumsum(np.abs(data)))

    def cost(start, end):
        """ Cost function for normal distribution with variable mean

        Args:
            start (int): start index
            end (int): end index
        Returns:
            float: Cost, from start to end
        """
        cmm2_diff = cmm2[end] - cmm2[start]
        cmm_diff = pow(cmm[end] - cmm[start], 2)
        i_diff = end - start
        diff = cmm2_diff - cmm_diff
        return (diff/i_diff) * i_variance_2

    return cost

def normal_var(data, mean):
    """ Creates a segment cost function for a time series with a
        Normal distribution with changing variance

    Args:
        data (:obj:`list` of float): 1D time series data
        variance (float): variance
    Returns:
        function: Function with signature
            (int, int) -> float
            where the first arg is the starting index, and the second
            is the last arg. Returns the cost of that segment
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    cumm = [0.0]
    cumm.extend(np.cumsum(np.power(np.abs(data - mean), 2)))

    def cost(s, t):
        """ Cost function for normal distribution with variable variance

        Args:
            start (int): start index
            end (int): end index
        Returns:
            float: Cost, from start to end
        """
        dist = float(t - s)
        diff = cumm[t] - cumm[s]
        return dist * np.log(diff/dist)

    return cost

def normal_meanvar(data):
    """ Creates a segment cost function for a time series with a
        Normal distribution with changing mean and variance

    Args:
        data (:obj:`list` of float): 1D time series data
    Returns:
        function: Function with signature
            (int, int) -> float
            where the first arg is the starting index, and the second
            is the last arg. Returns the cost of that segment
    """
    data = np.hstack(([0.0], np.array(data)))

    cumm = np.cumsum(data)
    cumm_sq = np.cumsum([val**2 for val in data])

    def cost(s, t):
        """ Cost function for normal distribution with variable variance

        Args:
            start (int): start index
            end (int): end index
        Returns:
            float: Cost, from start to end
        """
        ts_i = 1.0 / (t-s)
        mu = (cumm[t] - cumm[s]) * ts_i
        sig = (cumm_sq[t] - cumm_sq[s]) * ts_i - mu**2
        sig_i = 1.0 / sig
        return (t-s) * np.log(sig) + (cumm_sq[t] - cumm_sq[s]) * sig_i - 2*(cumm[t] - cumm[s])*mu*sig_i + ((t-s)*mu**2)*sig_i

    return cost

def poisson(data):
    """ Creates a segment cost function for a time series with a
        poisson distribution with changing mean

    Args:
        data (:obj:`list` of float): 1D time series data
    Returns:
        function: Function with signature
            (int, int) -> float
            where the first arg is the starting index, and the second
            is the last arg. Returns the cost of that segment
    """
    data = np.hstack(([0.0], np.array(data)))
    cumm = np.cumsum(data)

    def cost(s, t):
        """ Cost function for poisson distribution with changing mean

        Args:
            start (int): start index
            end (int): end index
        Returns:
            float: Cost, from start to end
        """
        diff = cumm[t]-cumm[s]
        if diff == 0:
            return -2 * diff * (- np.log(t-s) - 1)
        else:
            return -2 * diff * (np.log(diff) - np.log(t-s) - 1)

    return cost

def exponential(data):
    """ Creates a segment cost function for a time series with a
        exponential distribution with changing mean

    Args:
        data (:obj:`list` of float): 1D time series data
    Returns:
        function: Function with signature
            (int, int) -> float
            where the first arg is the starting index, and the second
            is the last arg. Returns the cost of that segment
    """
    data = np.hstack(([0.0], np.array(data)))
    cumm = np.cumsum(data)

    def cost(s, t):
        """ Cost function for exponential distribution with changing mean

        Args:
            start (int): start index
            end (int): end index
        Returns:
            float: Cost, from start to end
        """
        return -1*(t-s) * (np.log(t-s) - np.log(cumm[t] - cumm[s]))

    return cost


def PELT(data,cost,beta=None):

	'''
	data - dataset to determine changepoints for
	C - measure of fit/cost function we want to minimize
	beta - penalty constant, independent of number/location of changepoints
	'''
	n = len(data)
	if beta is None:
		beta = np.log(n)

	F = np.zeros(n+1)
	F[0] = -beta
	cp = [[]]
	R = [0]
	for tau_star in range(2,n+1):
		#for each possible segment - compute its cost
		segment_cost = np.array([F[tau] + cost(tau,tau_star) for tau in R])

		#compute cost full path fron 0 to tau_star using the previous segment totals:
		penalized_cost = segment_cost + np.ones(segment_cost.shape)*beta 
		#compute cost of each new possible set of changepoints
		 
		#select smallest cost - make this the cost of using the current point tau_star
		#in the final changepoint segmentation 
		
		#save smallest cost of getting from 0 to tau_star
		F[tau_star]  = np.min(penalized_cost)

		#get previous changepoint position in this series of changepoints
		tau_prev = R[np.argmin(penalized_cost)]

		#append this path to the list of changepoint candidate paths
		cp.append(cp[tau_prev]+[tau_prev])
		R = [r for i,r in enumerate(R) if segment_cost[i] < F[tau_star]] + [tau_star-1]
		
	#prune all points which are near each other - a segment must be at least length 1 (ie. spacing = 2)
	cps = cp[-1]
	points = [cps[i] for i in range(len(cps)-1) if cps[i+1] - cps[i] > 1] + [cps[-1], n-1]
	return points
	


if __name__ == "__main__":
	mus = [15.0,8.0,1.0]
	sigma = [1.0,1.0,1.0]

	data = np.array([])
	for i in range(15):
		data = np.append(data,np.random.normal(mus[np.random.randint(10)%len(mus)],sigma[np.random.randint(10)%len(sigma)],100))

	
	ys = np.asarray(data) 
	cost = normal_mean(ys,sigma[0]**2)
	cps = PELT(data=ys,cost=cost)

	plt.plot(data,alpha=0.3)
	comp = [i for i in range(len(data)) if i not in cps]

	plt.plot(cps,data[cps],"o",markersize=5,label="Changepoint")
	
	plt.xlabel("Time")
	plt.ylabel("Signal")
	plt.legend()
	plt.show()


	