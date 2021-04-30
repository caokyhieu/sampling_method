import numpy as np
import scipy.stats as stats
import pandas as pd
from utils import plot_hist
from abc import ABC,abstractmethod
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class ABCSampling(ABC):
    def __init__(self,data,threshold):
        
        self.threshold = threshold
        self.data = data
        self.accept = 0
        pass
    
    @abstractmethod
    def set_prior(self):
        """
        This method helps to set prior for object.
        This prior can sample proposal theta by sample method
        """

    @abstractmethod
    def generate_replica(self):
        """
        This method helps to generate replica data from proposal p(theta) from prior
        This method return replicated data
        parameters: proposal theta
        """ 

    @abstractmethod
    def distance(self,data,replica):
        """
        This method helps to calculate statistics and compare the distance between replica data and original data.
        input of this method is data and replica data
        output is the distance
        First try with Wassenstein Distance
        """
    
    def sample(self,n_samples,progress_bar=False):
        _range = tqdm(range(n_samples)) if progress_bar else range(n_samples)

        samples = []
        for i in _range:
            proposal = self.prior.sample(n_samples=None)
            replica = self.generate_replica(proposal)
            ### check proposal valid or not
            if self.distance(self.data,replica) < self.threshold:
                samples.append(proposal)
                self.accept +=1
                if progress_bar:
                    _range.set_description(f'Aceept rate: {self.accept/(i+1):.2f}')
        
        if progress_bar:
            _range.close()
    
        return samples        

class WassersteinABC(ABCSampling):

    def __init__(self,data,threshold,n_dim = 4):
        super().__init__(data,threshold)
        self.n_dim = n_dim
        
        self.set_prior()
        self.data_dim = np.array(data).shape[-1]
    
    def set_prior(self):
        self.prior = type('Prior', (object,), {'sample' : lambda n_samples: np.random.multivariate_normal(mean=np.array([0 for i in range(self.data_dim)]),cov=np.eye(self.data_dim),size=n_samples)})

    def generate_replica(self,theta):
        # thetas = self.prior.sample(n_samples=len(self.data))
        return np.array([np.random.normal(loc=theta,scale=2.0) for i in range(len(self.data))])

    def distance(self,data,replica):
        """
        Just use for one dimension data
        Wasserstein Distance very effective for capture distribution
        Improve by slice wasserstein distance: very impressive
        * Next move for multivariate data (2 dims)
        """
        weight = np.random.multivariate_normal(mean=np.random.normal(size=self.n_dim),cov = np.eye(self.n_dim),size=self.data_dim)
        weight = weight /np.sqrt(np.sum(weight**2,axis=0,keepdims=True))
        data = np.matmul(data,weight)
        replica = np.matmul(replica,weight)
      
        result = [stats.wasserstein_distance(data[:,i],replica[:,i]) for i in range(len(weight))]

        return np.mean(result)
        # return np.abs(np.mean(data) - np.mean(replica)) + np.abs(np.std(data) - np.std(replica))

if __name__ == '__main__':
    ### prior ~ N(0,1)
    data = np.array([np.random.normal(loc=theta,scale= 2.0) for theta in np.random.multivariate_normal(mean=[0,0],cov=[[1,0],[0,1]],size=1000)])
    ##-> posterior has mean = mean(data) * n /(sigma^2 + n)
    # print(f'Shape data: {np.array(data).shape}')
    threshold = 1.4
    n_dim = 10
    posterior = WassersteinABC(data,threshold,n_dim=n_dim)
    samples = posterior.sample(n_samples=10000,progress_bar=True)
    samples = pd.DataFrame(samples,columns=['x1','x2'])
    # print(np.array(samples).shape)
    # print(samples)
    sns_plot = sns.jointplot(data=samples, x="x1", y="x2")
    mean = np.mean(data,axis=0) *len(data)*(1/4 )/(1/4*len(data)+1)
    ax = sns_plot.ax_joint
    ax.axvline(x=mean[0],color="red")
    ax.axhline(y=mean[1],color="blue")
    sns_plot.savefig('ABC.png')
    # print(f'Accepted rate: {len(samples)/100000 * 100:.2f}%')
    # plt.hist(np.array(samples).squeeze(),bins=20)
    # plt.axvline(x=np.mean(samples),color="red")
    # plt.axvline(x=np.mean(data)*len(data)/(4+len(data)),color="purple")
    # plt.savefig('ABC.png')




            

            


        
    
