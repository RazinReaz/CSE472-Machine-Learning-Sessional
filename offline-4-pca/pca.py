import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from tqdm import tqdm
import os


class PCA():
    def __init__(self, n_components):
        self.n_components = n_components
        self.principal_components = None
        self.data = None

    def __standardize(self, data):
        """
        Standardizes the data by subtracting the mean and dividing by the standard deviation.
        :param data: The data to be standardized (an NxD matrix).
        :return: The standardized data (an NxD matrix).
        """
        assert len(data.shape) == 2, "The data must be a 2D matrix."

        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)  
        return (data - mean) / std
      
    def fit(self,data):
        """
        Fits the data to the PCA model
        :param data: The data to be fitted (an NxD matrix).
        :return: None
        """
        dimensions = data.shape[1]
        standardized_data = self.__standardize(data)
        self.data = standardized_data
        
        if dimensions <= self.n_components:
            return standardized_data
        
        # covariance_matrix = standardized_data.T.dot(standardized_data)/len(standardized_data)
        U, S, V_T = np.linalg.svd(standardized_data, full_matrices=False)
        self.principal_components = V_T[:self.n_components, :]


    def transform(self):
        """
        Transforms the data to the PCA model
        :param data: The data to be transformed (an NxD matrix).
        :return: The transformed data (an NxC matrix).
        """
        if self.data.shape[1] <= self.n_components:
            return self.data
        assert self.principal_components is not None, "Fit the data first."
        return self.data.dot(self.principal_components.T)    

class GMM():
    def __init__(self, n_components):
        self.n_components = n_components
        self.means = None
        self.covariances = None
        self.priors = None
        self.responsibilities = None

    def __get_log_likelihood(self, data):
        """
        Computes the log likelihood of the data given the parameters of the GMM.
        :param data: The data (an NxD matrix).
        :param means: The means of the K components (a KxD matrix).
        :param covariances: The covariance matrices of the K components (a KxDxD matrix).
        :param priors: The prior probabilities of the K components (a length-K vector).
        :return: The log likelihood of the data given the parameters of the GMM.
        """
        N, D = data.shape
        K = len(self.priors)
        
        log_gaussians = np.array([[np.log(multivariate_normal.pdf(data[j], mean=self.means[i], cov=self.covariances[i])) for i in range(K)] for j in range(N)])
        log_priors = np.log(self.priors).T
        ll = np.sum(np.logaddexp.reduce(log_priors + log_gaussians, axis=1))
        return ll
    
    def fit(self, data, max_iterations=100, tolerance=1e-6):
        """
        Fits the data to a Gaussian Mixture Model using the Expectation-Maximization algorithm
        :param data: The data to be fitted (an NxD matrix).
        :param max_iterations: The maximum number of iterations for convergence.
        :param tolerance: The tolerance for convergence based on log likelihood.
        :return: The parameters of the GMM.
        """
        # initialization
        N, D = data.shape
        self.means = np.random.rand(K, D)
        self.covariances = np.array([np.identity(D)] * K)
        self.priors = np.ones(K) / K
        self.responsibilities = np.zeros((K, N))
        
        prev_log_likelihood = None
        regularization = 1e-6 * np.identity(D)
        
        for iteration in tqdm(range(max_iterations)):
            # E-step
            for j in range(N):
                for i in range(K):
                    probability = multivariate_normal.pdf(data[j], mean=self.means[i], cov=self.covariances[i])
                    self.responsibilities[i, j] = self.priors[i] * probability
                self.responsibilities[:, j] /= np.sum(self.responsibilities[:, j])
            # M-step
            Ni = np.sum(self.responsibilities, axis=1, keepdims=True)    # (K, 1)
            self.priors = Ni.flatten() / N                               # (K,)  
            self.means = self.responsibilities.dot(data) / Ni                 # (K, D)
            for i in range(K):
                diff = data - self.means[i]                              
                self.covariances[i] = (sum(r * np.outer(d, d) for r, d in zip(self.responsibilities[i], diff)) + regularization) / Ni[i]
            
            if iteration % 5 == 0:
                self.plot_for_gif(data, "assets/gif-plots/", str(iteration))
            

            self.log_likelihood = self.__get_log_likelihood(data)
            if prev_log_likelihood is not None and self.log_likelihood - prev_log_likelihood < tolerance:
                print(f"\nConverged after {iteration} iterations.")
                break
            prev_log_likelihood = self.log_likelihood
        return self.means, self.covariances, self.priors, self.log_likelihood, self.responsibilities    

    def plot_for_gif(self, data, path, iteration):
        assert data.shape[1] == 2, "The data must be 2D."
        # if path doesnt exist, create it recursively
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        point_colors = np.zeros((data.shape[0], 3), dtype=np.uint8)

        for i in range(data.shape[0]):
            point_colors[i] = get_color(self.responsibilities[:, i].flatten(), gradient=True)
        
        plt.scatter(data[:,0], data[:,1], s=1, color=point_colors/255)
        min_x, max_x = np.min(data[:,0]), np.max(data[:,0])
        min_y, max_y = np.min(data[:,1]), np.max(data[:,1])
        for i in range(self.n_components):
            mean = self.means[i]
            covariance = self.covariances[i]
            x, y = np.meshgrid(np.linspace(min_x, max_x, 100), np.linspace(min_y, max_y, 100))
            pos = np.dstack((x, y))
            z = multivariate_normal.pdf(pos, mean=mean, cov=covariance)
            plt.contour(x, y, z)
        plt.savefig(path+iteration+".jpg")
        plt.close()

    def plot_2D(self, data, path, filename):
        assert data.shape[1] == 2, "The data must be 2D."

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        point_colors = np.zeros((data.shape[0], 3), dtype=np.uint8)

        for i in range(data.shape[0]):
            point_colors[i] = get_color(self.responsibilities[:, i].flatten(), gradient=True)
        
        plt.scatter(data[:,0], data[:,1], s=1, color=point_colors/255)
        plt.title(f"{filename} gmm with {self.n_components} components")
        plt.savefig(path+filename+".jpg")
        plt.close()
        

def plot(data, save = False, name = "plot.jpg"):
    assert data.shape[1] == 2, "The data must be 2D."
    plt.scatter(data[:,0], data[:,1], s=1, color='black')
    # plt.show()
    if save:
        plt.savefig(name)




def get_color(responsibility, gradient = False):
    colors = np.array([[255, 50, 50], [50, 255, 50], [50, 50, 255], [255, 255, 50], [50, 255, 255], [255, 50, 255], [85, 92,179], [152, 201,110]], dtype=np.uint8)
    if not gradient:
        return colors[np.argmax(responsibility)]
    
    point_color = np.zeros(3, dtype=np.float32)
    for i in range(len(responsibility)):
        point_color += responsibility[i] * colors[i]
    point_color = np.clip(point_color, 0, 255)
    return point_color

if __name__ == "__main__":

    input_files = ["100D.txt",
                    "2D_data_points_1.txt", 
                   "2D_data_points_2.txt", 
                   "3D_data_points.txt", 
                   "6D_data_points.txt"]
    
    for index in [0, 1, 2, 3, 4]:
        filename = input_files[index][:-4]
        data = np.loadtxt('data/'+input_files[index], delimiter=',')

        pca = PCA(n_components=2)
        pca.fit(data)
        reduced_data = pca.transform()
        
        plot(reduced_data, save = True, name = f"assets/plots/{filename}-reduced-plot.jpg")

        best_k, best_ll = 0, -np.inf
        lls = []
        Ks = range(3, 9)
        for K in Ks:
            print(f"Running GMM for {filename} with K = {K}")
            gmm = GMM(n_components=K)
            means, covariances, priors, log_likelihood, responsibilities = gmm.fit(reduced_data)
            lls.append(log_likelihood)
            if log_likelihood > best_ll:
                best_ll = log_likelihood
                best_k = K

            gmm.plot_2D(reduced_data, "assets/plots/", f"{filename}-gmm-{K}")
            os.system(f"convert -delay 100 -loop 0 assets/gif-plots/*.jpg assets/gif/{filename}-gmm-{K}.gif")
            
        print(f"\nBest K for {filename} is {best_k} with log likelihood {best_ll}")
        plt.plot(Ks, lls)
        plt.title(f"{filename} log likelihood")
        plt.savefig(f"assets/log-likelihoods/{filename}-log-likelihood.jpg")
        plt.close()
        