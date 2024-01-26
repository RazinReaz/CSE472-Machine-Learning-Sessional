import numpy as np
import matplotlib.pyplot as plt


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

def plot(data, save = False, name = "plot.jpg"):
    assert data.shape[1] == 2, "The data must be 2D."
    plt.scatter(data[:,0], data[:,1], s=1, color='black')
    plt.show()
    if save:
        plt.savefig(name)

def gaussian(x, mean, covariance):
    """
    Computes the probability density function of the multivariate normal distribution.
    :param x: The point at which the density is to be computed (a 1xD vector).
    :param mean: The mean of the multivariate normal distribution (a 1xD vector).
    :param covariance: The covariance matrix of the multivariate normal distribution (a DxD matrix).
    :return: The density at point x.
    """
    D = len(x)
    assert len(mean) == D, "The mean vector must have the same dimensionality as x."
    assert covariance.shape == (D, D), "The covariance matrix must be a square matrix with the same dimensionality as x."

    det = np.linalg.det(covariance)
    inv_covariance = np.linalg.inv(covariance)
    exponent = np.exp(-0.5 * np.dot(np.dot((x - mean), inv_covariance), (x - mean).T))
    return (1.0 / ((2 * np.pi) ** (len(x) / 2) * det ** 0.5)) * exponent

# def log_likelihood(data, means, covariances, weights):
#     num_components = len(means)
#     num_samples = len(data)
    
#     log_likelihood_value = 0.0
#     for i in range(num_samples):
#         likelihood_i = np.sum(weights[j] * gaussian(data[i], means[j], covariances[j]) for j in range(num_components))
#         log_likelihood_value += np.log(likelihood_i)
    
#     return log_likelihood_value

def expectation_maximization(data, K, max_iterations=10, tolerance=1e-6):
    """
    Fits the data to a Gaussian Mixture Model using the Expectation-Maximization algorithm
    :param data: The data to be fitted (an NxD matrix).
    :param K: The number of components in the GMM.
    :param max_iterations: The maximum number of iterations for convergence.
    :param tolerance: The tolerance for convergence based on log likelihood.
    :return: The parameters of the GMM.
    """
    # initialization
    N, D = data.shape
    means = np.random.rand(K, D)
    covariances = np.array([np.identity(D)] * K)
    priors = np.array([1/K] * K)
    responsibilities = np.zeros((K, N))
    
    prev_log_likelihood = None
    
    for iteration in range(max_iterations):
        # E-step
        for j in range(N):
            for i in range(K):
                responsibilities[i, j] = priors[i] * gaussian(data[j], means[i], covariances[i])
            responsibilities[:, j] = responsibilities[:, j] / np.sum(responsibilities[:, j])

        # M-step
        Ni = np.sum(responsibilities, axis=1, keepdims=True)    # (K, 1)
        means = responsibilities.dot(data) / Ni                # (K, D)
        print(f"means \n{means}")
        covariances = np.zeros((K, D, D))
        for i in range(K):
            diff = data - means[i]                              # (N, D)
            for j in range(N):
                covariances[i] += responsibilities[i, j] * np.dot(diff[j].T, diff[j])
            print(f"cov shape of {i}th component\n {covariances[i].shape}")
            print(covariances[i])
            covariances[i] /= Ni[i]
        priors = Ni / N
    # print(means)



def get_color(responsibilities):
    colors = np.array([[255, 0, 0], [0, 0, 255], [0, 255, 255],[255, 255, 0], [0, 255, 0], [255, 0, 255]], dtype=np.uint8)
    # K = len(responsibilities)
    # point_color = np.zeros(3)
    # for i in range(K):
    #     point_color += colors[i] * responsibilities[i]
    point_color = colors[np.argmax(responsibilities)]
    return point_color

if __name__ == "__main__":
    # nparray of colors rgb values
    # data = np.loadtxt("data/2D_data_points_1.txt", delimiter=",")
    # data = np.loadtxt("data/2D_data_points_2.txt", delimiter=",")
    # data = np.loadtxt("data/3D_data_points.txt", delimiter=",")
    data = np.loadtxt("data/6D_data_points.txt", delimiter=",")
    pca = PCA(n_components=2)
    pca.fit(data)
    reduced_data = pca.transform()
    plot(reduced_data, save = True, name = "plots/reduced-data.jpg")

    K=3
    expectation_maximization(reduced_data, K, max_iterations=10)
    # point_colors = np.zeros((reduced_data.shape[0], 3), dtype=np.uint8)
    # for i, x in enumerate(reduced_data):
    #     responsibilities = np.zeros(K)
    #     for j in range(K):
    #         responsibilities[j] = priors[j] * gaussian(x, means[j], covariances[j])
    #     responsibilities = responsibilities / np.sum(responsibilities)
    #     point_colors[i] = get_color(responsibilities)
    # plt.scatter(reduced_data[:,0], reduced_data[:,1], s=1, color=point_colors/255)
    # plt.show()
        
            
        
