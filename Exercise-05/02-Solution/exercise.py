# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_plot = np.load('x_plot.npy')
y_plot = np.load('y_plot.npy')

# the data noise is 1
sigma_y = 1

# hyperparameters
n_features = 5    # number of radial basis functions we want to use
lamb = 1e-3       # lambda regularization parameter
# the means of the Radial basis functions
features_means = np.linspace(np.min(x_plot), np.max(x_plot), n_features)   

def rbf(x, mean, sigma) -> float:
    return np.exp(-(np.linalg.norm(x-mean,axis=-1)**2)/(2*(sigma**2)))

def rbf_features(x: np.ndarray, means: np.ndarray, sigma:float) -> np.ndarray:
    """
    :param x: input parameter (shape: [N, d])
    :param means: means of each rbf function (shape: [k, d] (k=num features))
    :param sigma: bandwidth parameter. We use the same for all rbfs here
    :return : returns the radial basis features including the bias value 1 (shape: [N, k+1])
    """
    if len(x.shape) == 1:
        x = x.reshape((-1, 1))

    if len(means.shape) == 1:
        means = means.reshape((-1, 1))
    ############################################################
    # TODO Implement the normalized rbf features
    features: np.ndarray = np.ones((x.shape[0], means.shape[0] + 1))    #[N, k+1]
    N,k = x.shape[0], means.shape[0]
    
    for index, mean in enumerate(means):
        features[:,index] = rbf(x, mean, sigma)
    
    # normalize by dividing each value by the overall sum
    features[:,0:k] /= np.sum(features[:, 0:k])    
    
    ############################################################
    return features

feat_plot = plt.figure("Features")
feat_sigma = 0.6
y_featuers = rbf_features(x_plot, features_means, sigma=feat_sigma)
plt.plot(x_plot, y_featuers[:, :-1])

def posterior_distr(X: np.ndarray, y: np.ndarray, lamb:float, means: np.ndarray, sigma_feat:float):
    """
    :param x: input training data (shape: [N, d])
    :param y: output training data (shape: [N, 1])
    :param lamb: regularization factor (scalar)
    :param means: means of each rbf feature (shape: [k, d])
    :param sigma_feat: bandwidth of the features (scalar)
    :return : returns the posterior mean (shape: [k+1, 1])
                      the posterior covariance (shape: [k+1, k+1]) 
    """
    if len(y.shape) == 1:
        y = y.reshape((-1, 1))
    ############################################################
    # TODO Implement the posterior distribution
    k = means.shape[0]
    Phi = rbf_features(X, means, sigma_feat)
    post_mean = np.linalg.inv(Phi.T @ Phi + (sigma_y**2) * lamb * np.eye(k+1)) @ Phi.T @ y
    post_cov = (sigma_y**2) * np.linalg.inv(Phi.T @ Phi + (sigma_y**2) * lamb * np.eye(k+1))
    ############################################################
    return post_mean, post_cov

def predictive_distr(x: np.ndarray, y: np.ndarray, X: np.ndarray, lamb:float, means: np.ndarray, sigma_feat:float):
    """"
    :param x: input data (shape: [N, d])
    :param y: output training data (shape: [N, 1])
    :param X: input training data (shape: [N, d])
    :param means: means of each rbf feature (shape: [k, d])
    :param sigma_feat: bandwidth of the features (scalar)
    :return : returns the mean (shape: [N, d])
                      the variance (shape: [N])
                      of the predictive distribution
    """
    ############################################################
    # TODO Implement the predictive distribution
    
    Phi = rbf_features(X, means, sigma_feat)
    # post_mean, _ = posterior_distr(X,y, lamb, means, sigma_feat) # we could reuse, maybe?
    k = means.shape[0]
    phi_x = rbf_features(x, means, sigma_feat)
    mean_x = phi_x.dot(np.linalg.inv(Phi.T @ Phi + (sigma_y**2) * lamb * np.eye(k+1)) @ Phi.T @ y)
    var_x = (sigma_y**2) * (1 + phi_x @ np.linalg.inv(Phi.T @ Phi + (sigma_y**2) * lamb * np.eye(k+1)) @ phi_x.T)
    var_x = np.sum(var_x, axis=1) # this is a 100% wrong
    ############################################################
    return mean_x, var_x

def pred_lin_regr( weights: np.ndarray, input_features: np.ndarray):
    """
    :param x: input data (shape: [N, d])
    :param weights: weights for linear regression (shape: [k+1, 1])
    :param input_features: applied features on data to predict on (shape: [N, k+1])
    :return : returns the predictions to the inputs
    """
    return input_features @ weights

# first get the predictive distribution
pred_mean, pred_var = predictive_distr(x_plot, y_train, x_train, lamb=lamb, 
                                       means=features_means, sigma_feat=feat_sigma)

# plot the predictive distribution together with the 95%intervall
plt.figure('Predictve Distr')
plt.plot(x_plot, pred_mean, 'b')
plt.fill_between(np.squeeze(x_plot), np.squeeze(pred_mean)-2*np.sqrt(pred_var), 
                 np.squeeze(pred_mean)+2*np.sqrt(pred_var), alpha=0.2, color='blue')
plt.plot(x_train, y_train, 'or')
plt.plot(x_plot, y_plot, 'black')

# Calculate the posterior distribution for the weights now
post_mean, post_cov = posterior_distr(x_train, y_train, lamb=lamb, means=features_means, 
                                      sigma_feat=feat_sigma)
# sample 10 different models and plot them:
weights = np.random.multivariate_normal(mean=np.squeeze(post_mean), cov=post_cov, size=(10))
example_funcs = np.zeros((weights.shape[0], y_plot.shape[0]))
for i in range(weights.shape[0]):
    example_funcs[i] = pred_lin_regr(weights[i, :], rbf_features(x_plot, features_means, sigma=feat_sigma))
    plt.plot(x_plot, example_funcs[i], 'red', alpha=0.4)

sigma_kern = 1
inv_lamb = 1000 

def get_gaussian_kernel_matrix(x: np.ndarray, sigma, y) -> np.ndarray:
    """ Computes Kernel matrix K(x,y) between two sets of data points x, y  for a Gaussian Kernel with bandwidth sigma.
    If y is not given it is assumed to be equal to x, i.e. K(x,x) is computed
    :param x: matrix containing first set of points (shape: [N, data_dim])
    :param sigma: bandwidth of gaussian kernel
    :param y: matrix containing second set of points (shape: [M, data_dim])
    :return: kernel matrix K(x,y) (shape [M, N])
    """
    if y is None:
        y = x
    ### TODO ######################
    # DONE
    Kernel_matrix = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)**2
    
    Kernel_matrix /= (-2*sigma**2)
    return np.exp(Kernel_matrix)
    
    ###############################

def get_kernel_vec(x_prime: np.ndarray, x: np.ndarray, sigma: float) -> np.ndarray:
    """
    :param x_prime: input data (shape: [N_2 x d])
    :param x: input data (shape: [N_1, d])
    :param sigma: bandwidth of the kernel
    :return: return kernel vector 
            (shape: [N_2 x N_1])
    """
    if len(x_prime.shape) == 1:
        x_prime = x_prime.reshape((-1, 1))

    if len(x.shape) == 1:
        x = x.reshape((-1, 1))
    ############################################################
    # TODO Implement the kernel vector
    # kernel = np.exp(np.linalg.norm(x-x_prime)**2) / (2*(sigma**2))
    
    # use implementation from last exercise:
    kernel = get_gaussian_kernel_matrix(x_prime, sigma, x)
    ############################################################
    return kernel

def get_kernel_mat(X: np.ndarray, sigma: float) -> np.ndarray:
    """
    :param X: training data matrix (N_train, d)
    :sigma: bandwidth of the kernel(scalar)
    :return: the kernel matrix (N_train x N_train)
    """
    return get_kernel_vec(X, X, sigma)

def predictive_distr_gp(x: np.ndarray, y: np.ndarray, X: np.ndarray, sigma_kern:float, inv_lamb:float):
    """"
    :param x: input data (shape: [N_input, d])
    :param y: output training data (shape: [N_train, 1])
    :param X: input training data (shape: [N_train, d])
    :param sigma_kern: bandwidth of the kernel (scalar)
    :param inv_lamb: inverse of lambda (scalar)
    :return : returns the mean (shape: [N_input x 1])
                      the variance (shape: [N_input])
                      of the predictive distribution
    """
    ############################################################
    # TODO Implement the predictive distribution for GPs
    K = get_kernel_mat(X, sigma_kern)
    k_x = get_kernel_vec(x,X,sigma_kern)
    # pred_mean = get_kernel_vec(X,x,sigma_kern) * np.linalg.inv(K + sigma_y**2 * np.eye(K.shape[0])) * y
    
    # solutions
    # pred_mean = k_x.T * np.linalg.inv(K + sigma_y**2 * np.eye(K.shape[0])) * y
    pred_mean = k_x @ np.linalg.inv(K + sigma_y**2 * np.eye(K.shape[0])) @ y
    k_x_x = get_kernel_vec(x,x,sigma_kern)
    
    
    inv = np.linalg.inv(K + (sigma_y**2)*inv_lamb*np.eye(K.shape[0])) # [N_train, N_train]
    pred_var = rbf(x,x,sigma_kern) + sigma_y - k_x.T @ inv @ k_x
    ############################################################
    return pred_mean, pred_var

sigma_kern = 1              # standard deviation of function noise (given)
inv_lamb = 1000             # inverse lambda value -> equivalent to lambda = 1e-3
gp_fig = plt.figure()

# Let's go through the training data and add on training point to the system in each iteration and let's plot
# everything dynamically
x_dyn_train = []
y_dyn_train = []
for i in range(x_train.shape[0]):
    x_dyn_train.append(x_train[i])
    y_dyn_train.append(y_train[i])
    mean, var = predictive_distr_gp(x_plot, np.array(y_dyn_train), np.array(x_dyn_train), sigma_kern, inv_lamb)
    # print(f"Shapes: {mean.shape} and {var.shape}")
    if i % 5 == 0:
        plt.figure(gp_fig.number)
        gp_fig.clf()
        plt.plot(x_plot[:, 0], mean[:, 0])
        plt.fill_between(x_plot[:, 0], mean[:, 0] -2*np.sqrt(var), mean[:,0]+2*np.sqrt(var), 
                         alpha=0.2, edgecolor='r', facecolor='r')
        plt.plot(np.array(x_dyn_train), np.array(y_dyn_train), 'rx')
        plt.title('i='+ str(i))
        plt.pause(0.5)
    elif i == x_train.shape[0]-1:
        plt.figure(gp_fig.number)
        gp_fig.clf()
        plt.plot(x_plot[:, 0], mean[:, 0])
        plt.fill_between(x_plot[:, 0], mean[:, 0] -2*np.sqrt(var), mean[:,0]+2*np.sqrt(var), 
                         alpha=0.2, edgecolor='r', facecolor='r')
        plt.plot(np.array(x_dyn_train), np.array(y_dyn_train), 'rx')
        plt.title('i='+ str(i))
        plt.pause(0.5)

# now let's see the function approximation with all training data and compare to the ground truth function
mean, var = predictive_distr_gp(x_plot, y_train, x_train,sigma_kern, inv_lamb)

plt.figure()
plt.plot(x_plot[:, 0], mean[:, 0])
plt.fill_between(x_plot[:, 0], mean[:, 0] -2*np.sqrt(var), mean[:,0]+2*np.sqrt(var), 
                 alpha=0.2, edgecolor='r', facecolor='r')
plt.plot(np.array(x_train), np.array(y_train), 'rx')
plt.plot(x_plot, y_plot, 'g')

plt.legend(['mean prediction',  'training points', 'gt-function', '2 $\sigma$',])




