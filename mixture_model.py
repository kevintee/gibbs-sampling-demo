import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

def finite_mixture_model():
    # Generate some data
    k = 3
    mu = np.array([[2,3],[6,4],[4,5]])
    cov = np.array([[[1,0],[0,1]],
                   [[1,1],[-1,1]],
                   [[1,-1],[0,1]]])
    freq = [200, 100, 50]
    colors = ['ro', 'bo', 'go']

    X = np.zeros((0,2))

    for i in range(k):
        vals = np.random.multivariate_normal(mu[i], cov[i], freq[i])
        plt.plot(vals[:,0], vals[:,1], colors[i])
        X = np.vstack([X, vals])
    #plt.show()

    # Constants
    alpha = np.ones(k)/k
    sigma = 1
    rho = 1
    n,d = X.shape

    # Initialize variables
    pi = np.ones(k)/k
    z = np.random.multinomial(1, pi, size=n)
    mu = np.zeros((k, d))

    # Gibbs Sampler
    num_iter = 1000
    burn_in = 200
    samples = []
    for count in range(num_iter):
        print count

        # Update pi
        n_k = np.sum(z.T, axis=1, dtype=float)
        pi = np.random.dirichlet(alpha + n_k, 1)

        # Update mu, rho
        x_hat = np.dot(z.T, X) / n_k[:, None]
        mu_hat = x_hat/(1+sigma/(rho*n_k[:, None]))
        cov_hat = sigma*rho/(sigma+rho*n_k)
        mu = np.array([np.random.multivariate_normal(mu_hat[i], cov_hat[i]*np.eye(d), 1)[0] \
                       for i in range(k)])

        # Update z
        z_hat = np.zeros((n,k))
        for i in range(n):
            for c in range(k):
                z_hat[i,c] = np.prod(st.norm.pdf(X[i], mu[c], rho*np.ones(d)))
        z_hat = (z_hat*pi[:, None])[0]
        for i,x in enumerate(z_hat):
            z[i] = np.random.multinomial(1, x/sum(x))

        samples.append(mu)

    # Remove burn in
    samples = np.array(samples[burn_in:])

    # Find the median of each mu
    medians = np.median(samples, axis=0)
    plt.plot(medians[0,0], medians[0,1], 'ys')
    plt.plot(medians[1,0], medians[1,1], 'ys')
    plt.plot(medians[2,0], medians[2,1], 'ys')
    plt.show()

if __name__ == '__main__':
    finite_mixture_model()
