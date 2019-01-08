
import numpy as np 
import numpy.matlib


class PCA_parameters(object):
    def __init__(self, X,d,C):
        """
        X : data matrix NXD, N : Data points D : Dimensions must be transposed X.T
        d : Latent dimensionality
        C : Number of mixtures components
        
        """
        self.epsilon = 1e-3
        N,D = np.shape(X)
        # Initialization of mixtures variables
        self.data = X
        self.dim = D
        self.N = N
        self.C = C
        self.d = d
        self.R = np.zeros((N,C))
        # Initialization of mixtures components 
        self.mix = np.ones(C)/C
        # Initialization of means
        p = np.random.permutation(N)[0:C]
        self.means = X[p,:]
        # Initialization of noise variance
        # self.noise_var = (np.var(X))*np.ones((C, D))
        # self.noise_var =np.array( [np.var(X)] * C)
        self.noise_var = np.var(X) * np.ones(C)
        # Initialization of mixtures Weights W 
        self.W = np.random.randn(self.C,D,d)
        self.W_old = np.random.randn(self.C,D,d)
        
        self.M_i = np.random.randn(self.C,d,d)
        self.M_i_inv = np.random.randn(self.C,d,d)
        
        
    # def set_W(self, W):
    #     self.W = W
    # def set_mix(self, Mix):
    #     self.mix = Mix
    # def set_var(self, var):
    #     self.noise_var = var
    # def set_mean(self, mu):
    #     self.means = mu
        
    def get_W(self):
        return self.W  
    def get_mix(self):
        return self.mix 
    def get_var(self):
        return self.noise_var  
    def get_mean(self):
        return self.means  
        
    def E_step(self) :
        C,D,d = np.shape(self.W)
        self.proj = np.zeros((self.C,self.d, self.N))
        self.LogL = np.zeros((self.C,self.N))
        self.temp = np.zeros((self.N,self.C))
        
        for i in range(self.C) :

   #           # Compute M_i
            self.M_i[i,:,:] = self.noise_var[i]*np.eye(self.d) + np.dot(self.W[i,:,:].T, self.W[i,:,:])
            
            # Compute C_inverse_i
            self.M_i_inv[i,:,:] = np.linalg.inv(self.M_i[i,:,:])
            C_inv = np.eye(self.dim) - np.dot(np.dot(self.W[i,:,:], self.M_i_inv[i,:,:]),self.W[i,:,:].T )
            C_inv = C_inv / self.noise_var[i]
            
            # Compute R_ni 
            
            
            print("************* ")
            print(self.M_i[i,:,:])
            print()
            print(C_inv)
            print("|||||||||||||||||| ", i)
            
            
            const = -(self.dim/2)*np.log(2*np.pi) - 0.5*np.linalg.det(np.eye(self.dim) - np.dot(np.dot(self.W[i,:,:], self.M_i_inv[i,:,:]), self.W[i,:,:].T))
            Xm = self.data - np.matlib.repmat(self.means[i,:],self.N,1)
            temp = const - 0.5 * np.diag( np.matmul (Xm , np.matmul(C_inv, Xm.T )) )
            
            print("Temp 1 ")
            print()
            print(temp)
            
            self.temp[:,i] = temp 
            
            
        print()
        print("temp 2 ")
        print(self.temp)
            
        

        # print("*********** || ***************** ")
        # print()
        # print(np.matlib.repmat(self.temp.max(1),1,self.C))
        # print()
        # print(np.matlib.repmat(self.temp.max(1),1,self.C).reshape(s1,s2))
        # print()
        # print(np.matlib.repmat(self.temp.max(1),1,self.C).reshape(s2,s1).T)
        # print("*********** || ***************** ")
        
        # Computation of the Responsability Rni = prior*likelihood / sum(prior*likelihood)
        
        self.temp = self.temp + np.matlib.repmat(self.mix, self.N, 1)
        s1,s2 = np.shape(self.temp)
        
        Q = np.exp(self.temp - np.matlib.repmat(self.temp.max(1),1,self.C).reshape(s2,s1).T)
        denom = np.matlib.repmat(Q.sum(1),1,self.C).reshape(s2,s1).T
        Q = Q / denom
        
        print("Q = ")
        print(Q)
        # Update of mixtures coefficients 
        self.mix =  Q.mean(0)
        

        # M - step :
        for i in range(self.C) :
            
            # Update MEAN 


            numerator = Q[:,i].reshape((self.N,1)) * self.data
            numerator = numerator.sum(0)
            denominator = Q[:,i].sum()
            self.means[i,:] = numerator / denominator
            print("Medel värde total  = ", self.means[i,:] )
            
            
            # 
            # # Compute S_i
            # 


            Xm = self.data - self.means[i,:]
            print()
            print("Data = ", self.data)
            print()
            print("Data - Mean ", Xm)
            
            
            Rm = Q[:,i].reshape((self.N,1)) * Xm
            Rm = Q[:,i] * Xm.T
            print("Resp * X = ", Rm, " shape = ", np.shape(Rm))
            
            print("shape = ", np.shape(Xm))
            S_i = (1/(self.mix[i] * self.N)) * np.matmul(Rm, Xm)
            print("S_i = ", S_i)
            
            
            # Update Weight W
            
            self.W_old[i,:,:] = self.W[i,:,:]
            

            t1 = np.matmul(self.M_i_inv[i,:,:],self.W[i,:,:].T )
            print("t1 = ", t1)
            print()
            t2 = np.matmul(t1, S_i)
            print("t2 = ", t2)
            print("Weight W = ",self.W[i,:,:] )
            t3 = np.matmul(t2, self.W[i,:,:])
            
            
            t4 = self.noise_var[i]*np.eye(self.C) 
            print("t4 = ", t4)
            
            temporary = t3 + t4 

            inverse_cov = np.linalg.inv(temporary)
            sw = np.matmul(S_i,self.W[i,:,:])
            W_new = np.matmul(sw, inverse_cov)
            self.W[i,:,:] = W_new
            
            
            
            # Update sigma  self.W_old
            print("OLD *** Weight W = ",self.W_old[i,:,:] )
            print("NEW *** Weight W = ",self.W[i,:,:] )
            
            A = np.matmul(self.M_i_inv[i,:,:], self.W[i,:,:].T )
            B = np.matmul(self.W_old[i,:,:], A) 
            C = np.matmul(S_i,B) 
            sigma = S_i - C
            sigma = (1/(self.dim)) * np.trace(sigma)
            self.noise_var[i] = sigma
            
            print("Sigma = ", self.noise_var)
            
            

            
            # Rm = np.matmul (Q[:,i].reshape((self.N,1)) , Xm)
            # print(Rm)
            # s1,s2 = np.shape(self.data)
            # #??? mean = np.matlib.repmat(Q[:,i],1,self.dim).reshape(s2,s1).T
            # #mean = np.matlib.repmat(Q[:,i],1,self.dim).reshape(s2,s1).T
            # mean = np.matlib.repmat(Q[:,i],1,self.dim).reshape(s1,s2)
            # evidence = np.matmul(self.data.T, mean)
            # 
            # mean = np.matlib.repmat(Q[:,i],1,self.dim).reshape(s1,s2).T
            # mean = np.matlib.repmat(Q[:,i],1,self.dim).reshape(s2,s1)
            # evidence = np.matmul( mean, self.data)
            # 
            # # print("Evidence = ", evidence.sum(0) / Q.sum())
            # # print("Sum = ",Q.sum() )
            # num = evidence.sum(0)
            # denom = Q[:,i].sum()
            # print("Denom = ", denom)
            # mu = num / denom
            # self.means[i,:] = mu
            # print("New mu = ", mu)
            # 
            # # Update
            # Xm = self.data - np.matlib.repmat(self.means[i,:],self.N,1)
            # print("New X - mean = ", Xm)
            # # S_temp = np.matmul ( np.matmul(np.matlib.repmat(Q[:,i],1,self.dim), Xm.T ),Xm )
            # k = np.dot( Xm ,Xm.T)
            # print("X*X.T = ", k)
            # print("REpmat = ", np.matlib.repmat(Q[:,i],1,self.dim))
            # R = np.matlib.repmat(Q[:,i],1,self.dim).reshape(s2,s1)
            # print("Responsability 1 = ", R)
            # R = np.matlib.repmat(Q[:,i],1,self.dim).reshape(s1,s2).T
            # print("Responsability  = ", R)
            # S_temp =(1/(self.mix[i] * self.N)) * np.matmul ( R, k)
            # 
            # print("S = ", S_temp)
            # S_temp = np.matmul ( R, k)
            # 
            # print("S = ", S_temp)
            
            
            # print("MW == ", np.matmul(self.M_i_inv[i,:,:],self.W[i,:,:].T ))
            # print("S = ", S_temp)
            # t1 = np.matmul(self.M_i_inv[i,:,:],self.W[i,:,:].T )*S_temp
            # print("t1 = ", t1)
            # t3 = self.noise_var[i]*np.eye(self.C) 
            # print("t3 = ", t3)
            # t2 = np.dot(t1, self.W[i,:,:])

            # W_new = np.linalg.inv(self.noise_var[i]*np.eye(self.C) )
#             
# np.matlib.repmat(Q.sum(1),1,self.C).reshape(s2,s1).T
             
        # mean = np.matlib.repmat()
   #      
   # % Update means
   #  for k = 1:K
   #      means(k,:) = sum(X.*repmat(q(:,k),1,D),1)./sum(q(:,k));
   #  end
   #  % update covariances
   #  for k = 1:K
   #      Xm = X - repmat(means(k,:),N,1);
   #      covs(:,:,k) = (Xm.*repmat(q(:,k),1,D))'*Xm;
   #      covs(:,:,k) = covs(:,:,k)./sum(q(:,k));
   #  end

        
        
            
data = [[1,2 ,3 ,5],[ 5, 6 ,9 ,10],[ -2, 3, 4, 11]]
data = [[1,2 ,3 ],[ 5, 6 ,9 ],[ -2, 3, 4]]
data = [[1,2 ,3 ,5,0],[ 5, 6 ,9 ,10,-7],[ -2, 3, 4, 11, -8]]
data = np.array(data)
# data = data.T 

pca_model = PCA_parameters(data,2,2)

print(pca_model.get_W())
print()
print(pca_model.get_mix())
print()
print(pca_model.get_var())
print()
print(pca_model.get_mean())
print()
pca_model.E_step()

print("2 pi = ", pca_model.mix)
