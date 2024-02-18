import numpy as np
from scipy.misc import derivative


#settings go here
# Two possibilities for the file:
# i) If there is no correlation between x and y (xy_correlation = 0) just specify everything in four columns f2 F2 errorf2 errorF2
# ii) If there is a correlation specify 
infile = "./values150.txt"  
xy_correlation = 1 # Gibt es eine Korrelation zwischen den f2 und F2 Werten? Ja = 1, Nein = 0
relative_deltaF2 = 0.03 # Relativer Messfehler auf y Werte
relative_deltaf20 = 0.03 # Relativer Messfehler auf x Werte
temperature = 150 # Temperatur


# some functions to calculate the data and its covariance
M_oct = 112.24
M_eth = 28.05
Delta_Y = 0.5 # error on yield, always 0.5g

def partial_derivative(func, var=0, point=[]):
    args = point[:]
    def wraps(x):
        args[var] = x
        return func(*args)
    return derivative(wraps, point[var], dx = 1e-5)

def F2(w_oct):
    return (w_oct/M_oct/(w_oct/M_oct + (1.0 - w_oct)/M_eth))

def dF2dw_oct(w_oct):
    return (1.0/(M_eth*M_oct*((w_oct/M_oct - (w_oct-1.0)/M_eth))**2.0))

def f2(w_oct, Y,n0_oct,n0_eth):
    return (n0_oct - w_oct*Y/M_oct)/(n0_oct + n0_eth - w_oct*Y/M_oct-(1.0-w_oct)*Y/M_eth)

def df2dw_oct(w_oct, Y,n0_oct,n0_eth):
    return (partial_derivative(f2,0,[w_oct,Y,n0_oct,n0_eth]))

def df2d_Y(w_oct, Y,n0_oct,n0_eth):
    return (partial_derivative(f2,1,[w_oct,Y,n0_oct,n0_eth]))
    
def get_covariance(w_oct,Y,n0_oct,n0_eth,Delta_w_oct):
    covariance = np.zeros([2, 2])
    covariance[0][0] = (dF2dw_oct(w_oct)*Delta_w_oct)**2.0
    covariance[1][1] = (df2dw_oct(w_oct,Y,n0_oct,n0_eth)*Delta_w_oct)**2.0 + (df2d_Y(w_oct,Y,n0_oct,n0_eth)*Delta_Y)**2.0
    covariance[0][1] = (dF2dw_oct(w_oct)*Delta_w_oct)*(df2dw_oct(w_oct,Y,n0_oct,n0_eth)*Delta_w_oct)
    covariance[1][0] = covariance[0][1]
    return covariance    


data = np.loadtxt(infile)

# define number of measurements
N_measurements = len(data[:,0])

# define data:
X = np.zeros([N_measurements])
Y = np.zeros([N_measurements])
Yerror = np.zeros([N_measurements])
Xerror = np.zeros([N_measurements])

# define covariance
CXX = np.zeros([N_measurements,N_measurements])
CYY = np.zeros([N_measurements,N_measurements])
CXY = np.zeros([N_measurements,N_measurements])


n0eth = np.zeros([N_measurements])
n0oct = np.zeros([N_measurements])
w_oct = np.zeros([N_measurements])
Yield = np.zeros([N_measurements])
Delta_w_oct = np.zeros([N_measurements])
for i in range(N_measurements):
    n0eth[i] = data[i,0]
    n0oct[i] = data[i,1]      
    w_oct[i] = data[i,2] 
    Yield[i] = data[i,3]
    Delta_w_oct[i] = data[i,4]
for i in range(N_measurements):
    X[i] = f2(w_oct[i],Yield[i],n0oct[i],n0eth[i])
    Y[i] = F2(w_oct[i])
    cov = get_covariance(w_oct[i],Yield[i],n0oct[i],n0eth[i],Delta_w_oct[i])
    CYY[i][i] = cov[0][0]  
    CXX[i][i] = cov[1][1]
    CXY[i][i] = cov[0][1]
    Yerror[i] = np.sqrt(cov[0][0]) 
    Xerror[i] = np.sqrt(cov[1][1])

with open("./data.txt", 'w') as paramfile:
    paramfile.write("### independent variable   dependent variable\n")
    for i in range(N_measurements):
        paramfile.write(
            str(X[i]) + " " + str(Y[i]) + "\n")

with open("./X_covariance.txt", 'w') as paramfile:
    for i in range(N_measurements):
        aux_string = ''
        for j in range (N_measurements):
            if(i!=j):
                aux_string += ' 0 '
            else:
                aux_string += str(CXX[i][i])
        paramfile.write(
            str(aux_string + "\n"))

with open("./Y_covariance.txt", 'w') as paramfile:
    for i in range(N_measurements):
        aux_string = ''
        for j in range (N_measurements):
            if(i!=j):
                aux_string += ' 0 '
            else:
                aux_string += str(CYY[i][i])
        paramfile.write(
            str(aux_string + "\n"))

with open("./XY_covariance.txt", 'w') as paramfile:
    for i in range(N_measurements):
        aux_string = ''
        for j in range (N_measurements):
            if(i!=j):
                aux_string += ' 0 '
            else:
                aux_string += str(CXY[i][i])
        paramfile.write(
            str(aux_string + "\n"))
