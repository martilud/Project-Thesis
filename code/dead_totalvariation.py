def gradient(p,f,lam):
    grad = np.zeros_like(p)
    for i in range(1,nx-1):
       for j in range(1,ny-1):
            grad[i,j] = lam*(1.0/(hx*hx)*(p[i+1,j] - 2*p[i,j] + p[i-1,j]) + 1.0/(hy*hy)*(p[i,j+1] - 2*p[i,j] + p[i,j-1])) - (0.5/hx*(f[i+1,j] - f[i-1,j]) + 0.5/hy*(f[i,j+1]-f[i,j-1]))
    return grad

def betterGradient(p,f,lam):
    grad = np.zeros_like(p)
    div = divp(p)
    what = np.linalg.norm(f - lam*div)
    for i in range(1,nx-1):
       for j in range(1,ny-1):
            grad[i,j] = -lam*(1.0/(hx*hx)*(p[i+1,j] - 2*p[i,j] + p[i-1,j]) + 1.0/(hy*hy)*(p[i,j+1] - 2*p[i,j] + p[i,j-1]))*what
    return grad

def evenBetterGradient(p,f,lam):
    grad = np.zeros_like(p)
    # x - borders
    # y - borders
    # Inside
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            grad[i,j,0] = lam*((p[i+1,j,0] - 2*p[i,j,0] + p[i-1,j,0]) + \
                (p[i+1,j+1,1] - p[i-1,j+1,1] - p[i+1,j-1,1] + p[i-1,j-1,1])*0.25) - \
                (f[i+1,j] - f[i-1,j])*0.5 
            grad[i,j,1] = lam*((p[i,j+1,1] - 2*p[i,j,1] + p[i,j-1,1]) + \
                (p[i+1,j+1,0] - p[i-1,j+1,0] - p[i+1,j-1,0] + p[i-1,j-1,0])*0.25) - \
                (f[i,j+1] - f[i,j-1])*0.5
    return grad


def divp(p):
    div = np.zeros((nx,ny))
    div[:,0,0] = p[:,0,0]   

    for i in range(1,nx-1):
       for j in range(1,ny-1):
            div[i,j] = (0.5*(p[i+1,j,0] - p[i-1,j,0]) + 0.5*(p[i,j+1,1]-p[i,j-1,1]))
    return div 

def projp(p,dp,tau):
    for i in range(nx):
        for j in range(ny):
            p[i,j,0] = (p[i,j,0] + tau*dp[i,j,0])/(1 + tau*proj)
            p[i,j,1] = (p[i,j,1] + tau*dp[i,j,1])/(1 + tau*proj)
            #p[i,j,0] = (p[i,j,0] + tau*dp[i,j,0])/max(1.0,abs(p[i,j,0] + tau*dp[i,j,0]))
            #p[i,j,1] = (p[i,j,1] + tau*dp[i,j,1])/max(1.0,abs(p[i,j,1] + tau*dp[i,j,1]))
    return p

def projectedGradient(f, lam = 0.1):
    tau = 0.125
    maxiterations = 50
    p = np.zeros((nx,ny,dim))
    for i in range(maxiterations):
        print(i)
        print(nx*20*np.linalg.norm(divp(p)))
        dp = evenBetterGradient(p,f,lam)
        p = projp(p,dp,tau)
        print(p)
    return f - lam*divp(p) 

