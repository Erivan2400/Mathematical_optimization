from numpy import array, zeros, linalg, dot

def fc(x):
    return (x[0]-2)**2 + (x[1]-3)**2  # coloque a função aqui

def gd(x):
    gd = zeros((N,1))
    for i in range(N):
        x[i] += Dx[i]
        fcx1 = fc(x)
        x[i] -= 2*Dx[i]
        fcx0 = fc(x)
        x[i] += Dx[i]
        gd[i] = (fcx1-fcx0)/(2*Dx[i])
    return gd

def hs(x):
    hs = zeros((N,N))
    for i in range(N):
        x[i] += Dx[i]
        fcx1 = gd(x)
        x[i] -= 2*Dx[i]
        fcx0 = gd(x)
        x[i] += Dx[i]
        hs[0:N,i:i+1] = (fcx1-fcx0)/(2*Dx[i])
    return hs

def NRm(x):
    it=0
    B=1
    x = array(x, dtype=float).reshape(N,1)
    while True:
        x1 = x - dot(linalg.inv(hs(x)), gd(x))
        if max(abs(gd(x))) < 1e-10:
            return x1
            break
        if max(abs(gd(x1))) < max(abs(gd(x))):
           x=x1
           B=1
        else:B=B*0.5
        it += 1
        if it >100:
            return 'Não convergiu'
            break

x = [20, 20] # chute inicial com a quantidade de variáveis da função
Dx = [1e-5, 1e-6] # discretização em cada variável

N = len(Dx)

print(NRm(x))

