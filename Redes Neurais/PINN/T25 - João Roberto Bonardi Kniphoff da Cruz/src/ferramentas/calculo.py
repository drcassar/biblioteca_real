from torch.func import jacrev, jacfwd, vmap


class OperadoresDiferenciais():
    """Guarda os operadores diferenciais"""

    @staticmethod
    def jacobiana(modelo, X):
        def f(x):
            return modelo(x.unsqueeze(0)).squeeze(0) ### Por que isso?

        return vmap(jacrev(f))(X)
    
    @staticmethod
    def hessiana(modelo, X):
        def f(x):
            return modelo(x.unsqueeze(0)).squeeze(0) ### Por que isso?

        return vmap(jacfwd(jacrev(f)))(X)
    
    @staticmethod
    def extrair_derivadas_total(jacob, hess):
        u_t  = jacob[:, 0, 1:2]

        u_tt = hess[:, 0, 1, 1:2]
        u_xx = hess[:, 0, 0, 0:1]

        w_xx = hess[:, 1, 0, 0:1]

        return u_t, u_tt, u_xx, w_xx

    @staticmethod
    def extrair_derivadas_jacob(jacob):
        u_t  = jacob[:, 0, 1:2]

        return u_t
    
    @staticmethod
    def extrair_derivadas_hess(hess):
        u_tt = hess[:, 0, 1, 1:2]
        u_xx = hess[:, 0, 0, 0:1]

        w_xx = hess[:, 1, 0, 0:1]

        return u_tt, u_xx, w_xx