import torch


class GradNorm():
    """Guarda o relativo à normalização dos gradiantes"""

    @ staticmethod
    def calcula_gardnorm(rede, loss_fisica, loss_contorno, loss_inicio, lambdas):
        """Realisa o balanceamento dos gradientes de cada perda"""

        grads = []

        for i, loss_i in enumerate([loss_fisica, loss_contorno, loss_inicio]):
            grad = torch.autograd.grad(
                lambdas[i] * loss_i,
                rede.last_layer.parameters(),
                retain_graph=True,
                create_graph=True
            )
            
            grad_norm = torch.norm(torch.stack([
                g.norm() for g in grad
            ]))
            
            grads.append(grad_norm)

        G = torch.stack(grads)

        losses = torch.tensor([
            loss_fisica.item(),
            loss_contorno.item(),
            loss_inicio.item()
        ], device=rede.device)

        r = losses / rede.L0

        G_avg = G.mean()
        target = G_avg * (r ** rede.alpha)

        loss_grad = torch.sum(torch.abs(G - target)) ### Acho que eu posso usar o func_perda

        return loss_grad