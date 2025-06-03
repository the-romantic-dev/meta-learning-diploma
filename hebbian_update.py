import numpy as np
import torch


# @sat('Обновление весов')
def hebbian_update(heb_rule: str,
                   heb_coeffs: torch.Tensor,
                   weights: list[torch.Tensor],
                   outputs: list[torch.Tensor]):
    """
    vectored Hebbian update: weights[z] shape = (out_neurons, in_neurons);
    outputs[z] length = in_neurons, outputs[z+1] length = out_neurons;
    heb_coeffs packed per-connection in row-major of w (col-major in original loops).
    """
    updated_weights = []
    offset = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for z, w in enumerate(weights):
        w = w.to(device)
        popsize, out_dim, in_dim = w.shape
        block_size = out_dim * in_dim
        coeffs = heb_coeffs[:, offset:offset + block_size, :].permute(2, 0, 1).to(device)
        coeffs = coeffs.reshape(coeffs.shape[0], coeffs.shape[1], out_dim, in_dim)
        o_in = outputs[z + 1].to(device).unsqueeze(2)
        o_out = outputs[z].to(device).unsqueeze(1)
        # reshape coefficients and compute delta
        if heb_rule == 'A':
            A = coeffs.view(out_dim, in_dim).to(device)
            delta = A * (o_in * o_out)
        elif heb_rule == 'AD':
            A, D = [coeff.view(out_dim, in_dim).to(device) for coeff in coeffs.view(-1, 2).T]
            delta = A * (o_in * o_out) + D
        elif heb_rule == 'AD_lr':
            A, D, lr = [coeff.view(out_dim, in_dim).to(device) for coeff in coeffs.view(-1, 3).T]
            delta = lr * (A * (o_in * o_out) + D)
        elif heb_rule == 'ABC':
            A, B, C = [coeff.view(out_dim, in_dim).to(device) for coeff in coeffs.view(-1, 3).T]
            delta = (A * o_in * o_out + B * o_out + C * o_in)
        elif heb_rule == 'ABC_lr':
            A, B, C, lr = [coeff.view(out_dim, in_dim).to(device) for coeff in coeffs.view(-1, 4).T]
            delta = lr * (A * o_in * o_out + B * o_out + C * o_in)
        elif heb_rule == 'ABCD':
            A, B, C, D = [coeff.view(out_dim, in_dim).to(device) for coeff in coeffs.view(-1, 4).T]
            delta = (A * o_in * o_out + B * o_out + C * o_in + D)
        elif heb_rule == 'ABCD_lr':
            A, B, C, lr, D = coeffs
            delta = lr * (A * o_in * o_out + B * o_out + C * o_in + D)
        elif heb_rule == 'ABCD_lr_D_out':
            A, B, C, lr, D = [coeff.view(out_dim, in_dim).to(device) for coeff in coeffs.view(-1, 5).T]
            delta = (lr * (A * o_in * o_out + B * o_out + C * o_in) + D)
        elif heb_rule == 'ABCD_lr_D_in_and_out':
            A, B, C, lr, D_in, D_out = [coeff.view(out_dim, in_dim).to(device)   for coeff in coeffs.view(-1, 6).T]
            delta = (lr * (A * o_in * o_out + B * o_out + C * o_in + D_in) + D_out)
        else:
            raise ValueError(f"Unknown heb_rule '{hebbian_update.__name__}'")
        # In-place update
        w = w + delta
        updated_weights.append(w)
        offset += block_size

    return updated_weights
