
import numpy as np
import gymnasium as gym

def hebbian_update(heb_rule: str,
                       heb_coeffs: np.ndarray,
                       weights: list[np.ndarray],
                       outputs: list[np.ndarray]):
    """
    vectored Hebbian update: weights[z] shape = (out_neurons, in_neurons);
    outputs[z] length = in_neurons, outputs[z+1] length = out_neurons;
    heb_coeffs packed per-connection in row-major of w (col-major in original loops).
    """
    heb_offset = 0
    for z, w in enumerate(weights):
        out_dim, in_dim = w.shape
        block_size = out_dim * in_dim
        coeffs = heb_coeffs[heb_offset:heb_offset+block_size]
        # reshape to per-connection tensors
        if heb_rule == 'A':
            A = coeffs.reshape(out_dim, in_dim)
            delta = A * (outputs[z+1][:, None] * outputs[z][None, :])

        elif heb_rule == 'AD':
            A, D = coeffs.reshape(-1, 2).T
            A = A.reshape(out_dim, in_dim)
            D = D.reshape(out_dim, in_dim)
            delta = A * (outputs[z+1][:, None] * outputs[z][None, :]) + D

        elif heb_rule == 'AD_lr':
            A, D, lr = coeffs.reshape(-1, 3).T
            A, D, lr = [x.reshape(out_dim, in_dim) for x in (A, D, lr)]
            delta = lr * (A * (outputs[z+1][:, None] * outputs[z][None, :]) + D)

        elif heb_rule == 'ABC':
            A, B, C = coeffs.reshape(-1, 3).T
            A, B, C = [x.reshape(out_dim, in_dim) for x in (A, B, C)]
            delta = A * (outputs[z+1][:, None] * outputs[z][None, :]) \
                  + B * outputs[z][None, :] \
                  + C * outputs[z+1][:, None]

        elif heb_rule == 'ABC_lr':
            A, B, C, lr = coeffs.reshape(-1, 4).T
            A, B, C, lr = [x.reshape(out_dim, in_dim) for x in (A, B, C, lr)]
            delta = lr * (A * (outputs[z+1][:, None] * outputs[z][None, :]) \
                         + B * outputs[z][None, :] \
                         + C * outputs[z+1][:, None])

        elif heb_rule == 'ABCD':
            A, B, C, D = coeffs.reshape(-1, 4).T
            A, B, C, D = [x.reshape(out_dim, in_dim) for x in (A, B, C, D)]
            delta = A * (outputs[z+1][:, None] * outputs[z][None, :]) \
                  + B * outputs[z][None, :] \
                  + C * outputs[z+1][:, None] \
                  + D

        elif heb_rule == 'ABCD_lr':
            A, B, C, lr, D = coeffs.reshape(-1, 5).T
            A, B, C, lr, D = [x.reshape(out_dim, in_dim) for x in (A, B, C, lr, D)]
            delta = lr * (A * (outputs[z+1][:, None] * outputs[z][None, :]) \
                         + B * outputs[z][None, :] \
                         + C * outputs[z+1][:, None] \
                         + D)

        elif heb_rule == 'ABCD_lr_D_out':
            A, B, C, lr, D = coeffs.reshape(-1, 5).T
            A, B, C, lr, D = [x.reshape(out_dim, in_dim) for x in (A, B, C, lr, D)]
            delta = lr * (A * (outputs[z+1][:, None] * outputs[z][None, :]) \
                         + B * outputs[z][None, :] \
                         + C * outputs[z+1][:, None]) \
                  + D

        elif heb_rule == 'ABCD_lr_D_in_and_out':
            A, B, C, lr, D_in, D_out = coeffs.reshape(-1, 6).T
            A, B, C, lr, D_in, D_out = [
                x.reshape(out_dim, in_dim)
                for x in (A, B, C, lr, D_in, D_out)
            ]
            delta = lr * (A * (outputs[z+1][:, None] * outputs[z][None, :]) \
                         + B * outputs[z][None, :] \
                         + C * outputs[z+1][:, None] \
                         + D_in) \
                  + D_out

        else:
            raise ValueError(f"Unknown heb_rule '{heb_rule}'")

        # обновляем слой
        w += delta
        heb_offset += block_size

    return tuple(weights)