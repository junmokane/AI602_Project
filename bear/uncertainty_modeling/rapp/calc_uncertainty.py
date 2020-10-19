import torch

def uncertainty(data, M, k=5):
    '''
    data : data will be fed to M (batch, ...)
    M : deep ensemble model with dropout=True
    k : iteration for calculating outputs
    '''
    M.eval()
    output_list = []
    for _ in range(k):
        output = M(data)  # [N, 1]
        output_list.append(output)
    output_final = torch.cat(output_list, dim=1) # [N, k]
    output_mean = output_final.mean(dim=1, keepdim=False)  # [N]
    output_var = output_final.var(dim=1, keepdim=False)  # [N]
    return output_mean, output_var


# MakinaRocks
def get_diffs(x, model, batch_size=256):
    model.eval()

    batchified = x.split(batch_size)
    stacked = []
    for _x in batchified:
        model.eval()
        diffs = []
        _x = _x.to(next(model.parameters()).device).float()
        x_tilde = model(_x)
        diffs.append((x_tilde - _x).cpu())

        for layer in model.enc_layer_list:
            _x = layer(_x)
            x_tilde = layer(x_tilde)
            diffs.append((x_tilde - _x).cpu())

        stacked.append(diffs)

    stacked = list(zip(*stacked))
    diffs = [torch.cat(s, dim=0).numpy() for s in stacked]

    return diffs
