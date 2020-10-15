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