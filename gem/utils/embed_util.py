import numpy as np

def reorient(embed1, embed2):
    assert embed1.shape[1] == embed2.shape[1], ('Embedding dimension should be the same for both embeddings')
    n1, d = embed1.shape
    n2, d = embed2.shape
    if(n1 > n2):
        S = np.dot(embed2.T, embed1[0:n2,:])
    else:
        S = np.dot(embed2[0:n1,:].T, embed1)
    u, sig, v = np.linalg.svd(S)
    R = np.dot(u,v)
    reoriented_embed2 = np.dot(embed2, R)
    return reoriented_embed2, R