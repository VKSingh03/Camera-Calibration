import numpy as np

def gram_schmidt(M):
    Q = np.array(M, dtype=np.float64)
    # for j in range(num_rows-1,-1,-1):
    #     for i in range(j):
    #         Q[i, :] -= np.dot(Q[j, :], Q[i, :]) * Q[j, :]
    #     Q[j, :] /= np.linalg.norm(Q[j, :])
    r3=M[0,:]
    r2=M[1,:]
    r1=M[2,:]
    u1=r1
    e1=(u1/np.linalg.norm(u1))
    u2=r2-(r2@e1)*e1
    e2=(u2/np.linalg.norm(u2))
    u3=r3-(r3@e1)*e1-(r3@e2)*e2
    e3=(u3/np.linalg.norm(u3))
    Q=np.vstack((e3,e2,e1))
    Q=Q.reshape((3,3))
    return Q

def decompose_camera_matrix(P):
    M = P[:,:3]
    # A = P[:,3]
    # T = -np.linalg.inv(M)@A
    U, s, V = np.linalg.svd(P)
    C = V.T[:,3].reshape(4,1)
    Q = gram_schmidt(M)
    K = np.dot(M,np.linalg.inv(Q))
    return Q, K, C

def ProjectionMatrix(wp, cp):
    rows = []
    for i in range(cp.shape[0]):
        row1 = [wp[i][0], wp[i][1], wp[i][2], 1, 0,        0,        0,        0, -cp[i][0]*wp[i][0], -cp[i][0]*wp[i][1], -cp[i][0]*wp[i][2], -cp[i][0]]
        row2 = [0,        0,        0,        0, wp[i][0], wp[i][1], wp[i][2], 1, -cp[i][1]*wp[i][0], -cp[i][1]*wp[i][1], -cp[i][1]*wp[i][2], -cp[i][1]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)
    U, s, V = np.linalg.svd(rows)
    P = V[-1].reshape(3,4)
    # Normalizing P matrix with its last element. 
    P = P/P[2,3]
    return P

if __name__ == "__main__": 
    camera_points = np.array([[757,213,1],[758,415,1],[758,686,1],[759,966,1],[1190,172,1],[329,1041,1],[1204,850,1],[340,159,1]])
    world_points = np.array([[0,0,0,1],[0,3,0,1],[0,7,0,1],[0,11,0,1],[7,1,0,1],[0,11,7,1],[7,9,0,1],[0,1,7,1]]) 

    error_euclidean = []

    P= ProjectionMatrix(world_points,camera_points) 
    print("Projection Matrix: \n", P, "\n\n") 

    R, K, T = decompose_camera_matrix(P) 
    # R = R /R[2,2]

    K=K/K[2,2]

    
    print("Intrinsic Camera Matrix: \n ", K, "\n\n") 
    print("Rotational Matrix: \n ", R, "\n\n") 
    print("Translational Matrix: \n ", T, "\n\n") 
    # T = np.dot(np.linalg.inv(K),(P[:,3]))
    T = T/T[3]
    print("Normalized Translation Matrix: \n ", T, "\n\n")

    Rproj = np.dot(P,world_points.T)
    Rproj = Rproj.T
    for i in range(len(Rproj)):
        Rproj[i]= Rproj[i]/Rproj[i][2]
    error = Rproj - camera_points
    
    for i in error: 
        error_euclidean.append(np.sqrt(i[0]**2+i[1]**2+i[2]**2))
    # error_euclidean = np.mean(error_euclidean)
    error_euclidean = np.array([error_euclidean])
    # print("Reprojection error for each point in sequence [x,y,z]: \n", error[:,:1]/error, "\n\n")

    print("Reprojection error (Distance in pixel units) for each point in sequence: \n", error_euclidean.T, "\n\n")

    mean_error = np.abs(np.mean(error,axis=0))

    print("Mean Error [x, y, z]: \n", mean_error)