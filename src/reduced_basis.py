from __init__ import volta_logger
import numpy as np


class ReducedBasis():
    """
    Class for the reduced basis needed for Gappy POD and PPCA-Gaussian Processes

    Fields:
    -------
    base: numpy.ndarray
        The POD basis.
    singular: numpy.ndarray
        The singular values associated to each singular vector in the basis.
    dim: int
        Dimension of the base. It is the original number of snapshots used to build the basis.
    """

    def __init__(self,
                 base = [],
                 singular = [],
                 mean = [],
                 dim = 0):
        self.base = base
        self.singular = singular
        self.dim = dim
        self.mean = mean

    @classmethod
    def from_snapshots(cls,
                       snapshot_matrix,
                       modes = None,
                       centered = False,
                       mean = None,
                       logger = volta_logger):
        """
        Constructor from a snapshot matrix. SVD computed from the snapshot matrix.

        Parameters:
        -----------
        snapshot_matrix: numpy.ndarray
            Snapshot matrix from which the SVD is computed.
        modes: int
            Number of modes in the basis. Defaults to None, meaning that all the modes
            are selected.
        centered: boolean
            Boolean that indicates if the snapshot is centered or not. Defaults to False.
        mean: numpy.ndarray
            Mean of the snapshot matrix. This parameter should be given if the snapshot matrix is centered.
            Defaults to None in order to be consistent with centered = False as default.
        """
        if centered:
            V, S, W = np.linalg.svd(snapshot_matrix,
                                    full_matrices = False)
        else:
            mean = snapshot_matrix.mean(axis = 1).reshape(-1, 1)
            V, S, W = np.linalg.svd(snapshot_matrix - mean,
                                    full_matrices = False)
        if modes is None:
            modes = V.shape[1]
        else:
            V = V[:, :modes]
            S = S[:modes]
        if mean is not None:
            return cls(V, S, mean.reshape(-1,), modes)
        else:
            logger.warning("No mean was provided")
            return cls(V, S, [], modes)

    def number_of_modes(self,
                        exp_variance = 0.95,
                        nb_max = 100):
        """
        Find the number of modes needed to express the desired variance in the data.
        The variance is computed from the singular values s_i :
            v_j = \frac{ \sum_{i=1}^j s_i }{\sum_i s_i}

        Parameters:
        -----------
        variance: float
                The desired expressed variance. Defaults to 0.95.

        Returns:
        n: int
                The number of modes needed to express the desired variance.
        """
        return np.min([nb_max, np.flatnonzero(self.singular.cumsum() / self.singular.sum() >= exp_variance)[0] + 1])

    def reduced_coordinates(self,
                            field,
                            modes = None,
                            centered = False,
                            restriction = None,
                            logger = volta_logger):
        """
        Compute the reduced coordinates of a field relative to the basis.

        Parameters
        ----------
        field : numpy.ndarray
            Physical field. It could be a vector with dimensions (n,) or a matrix with dimensions (n, m).
            When using a restriction, the user can input a 
                - restricted field : dimensions (r,) or (r, m) 
                - whole field : dimensions (n,) or (n,m).
        modes: int
            Number of POD modes to use. Default is the original dimension of the basis.
        centered: boolean
            Boolean that indicates if the mean has been removed or not. Defaults to False.
        restriction: list of int
            Restriction of the base to a number of nodes of the mesh (r nodes).
            Default is to use the whole mesh (n nodes).

        Returns:
        --------
        alpha: numpy.ndarray
            Reduced coordinates of field.
        """
        if not centered:
            if field.ndim == 1:
                field = field - self.mean
            else:
                field = field - self.mean.reshape(-1, 1)
        if modes is None:
            modes = self.dim
        if restriction is None:
            phi = self.base
        else:
            phi = self.base[restriction, :]
            logger.debug("When there is a restriction, the dimension of field must be checked.")
            if field.shape[0] != phi.shape[0]:
                field = field[restriction]
        B = np.linalg.inv(phi[:, :modes].T @ phi[:, :modes])
        alpha = np.linalg.multi_dot([B, phi[:, :modes].T, field])
        return alpha

    def to_original_space(self,
                          alpha,
                          modes = None,
                          restriction = None,
                          logger = volta_logger):
        """
        Bring a point in the reduced space to its original space.

        Parameters
        ----------
        alpha : numpy.array
            Coordinates in the reduced space
        modes: int
            Number of POD modes to use. Default is the original dimension of the basis.
        restriction: list(int)
            Restriction of the base to a number of nodes of the mesh.
            Default is to use the whole mesh.

        Returns
        -------
        resu : numpy.array
            Result in the original space.
        """
        if modes is None:
            modes = self.dim
        if restriction is None:
            phi = self.base
        else:
            phi = self.base[restriction, :]
        if alpha.ndim == 1:
            resu = self.mean + phi[:, :modes] @ alpha
        else:
            logger.debug("We are dealing with a matrix shaped alpha and the mean should be reshaped")
            resu = self.mean.reshape(-1, 1) + phi[:, :modes] @ alpha
        return resu
